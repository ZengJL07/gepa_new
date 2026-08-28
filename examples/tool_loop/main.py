"""GEPA optimization of the initial prompt for a multi-turn tool-feedback task.

Reuses the aime_math skeleton (strategy toggle, TruncationTrackingLM, LM config)
but replaces per-example evaluation with a multi-turn episode: the model emits
XML-style ``<call>``/``<final>`` actions, tools produce local feedback, and the
loop continues under a dual budget (max turns + total tokens including tool
feedback). The optimized component is the initial (system) prompt.

Batch sampling is selected by GEPA_SAMPLING_STRATEGY exactly like aime_math, so
the custom CapabilityTransferUCBSampling plugs in unchanged.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
import requests

from examples.aime_math.main import build_apex_components, is_apex
from examples.aime_math.utils import configure_solver_lm, reset_truncation_flag, truncation_hit
from examples.tool_loop.envs.base import EnvError
from examples.tool_loop.profiles import get_profile
from examples.tool_loop.task_env import Episode, run_env_episode, run_episode
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)
from gepa.strategies.capability_transfer_sampling import CapabilityTransferUCBSampling
from gepa.strategies.proposal_sampling import IndependentSampling

# Selects the task profile: "guess" (synthetic, offline) | "textcraft" | "alfworld".
_TASK = os.environ.get("TOOL_LOOP_TASK", "guess").lower()
_PROFILE = get_profile(_TASK)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_float_opt(name: str) -> float | None:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else None


def _cfg_int(env_name: str, default_key: str) -> int:
    """Resolve an int config: env override wins, else the profile's default."""
    return _env_int(env_name, _PROFILE.defaults[default_key])


# Episode budget (dual): max model turns and max cumulative tokens (incl. tool feedback).
# Profile default, overridable per run via env.
_MAX_TURNS = _cfg_int("TOOL_LOOP_MAX_TURNS", "max_turns")
_MAX_TOKENS = _cfg_int("TOOL_LOOP_MAX_TOKENS", "max_tokens")
_SOLVER_MODEL = os.environ.get("AIME_DEEPSEEK_MODEL", "openai/deepseek-v4-flash")


def _make_generate():
    """Return a ``generate(messages) -> str`` bound to the configured dspy LM."""

    def generate(messages: list[dict[str, str]]) -> str:
        lm = dspy.settings.lm
        out = lm(messages=messages)
        if not out:
            return ""
        first = out[0]
        return first if isinstance(first, str) else str(first)

    return generate


def _run_one(prompt: str, example):
    """Run a single episode and score it. Shared by evaluate() and eval passes."""
    reset_truncation_flag()
    generate = _make_generate()
    if _PROFILE.kind == "answer":
        episode = run_episode(
            generate,
            prompt,
            example,
            max_turns=_MAX_TURNS,
            max_total_tokens=_MAX_TOKENS,
            count_tokens=_token_counter(),
            truncated=truncation_hit,
        )
    else:
        episode = run_env_episode(
            generate,
            prompt,
            _PROFILE.make_session(example),
            max_turns=_MAX_TURNS,
            max_total_tokens=_MAX_TOKENS,
            count_tokens=_token_counter(),
            truncated=truncation_hit,
        )
    return episode, *_PROFILE.scorer(episode, example)


def _run_one_resilient(prompt: str, example):
    """``_run_one``, but an env transport failure scores 0 instead of killing the run.

    Without this, one dropped connection to the env server takes down the entire
    optimization: the exception propagates out of the worker thread, through the
    adapter, and out of ``core.engine.run()`` (raise_on_exception defaults True),
    so the whole iteration's work — and every eval already paid for in it — is
    discarded. A single unlucky ``/step`` should cost one episode, not the run.

    Scoped to env/transport errors on purpose. Bugs in our own scoring or loop
    logic still propagate, because silently scoring those 0 would corrupt the
    search with no signal that anything went wrong.
    """
    try:
        return _run_one(prompt, example)
    except (EnvError, requests.RequestException) as e:
        item = getattr(example, "item_id", None) or getattr(example, "input", "?")
        print(f"[tool_loop] WARNING: env failure on {item}, scoring 0: {type(e).__name__}: {e}")
        episode = Episode(
            messages=[],
            final_answer=None,
            turns_used=0,
            tokens_used=0,
            stop_reason="env_error",
            max_turns=_MAX_TURNS,
            max_total_tokens=_MAX_TOKENS,
        )
        return episode, 0.0, f"The episode could not run: the environment server failed ({type(e).__name__}: {e})."


_COUNTER = None


def _token_counter():
    """Lazily build one litellm counter bound to the solver model, then reuse it."""
    global _COUNTER
    if _COUNTER is None:
        import litellm

        def _count(text: str) -> int:
            try:
                return int(litellm.token_counter(model=_SOLVER_MODEL, text=text or ""))
            except Exception:
                return len((text or "").split())

        _COUNTER = _count
    return _COUNTER


def evaluate(candidate: str, example) -> tuple[float, SideInfo]:
    """GEPA evaluator: run the tool loop with ``candidate`` as the initial prompt."""
    episode, score, feedback = _run_one_resilient(candidate, example)
    return score, {
        "score": score,
        "input": getattr(example, "input", str(example)),
        "output": episode.final_answer or "",
        "trajectory": episode.messages,
        "stop_reason": episode.stop_reason,
        "turns": episode.turns_used,
        "tokens": episode.tokens_used,
        "execution_feedback": feedback,
    }


def _resolve_max_workers() -> int:
    """Episode concurrency: env override, else the task profile default.

    Same resolution order as :func:`main` so the held-out test passes run at the
    concurrency the run was configured with.
    """
    return _env_int("TOOL_LOOP_MAX_WORKERS", _env_int("AIME_MAX_WORKERS", _PROFILE.defaults["max_workers"]))


def evaluate_on_dataset(prompt: str, dataset, on_episode=None, max_workers: int | None = None) -> float:
    """Mean score of ``prompt`` over ``dataset`` (baseline / optimized eval).

    Runs episodes concurrently. This used to be a serial loop, which made the
    post-search test passes the slowest part of a run by far — with TEST_SIZE=500
    and up to 20 LLM round-trips per episode, one prompt meant ~10k sequential
    API calls while the search phase itself was already parallel.

    Concurrency is safe here for the same reason it is during the search: GEPA's
    adapter already evaluates via a thread pool, and the truncation flag the
    scorer reads is thread-local by design (see examples/aime_math/utils.py), so
    each worker sees only its own episode's truncation state.

    ``on_episode(index, example, episode, score, feedback)`` — optional hook,
    used to persist trajectories. It is called as results arrive (so progress
    streams) but serialized under a lock, so implementations need not be
    thread-safe. Arrival order is nondeterministic; callers that persist rows
    should sort by ``index``.
    """
    if not dataset:
        return 0.0

    workers = max_workers if max_workers is not None else _resolve_max_workers()
    workers = max(1, min(workers, len(dataset)))

    if workers == 1:
        total = 0.0
        for i, ex in enumerate(dataset):
            episode, score, feedback = _run_one_resilient(prompt, ex)
            total += score
            if on_episode is not None:
                on_episode(i, ex, episode, score, feedback)
        return total / len(dataset)

    hook_lock = threading.Lock()
    scores: list[float] = [0.0] * len(dataset)

    def _one(index: int, example):
        episode, score, feedback = _run_one_resilient(prompt, example)
        scores[index] = score
        if on_episode is not None:
            with hook_lock:
                on_episode(index, example, episode, score, feedback)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, i, ex) for i, ex in enumerate(dataset)]
        for future in as_completed(futures):
            # Re-raise here rather than silently scoring 0: a crash in the final
            # eval is a real failure, not a wrong answer.
            future.result()

    return sum(scores) / len(dataset)


def _configure_reflection_cache(cache_dir: str) -> None:
    """Put GEPA's reflection LM behind a disk cache so a re-run replays.

    GEPA's reflection LM is :class:`gepa.lm.LM`, which calls
    ``litellm.completion`` / ``litellm.batch_completion`` directly and passes no
    ``caching`` kwarg — dspy's cache does not see it. A *global* ``litellm.cache``
    does intercept both paths (verified), keyed on the full request (model +
    messages + sampling params), which is exactly the replay semantics we want:
    same reflective dataset in, same candidate text out.

    Best-effort: a cache is an optimization, so any failure here degrades to an
    uncached (still correct, just slower and paid-for) run.
    """
    try:
        import litellm
        from litellm.caching import Cache

        os.makedirs(cache_dir, exist_ok=True)
        litellm.cache = Cache(type="disk", disk_cache_dir=cache_dir)
        litellm.enable_cache()
        print(f"[tool_loop] reflection LM cache -> {cache_dir}")
    except Exception as e:  # noqa: BLE001 — never fail a run over a cache
        print(f"[tool_loop] WARNING: reflection LM cache disabled ({type(e).__name__}: {e})")


def _salvage_best_prompt(run_dir: str, fallback: str) -> str:
    """Best candidate GEPA had persisted before a crash, else ``fallback``.

    GEPA core writes ``candidates.json`` (all accepted candidates, best last) at
    every iteration boundary, so a crashed run still has its search results on
    disk. Reading them back is the difference between losing an entire run and
    losing only the iteration that failed.
    """
    path = os.path.join(run_dir, "candidates.json")
    try:
        with open(path, encoding="utf-8") as f:
            candidates = json.load(f)
        if not candidates:
            return fallback
        last = candidates[-1]
        # Candidates are {component: text}; this task optimizes a single component.
        if isinstance(last, dict):
            return next(iter(last.values()), fallback)
        return str(last)
    except (OSError, ValueError, StopIteration):
        return fallback


def _write_json(run_dir: str, name: str, payload) -> None:
    """Write one artifact into ``run_dir``, warning (not raising) on failure.

    A finished optimization run is expensive; a bad path or a value that will not
    serialize must never lose the result, so persistence failures degrade to a
    warning on stdout.
    """
    path = os.path.join(run_dir, name)
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if name.endswith(".json"):
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            else:
                f.write(payload)
        print(f"[tool_loop] wrote {path}")
    except OSError as e:
        print(f"[tool_loop] WARNING: could not write {path}: {e}")


def _episode_collector() -> tuple[list, object]:
    """Return ``(rows, on_episode)`` — the list and the hook that fills it.

    Held-out test episodes are scored outside GEPA's budget, so nothing in the
    engine's own run_dir records them. Without this the final baseline/optimized
    numbers land on stdout only and the per-episode detail behind them is lost.

    ``rows`` is appended in completion order, which is nondeterministic under
    concurrent evaluation — sort by ``index`` before persisting so the artifact
    is diffable across runs.
    """
    rows: list = []

    def collect(i, ex, episode, score, feedback):
        rows.append(
            {
                "index": i,
                "item_id": getattr(ex, "item_id", None),
                "env_index": getattr(ex, "env_index", None),
                "score": score,
                "stop_reason": episode.stop_reason,
                "reward": episode.reward,
                "env_done": episode.env_done,
                "turns_used": episode.turns_used,
                "max_turns": episode.max_turns,
                "tokens_used": episode.tokens_used,
                "max_total_tokens": episode.max_total_tokens,
                "tool_calls": episode.tool_calls,
                "format_errors": episode.format_errors,
                "feedback": feedback,
                "trace": episode.trace,
                "messages": episode.messages,
            }
        )

    return rows, collect


def build_sampling_strategy(minibatch_size: int, n_parallel: int, seed: int):
    """Identical toggle to aime_math: baseline IndependentSampling vs our method."""
    strategy = os.environ.get("GEPA_SAMPLING_STRATEGY", "baseline").lower()
    if strategy in ("baseline", "independent", "default"):
        return None if n_parallel <= 1 else IndependentSampling(n=n_parallel)
    if strategy in ("capability_transfer", "captransfer", "ct"):
        return CapabilityTransferUCBSampling(
            n=n_parallel,
            minibatch_size=minibatch_size,
            tau=_env_float("GEPA_CT_TAU", 0.5),
            alpha=_env_float("GEPA_CT_ALPHA", 1.0),
            beta=_env_float("GEPA_CT_BETA", 1.0),
            exploration_weight=_env_float("GEPA_CT_LAMBDA", 0.2),
            cold_start_bonus=_env_float("GEPA_CT_COLD_START_BONUS", 0.2),
            seed=seed,
        )
    if strategy in ("apex",):
        # Built in main() by build_apex_components: APEX also installs an
        # evaluation policy, a candidate selector and a reflection strategy.
        return None
    raise ValueError(
        f"Unknown GEPA_SAMPLING_STRATEGY={strategy!r}; expected 'baseline', 'capability_transfer' or 'apex'."
    )


def main():
    initial_prompt = os.environ.get("TOOL_LOOP_INITIAL_PROMPT", _PROFILE.seed_prompt)

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    reflection_model = os.environ.get("AIME_REFLECTION_MODEL", "openai/deepseek-v4-pro")

    seed = _env_int("AIME_SEED", 0)
    max_metric_calls = _env_int("AIME_MAX_METRIC_CALLS", 40)
    # Shared with the held-out test passes, so search and final eval always run
    # at the same concurrency (see _resolve_max_workers for the override order).
    max_workers = _resolve_max_workers()
    minibatch_size = _env_int("AIME_REFLECTION_MINIBATCH_SIZE", 3)
    n_parallel = _env_int("AIME_NUM_PARALLEL_PROPOSALS", 1)
    solver_max_tokens = _env_int("AIME_SOLVER_MAX_TOKENS", 2000)
    solver_temperature = _env_float("AIME_SOLVER_TEMPERATURE", 1.0)
    reflection_temperature = _env_float_opt("AIME_REFLECTION_TEMPERATURE")

    cache_dir = os.environ.get("AIME_CACHE_DIR")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Solver LM (dspy-managed) — every per-turn episode call.
        dspy.configure_cache(disk_cache_dir=os.path.join(cache_dir, "solver"))
        # Reflection LM. NOT covered by dspy's cache: GEPA's reflection LM is
        # gepa.lm.LM calling litellm directly, so it bypasses dspy entirely.
        # Leaving it uncached breaks replay for the whole run, not just itself:
        # the reflector samples a DIFFERENT candidate text each time, which
        # changes the candidate hash, which misses the fitness cache, which
        # re-runs the episodes and misses the solver cache. Caching this one
        # call is what makes the other two caches effective on a re-run.
        _configure_reflection_cache(os.path.join(cache_dir, "reflection"))

    configure_solver_lm(_SOLVER_MODEL, api_key, api_base, max_tokens=solver_max_tokens, temperature=solver_temperature)

    train_n = _cfg_int("TOOL_LOOP_TRAIN_N", "train_n")
    val_n = _cfg_int("TOOL_LOOP_VAL_N", "val_n")
    test_n = _cfg_int("TOOL_LOOP_TEST_N", "test_n")
    trainset, valset, testset = _PROFILE.load_splits(train_n=train_n, val_n=val_n, test_n=test_n, seed=seed)

    sampling_strategy = build_sampling_strategy(minibatch_size, n_parallel, seed)

    reflection_lm_kwargs = {"api_key": api_key, "api_base": api_base}
    if reflection_temperature is not None:
        reflection_lm_kwargs["temperature"] = reflection_temperature

    # APEX uses a SINGLE development set D for both mutation and selection: its
    # nine buckets intersect tiers derived from the history H with outcomes on
    # the same examples, so a disjoint train/val split leaves them ill-defined.
    # Merge train+val into D and pass valset=None (GEPA then reuses the
    # trainset); the test split stays held out, so total data is unchanged.
    #
    # This task binarizes cleanly: score_env_episode returns 1.0 iff the env task
    # was solved and 0.0 otherwise, so Section 3.1's "only a perfect score yields
    # a pass" is exact here rather than an approximation of a continuous metric.
    apex = None
    engine_extra: dict = {"sampling_strategy": sampling_strategy}
    reflection_extra: dict = {}
    optimize_valset = valset
    if is_apex():
        from gepa.gepa_launcher import make_litellm_lm

        dataset_d = list(trainset) + list(valset)
        optimize_valset = None
        apex = build_apex_components(
            dataset_size=len(dataset_d),
            minibatch_size=minibatch_size,
            n_parallel=n_parallel,
            reflection_lm=make_litellm_lm(reflection_model, **reflection_lm_kwargs),
        )
        trainset = dataset_d
        engine_extra = {
            "sampling_strategy": apex["sampling_strategy"],
            "val_evaluation_policy": apex["val_evaluation_policy"],
            "candidate_selection_strategy": apex["candidate_selection_strategy"],
            "acceptance_criterion": apex["acceptance_criterion"],
        }
        reflection_extra = {
            "reflection_strategy": apex["reflection_strategy"],
            "perfect_score": apex["perfect_score"],
        }

    print(
        f"[tool_loop] task={_TASK} strategy={os.environ.get('GEPA_SAMPLING_STRATEGY', 'baseline')} "
        f"n_parallel={n_parallel} minibatch_size={minibatch_size} seed={seed} "
        f"max_metric_calls={max_metric_calls} max_turns={_MAX_TURNS} max_tokens={_MAX_TOKENS} "
        f"max_workers={max_workers} "
        + (
            f"|D|={len(trainset)} (train+val merged) N_eval={apex['n_eval']} test={len(testset)}"
            if apex is not None
            else f"train/val/test={len(trainset)}/{len(valset)}/{len(testset)}"
        )
    )

    run_dir = os.environ.get("AIME_RUN_DIR", f"outputs/tool_loop_{_TASK}")
    # write_agent_state also registers ReflectiveDatasetDumpCallback, which
    # persists iterations/<id>/reflective_dataset.json — the exact <side_info>
    # the reflection LM was shown. Nothing else records it, so without this the
    # feedback actually driving the search is unrecoverable after the run.
    write_agent_state = os.environ.get("TOOL_LOOP_WRITE_AGENT_STATE", "true").lower() in ("1", "true", "yes")

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=run_dir,
            seed=seed,
            max_metric_calls=max_metric_calls,
            track_best_outputs=True,
            parallel=True,
            max_workers=max_workers,
            cache_evaluation=True,
            write_agent_state=write_agent_state,
            **engine_extra,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_model,
            reflection_lm_kwargs=reflection_lm_kwargs,
            reflection_minibatch_size=minibatch_size,
            **reflection_extra,
        ),
    )

    # Config snapshot, written BEFORE the run so a crashed or killed run still
    # leaves a record of what it was trying to do.
    run_config = {
        "task": _TASK,
        "strategy": os.environ.get("GEPA_SAMPLING_STRATEGY", "baseline"),
        "seed": seed,
        "max_metric_calls": max_metric_calls,
        "max_workers": max_workers,
        "n_parallel_proposals": n_parallel,
        "reflection_minibatch_size": minibatch_size,
        "episode_max_turns": _MAX_TURNS,
        "episode_max_tokens": _MAX_TOKENS,
        "solver_model": _SOLVER_MODEL,
        "solver_max_tokens": solver_max_tokens,
        "solver_temperature": solver_temperature,
        "reflection_model": reflection_model,
        "reflection_temperature": reflection_temperature,
        "api_base": api_base,
        "split_requested": {"train_n": train_n, "val_n": val_n, "test_n": test_n},
        "initial_prompt": initial_prompt,
    }
    if apex is not None:
        # Under APEX trainset IS D (train+val merged) and there is no separate
        # valset, so reporting train/val separately would double-count.
        run_config["split_sizes"] = {"D": len(trainset), "test": len(testset)}
        run_config["apex"] = {
            "n_eval": apex["n_eval"],
            "lookback": apex["sampling_strategy"].lookback,
            "alpha_0": _env_float("GEPA_APEX_ALPHA0", 0.2),
            "beta": _env_float("GEPA_APEX_BETA", 0.03),
            "perfect_score": apex["perfect_score"],
            "minibatch_size": apex["sampling_strategy"].minibatch_size,
        }
    else:
        run_config["split_sizes"] = {"train": len(trainset), "val": len(valset), "test": len(testset)}
    if _TASK != "guess":
        item_ids: dict[str, list] = {"test": [getattr(e, "item_id", None) for e in testset]}
        if apex is not None:
            item_ids["D"] = [getattr(e, "item_id", None) for e in trainset]
        else:
            item_ids["train"] = [getattr(e, "item_id", None) for e in trainset]
            item_ids["val"] = [getattr(e, "item_id", None) for e in valset]
        run_config["item_ids"] = item_ids
    _write_json(run_dir, "run_config.json", run_config)

    started = time.time()
    try:
        result = optimize_anything(
            seed_candidate=initial_prompt,
            evaluator=evaluate,
            dataset=trainset,
            valset=optimize_valset,
            config=gepa_config,
        )
    except BaseException as e:
        # Salvage whatever the search produced before dying. GEPA persists its
        # state at every iteration boundary, so the best candidate found so far is
        # recoverable — but only if we look. Previously an exception here skipped
        # every _write_json below, so a run that crashed in iteration 3 left no
        # summary, no scores, and no best_prompt: hours of paid-for search lost to
        # one dropped socket.
        elapsed = time.time() - started
        print(f"\n[tool_loop] run FAILED after {elapsed:.0f}s: {type(e).__name__}: {e}")
        salvaged = _salvage_best_prompt(run_dir, initial_prompt)
        _write_json(run_dir, "best_prompt.txt", salvaged)
        _write_json(
            run_dir,
            "summary.json",
            {
                **run_config,
                "elapsed_seconds": round(elapsed, 1),
                "failed": True,
                "error": f"{type(e).__name__}: {e}",
                "best_prompt": salvaged,
                "note": "Search crashed; best_prompt salvaged from candidates.json. No test scores.",
            },
        )
        raise
    elapsed = time.time() - started

    # Under APEX the run's answer is P_curr (Algorithm 1 line 20), not
    # GEPAResult.best_idx. The latter is max(val_aggregate_scores), i.e. the
    # argmax over each candidate's average on *its own* coverage -- and with
    # subset evaluation those coverages are different, deliberately easier
    # samples of D (Eq. 10 favors B[M,1]/B[M,0] and never samples B[H,0]), so
    # they cannot be ranked against each other. P_curr instead only ever moved
    # by winning a comparison on one shared D_eval.
    apex_p_curr_idx = None
    if apex is not None:
        apex_p_curr_idx = apex["val_evaluation_policy"].current_best_idx
        best_prompt = result.candidates[apex_p_curr_idx]
        if isinstance(best_prompt, dict) and len(best_prompt) == 1:
            best_prompt = next(iter(best_prompt.values()))
    else:
        best_prompt = result.best_candidate
    # Written before the test evals: the optimized prompt is the run's primary
    # artifact and must survive even if the held-out eval then fails. This is
    # also the file the eval_* scripts read via TOOL_LOOP_OPTIMIZED_PROMPT_FILE.
    _write_json(run_dir, "best_prompt.txt", best_prompt)

    if os.environ.get("AIME_SKIP_BASELINE_EVAL", "false").lower() in ("1", "true", "yes"):
        baseline_score = None
    else:
        print("\nEvaluating Baseline (Initial Prompt)...")
        baseline_rows, baseline_collect = _episode_collector()
        baseline_score = evaluate_on_dataset(initial_prompt, testset, on_episode=baseline_collect)
        # Sorted: concurrent evaluation appends in completion order.
        _write_json(run_dir, "test_episodes_baseline.json", sorted(baseline_rows, key=lambda r: r["index"]))

    print("\nEvaluating Best Optimized Program...")
    print(f"Best Prompt Found:\n{best_prompt}")

    optimized_rows, optimized_collect = _episode_collector()
    optimized_score = evaluate_on_dataset(best_prompt, testset, on_episode=optimized_collect)
    _write_json(run_dir, "test_episodes_optimized.json", sorted(optimized_rows, key=lambda r: r["index"]))

    if baseline_score is not None:
        print(f"Baseline Score: {baseline_score:.2%}")
    print(f"Optimized Score: {optimized_score:.2%}")
    if baseline_score is not None:
        print(f"Improvement: {optimized_score - baseline_score:.2%}")

    # Field names differ by result shape: passing a legacy GEPAConfig (as we do)
    # returns a GEPAResult (total_metric_calls / val_aggregate_scores[best_idx]),
    # whereas an OptimizeAnythingConfig returns a Result (total_evals /
    # best_score). Try both so the summary is never silently None.
    total_evals = getattr(result, "total_evals", None)
    if total_evals is None:
        total_evals = getattr(result, "total_metric_calls", None)
    if apex_p_curr_idx is not None:
        # Report P_curr's own score so best_prompt and best_val_score describe
        # the same candidate. Still an average over that candidate's coverage,
        # so read it as "P_curr on the ids it was scored on", not as an estimate
        # over all of D.
        try:
            best_val_score = result.val_aggregate_scores[apex_p_curr_idx]
        except (AttributeError, IndexError, TypeError):
            best_val_score = None
    else:
        best_val_score = getattr(result, "best_score", None)
        if best_val_score is None:
            try:
                best_val_score = result.val_aggregate_scores[result.best_idx]
            except (AttributeError, IndexError, TypeError):
                best_val_score = None

    apex_diagnostics = {}
    if apex is not None:
        policy = apex["val_evaluation_policy"]
        apex_diagnostics = {
            "apex_p_curr_idx": apex_p_curr_idx,
            "apex_alpha_final": policy.alpha,
            # How often D_eval came out empty and fell back to all of D -- the
            # paper's lines 12-13 never sample B[H,0], so a lineage that fails
            # everything leaves nothing to select. Each occurrence costs |D|
            # metric calls instead of N.
            "apex_degenerate_fallbacks": policy.degenerate_fallbacks,
            # GEPAResult's own pick, kept for comparison: a large gap between
            # this and apex_p_curr_idx means absolute per-candidate averages were
            # indeed ranking easier subsets higher.
            "gepa_best_idx_for_reference": getattr(result, "best_idx", None),
        }

    _write_json(
        run_dir,
        "summary.json",
        {
            **run_config,
            "elapsed_seconds": round(elapsed, 1),
            "total_evals": total_evals,
            "best_val_score": best_val_score,
            "best_prompt": best_prompt,
            "test_baseline_score": baseline_score,
            "test_optimized_score": optimized_score,
            "test_improvement": None if baseline_score is None else optimized_score - baseline_score,
            **apex_diagnostics,
        },
    )


if __name__ == "__main__":
    main()
