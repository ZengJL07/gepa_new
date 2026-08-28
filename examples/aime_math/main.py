import os
from types import SimpleNamespace

import dspy

from examples.aime_math.utils import (
    build_side_info,
    configure_solver_lm,
    evaluate_on_dataset,
    load_math_dataset,
    math_metric,
    reset_truncation_flag,
    run_llm,
    split_sizes_from_env,
    truncation_hit,
)

# Stand-in prediction for the failure path, where no prediction object exists but
# side_info still has to carry the example's image/annotation context.
_EMPTY_PREDICTION = SimpleNamespace(answer="", reasoning="")
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)
from gepa.strategies.acceptance import AlwaysAcceptance
from gepa.strategies.apex_candidate_selector import ApexCurrentBestSelector
from gepa.strategies.apex_eval_policy import ApexRankSensitivePolicy
from gepa.strategies.apex_reflection import ApexTwoStepReflection
from gepa.strategies.apex_sampling import ApexDynamicSampling
from gepa.strategies.apex_stratification import RejectedHistoryTracker
from gepa.strategies.capability_transfer_sampling import CapabilityTransferUCBSampling
from gepa.strategies.proposal_sampling import IndependentSampling


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_float_opt(name: str) -> float | None:
    """Read an optional float env var; None when unset (so we can preserve a
    provider's own default instead of forcing a value)."""
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else None


def evaluate(candidate: str, example) -> tuple[float, SideInfo]:
    """Evaluate a candidate on a single example.

    Reasoning models can exhaust ``max_tokens`` inside their hidden reasoning
    and return an empty answer, which makes dspy's adapter raise
    ``AdapterParseError``. A prompt that provokes that is a bad prompt, so we
    score it 0 and feed the failure back to the optimizer.

    Config/infrastructure errors (bad request, auth, model-not-found) are NOT
    a prompt problem — scoring them 0 would silently zero out the whole run
    (as happened when max_tokens=32000 exceeded a provider's 8192 ceiling). We
    re-raise those so the run fails loudly instead.
    """
    reset_truncation_flag()
    try:
        prediction = run_llm(example, candidate)
    except Exception as e:
        msg = str(e)
        ename = type(e).__name__
        # Genuine config/infrastructure errors are NOT a prompt problem; scoring
        # them 0 would silently zero out the whole run (as happened when
        # max_tokens=32000 exceeded a provider's 8192 ceiling). Match the
        # specific 400 signatures — a max_tokens *range* rejection, auth, and
        # model-not-found — and re-raise so the run fails loudly.
        config_markers = (
            "Range of max_tokens",  # e.g. "Range of max_tokens should be [1, 8192]"
            "max_tokens is too large",
            "AuthenticationError",
            "invalid_api_key",
            "model_not_found",
            "NotFoundError",
        )
        if any(m in ename or m in msg for m in config_markers):
            raise
        # Everything else — AdapterParseError on an empty/null response, a
        # response truncated at max_tokens, un-serializable output — means the
        # model burned its token budget (often runaway reasoning) without
        # emitting a final answer. That is a bad prompt: score it 0 and tell the
        # optimizer to steer the model toward a concise, terminated answer.
        return 0.0, build_side_info(
            example,
            _EMPTY_PREDICTION,
            0.0,
            (
                f"The model failed to return a parseable answer ({ename}: {e}). "
                "This usually means the response hit its max_tokens limit — often from "
                "runaway reasoning — before producing a final answer. Make the "
                "prompt guide the model to reason concisely and always end with the "
                "required final answer."
            ),
        )

    # A response truncated at max_tokens (finish_reason == "length") is scored 0
    # even if a partial answer parsed: hitting the ceiling signals runaway
    # reasoning, and any integer that survived truncation is not a reliable
    # final answer.
    if truncation_hit():
        return 0.0, build_side_info(
            example,
            prediction,
            0.0,
            (
                "The model's response was truncated at the max_tokens limit — likely "
                "runaway reasoning — so its answer is unreliable and scored 0. Make the "
                "prompt guide the model to reason concisely and end with the required "
                "final answer."
            ),
        )

    score, feedback = math_metric(example, prediction)

    return score, build_side_info(example, prediction, score, feedback)


def build_sampling_strategy(minibatch_size: int, n_parallel: int, seed: int):
    """Pick the batch-sampling strategy from GEPA_SAMPLING_STRATEGY.

    - "baseline"          -> IndependentSampling(n_parallel): stock GEPA, each
      task picks its own parent and a random epoch-shuffled minibatch. When
      n_parallel == 1 we return None so GEPA uses its default
      SingleMutationSampling (identical behavior, no wrapper).
    - "capability_transfer" -> CapabilityTransferUCBSampling: our method.
    - "apex" is handled by build_apex_components instead, since it also
      installs an evaluation policy, a candidate selector and a reflection
      strategy (see that function).
    """
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
            usability_weight=_env_float("GEPA_CT_USABILITY_WEIGHT", 1.0),
            alpha_u=_env_float("GEPA_CT_U_ALPHA", 1.0),
            beta_u=_env_float("GEPA_CT_U_BETA", 1.0),
            seed=seed,
        )
    if strategy in ("apex",):
        return None  # built separately; see build_apex_components
    raise ValueError(
        f"Unknown GEPA_SAMPLING_STRATEGY={strategy!r}; expected 'baseline', 'capability_transfer' or 'apex'."
    )


def is_apex() -> bool:
    return os.environ.get("GEPA_SAMPLING_STRATEGY", "baseline").lower() == "apex"


def build_apex_components(dataset_size: int, minibatch_size: int, n_parallel: int, reflection_lm):
    """Build the four APEX components (arXiv:2606.11459v1).

    APEX replaces four GEPA strategies at once, all sharing one evaluation
    history ``H`` as the paper requires:

    * ``ApexDynamicSampling``      -- Section 4.2, trajectory-guided mutation
    * ``ApexRankSensitivePolicy``  -- Section 4.3, rank-sensitive evaluation
    * ``ApexCurrentBestSelector``  -- Algorithm 1 lines 16-17, hill-climbing
    * ``ApexTwoStepReflection``    -- Appendix C, Critique -> Mutate

    ``GEPA_APEX_N_EVAL`` defaults to 18% of ``|D|``, preserving the paper's
    ratio (N=100 against |D| of 500-700) rather than its absolute value: with
    ``N >= |D|`` the rank-sensitive policy degrades to full evaluation and
    Section 4.3 stops doing anything.

    Requires ``valset=None`` so trainset and valset are the same ``D``; the nine
    buckets need tiers and current outcomes over the same examples.
    """
    perfect_score = _env_float("GEPA_APEX_PERFECT_SCORE", 1.0)
    lookback = _env_int("GEPA_APEX_LOOKBACK", 5)
    default_n_eval = max(1, round(0.18 * dataset_size))
    n_eval = _env_int("GEPA_APEX_N_EVAL", default_n_eval)

    # One shared H: the mutation and selection stages must stratify identically.
    history = RejectedHistoryTracker(perfect_score=perfect_score)

    policy = ApexRankSensitivePolicy(
        n_eval=n_eval,
        alpha_0=_env_float("GEPA_APEX_ALPHA0", 0.2),
        beta=_env_float("GEPA_APEX_BETA", 0.03),
        lookback=lookback,
        perfect_score=perfect_score,
        history=history,
    )
    sampling = ApexDynamicSampling(
        n=n_parallel,
        minibatch_size=minibatch_size,
        lookback=lookback,
        perfect_score=perfect_score,
        history=history,
    )
    return {
        "sampling_strategy": sampling,
        "val_evaluation_policy": policy,
        "candidate_selection_strategy": ApexCurrentBestSelector(policy),
        "reflection_strategy": ApexTwoStepReflection(reflection_lm),
        # Algorithm 1 evaluates P_new exactly once, on D_eval (line 15), and the
        # decision is line 16. The error batch E of line 5 feeds Critique only
        # (line 6) -- it is never scored against the parent. GEPA's default
        # minibatch gate would add a second, earlier decision on those m=3
        # examples, all of which are by construction ones P_curr already fails,
        # so the parent's sum is ~0 and the gate reduces to "solve at least one
        # of three hard cases" -- noisy, and it discards candidates before the
        # D_eval comparison can judge them.
        "acceptance_criterion": AlwaysAcceptance(),
        "perfect_score": perfect_score,
        "n_eval": n_eval,
    }


def main():
    initial_prompt = os.environ.get(
        "AIME_INITIAL_PROMPT",
        "Solve the math problem carefully. Break down the steps and provide the final answer as a single number.",
    )

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    solver_model = os.environ.get("AIME_DEEPSEEK_MODEL", "openai/deepseek-v4-flash")
    reflection_model = os.environ.get("AIME_REFLECTION_MODEL", "openai/deepseek-v4-pro")

    seed = _env_int("AIME_SEED", 0)
    max_metric_calls = _env_int("AIME_MAX_METRIC_CALLS", 5)
    max_workers = _env_int("AIME_MAX_WORKERS", 32)
    minibatch_size = _env_int("AIME_REFLECTION_MINIBATCH_SIZE", 3)
    n_parallel = _env_int("AIME_NUM_PARALLEL_PROPOSALS", 1)
    # Must not exceed the provider's ceiling (e.g. Aliyun qwen3-8b caps at 8192);
    # 32000 works for deepseek but 400s elsewhere. Keep it a knob.
    solver_max_tokens = _env_int("AIME_SOLVER_MAX_TOKENS", 8000)
    solver_temperature = _env_float("AIME_SOLVER_TEMPERATURE", 1.0)
    # Reflection temperature is optional: unset -> use the provider default
    # (GEPA's prior behaviour), matching how reflection_lm was configured before.
    reflection_temperature = _env_float_opt("AIME_REFLECTION_TEMPERATURE")

    # Point dspy's on-disk LM cache at a chosen dir so it can be shared with
    # eval runs (unset -> dspy default ~/.dspy_cache). Must precede LM config.
    cache_dir = os.environ.get("AIME_CACHE_DIR")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        dspy.configure_cache(disk_cache_dir=cache_dir)

    configure_solver_lm(solver_model, api_key, api_base, max_tokens=solver_max_tokens, temperature=solver_temperature)

    dataset_name = os.environ.get("AIME_DATASET", "aime")
    sizes = split_sizes_from_env()
    trainset, valset, testset = load_math_dataset(name=dataset_name, sizes=sizes, seed=seed)

    sampling_strategy = build_sampling_strategy(minibatch_size, n_parallel, seed)

    reflection_lm_kwargs = {"api_key": api_key, "api_base": api_base}
    if reflection_temperature is not None:
        reflection_lm_kwargs["temperature"] = reflection_temperature

    # APEX operates on a single dataset D used for both mutation and selection
    # (its nine buckets intersect tiers from H with outcomes on the same
    # examples), so train and val are merged and only the test split is held
    # out. Other strategies keep the usual train/val split.
    apex = None
    engine_extra: dict = {}
    reflection_extra: dict = {}
    optimize_valset = valset
    if is_apex():
        from gepa.gepa_launcher import make_litellm_lm

        dataset_d = list(trainset) + list(valset)
        optimize_valset = None  # GEPA then reuses the trainset as the valset
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
    else:
        engine_extra = {"sampling_strategy": sampling_strategy}
        # Parent selection. The default "pareto" samples a parent in proportion to
        # how many val examples it sits on the front for — which on a task where
        # many examples are solved by *every* candidate flattens the selection
        # pressure almost to uniform: measured on pptblank, the best candidate
        # (val 0.812) was only 1.28x more likely to be picked than the worst
        # (0.625), because 16 of 48 examples handed every candidate a front slot.
        # "current_best" hill-climbs on the aggregate score instead.
        selector = os.environ.get("GEPA_CANDIDATE_SELECTOR")
        if selector:
            engine_extra["candidate_selection_strategy"] = selector

    print(
        f"[AIME] dataset={dataset_name} sizes={sizes} "
        f"strategy={os.environ.get('GEPA_SAMPLING_STRATEGY', 'baseline')} "
        f"n_parallel={n_parallel} minibatch_size={minibatch_size} seed={seed} "
        f"max_metric_calls={max_metric_calls} "
        + (
            f"|D|={len(trainset)} (train+val merged) N_eval={apex['n_eval']} test={len(testset)}"
            if apex is not None
            else f"train/val/test={len(trainset)}/{len(valset)}/{len(testset)}"
        )
    )

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=os.environ.get("AIME_RUN_DIR", "outputs/aime_math"),
            seed=seed,
            max_metric_calls=max_metric_calls,
            track_best_outputs=True,
            parallel=True,
            max_workers=max_workers,
            cache_evaluation=True,
            **engine_extra,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_model,
            reflection_lm_kwargs=reflection_lm_kwargs,
            reflection_minibatch_size=minibatch_size,
            **reflection_extra,
        ),
    )

    result = optimize_anything(
        seed_candidate=initial_prompt,
        evaluator=evaluate,
        dataset=trainset,
        valset=optimize_valset,
        config=gepa_config,
    )

    if os.environ.get("AIME_SKIP_BASELINE_EVAL", "false").lower() in ("1", "true", "yes"):
        baseline_score = None
    else:
        print("\nEvaluating Baseline (Initial Prompt)...")
        baseline_score = evaluate_on_dataset(initial_prompt, testset)

    # Optimized Evaluation
    print("\nEvaluating Best Optimized Program...")
    if apex is not None:
        # Algorithm 1 line 20 returns P_curr, not an argmax over the pool.
        # GEPAResult.best_idx ranks each candidate's average on its own coverage,
        # and under subset evaluation those coverages are different (and easier)
        # samples of D, so they are not comparable across candidates.
        p_curr_idx = apex["val_evaluation_policy"].current_best_idx
        best_prompt = result.candidates[p_curr_idx]
        if isinstance(best_prompt, dict) and len(best_prompt) == 1:
            best_prompt = next(iter(best_prompt.values()))
        print(f"[APEX] P_curr = candidate {p_curr_idx} (GEPAResult.best_idx = {result.best_idx})")
    else:
        best_prompt = result.best_candidate
    print(f"Best Prompt Found:\n{best_prompt}")

    optimized_score = evaluate_on_dataset(best_prompt, testset)

    if baseline_score is not None:
        print(f"Baseline Score: {baseline_score:.2%}")
    print(f"Optimized Score: {optimized_score:.2%}")
    if baseline_score is not None:
        print(f"Improvement: {optimized_score - baseline_score:.2%}")


if __name__ == "__main__":
    main()
