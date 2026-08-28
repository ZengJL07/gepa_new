"""Standalone evaluator for a tool-loop task (no optimization).

Scores two prompts on the same split and reports the delta: a baseline
(the profile's seed prompt, overridable via TOOL_LOOP_EVAL_PROMPT) followed by an
optimized prompt (TOOL_LOOP_OPTIMIZED_PROMPT / _FILE). Reuses the exact episode
loop, session builder, and scorer from ``main`` via the task profile, so numbers
are comparable to an optimization run. Example — score guess on its whole test:

    TOOL_LOOP_TASK=guess TOOL_LOOP_EVAL_SPLIT=test \
    AIME_DEEPSEEK_MODEL=openai/deepseek-v4-flash DEEPSEEK_API_KEY=... \
    python -m examples.tool_loop.eval_dataset

Env:
- TOOL_LOOP_TASK          which profile: guess | textcraft | alfworld | sciworld
                          (default guess)
- TOOL_LOOP_EVAL_SPLIT    split to score: train | val | test (default test)
- TOOL_LOOP_EVAL_K        size of the scored split: >0 = that many examples,
                          <=0 or unset = the WHOLE split (test: 100 textcraft /
                          200 alfworld / 200 sciworld; train: 374 / 2420 / 2059).
                          This is the single size knob for eval — it overrides
                          the profile default.
- TOOL_LOOP_EVAL_PROMPT / _FILE       baseline prompt (file wins over the inline
                          string; default: the profile's seed prompt).
- TOOL_LOOP_OPTIMIZED_PROMPT / _FILE  optimized prompt to compare (file wins).
                          Unset -> reuse the baseline (delta 0), so a bare run still works.
                          Both prompts are specifiable independently, so you can
                          compare two GEPA artifacts directly, not just
                          seed-vs-optimized. An empty prompt file is an error.
- TOOL_LOOP_MAX_TURNS / _MAX_TOKENS   episode budget (default: profile).
- TOOL_LOOP_{TRAIN,VAL,TEST}_N        split sizes (default: profile).
- TOOL_LOOP_EVAL_OUTPUT   path to write a JSON results file (default: print only).
- AIME_CACHE_DIR, AIME_SEED, AIME_SOLVER_*, and the usual model/API env vars.
"""

import json
import os

import dspy

from examples.aime_math.utils import configure_solver_lm
from examples.tool_loop import main as tl_main
from examples.tool_loop.profiles import get_profile


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _resolve_prompt(file_var: str, text_var: str, default: str) -> str:
    """Resolve one prompt: file > inline string > default.

    A file wins over an inline string so a run can point at a GEPA artifact
    without having to unset a leftover inline override. An empty or whitespace-only
    file is an error rather than a silent fall back to the default: scoring the
    seed prompt while believing you scored an optimized one is worse than failing.
    """
    path = os.environ.get(file_var)
    if path:
        with open(path, encoding="utf-8") as f:
            text = f.read().rstrip("\n")
        if not text.strip():
            raise ValueError(f"{file_var}={path!r} is empty; refusing to score an empty prompt.")
        return text
    return os.environ.get(text_var, default)


def main():
    task = os.environ.get("TOOL_LOOP_TASK", "guess").lower()
    profile = get_profile(task)

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    solver_model = os.environ.get("AIME_DEEPSEEK_MODEL", "openai/deepseek-v4-flash")
    solver_max_tokens = _env_int("AIME_SOLVER_MAX_TOKENS", 2000)
    solver_temperature = _env_float("AIME_SOLVER_TEMPERATURE", 1.0)

    split_name = os.environ.get("TOOL_LOOP_EVAL_SPLIT", "test").lower()
    if split_name not in ("train", "val", "test"):
        raise ValueError(f"Unknown TOOL_LOOP_EVAL_SPLIT={split_name!r}; expected train|val|test.")
    k = _env_int("TOOL_LOOP_EVAL_K", 0) or None
    seed = _env_int("AIME_SEED", 0)

    # Both prompts are independently specifiable, by file or inline. The baseline
    # defaults to the profile's seed prompt; the optimized one defaults to whatever
    # the baseline resolved to, which makes the delta 0 by construction and is the
    # signal that no optimized prompt was supplied.
    baseline_prompt = _resolve_prompt(
        "TOOL_LOOP_EVAL_PROMPT_FILE", "TOOL_LOOP_EVAL_PROMPT", profile.seed_prompt
    )
    optimized_prompt = _resolve_prompt(
        "TOOL_LOOP_OPTIMIZED_PROMPT_FILE", "TOOL_LOOP_OPTIMIZED_PROMPT", baseline_prompt
    )
    if optimized_prompt == baseline_prompt:
        print("[eval] NOTE: optimized prompt == baseline prompt, so delta will be 0 by construction.")

    cache_dir = os.environ.get("AIME_CACHE_DIR")
    if cache_dir:
        # Must be the SAME "solver" subdirectory main.py uses, or an eval run
        # cannot reuse the episodes a training run already paid for. There is no
        # reflection LM here, so only the solver cache applies.
        solver_cache = os.path.join(cache_dir, "solver")
        os.makedirs(solver_cache, exist_ok=True)
        dspy.configure_cache(disk_cache_dir=solver_cache)

    configure_solver_lm(
        solver_model, api_key, api_base, max_tokens=solver_max_tokens, temperature=solver_temperature
    )

    # Size the split under evaluation. TOOL_LOOP_EVAL_K is the single knob for the
    # evaluated split: >0 = that many examples, <=0 (default) = the WHOLE split.
    # The other two splits are loaded at their profile size (only for train/val
    # carving); they are not evaluated here. Setting the evaluated split's size
    # from EVAL_K keeps a single source of truth (no double-trim surprise).
    eval_size = k if k is not None else 0  # 0 => whole split (loader sentinel)
    sizes = {
        "train": profile.defaults["train_n"],
        "val": profile.defaults["val_n"],
        "test": profile.defaults["test_n"],
    }
    if split_name == "val" and eval_size <= 0:
        # val is a carve-out of train, not a standalone corpus; there is no
        # "whole val". Keep the profile val size when EVAL_K is unset.
        eval_size = profile.defaults["val_n"]
    sizes[split_name] = eval_size
    trainset, valset, testset = profile.load_splits(
        train_n=sizes["train"], val_n=sizes["val"], test_n=sizes["test"], seed=seed
    )
    dataset = {"train": trainset, "val": valset, "test": testset}[split_name]
    n = len(dataset)

    print(
        f"[eval] task={task} split={split_name} n={n} model={solver_model} "
        f"max_turns={tl_main._MAX_TURNS} episode_max_tokens={tl_main._MAX_TOKENS} "
        f"solver_max_tokens={solver_max_tokens}"
    )

    # Optional per-episode trace dump for failure analysis. Set TOOL_LOOP_TRACE_DIR
    # to a directory; each episode is written as <label>_<index>.json with the full
    # conversation, stop_reason, reward, turns, tokens, and feedback.
    trace_dir = os.environ.get("TOOL_LOOP_TRACE_DIR")
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)

    def _run(label: str, prompt: str) -> float:
        print(f"\n[eval] --- {label} ---")
        rows = []

        done = [0]

        def _collect(i, ex, episode, score, feedback):
            # Episodes run concurrently, so these lines arrive out of order —
            # include a k/n counter so progress is still readable.
            done[0] += 1
            item = getattr(ex, "item_id", getattr(ex, "input", i))
            print(
                f"[eval]   [{done[0]}/{n}] #{i} item={item} score={score:.0f} "
                f"stop={episode.stop_reason} "
                f"reward={episode.reward} turns={episode.turns_used} tokens={episode.tokens_used} "
                f"tool_calls={episode.tool_calls} format_errors={episode.format_errors}"
            )
            if trace_dir:
                rows.append({
                    "index": i,
                    "item_id": getattr(ex, "item_id", None),
                    "env_index": getattr(ex, "env_index", None),
                    "score": score,
                    "stop_reason": episode.stop_reason,
                    "reward": episode.reward,
                    "env_done": episode.env_done,
                    "turns_used": episode.turns_used,
                    "tokens_used": episode.tokens_used,
                    "tool_calls": episode.tool_calls,
                    "format_errors": episode.format_errors,
                    "feedback": feedback,
                    "trace": episode.trace,
                    "messages": episode.messages,
                })

        s = tl_main.evaluate_on_dataset(prompt, dataset, on_episode=_collect)
        print(f"[eval] {label} score = {s:.4f} ({s * n:.1f}/{n})")
        if trace_dir:
            path = os.path.join(trace_dir, f"{label}.json")
            rows.sort(key=lambda r: r["index"])  # completion order -> dataset order
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            print(f"[eval] {label} traces -> {path}")
        return s

    baseline_score = _run("baseline", baseline_prompt)
    optimized_score = _run("optimized", optimized_prompt)
    delta = optimized_score - baseline_score
    print(
        f"\n[eval] summary on {task}/{split_name} (n={n}): "
        f"baseline={baseline_score:.4f}, optimized={optimized_score:.4f}, delta={delta:+.4f}"
    )

    output_path = os.environ.get("TOOL_LOOP_EVAL_OUTPUT")
    if output_path:
        results = {
            "task": task,
            "split": split_name,
            "n": n,
            "seed": seed,
            "model": solver_model,
            "max_turns": tl_main._MAX_TURNS,
            "episode_max_tokens": tl_main._MAX_TOKENS,
            "solver_max_tokens": solver_max_tokens,
            "baseline_prompt": baseline_prompt,
            "optimized_prompt": optimized_prompt,
            "baseline_score": baseline_score,
            "optimized_score": optimized_score,
            "delta": delta,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[eval] results written to {output_path}")


if __name__ == "__main__":
    main()
