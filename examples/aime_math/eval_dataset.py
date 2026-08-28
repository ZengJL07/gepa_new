"""Standalone evaluator: score a model on a dataset split (no optimization).

Evaluates two prompts on the same split and reports the delta: a baseline
(``AIME_EVAL_PROMPT``, default the generic solve prompt) followed by a
GEPA-optimized prompt (``OPTIMIZED_PROMPT``). Reuses the same solver LM and
grading path as the optimization run, so numbers are comparable. Example:
evaluate deepseek on the first 50 MATH-500 test items.

    AIME_DATASET=math500 AIME_EVAL_K=50 AIME_EVAL_SPLIT=test \
    AIME_DEEPSEEK_MODEL=openai/deepseek-v4-flash DEEPSEEK_API_KEY=... \
    python -m examples.aime_math.eval_dataset

Env:
- AIME_DATASET       dataset name (default "aime")
- AIME_EVAL_SPLIT    which split to score: train | val | test (default "test")
- AIME_EVAL_K        first-k trim per split (default 0 = whole split)
- AIME_EVAL_PROMPT   baseline solver instructions (default: the generic solve prompt).
- AIME_OPTIMIZED_PROMPT  optimized solver instructions to compare against the
                     baseline. Unset -> the built-in OPTIMIZED_PROMPT default.
- AIME_OPTIMIZED_PROMPT_FILE  path to a file whose contents are the optimized
                     prompt (takes precedence over AIME_OPTIMIZED_PROMPT). Handy
                     for pasting a long GEPA-produced prompt without shell quoting.
- AIME_SOLVER_TEMPERATURE  solver sampling temperature (default 1.0)
- AIME_CACHE_DIR     on-disk LM cache dir (default: dspy's ~/.dspy_cache).
                     Point runs at the same dir to reuse cached completions.
- AIME_EVAL_OUTPUT   path to write a JSON results file (default: no file, print only).
- AIME_SOLVER_MAX_TOKENS, AIME_SEED, and the usual model/API env vars.
"""

import json
import os

import dspy

from examples.aime_math.utils import configure_solver_lm, evaluate_on_dataset, load_math_dataset

DEFAULT_PROMPT = (
    "Solve the math problem carefully. Break down the steps and provide the final answer as a single number."
)

# A GEPA-optimized prompt, evaluated right after the baseline for comparison.
# Overridable at runtime via AIME_OPTIMIZED_PROMPT / AIME_OPTIMIZED_PROMPT_FILE.
OPTIMIZED_PROMPT = (
    "Solve the math problem carefully using systematic, thorough reasoning. Identify all constraints on the "
    "variables. Explore different structural approaches: consider number theory, combinatorial enumeration, "
    "algebraic manipulation, geometric reasoning, or case analysis as appropriate. Do not prematurely fix on a "
    "single strategy — test multiple routes. Before concluding, verify that no other configuration yields a more "
    "extreme (smaller or larger) result by checking all viable candidates within the constrained space. Pay "
    "special attention to edge cases where the optimal outcome may arise from an unexpected combination (e.g., a "
    "very small numerator rather than a very large denominator). Provide the final answer as a single number. "
    "Ensure your solution is complete and rigorous."
)


def _resolve_optimized_prompt() -> str:
    """Pick the optimized prompt: file > env string > built-in default.

    AIME_OPTIMIZED_PROMPT_FILE wins (read verbatim, trailing newline stripped),
    then AIME_OPTIMIZED_PROMPT, then the module default OPTIMIZED_PROMPT.
    """
    path = os.environ.get("AIME_OPTIMIZED_PROMPT_FILE")
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read().rstrip("\n")
    return os.environ.get("AIME_OPTIMIZED_PROMPT", OPTIMIZED_PROMPT)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    solver_model = os.environ.get("AIME_DEEPSEEK_MODEL", "openai/deepseek-v4-flash")
    solver_max_tokens = _env_int("AIME_SOLVER_MAX_TOKENS", 8000)
    solver_temperature = _env_float("AIME_SOLVER_TEMPERATURE", 1.0)

    dataset_name = os.environ.get("AIME_DATASET", "aime")
    split_name = os.environ.get("AIME_EVAL_SPLIT", "test").lower()
    k = _env_int("AIME_EVAL_K", 0) or None
    seed = _env_int("AIME_SEED", 0)
    prompt = os.environ.get("AIME_EVAL_PROMPT", DEFAULT_PROMPT)
    optimized_prompt = _resolve_optimized_prompt()

    if split_name not in ("train", "val", "test"):
        raise ValueError(f"Unknown AIME_EVAL_SPLIT={split_name!r}; expected train|val|test.")
    # AIME_EVAL_K caps only the split being evaluated.
    sizes = {
        "train": (k, None, None),
        "val": (None, k, None),
        "test": (None, None, k),
    }[split_name]

    cache_dir = os.environ.get("AIME_CACHE_DIR")
    output_path = os.environ.get("AIME_EVAL_OUTPUT")

    # Point dspy's on-disk LM cache at a chosen dir so repeated runs reuse
    # completions (unset -> dspy's default ~/.dspy_cache). Must happen before the
    # LM is configured/called.
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        dspy.configure_cache(disk_cache_dir=cache_dir)

    configure_solver_lm(
        solver_model, api_key, api_base, max_tokens=solver_max_tokens, temperature=solver_temperature
    )

    trainset, valset, testset = load_math_dataset(name=dataset_name, sizes=sizes, seed=seed)
    dataset = {"train": trainset, "val": valset, "test": testset}[split_name]

    print(
        f"[eval] dataset={dataset_name} split={split_name} n={len(dataset)} "
        f"model={solver_model} max_tokens={solver_max_tokens}"
    )

    n = len(dataset)

    def _run(label: str, p: str) -> float:
        print(f"\n[eval] --- {label} ---")
        s = evaluate_on_dataset(p, dataset)
        print(f"[eval] {label} accuracy = {s:.4f} ({s * n:.0f}/{n})")
        return s

    baseline_score = _run("baseline", prompt)
    optimized_score = _run("optimized", optimized_prompt)

    delta = optimized_score - baseline_score
    print(
        f"\n[eval] summary on {dataset_name}/{split_name} (n={n}): "
        f"baseline={baseline_score:.4f}, optimized={optimized_score:.4f}, "
        f"delta={delta:+.4f}"
    )

    if output_path:
        results = {
            "dataset": dataset_name,
            "split": split_name,
            "n": n,
            "seed": seed,
            "model": solver_model,
            "solver_max_tokens": solver_max_tokens,
            "solver_temperature": solver_temperature,
            "cache_dir": cache_dir,
            "baseline_prompt": prompt,
            "optimized_prompt": optimized_prompt,
            "baseline_accuracy": baseline_score,
            "optimized_accuracy": optimized_score,
            "delta": delta,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[eval] results written to {output_path}")


if __name__ == "__main__":
    main()
