#!/usr/bin/env bash
# Evaluate a solver model on a dataset split, no optimization. Scores a baseline
# prompt and a fixed optimized prompt on the same split and reports the delta.
# Default: deepseek on the first 100 MATH-500 test items.
#
# Every configurable knob is listed below with its default. Override any of them
# from the environment, e.g.  AIME_EVAL_K=50 ./scripts/formal/eval_math500.sh
set -euo pipefail

REPO_ROOT="/home/jlzeng/code/gepa_new"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# --- What to evaluate -----------------------------------------------------
# These MUST match the training runs (run_math500_gepa_np.sh /
# run_math500_gepa_captransfer.sh) so the scored test set is byte-for-byte the
# same problems. For math500 the test split is records[350:] of a seed-shuffled
# 500, then trimmed to the first K — it is independent of the train/val trims,
# so pinning (dataset, seed, test-K) alone fixes the exact test items.
export AIME_DATASET="${AIME_DATASET:-math500}"          # registered dataset: aime | math500
export AIME_EVAL_SPLIT="${AIME_EVAL_SPLIT:-test}"       # which split to score: train | val | test
export AIME_EVAL_K="${AIME_EVAL_K:-100}"               # == AIME_TEST_K in the training scripts (first-K of test)
export AIME_SEED="${AIME_SEED:-42}"                     # == AIME_SEED in the training scripts (split shuffle seed)

# --- Prompts --------------------------------------------------------------
# Baseline (initial) prompt: keep identical to the training scripts' seed prompt
# so "baseline" here reproduces the pre-optimization number.
export AIME_EVAL_PROMPT="${AIME_EVAL_PROMPT:-Solve the math problem carefully. Break down the steps and provide the final answer as a single number.}"
# Optimized prompt to compare against the baseline. Provide it inline via
# AIME_OPTIMIZED_PROMPT, or point AIME_OPTIMIZED_PROMPT_FILE at a file holding
# the GEPA-produced prompt (avoids shell quoting for long prompts). Unset both
# to fall back to the built-in default in eval_dataset.py.
#
# The default is kept in a SINGLE-QUOTED variable so LaTeX is preserved verbatim.
# Do NOT inline it directly in the ${VAR:-default} below: bash matches braces
# inside the expansion, so a literal "\frac{a}{b}" there gets mangled to
# "\frac{a{b}" plus a stray "}". Single quotes disable all of that.
_AIME_OPTIMIZED_PROMPT_DEFAULT='Solve the math problem carefully. Break down the steps, and provide the final answer using correct mathematical notation (for fractions, radicals, and other non-integer expressions use LaTeX, e.g., \frac{a}{b} or \sqrt{c}). Do not wrap the final answer in \boxed{} or any other formatting command. For an integer answer, output the plain integer; for a fraction, output a LaTeX fraction using \frac (not \dfrac). For multiple-choice questions where you must select a letter, output the letter exactly as the reference would: inside \text{} with parentheses, e.g., \text{(C)}. Ensure the final answer exactly matches the expected format, including matching the exact LaTeX style (e.g., \frac vs \dfrac) as in the reference solutions.'
# _AIME_OPTIMIZED_PROMPT_DEFAULT='Solve the math problem step by step, then provide the final answer. The final answer must be formatted exactly as required by the problem, using proper mathematical notation. This includes LaTeX fractions (e.g., \frac{a}{b}), interval notation (e.g., x \in [a,b]), and any units or symbols specified in the problem, such as the degree symbol (e.g., 90^\circ) for angles, percent signs for percentages, or units like cm, m, etc. Output only the final answer, with absolutely no additional text of any kind—no explanations, no completion markers (e.g., [[ ## completed ## ]]), no extra whitespace beyond what is necessary for the answer.'
export AIME_OPTIMIZED_PROMPT="${AIME_OPTIMIZED_PROMPT:-$_AIME_OPTIMIZED_PROMPT_DEFAULT}"
# export AIME_OPTIMIZED_PROMPT_FILE="${AIME_OPTIMIZED_PROMPT_FILE:-$REPO_ROOT/scripts/formal/optimized_prompt.txt}"

# --- Solver ---------------------------------------------------------------
# CoT off: use a bare dspy.Predict (answer only), no explicit reasoning field.
# Redundant + truncation-prone for models that already reason internally.
export AIME_SOLVER_USE_COT="${AIME_SOLVER_USE_COT:-0}"          # 1 = wrap in ChainOfThought, 0 = bare Predict
export AIME_SOLVER_TEMPERATURE="${AIME_SOLVER_TEMPERATURE:-0.2}"  # solver sampling temperature
# Keep within the provider ceiling (deepseek allows 32000; Aliyun qwen3-8b 8192).
# MUST match the training scripts (8191) so requests hash identically and the
# on-disk LM cache is reused instead of re-billed. max_tokens is part of the
# dspy cache key, so any mismatch = total cache miss.
export AIME_SOLVER_MAX_TOKENS="${AIME_SOLVER_MAX_TOKENS:-8191}"   # solver max output tokens
export AIME_EVAL_NUM_THREADS="${AIME_EVAL_NUM_THREADS:-16}"       # concurrent eval requests (provider rate limit)

# --- Where to save --------------------------------------------------------
# Paths encode the solver temperature so runs at different temperatures don't
# overwrite each other.
TEST_ROOT="$REPO_ROOT/examples/aime_math/test/eval/${AIME_DATASET}_${AIME_EVAL_SPLIT}"
# On-disk LM cache: reused across runs to avoid paying for repeat completions.
export AIME_CACHE_DIR="${AIME_CACHE_DIR:-$TEST_ROOT/lm_cache}"
# JSON results file (config + baseline/optimized accuracy + delta).
export AIME_EVAL_OUTPUT="${AIME_EVAL_OUTPUT:-$TEST_ROOT/results_seed_${AIME_SEED}_ori/t_${AIME_SOLVER_TEMPERATURE}.json}"
mkdir -p "$TEST_ROOT"

# --- Models / API ---------------------------------------------------------
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Please set DEEPSEEK_API_KEY}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
export AIME_DEEPSEEK_MODEL="${AIME_DEEPSEEK_MODEL:-openai/deepseek-v4-flash}"  # solver model

python -m examples.aime_math.eval_dataset
