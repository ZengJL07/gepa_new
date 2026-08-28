#!/usr/bin/env bash
# Evaluate a solver model on the HMMT dataset, no optimization.
# train = FlagEval/HMMT_2025 (has solutions), val = hmmt_feb_2026,
# test = hmmt_feb_2023 + hmmt_feb_2024. Default: deepseek on the whole test split.
set -euo pipefail

REPO_ROOT="/home/jlzeng/code/gepa_new"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# --- What to evaluate -----------------------------------------------------
export AIME_DATASET="${AIME_DATASET:-hmmt}"
export AIME_EVAL_SPLIT="${AIME_EVAL_SPLIT:-test}"   # train | val | test
export AIME_EVAL_K="${AIME_EVAL_K:-0}"              # 0 = whole split
export AIME_SEED="${AIME_SEED:-42}"

# --- Solver ---------------------------------------------------------------
# HMMT answers are LaTeX (fractions/radicals); grading uses math_verify.
export AIME_SOLVER_USE_COT="${AIME_SOLVER_USE_COT:-1}"
export AIME_SOLVER_MAX_TOKENS="${AIME_SOLVER_MAX_TOKENS:-8000}"

# --- Models / API ---------------------------------------------------------
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Please set DEEPSEEK_API_KEY}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
export AIME_DEEPSEEK_MODEL="${AIME_DEEPSEEK_MODEL:-openai/deepseek-v4-flash}"

python -m examples.aime_math.eval_dataset
