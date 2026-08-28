#!/usr/bin/env bash
# Baseline: stock GEPA batch sampling (IndependentSampling / epoch-shuffled).
set -euo pipefail

REPO_ROOT="/home/jlzeng/code/gepa_new"
TEST_ROOT="$REPO_ROOT/examples/aime_math/test/aime_formal/gepa_baseline"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p "$TEST_ROOT"

# --- Batch sampling strategy ---------------------------------------------
export GEPA_SAMPLING_STRATEGY="baseline"

# --- Solver ---------------------------------------------------------------
# CoT adds an explicit reasoning field before the answer (helps non-reasoning
# models; redundant + truncation-prone for models that reason internally).
export AIME_SOLVER_USE_COT="${AIME_SOLVER_USE_COT:-1}"
# Solver max_tokens. MUST stay within the provider's ceiling: Aliyun qwen3-8b
# rejects anything above 8192 with a 400 (deepseek allows up to 32000).
export AIME_SOLVER_MAX_TOKENS="${AIME_SOLVER_MAX_TOKENS:-8000}"
# Per-model sampling temperature. Reflection temperature is optional: leave it
# unset to use the provider default (GEPA's prior behaviour).
export AIME_SOLVER_TEMPERATURE="${AIME_SOLVER_TEMPERATURE:-1.0}"
# export AIME_REFLECTION_TEMPERATURE="1.0"

# --- Run configuration ---------------------------------------------------
export AIME_SEED="${AIME_SEED:-42}"
export AIME_MAX_METRIC_CALLS="${AIME_MAX_METRIC_CALLS:-500}"
export AIME_MAX_WORKERS="${AIME_MAX_WORKERS:-15}"
export AIME_NUM_PARALLEL_PROPOSALS="${AIME_NUM_PARALLEL_PROPOSALS:-5}"
export AIME_REFLECTION_MINIBATCH_SIZE="${AIME_REFLECTION_MINIBATCH_SIZE:-3}"
export AIME_SKIP_BASELINE_EVAL="${AIME_SKIP_BASELINE_EVAL:-false}"
export AIME_RUN_DIR="${AIME_RUN_DIR:-$TEST_ROOT/te_qwen_3/aime26/run_seed_${AIME_SEED}}"

# --- Models / API --------------------------------------------------------
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Please set DEEPSEEK_API_KEY}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://llm-4qlgg8s0leay0d1d.cn-beijing.maas.aliyuncs.com/compatible-mode/v1}"
export AIME_DEEPSEEK_MODEL="${AIME_DEEPSEEK_MODEL:-openai/qwen3-8b}"
export AIME_REFLECTION_MODEL="${AIME_REFLECTION_MODEL:-openai/deepseek-v4-pro}"

python -m examples.aime_math.main
