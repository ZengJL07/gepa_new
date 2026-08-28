#!/usr/bin/env bash
# Baseline: stock GEPA batch sampling (IndependentSampling / epoch-shuffled).
# Task: multi-turn tool-feedback loop; the optimized component is the initial prompt.
set -euo pipefail

REPO_ROOT="/home/jlzeng/code/gepa_new"
TEST_ROOT="$REPO_ROOT/examples/tool_loop/test/gepa_baseline"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p "$TEST_ROOT"

# --- Batch sampling strategy ---------------------------------------------
export GEPA_SAMPLING_STRATEGY="baseline"

# --- Initial prompt (the sole optimized component) -----------------------
export TOOL_LOOP_INITIAL_PROMPT="${TOOL_LOOP_INITIAL_PROMPT:-You can solve the task by calling tools and reading their feedback. To call a tool, output exactly: <call name=\"TOOL\">{\"arg\": value}</call>. When you know the answer, output: <final>ANSWER</final>. Emit one action per turn and nothing else.}"

# --- Episode budget (dual: turns + total tokens incl. tool feedback) -----
export TOOL_LOOP_MAX_TURNS="${TOOL_LOOP_MAX_TURNS:-6}"
export TOOL_LOOP_MAX_TOKENS="${TOOL_LOOP_MAX_TOKENS:-8000}"

# --- Dataset sizes -------------------------------------------------------
# Defaults match run_math500_gepa.sh (train=40, val=45, test=100). The guess
# task is generated, so any size is available.
export TOOL_LOOP_TRAIN_N="${TOOL_LOOP_TRAIN_N:-40}"
export TOOL_LOOP_VAL_N="${TOOL_LOOP_VAL_N:-45}"
export TOOL_LOOP_TEST_N="${TOOL_LOOP_TEST_N:-100}"

# --- Solver ---------------------------------------------------------------
# Per-turn model max_tokens. MUST stay within the provider's ceiling.
export AIME_SOLVER_MAX_TOKENS="${AIME_SOLVER_MAX_TOKENS:-2000}"
export AIME_SOLVER_TEMPERATURE="${AIME_SOLVER_TEMPERATURE:-1.0}"
# export AIME_REFLECTION_TEMPERATURE="1.0"

# --- Run configuration ---------------------------------------------------
export AIME_SEED="${AIME_SEED:-42}"
export AIME_MAX_METRIC_CALLS="${AIME_MAX_METRIC_CALLS:-500}"
export AIME_MAX_WORKERS="${AIME_MAX_WORKERS:-15}"
# Must match the captransfer script: n_parallel changes how many candidates are
# proposed per iteration, so a mismatch would confound the strategy comparison.
export AIME_NUM_PARALLEL_PROPOSALS="${AIME_NUM_PARALLEL_PROPOSALS:-5}"
export AIME_REFLECTION_MINIBATCH_SIZE="${AIME_REFLECTION_MINIBATCH_SIZE:-3}"
# true = skip scoring the SEED prompt on test after the search (the optimized
# prompt is still scored). Both test passes are outside AIME_MAX_METRIC_CALLS,
# so this halves the post-search cost; summary.json then records
# test_baseline_score/test_improvement as null.
export AIME_SKIP_BASELINE_EVAL="${AIME_SKIP_BASELINE_EVAL:-false}"
export AIME_RUN_DIR="${AIME_RUN_DIR:-$TEST_ROOT/run_seed_${AIME_SEED}}"
# Persist the reflective datasets (iterations/<id>/reflective_dataset.json).
export TOOL_LOOP_WRITE_AGENT_STATE="${TOOL_LOOP_WRITE_AGENT_STATE:-1}"
# On-disk LM cache (solver/ + reflection/ subdirs) so a re-run replays instead of
# paying again. Per-strategy: TEST_ROOT differs from the captransfer script, so
# neither strategy answers from the other's completions. Set to "" to disable.
export AIME_CACHE_DIR="${AIME_CACHE_DIR:-$TEST_ROOT/lm_cache}"

# --- Models / API --------------------------------------------------------
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Please set DEEPSEEK_API_KEY}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
export AIME_DEEPSEEK_MODEL="${AIME_DEEPSEEK_MODEL:-openai/deepseek-v4-flash}"
export AIME_REFLECTION_MODEL="${AIME_REFLECTION_MODEL:-openai/deepseek-v4-pro}"

# Mirror stdout+stderr into the run dir (config echo, per-iteration progress,
# final scores, optimized prompt) instead of only the terminal scrollback.
mkdir -p "$AIME_RUN_DIR"
export PYTHONUNBUFFERED=1
python -m examples.tool_loop.main 2>&1 | tee -a "$AIME_RUN_DIR/stdout.log"
exit "${PIPESTATUS[0]}"
