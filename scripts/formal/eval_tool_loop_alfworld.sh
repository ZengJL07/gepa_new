#!/usr/bin/env bash
# Evaluate prompts on AlfWorld (no optimization), like eval_hmmt.sh.
# Scores a baseline prompt then an optimized prompt on one split and prints the
# delta. Set BASELINE_PROMPT_FILE / OPTIMIZED_PROMPT_FILE in EVAL CONFIG below.
#
# PREREQUISITE: the AlfWorld server (:36002, ALFWORLD_DATA set) must be running —
# see _tool_loop_env_common.sh. Budget/splits default from profiles.py.
set -euo pipefail

# ========================== EVAL CONFIG ================================
# EDIT THESE VALUES. They are the source of truth for a run — the script does
# NOT read these from the environment, so a stale `export EVAL_SIZE=...` in your
# shell cannot silently override them.
# --- Evaluation set -----------------------------------------------------
# How many examples to score. 0 = the WHOLE split (AlfWorld test=200, train=2420).
EVAL_SIZE=100
EVAL_SPLIT=test                  # test | train | val
# --- Episode budget -----------------------------------------------------
MAX_TURNS=20                     # max model turns per episode
MAX_TOKENS=12287                 # cumulative tokens per episode (incl. feedback + reasoning)
# --- Concurrency --------------------------------------------------------
MAX_WORKERS=15                   # episodes scored in parallel (profile default is 15)
# --- Cache --------------------------------------------------------------
# Solver-completion cache, so re-scoring the same prompts is free. MUST match the
# budget it was recorded under: a completion cached at a different MAX_TURNS /
# SOLVER_MAX_TOKENS would be replayed here as if it were valid. Keep this tag in
# step with RUN_TAG in _tool_loop_env_common.sh to share the training run's cache;
# set CACHE_DIR="" to disable.
# Which training run's cache to share: the strategy dir plus its RUN_TAG.
# CACHE_TAG="" disables the cache. (REPO_ROOT is prepended below, once it is set.)
# NOTE: keep CACHE_TAG consistent with THIS script's MAX_TURNS / MAX_TOKENS /
# SOLVER_MAX_TOKENS above, not merely copied from the training script — a tag that
# claims turns10 while this script runs 20 turns is how a stale completion gets
# replayed as if it were valid. Default is "" (no cache) so this is opt-in.
CACHE_STRATEGY_DIR=examples/tool_loop/test/alfworld_gepa_baseline
CACHE_TAG=""
# --- The two prompts being compared -------------------------------------
# Both are optional and independent, so this script can compare any two prompts —
# not just seed-vs-optimized. Paths are relative to REPO_ROOT (or absolute).
#   BASELINE_PROMPT_FILE   left side.  Empty => the profile's seed prompt.
#   OPTIMIZED_PROMPT_FILE  right side. Empty => same as baseline (delta 0).
# main.py writes best_prompt.txt into its run dir, which includes RUN_TAG, e.g.
#   examples/tool_loop/test/alfworld_gepa_captransfer/turns10_tok8191_v45_tr40/run_seed_42/best_prompt.txt
BASELINE_PROMPT_FILE=""
OPTIMIZED_PROMPT_FILE=""
# --- Solver (per-turn model call) ---------------------------------------
SOLVER_MAX_TOKENS=12287          # per-turn generation cap
SOLVER_TEMPERATURE=1.0
# =======================================================================

REPO_ROOT="/home/jlzeng/code/gepa_new"
cd "$REPO_ROOT"
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# --- Task + what to evaluate ---------------------------------------------
export TOOL_LOOP_TASK="alfworld"
export TOOL_LOOP_EVAL_SPLIT="$EVAL_SPLIT"
export TOOL_LOOP_EVAL_K="$EVAL_SIZE"
export TOOL_LOOP_MAX_TURNS="$MAX_TURNS"
export TOOL_LOOP_MAX_TOKENS="$MAX_TOKENS"
export TOOL_LOOP_MAX_WORKERS="$MAX_WORKERS"
export AIME_SEED="${AIME_SEED:-42}"

echo "[eval] effective config: EVAL_SIZE=$EVAL_SIZE EVAL_SPLIT=$EVAL_SPLIT " \
     "MAX_TURNS=$MAX_TURNS MAX_TOKENS=$MAX_TOKENS SOLVER_MAX_TOKENS=$SOLVER_MAX_TOKENS" \
     "MAX_WORKERS=$MAX_WORKERS"

# Resolve the two prompt files. Relative paths are taken against REPO_ROOT, and
# existence is checked here so a typo fails before the env server spins up.
_resolve_prompt_file() {
  local label="$1" value="$2"
  [[ -z "$value" ]] && return 0
  [[ "$value" != /* ]] && value="$REPO_ROOT/$value"
  if [[ ! -f "$value" ]]; then
    echo "[eval] ERROR: $label file not found: $value" >&2
    exit 2
  fi
  echo "$value"
}

BASELINE_PROMPT_FILE="$(_resolve_prompt_file BASELINE_PROMPT_FILE "$BASELINE_PROMPT_FILE")"
OPTIMIZED_PROMPT_FILE="$(_resolve_prompt_file OPTIMIZED_PROMPT_FILE "$OPTIMIZED_PROMPT_FILE")"

if [[ -n "$BASELINE_PROMPT_FILE" ]]; then
  export TOOL_LOOP_EVAL_PROMPT_FILE="$BASELINE_PROMPT_FILE"
else
  unset TOOL_LOOP_EVAL_PROMPT_FILE
fi
if [[ -n "$OPTIMIZED_PROMPT_FILE" ]]; then
  export TOOL_LOOP_OPTIMIZED_PROMPT_FILE="$OPTIMIZED_PROMPT_FILE"
else
  unset TOOL_LOOP_OPTIMIZED_PROMPT_FILE
fi
echo "[eval] baseline  prompt = ${TOOL_LOOP_EVAL_PROMPT_FILE:-<profile seed prompt>}"
echo "[eval] optimized prompt = ${TOOL_LOOP_OPTIMIZED_PROMPT_FILE:-<same as baseline: delta will be 0>}"

# --- Environment server + data -------------------------------------------
export TOOL_LOOP_DATA_ROOT="${TOOL_LOOP_DATA_ROOT:-/home/jlzeng/code/AgentGym}"
# export TOOL_LOOP_ENV_SERVER="http://127.0.0.1:36002"   # else profile default

# --- Solver ---------------------------------------------------------------
export AIME_SOLVER_MAX_TOKENS="$SOLVER_MAX_TOKENS"
export AIME_SOLVER_TEMPERATURE="$SOLVER_TEMPERATURE"

# --- Models / API ---------------------------------------------------------
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Please set DEEPSEEK_API_KEY}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
export AIME_DEEPSEEK_MODEL="${AIME_DEEPSEEK_MODEL:-openai/deepseek-v4-flash}"

# --- Env server (auto start/stop; TOOL_LOOP_MANAGE_SERVER=0 to opt out) ---
export TEST_ROOT="${TEST_ROOT:-$REPO_ROOT/examples/tool_loop/test/alfworld_eval}"
mkdir -p "$TEST_ROOT"

# eval_dataset.py appends the "solver" subdirectory itself, matching main.py, so
# a training run and an eval run share one solver cache.
if [[ -n "$CACHE_TAG" ]]; then
  export AIME_CACHE_DIR="$REPO_ROOT/$CACHE_STRATEGY_DIR/lm_cache/$CACHE_TAG"
else
  unset AIME_CACHE_DIR
fi
echo "[eval] lm_cache = ${AIME_CACHE_DIR:-disabled}"

# --- What to persist ------------------------------------------------------
# Summary (both scores + both prompts + config) and one JSON per pass holding
# every episode's messages/trace/feedback, for failure analysis.
export TOOL_LOOP_EVAL_OUTPUT="${TOOL_LOOP_EVAL_OUTPUT:-$TEST_ROOT/eval_seed_${AIME_SEED}.json}"
export TOOL_LOOP_TRACE_DIR="${TOOL_LOOP_TRACE_DIR:-$TEST_ROOT/traces}"

# shellcheck source=_tool_loop_server.sh
source "$(dirname "$0")/_tool_loop_server.sh"
tool_loop_start_server "$TOOL_LOOP_TASK"

# Mirror stdout+stderr next to the results; PIPESTATUS so tee doesn't mask a
# python failure.
export PYTHONUNBUFFERED=1
python -m examples.tool_loop.eval_dataset 2>&1 | tee -a "$TEST_ROOT/stdout.log"
exit "${PIPESTATUS[0]}"
