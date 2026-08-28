#!/usr/bin/env bash
# Evaluate prompts on TextCraft (no optimization), like eval_hmmt.sh.
# Scores a baseline prompt then an optimized prompt on one split and prints the
# delta. Point TOOL_LOOP_OPTIMIZED_PROMPT_FILE at a GEPA-produced prompt to compare.
#
# PREREQUISITE: the TextCraft server (:36001) must be running — see
# _tool_loop_env_common.sh. Budget/splits default from profiles.py.
set -euo pipefail

# ========================== EVAL CONFIG ================================
# EDIT THESE VALUES. They are the source of truth for a run — the script does
# NOT read these from the environment, so a stale `export EVAL_SIZE=...` in your
# shell cannot silently override them.
# --- Evaluation set -----------------------------------------------------
# How many examples to score. 0 = the WHOLE split (TextCraft test=100, train=374).
EVAL_SIZE=50
EVAL_SPLIT=test                  # test | train | val
# --- Episode budget -----------------------------------------------------
MAX_TURNS=5                      # max model turns per episode
MAX_TOKENS=8000                  # cumulative tokens per episode (incl. feedback + reasoning)
# --- Concurrency --------------------------------------------------------
MAX_WORKERS=15                   # episodes scored in parallel (profile default is 15)
# --- Solver (per-turn model call) ---------------------------------------
SOLVER_MAX_TOKENS=2000           # per-turn generation cap
SOLVER_TEMPERATURE=1.0
# =======================================================================

REPO_ROOT="/home/jlzeng/code/gepa_new"
cd "$REPO_ROOT"
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# --- Task + what to evaluate ---------------------------------------------
export TOOL_LOOP_TASK="textcraft"
export TOOL_LOOP_EVAL_SPLIT="$EVAL_SPLIT"
export TOOL_LOOP_EVAL_K="$EVAL_SIZE"
export TOOL_LOOP_MAX_TURNS="$MAX_TURNS"
export TOOL_LOOP_MAX_TOKENS="$MAX_TOKENS"
export TOOL_LOOP_MAX_WORKERS="$MAX_WORKERS"
export AIME_SEED="${AIME_SEED:-42}"

echo "[eval] effective config: EVAL_SIZE=$EVAL_SIZE EVAL_SPLIT=$EVAL_SPLIT " \
     "MAX_TURNS=$MAX_TURNS MAX_TOKENS=$MAX_TOKENS SOLVER_MAX_TOKENS=$SOLVER_MAX_TOKENS" \
     "MAX_WORKERS=$MAX_WORKERS"

# Optimized prompt to compare against the baseline (seed prompt). Unset => the
# optimized prompt falls back to the baseline, so the delta is 0 BY CONSTRUCTION
# (both passes score the same prompt) — uncomment to make the comparison real.
# main.py writes best_prompt.txt into its AIME_RUN_DIR:
# export TOOL_LOOP_OPTIMIZED_PROMPT_FILE="$REPO_ROOT/examples/tool_loop/test/textcraft_gepa_captransfer/run_seed_42/best_prompt.txt"

# --- Environment server + data -------------------------------------------
export TOOL_LOOP_DATA_ROOT="${TOOL_LOOP_DATA_ROOT:-/home/jlzeng/code/AgentGym}"
# export TOOL_LOOP_ENV_SERVER="http://127.0.0.1:36001"   # else profile default

# --- Solver ---------------------------------------------------------------
export AIME_SOLVER_MAX_TOKENS="$SOLVER_MAX_TOKENS"
export AIME_SOLVER_TEMPERATURE="$SOLVER_TEMPERATURE"

# --- Models / API ---------------------------------------------------------
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Please set DEEPSEEK_API_KEY}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
export AIME_DEEPSEEK_MODEL="${AIME_DEEPSEEK_MODEL:-openai/deepseek-v4-flash}"

# --- Env server (auto start/stop; TOOL_LOOP_MANAGE_SERVER=0 to opt out) ---
export TEST_ROOT="${TEST_ROOT:-$REPO_ROOT/examples/tool_loop/test/textcraft_eval}"
mkdir -p "$TEST_ROOT"

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
