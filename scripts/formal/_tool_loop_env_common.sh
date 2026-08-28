#!/usr/bin/env bash
# Shared launcher for AgentGym env tasks (textcraft / alfworld) under GEPA.
# Sourced by the per-task/per-strategy wrappers. It relies on these vars being
# set by the caller BEFORE sourcing:
#   TOOL_LOOP_TASK          textcraft | alfworld
#   GEPA_SAMPLING_STRATEGY  baseline | capability_transfer
#   TEST_ROOT               where to put the run dir
# Budget/concurrency defaults live in examples/tool_loop/profiles.py. Tune a run
# via the friendly RUN CONFIG knobs below (TRAIN_SIZE/VAL_SIZE/TEST_SIZE,
# MAX_TURNS/MAX_TOKENS/MAX_WORKERS, METRIC_CALLS, SEED, SOLVER_*) — each is an
# env override, e.g.  TRAIN_SIZE=40 METRIC_CALLS=80 bash run_..._gepa.sh
#
# ENV SERVER — started/stopped automatically by _tool_loop_server.sh (sourced
# below). It reuses an already-running server if one answers on the port, else
# launches one in the task's conda env and reaps it on exit. Set
# TOOL_LOOP_MANAGE_SERVER=0 to skip management and assume it is already up.
# One-time setup still required:
#   TextCraft: rebuild the machine-specific remap once per host:
#       conda run -n agentenv-textcraft python AgentGym/data/build_textcraft_goal_index.py
#   AlfWorld: ALFWORLD_DATA must point at the alfworld data (default ~/.cache/alfworld);
#       the first /reset per game compiles a TextWorld game (tens of seconds).
set -euo pipefail

# ========================== RUN CONFIG =================================
# Friendly knobs for a GEPA optimization run. Leave a value blank to fall back
# to the task profile default (examples/tool_loop/profiles.py). All are also
# overridable from the environment, e.g.  TRAIN_SIZE=40 bash run_..._gepa.sh
#
# --- Dataset sizes ------------------------------------------------------
# Defaults match the math scripts (run_math500_gepa.sh: train=40, val=45,
# test=100) so budget and split sizes are comparable across tasks. Set to 0 for
# the whole split; blank to fall back to the profile default.
#   Upper bounds: TextCraft train=374/test=100, AlfWorld train=2420/test=200.
TRAIN_SIZE="${TRAIN_SIZE:-40}"     # training examples GEPA optimizes on
VAL_SIZE="${VAL_SIZE:-45}"         # validation examples for candidate selection
TEST_SIZE="${TEST_SIZE:-100}"      # held-out test examples for final eval
MAX_WORKERS="${MAX_WORKERS:-15}"
# --- Episode budget (blank => profile default) --------------------------
MAX_TURNS="${MAX_TURNS:-40}"       # max model turns per episode
MAX_TOKENS="${MAX_TOKENS:-18431}"   # cumulative tokens per episode (incl. 
# --- Search budget / reproducibility ------------------------------------
METRIC_CALLS="${METRIC_CALLS:-1000}"  # total evaluator calls (GEPA search budget)
SEED="${SEED:-42}"
# --- Final held-out test eval -------------------------------------------
# After the search, main.py scores the seed prompt AND the optimized prompt on
# the test split, then reports the delta. Both passes are OUTSIDE the
# METRIC_CALLS budget, so each costs a further TEST_SIZE episodes.
#
# 1 = skip the seed-prompt pass (only the optimized prompt is scored).
#     Halves the post-search cost. Use when the seed's test score is already
#     known — e.g. from a previous run, or from eval_tool_loop_alfworld.sh,
#     which scores a baseline and an optimized prompt on the same split.
#     summary.json then records test_baseline_score/test_improvement as null.
# 0 = score both (default), so the run is self-contained.
SKIP_BASELINE_EVAL="${SKIP_BASELINE_EVAL:-0}"
# --- Solver (per-turn model call) ---------------------------------------
# Per-turn generation cap. Must match the eval scripts' SOLVER_MAX_TOKENS,
# otherwise a prompt is optimized under a tighter per-turn ceiling than it is
# scored under: a turn that exceeds this cap ends the episode as
# stop_reason="truncated" (score 0). At the old value of 2000, ~3% of observed
# AlfWorld turns would truncate during training but survive at eval time.
SOLVER_MAX_TOKENS="${SOLVER_MAX_TOKENS:-12287}"  # per-turn generation cap
SOLVER_TEMPERATURE="${SOLVER_TEMPERATURE:-1.0}"
# --- Caching / run isolation --------------------------------------------
# EDIT RUN_TAG WHENEVER YOU CHANGE THE BUDGET OR THE SPLITS.
#
# A run keeps two caches, in two different places:
#   <run_dir>/fitness_cache          (prompt, example) -> score
#   <CACHE_DIR>/{solver,reflection}  LM completions
# RUN_TAG is inserted into BOTH paths, so runs with different tags cannot read
# each other's results.
#
# Why this is a correctness knob, not a tidiness one: the score cache is keyed on
# (candidate_text, example) ONLY — the episode budget is NOT part of the key. An
# entry recorded at max_turns=20/max_tokens=12287 is therefore served verbatim to
# a run configured for 10/8191, reporting a score that episode could not have
# achieved under the tighter budget. Reusing a tag across budgets is silently
# wrong, not merely slow.
#
#   RUN_TAG="turns20_tok12287"  name the configuration (recommended)
#   RUN_TAG=""                  no tag: the legacy shared location
#   CACHE_DIR=""                disable the LM caches for this run
#   CACHE_DIR=/path/to/cache    put the LM caches somewhere specific
# ${RUN_TAG-...} (not :-) so an explicit RUN_TAG="" means "no tag" instead of
# being refilled with the default.
# Default is DERIVED from the budget/splits above rather than hardcoded, because
# those values now differ per task (ScienceWorld runs 40/24000, others 10/8191).
# A hardcoded tag would label a sciworld run "turns10" and, worse, make two
# different budgets share one cache.
RUN_TAG="${RUN_TAG-turns${MAX_TURNS}_tok${MAX_TOKENS}_v${VAL_SIZE}_tr${TRAIN_SIZE}}"
CACHE_DIR="${CACHE_DIR-__derive__}"   # __derive__ => $TEST_ROOT/lm_cache[/$RUN_TAG]
# =======================================================================

REPO_ROOT="/home/jlzeng/code/gepa_new"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p "$TEST_ROOT"

# --- AgentGym resources (Resource layer) ---------------------------------
export TOOL_LOOP_DATA_ROOT="${TOOL_LOOP_DATA_ROOT:-/home/jlzeng/code/AgentGym}"
# TOOL_LOOP_ENV_SERVER: leave unset to use the profile default (:36001 / :36002).
# export TOOL_LOOP_ENV_SERVER="http://127.0.0.1:36001"

# --- Map friendly knobs -> internal env; blank knobs stay unset so main.py
#     falls back to the profile default. ----------------------------------
TOOL_LOOP_MAX_TURNS="$MAX_TURNS"
TOOL_LOOP_MAX_TOKENS="$MAX_TOKENS"
TOOL_LOOP_MAX_WORKERS="$MAX_WORKERS"
TOOL_LOOP_TRAIN_N="$TRAIN_SIZE"
TOOL_LOOP_VAL_N="$VAL_SIZE"
TOOL_LOOP_TEST_N="$TEST_SIZE"
for v in TOOL_LOOP_MAX_TURNS TOOL_LOOP_MAX_TOKENS TOOL_LOOP_MAX_WORKERS \
         TOOL_LOOP_TRAIN_N TOOL_LOOP_VAL_N TOOL_LOOP_TEST_N; do
  if [[ -z "${!v}" ]]; then unset "$v"; else export "${v?}"; fi
done

# --- Solver --------------------------------------------------------------
export AIME_SOLVER_MAX_TOKENS="$SOLVER_MAX_TOKENS"
export AIME_SOLVER_TEMPERATURE="$SOLVER_TEMPERATURE"

# --- Run configuration ---------------------------------------------------
export AIME_SEED="$SEED"
export AIME_MAX_METRIC_CALLS="$METRIC_CALLS"
export AIME_NUM_PARALLEL_PROPOSALS="${AIME_NUM_PARALLEL_PROPOSALS:-5}"
export AIME_REFLECTION_MINIBATCH_SIZE="${AIME_REFLECTION_MINIBATCH_SIZE:-3}"
# SKIP_BASELINE_EVAL (0/1) is the friendly knob; main.py reads the true/false
# form. An explicit AIME_SKIP_BASELINE_EVAL still wins, for parity with the
# other AIME_* overrides.
if [[ -z "${AIME_SKIP_BASELINE_EVAL:-}" ]]; then
  case "$SKIP_BASELINE_EVAL" in
    1|true|True|yes|YES) AIME_SKIP_BASELINE_EVAL="true" ;;
    0|false|False|no|NO|"") AIME_SKIP_BASELINE_EVAL="false" ;;
    *) echo "[config] ERROR: SKIP_BASELINE_EVAL must be 0 or 1, got '$SKIP_BASELINE_EVAL'" >&2; exit 2 ;;
  esac
fi
export AIME_SKIP_BASELINE_EVAL
# RUN_TAG (set in RUN CONFIG above) segments the run dir, which is also where the
# score cache lives. An explicit AIME_RUN_DIR still wins.
export AIME_RUN_DIR="${AIME_RUN_DIR:-$TEST_ROOT/${RUN_TAG:+$RUN_TAG/}run_seed_${AIME_SEED}}"
# Persist the reflective datasets (iterations/<id>/reflective_dataset.json) —
# the actual <side_info> the reflection LM saw. 0 to skip (saves disk).
export TOOL_LOOP_WRITE_AGENT_STATE="${TOOL_LOOP_WRITE_AGENT_STATE:-1}"
# On-disk LM cache, so re-running the same config replays instead of paying for
# the completions again. main.py splits it into solver/ and reflection/
# subdirectories (the reflection LM bypasses dspy's cache and needs its own).
#
# PER-STRATEGY by construction: TEST_ROOT already differs between the baseline
# and captransfer scripts. That is deliberate — a shared cache would let the
# second strategy answer from the first strategy's completions whenever they
# happened to propose the same candidate text, making a per-strategy API-cost
# comparison meaningless. Point both at one directory only if you want maximum
# cache reuse and do not care about attributing cost.
#
# CACHE_DIR is set in RUN CONFIG above. The "__derive__" sentinel (rather than an
# empty default) is what lets CACHE_DIR="" mean "disable" instead of being
# refilled: with :- an explicit empty string is indistinguishable from unset.
# RUN_TAG is in the derived path for the same reason it is in the run dir — a
# completion recorded under a different per-turn token cap should not be replayed
# into a run with a different one.
if [[ "$CACHE_DIR" == "__derive__" ]]; then CACHE_DIR="$TEST_ROOT/lm_cache${RUN_TAG:+/$RUN_TAG}"; fi
if [[ -n "$CACHE_DIR" ]]; then
  export AIME_CACHE_DIR="${AIME_CACHE_DIR:-$CACHE_DIR}"
else
  unset AIME_CACHE_DIR
fi

# Introspection hook: print the resolved paths and exit WITHOUT starting a server
# or spending anything. Lets a wrapper (run_tool_loop_alfworld_both.sh) learn the
# run_dir this script would use instead of duplicating the formula and drifting.
if [[ -n "${TOOL_LOOP_PRINT_CONFIG:-}" ]]; then
  echo "RUN_TAG=$RUN_TAG"
  echo "AIME_RUN_DIR=$AIME_RUN_DIR"
  echo "AIME_CACHE_DIR=${AIME_CACHE_DIR:-}"
  exit 0
fi

# --- Models / API --------------------------------------------------------
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Please set DEEPSEEK_API_KEY}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
export AIME_DEEPSEEK_MODEL="${AIME_DEEPSEEK_MODEL:-openai/deepseek-v4-flash}"
export AIME_REFLECTION_MODEL="${AIME_REFLECTION_MODEL:-openai/deepseek-v4-pro}"

# --- Env server (auto start/stop; TOOL_LOOP_MANAGE_SERVER=0 to opt out) ---
# shellcheck source=_tool_loop_server.sh
source "$(dirname "${BASH_SOURCE[0]}")/_tool_loop_server.sh"
tool_loop_start_server "$TOOL_LOOP_TASK"

# Mirror stdout+stderr into the run dir. Without this the config echo, the
# per-iteration progress, the final scores, and the optimized prompt live only
# in the terminal scrollback. PYTHONUNBUFFERED keeps the tee'd log in real order.
mkdir -p "$AIME_RUN_DIR"
export PYTHONUNBUFFERED=1
# Echoed into the log so a run is self-describing: stdout.log is appended across
# runs, and the final scores are meaningless without knowing whether the seed
# prompt was scored at all.
echo "[config] task=$TOOL_LOOP_TASK strategy=$GEPA_SAMPLING_STRATEGY seed=$SEED" \
     "metric_calls=$METRIC_CALLS skip_baseline_eval=$AIME_SKIP_BASELINE_EVAL" \
     "test_size=${TOOL_LOOP_TEST_N:-profile} run_tag=${RUN_TAG:-none}"
echo "[config] run_dir  = $AIME_RUN_DIR   (holds fitness_cache: score cache)"
echo "[config] lm_cache = ${AIME_CACHE_DIR:-disabled}"
python -m examples.tool_loop.main 2>&1 | tee -a "$AIME_RUN_DIR/stdout.log"
# tee is last in the pipe, so its exit status would mask a python failure.
exit "${PIPESTATUS[0]}"
