#!/usr/bin/env bash
# Run BOTH AlfWorld strategies back to back: baseline (stock GEPA sampling) then
# ours (capability-transfer UCB). The two runs share every knob except the
# sampling strategy, so the difference in their test scores is attributable to
# the strategy alone.
#
# Usage:
#   bash scripts/formal/run_tool_loop_alfworld_both.sh
#   SEED=7 bash scripts/formal/run_tool_loop_alfworld_both.sh          # one seed
#   SEEDS="42 7 13" bash scripts/formal/run_tool_loop_alfworld_both.sh # sweep
#   TRAIN_SIZE=4 VAL_SIZE=3 TEST_SIZE=3 METRIC_CALLS=12 \
#     bash scripts/formal/run_tool_loop_alfworld_both.sh              # smoke test
#
# Every knob from _tool_loop_env_common.sh's RUN CONFIG block (TRAIN_SIZE,
# VAL_SIZE, TEST_SIZE, MAX_TURNS, MAX_TOKENS, MAX_WORKERS, METRIC_CALLS,
# SOLVER_*) is forwarded to both child runs unchanged.
#
# PREREQUISITE: ALFWORLD_DATA must point at the AlfWorld data (default
# ~/.cache/alfworld). The env server on :36002 is started/stopped automatically
# by each child script; see _tool_loop_server.sh.
set -uo pipefail   # NOTE: no -e; a failed strategy must not skip the other one.

REPO_ROOT="/home/jlzeng/code/gepa_new"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Seeds to run. SEEDS wins; else a single SEED; else 42.
SEEDS="${SEEDS:-${SEED:-42}}"
# Order matters only for reading the log; both are independent.
STRATEGIES=(baseline captransfer)

# RUN_TAG segments each run's dir AND its caches, and we build the run_dir paths
# here, so we must agree with the children on its value. Rather than duplicating
# the child's default (which would silently drift), ask the child what it would
# use: sourcing it with TOOL_LOOP_PRINT_CONFIG=1 makes it print and exit before
# doing any work.
if [[ -z "${RUN_TAG+set}" ]]; then
  RUN_TAG="$(TOOL_LOOP_PRINT_CONFIG=1 TOOL_LOOP_MANAGE_SERVER=0 \
             DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-probe}" \
             bash "$SCRIPT_DIR/run_tool_loop_alfworld_gepa.sh" 2>/dev/null \
             | sed -n 's/^RUN_TAG=//p' | head -1)"
fi
export RUN_TAG

SUMMARY_ROOT="${SUMMARY_ROOT:-$REPO_ROOT/examples/tool_loop/test/alfworld_both}"
mkdir -p "$SUMMARY_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
COMBINED_LOG="$SUMMARY_ROOT/run_${STAMP}.log"

# Tee everything below into the combined log. Each child ALSO writes its own
# stdout.log into its run_dir; this file is the cross-strategy view. exec is
# used (rather than piping the child runs) so the final summary table is
# captured too. PYTHONUNBUFFERED keeps the interleaving faithful.
exec > >(tee -a "$COMBINED_LOG") 2>&1
export PYTHONUNBUFFERED=1

echo "[both] repo=$REPO_ROOT"
echo "[both] seeds=$SEEDS strategies=${STRATEGIES[*]}"
echo "[both] combined log -> $COMBINED_LOG"
echo "[both] forwarded RUN CONFIG overrides:"
for v in TRAIN_SIZE VAL_SIZE TEST_SIZE MAX_TURNS MAX_TOKENS MAX_WORKERS \
         METRIC_CALLS SKIP_BASELINE_EVAL SOLVER_MAX_TOKENS SOLVER_TEMPERATURE \
         AIME_DEEPSEEK_MODEL AIME_REFLECTION_MODEL; do
  [[ -n "${!v:-}" ]] && echo "[both]   $v=${!v}"
done
echo

declare -a RESULTS=()   # "seed strategy exit_code run_dir"
FAILURES=0

for seed in $SEEDS; do
  for strat in "${STRATEGIES[@]}"; do
    script="$SCRIPT_DIR/run_tool_loop_alfworld_gepa.sh"
    [[ "$strat" == "captransfer" ]] && script="$SCRIPT_DIR/run_tool_loop_alfworld_gepa_captransfer.sh"

    # Set AIME_RUN_DIR ourselves rather than re-deriving the child's default.
    # The child's path is TEST_ROOT/[RUN_TAG/]run_seed_N — mirroring that formula
    # here means it silently breaks whenever the child's layout changes (it did:
    # RUN_TAG was added and this lookup started missing every summary.json).
    # Passing the value down makes this script the single source of truth.
    test_root="$REPO_ROOT/examples/tool_loop/test/alfworld_gepa_${strat}"
    run_dir="$test_root/${RUN_TAG:+$RUN_TAG/}run_seed_${seed}"

    echo "======================================================================"
    echo "[both] seed=$seed strategy=$strat"
    echo "[both] script  = $script"
    echo "[both] run_dir = $run_dir"
    echo "======================================================================"

    start=$(date +%s)
    # SEED is the knob _tool_loop_env_common.sh reads. AIME_RUN_DIR is passed
    # explicitly so the child writes exactly where the summary table looks.
    SEED="$seed" AIME_RUN_DIR="$run_dir" bash "$script"
    code=$?
    elapsed=$(( $(date +%s) - start ))

    if (( code == 0 )); then
      echo "[both] OK   seed=$seed $strat in ${elapsed}s"
    else
      echo "[both] FAIL seed=$seed $strat exit=$code after ${elapsed}s"
      FAILURES=$(( FAILURES + 1 ))
    fi
    RESULTS+=("$seed $strat $code $run_dir")
    echo
  done
done

echo "======================================================================"
echo "[both] all runs finished; $FAILURES failure(s)"
echo "======================================================================"
printf '%-6s %-12s %-6s %-9s %-9s %s\n' SEED STRATEGY EXIT BASELINE OPTIMIZED RUN_DIR
for row in "${RESULTS[@]}"; do
  read -r seed strat code run_dir <<< "$row"
  base="-"; opt="-"
  summary="$run_dir/summary.json"
  if [[ -f "$summary" ]]; then
    # Read the two held-out test scores main.py wrote. python3 (not jq) to avoid
    # a dependency the rest of these scripts don't have.
    # "skipped" (SKIP_BASELINE_EVAL=1) is distinguished from "-" (no summary /
    # unreadable), so a missing number is never mistaken for a deliberate skip.
    read -r base opt < <(python3 -c "
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    def f(v): return '-' if v is None else f'{v:.4f}'
    b = d.get('test_baseline_score')
    b = 'skipped' if b is None and d.get('test_optimized_score') is not None else f(b)
    print(b, f(d.get('test_optimized_score')))
except Exception:
    print('- -')
" "$summary")
  fi
  printf '%-6s %-12s %-6s %-9s %-9s %s\n' "$seed" "$strat" "$code" "$base" "$opt" "$run_dir"
done

echo
echo "[both] per-run artifacts: summary.json, best_prompt.txt, run_config.json,"
echo "[both]   test_episodes_{baseline,optimized}.json, run_log.txt, stdout.log,"
echo "[both]   iterations/<id>/reflective_dataset.json"
echo "[both] combined log: $COMBINED_LOG"

exit $(( FAILURES > 0 ? 1 : 0 ))
