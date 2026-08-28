#!/usr/bin/env bash
# APEX: dynamic data selection (arXiv:2606.11459v1), on MATH-500. Config kept
# identical to run_math500_gepa_captransfer.sh except for the strategy block.
#
# NOTE on the dataset: APEX uses a SINGLE development set D for both mutation
# and selection -- its nine buckets B[tier, s] intersect tiers derived from the
# history H with outcomes on the same examples, so a disjoint train/val split
# makes them ill-defined. main.py therefore merges train+val into D (40+45=85)
# and passes valset=None; only the test split (100) stays held out. Total data
# touched is identical to the baseline runs, so the comparison stays fair.
set -euo pipefail

REPO_ROOT="/home/jlzeng/code/gepa_new"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# --- Dataset --------------------------------------------------------------
export AIME_DATASET="${AIME_DATASET:-math500}"          # registered dataset: aime | math500
# Per-split size caps (0/empty = full split). AIME_TRIM_K caps all three splits;
# AIME_TRAIN_K / AIME_VAL_K / AIME_TEST_K override an individual split (unset =
# inherit AIME_TRIM_K). train+val are merged into D under the apex strategy.
export AIME_TRIM_K="${AIME_TRIM_K:-0}"                  # cap for every split
export AIME_TRAIN_K="${AIME_TRAIN_K:-40}"               # train split size  -> D
export AIME_VAL_K="${AIME_VAL_K:-45}"                   # val split size    -> D  (|D| = 85)
export AIME_TEST_K="${AIME_TEST_K:-100}"                # test split size (held out)

# --- Batch sampling strategy ---------------------------------------------
export GEPA_SAMPLING_STRATEGY="apex"    # baseline (stock GEPA) | capability_transfer | apex
# Lineage lookback window k (paper: 5). Table 4 reports 50.3 / 52.3 / 50.6 for
# k = 3 / 5 / 10 -- a wider window reintroduces stale signals.
export GEPA_APEX_LOOKBACK="${GEPA_APEX_LOOKBACK:-5}"
# Per-iteration evaluation budget N for the rank-sensitive policy. Unset ->
# 18% of |D| (= 15 at |D|=85), preserving the paper's ratio (N=100 against |D|
# of 500-700) rather than its absolute value: with N >= |D| the policy degrades
# to full evaluation and Section 4.3 stops doing anything.
# export GEPA_APEX_N_EVAL="15"
# Anchor ratio schedule (Eq. 9/10). Paper: alpha_0 = 0.2, beta = 0.03.
export GEPA_APEX_ALPHA0="${GEPA_APEX_ALPHA0:-0.2}"
export GEPA_APEX_BETA="${GEPA_APEX_BETA:-0.03}"
# Pass threshold for binarizing scores. Section 3.1 counts only a perfect score
# as a pass, so this tracks the metric's maximum rather than being tuned.
export GEPA_APEX_PERFECT_SCORE="${GEPA_APEX_PERFECT_SCORE:-1.0}"

# --- Prompt ---------------------------------------------------------------
# Seed prompt GEPA starts optimizing from (also the baseline eval prompt).
export AIME_INITIAL_PROMPT="${AIME_INITIAL_PROMPT:-Solve the math problem carefully. Break down the steps and provide the final answer as a single number.}"

# --- Solver ---------------------------------------------------------------
# CoT off: use a bare dspy.Predict (answer only), no explicit reasoning field.
# Redundant + truncation-prone for models that already reason internally.
export AIME_SOLVER_USE_COT="${AIME_SOLVER_USE_COT:-0}"          # 1 = ChainOfThought, 0 = bare Predict
export AIME_SOLVER_TEMPERATURE="${AIME_SOLVER_TEMPERATURE:-0.2}"  # solver sampling temperature
# Keep within the provider ceiling (deepseek allows 32000; Aliyun qwen3-8b 8192).
export AIME_SOLVER_MAX_TOKENS="${AIME_SOLVER_MAX_TOKENS:-8191}"   # solver max output tokens
# Reflection temperature is optional; leave unset to use the provider default.
# export AIME_REFLECTION_TEMPERATURE="1.0"                       # reflection model temperature

# --- Where to save --------------------------------------------------------
# Run dir encodes the solver temperature so runs at different temperatures land
# in separate directories.
TEST_ROOT="$REPO_ROOT/examples/aime_math/test/${AIME_DATASET}_formal/gepa_apex"
export AIME_SEED="${AIME_SEED:-42}"                     # split shuffle + optimizer seed
export AIME_RUN_DIR="${AIME_RUN_DIR:-$TEST_ROOT/t_${AIME_SOLVER_TEMPERATURE}/run_seed_${AIME_SEED}}"
# Share the on-disk LM cache with eval runs to avoid paying for repeat completions.
export AIME_CACHE_DIR="${AIME_CACHE_DIR:-$TEST_ROOT/lm_cache}"
mkdir -p "$TEST_ROOT"

# --- Run configuration ----------------------------------------------------
export AIME_MAX_METRIC_CALLS="${AIME_MAX_METRIC_CALLS:-1000}"        # BUDGET: total metric (rollout) calls
export AIME_MAX_WORKERS="${AIME_MAX_WORKERS:-15}"                   # concurrent solver requests
export AIME_NUM_PARALLEL_PROPOSALS="${AIME_NUM_PARALLEL_PROPOSALS:-5}"  # candidates proposed per iteration
export AIME_REFLECTION_MINIBATCH_SIZE="${AIME_REFLECTION_MINIBATCH_SIZE:-3}"  # train_batch size per proposal (paper's m)
export AIME_EVAL_NUM_THREADS="${AIME_EVAL_NUM_THREADS:-16}"         # concurrency of the final test eval
export AIME_SKIP_BASELINE_EVAL="${AIME_SKIP_BASELINE_EVAL:-false}"  # true = skip scoring the seed prompt on test

# --- Models / API ---------------------------------------------------------
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Please set DEEPSEEK_API_KEY}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
export AIME_DEEPSEEK_MODEL="${AIME_DEEPSEEK_MODEL:-openai/deepseek-v4-flash}"   # solver model
export AIME_REFLECTION_MODEL="${AIME_REFLECTION_MODEL:-openai/deepseek-v4-pro}"  # reflection model

python -m examples.aime_math.main
