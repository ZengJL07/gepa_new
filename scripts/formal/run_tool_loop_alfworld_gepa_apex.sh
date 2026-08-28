#!/usr/bin/env bash
# AlfWorld + GEPA, APEX dynamic data selection (arXiv:2606.11459v1).
# Optimized component = initial prompt. Budget/concurrency/splits come from the
# shared RUN CONFIG block in _tool_loop_env_common.sh, so this run and the
# baseline run differ ONLY in the data-selection strategy (which for APEX means
# four strategies at once: sampling, evaluation policy, candidate selector and
# reflection). Server prerequisites (ALFWORLD_DATA + :36002) are listed there.
#
# DATASET — APEX uses a SINGLE development set D for both mutation and
# selection: its nine buckets B[tier, s] intersect tiers derived from the
# history H with outcomes on the same examples, so a disjoint train/val split
# leaves them ill-defined. main.py therefore merges train+val into D
# (40+45 = 85) and passes valset=None; only the test split (100) is held out.
# Total data touched is identical to the baseline run, so the comparison is fair.
#
# BINARIZATION — this task is a natural fit: score_env_episode returns 1.0 iff
# the env task was solved and 0.0 otherwise, so Section 3.1's "only a perfect
# score yields a pass" is exact here rather than an approximation of a
# continuous metric (as it is on MATH-500).
set -euo pipefail
export TOOL_LOOP_TASK="alfworld"
export GEPA_SAMPLING_STRATEGY="apex"

# --- Episode budget ------------------------------------------------------
# Aligned with the captransfer run this is compared against
# (test/alfworld_gepa_captransfer/turns10_tok8191_v45_tr40/run_seed_42, whose
# run_config.json records episode_max_turns=10 / episode_max_tokens=8191), NOT
# with the shared RUN CONFIG defaults of 40/18431. The episode budget is not part
# of the fitness-cache key, so a score recorded at 40/18431 would be served
# verbatim to a run configured for 10/8191 -- comparing against that run requires
# matching it here.
MAX_TURNS="${MAX_TURNS:-10}"
MAX_TOKENS="${MAX_TOKENS:-8191}"

# --- APEX hyperparameters (paper values) ---------------------------------
# Lineage lookback window k. Table 4: 50.3 / 52.3 / 50.6 for k = 3 / 5 / 10 --
# a wider window reintroduces the stale signals the method exists to discard.
export GEPA_APEX_LOOKBACK="${GEPA_APEX_LOOKBACK:-5}"
# Per-iteration evaluation budget N for the rank-sensitive policy. Unset =>
# 18% of |D| (= 15 at |D|=85), preserving the paper's RATIO (N=100 against |D|
# of 500-700) rather than its absolute value: at N >= |D| the policy degrades to
# full evaluation and Section 4.3 stops doing anything at all.
# export GEPA_APEX_N_EVAL="15"
# Anchor ratio schedule (Eq. 9 / Eq. 10). Paper: alpha_0 = 0.2, beta = 0.03.
export GEPA_APEX_ALPHA0="${GEPA_APEX_ALPHA0:-0.2}"
export GEPA_APEX_BETA="${GEPA_APEX_BETA:-0.03}"
# Pass threshold. 1.0 is the exact success value of this task's scorer, not a
# tuned knob -- see BINARIZATION above.
export GEPA_APEX_PERFECT_SCORE="${GEPA_APEX_PERFECT_SCORE:-1.0}"

# n_parallel / minibatch / budget / split sizes / episode budget / models all
# come from the shared RUN CONFIG, so this differs from the baseline ONLY in the
# strategy block above.
TEST_ROOT="/home/jlzeng/code/gepa_new/examples/tool_loop/test/alfworld_gepa_apex"

# RUN_TAG is left to the shared formula (turns<N>_tok<N>_v<VAL>_tr<TRAIN>) so this
# run lands under the same tag as the captransfer run it is compared against,
# making the two directories trivially pairable. The tag names a train/val split
# that APEX does not literally have (it merges the two into D=85), but the split
# SIZES it encodes are the ones requested, and TEST_ROOT already isolates both the
# fitness cache and the LM cache per strategy, so nothing can be cross-served.

# shellcheck source=_tool_loop_env_common.sh
source "$(dirname "$0")/_tool_loop_env_common.sh"
