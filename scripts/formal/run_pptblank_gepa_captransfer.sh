#!/usr/bin/env bash
# PPT excessive-whitespace detection: capability-transfer UCB batch sampling.
#
# Requires the local dataset first:
#   python -m examples.aime_math.prepare_pptblank
#
# Both solver and reflector must be vision models (the solver reads the slide, the
# reflector reads the box-annotated slide). Routed through api.luminai.cc.
set -euo pipefail

REPO_ROOT="/home/jlzeng/code/gepa_new"
TEST_ROOT="$REPO_ROOT/examples/aime_math/test/pptblank_formal/gepa_captransfer"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p "$TEST_ROOT"

# --- Dataset --------------------------------------------------------------
export AIME_DATASET="pptblank"
# 45 train / 48 val / 138 test, split by source .pptx (two decks per style in
# train and val) so slides sharing a template never straddle splits.

# --- Parent selection -----------------------------------------------------
# Left unset = GEPA's default "pareto", keeping this run comparable to the math
# tasks. Note what pareto does here: it samples a parent in proportion to how many
# val examples the candidate is on the Pareto front for, and on this task that is
# nearly uniform — 16 of 48 val examples are solved by *every* candidate, so each
# hands every candidate a front slot and the strong/weak gap washes out (measured:
# the best candidate was only 1.28x more likely to be picked than the worst).
# Set to "current_best" to hill-climb on the aggregate score, or "top_k_pareto"
# for a middle ground that keeps some diversity. See examples/aime_math/README.md.
# export GEPA_CANDIDATE_SELECTOR="current_best"

# --- Batch sampling strategy ---------------------------------------------
export GEPA_SAMPLING_STRATEGY="capability_transfer"
export GEPA_CT_TAU="${GEPA_CT_TAU:-0.5}"
export GEPA_CT_ALPHA="${GEPA_CT_ALPHA:-1.0}"
export GEPA_CT_BETA="${GEPA_CT_BETA:-1.0}"
export GEPA_CT_LAMBDA="${GEPA_CT_LAMBDA:-0.2}"
export GEPA_CT_COLD_START_BONUS="${GEPA_CT_COLD_START_BONUS:-0.2}"

# --- Solver ---------------------------------------------------------------
# CoT stays on: the verdict benefits from an explicit scan of the layout before
# committing, and the reasoning field also lands in the reflection payload.
export AIME_SOLVER_USE_COT="${AIME_SOLVER_USE_COT:-1}"
# gemini-3-flash spends ~70-300 tokens of hidden reasoning on this task even for a
# one-word answer, so a tight ceiling would truncate before the verdict appears.
export AIME_SOLVER_MAX_TOKENS="${AIME_SOLVER_MAX_TOKENS:-2000}"
# Deterministic verdicts: this is a classification task, not generation.
export AIME_SOLVER_TEMPERATURE="${AIME_SOLVER_TEMPERATURE:-0.0}"

# --- Seed prompt ----------------------------------------------------------
# The reviewer's own checklist, translated, with the output format adapted to the
# single-word verdict the scorer reads.
export AIME_INITIAL_PROMPT="$(cat "$REPO_ROOT/examples/aime_math/prompts/pptblank_seed.txt")"

# --- Run configuration ----------------------------------------------------
export AIME_SEED="${AIME_SEED:-42}"
export AIME_MAX_METRIC_CALLS="${AIME_MAX_METRIC_CALLS:-2500}"
# Lower than the text tasks: every call ships a ~75 KB base64 image, and the
# upstream pool returns transient RateLimitError under heavy concurrency.
export AIME_MAX_WORKERS="${AIME_MAX_WORKERS:-6}"
export AIME_EVAL_NUM_THREADS="${AIME_EVAL_NUM_THREADS:-6}"
export AIME_NUM_PARALLEL_PROPOSALS="${AIME_NUM_PARALLEL_PROPOSALS:-2}"
export AIME_REFLECTION_MINIBATCH_SIZE="${AIME_REFLECTION_MINIBATCH_SIZE:-3}"
export AIME_SKIP_BASELINE_EVAL="${AIME_SKIP_BASELINE_EVAL:-false}"
export AIME_RUN_DIR="${AIME_RUN_DIR:-$TEST_ROOT/run_seed_${AIME_SEED}}"

# --- Models / API --------------------------------------------------------
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Please set DEEPSEEK_API_KEY}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.luminai.cc/v1}"
export AIME_DEEPSEEK_MODEL="${AIME_DEEPSEEK_MODEL:-openai/gemini-3-flash}"
export AIME_REFLECTION_MODEL="${AIME_REFLECTION_MODEL:-openai/claude-opus-4-6-thinking}"

python -m examples.aime_math.main
