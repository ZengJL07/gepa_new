#!/usr/bin/env bash
# TextCraft + GEPA, ours: capability-transfer UCB batch sampling.
# Optimized component = initial prompt. Budget/concurrency come from profiles.py
# (max_turns=20, max_tokens=16000, max_workers=8) unless overridden via env.
# See _tool_loop_env_common.sh for the server prerequisites.
set -euo pipefail
export TOOL_LOOP_TASK="textcraft"
export GEPA_SAMPLING_STRATEGY="capability_transfer"
export GEPA_CT_TAU="${GEPA_CT_TAU:-0.5}"
export GEPA_CT_ALPHA="${GEPA_CT_ALPHA:-1.0}"
export GEPA_CT_BETA="${GEPA_CT_BETA:-1.0}"
export GEPA_CT_LAMBDA="${GEPA_CT_LAMBDA:-0.2}"
export GEPA_CT_COLD_START_BONUS="${GEPA_CT_COLD_START_BONUS:-0.2}"
# n_parallel / minibatch / budget / split sizes come from the shared RUN CONFIG
# so this run and the baseline run differ ONLY in the sampling strategy.
TEST_ROOT="/home/jlzeng/code/gepa_new/examples/tool_loop/test/textcraft_gepa_captransfer"
# shellcheck source=_tool_loop_env_common.sh
source "$(dirname "$0")/_tool_loop_env_common.sh"
