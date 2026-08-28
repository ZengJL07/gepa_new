#!/usr/bin/env bash
# TextCraft + GEPA, baseline batch sampling (IndependentSampling / epoch-shuffled).
# Optimized component = initial prompt. Budget/concurrency come from profiles.py
# (max_turns=20, max_tokens=16000, max_workers=8) unless overridden via env.
# See _tool_loop_env_common.sh for the server prerequisites.
set -euo pipefail
export TOOL_LOOP_TASK="textcraft"
export GEPA_SAMPLING_STRATEGY="baseline"
TEST_ROOT="/home/jlzeng/code/gepa_new/examples/tool_loop/test/textcraft_gepa_baseline"
# shellcheck source=_tool_loop_env_common.sh
source "$(dirname "$0")/_tool_loop_env_common.sh"
