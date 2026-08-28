#!/usr/bin/env bash
# AlfWorld + GEPA, baseline batch sampling (IndependentSampling / epoch-shuffled).
# Optimized component = initial prompt. Budget/concurrency come from profiles.py
# (max_turns=20, max_tokens=12287 — deliberately tight so efficient planning is
# what's optimized; max_workers=3 because AlfWorld has big observations, slow
# reset, and no /close). Split sizes and budget are overridable via the RUN
# CONFIG block in _tool_loop_env_common.sh, which also lists server
# prerequisites (ALFWORLD_DATA + :36002).
set -euo pipefail
export TOOL_LOOP_TASK="alfworld"
export GEPA_SAMPLING_STRATEGY="baseline"
TEST_ROOT="/home/jlzeng/code/gepa_new/examples/tool_loop/test/alfworld_gepa_baseline"
# shellcheck source=_tool_loop_env_common.sh
source "$(dirname "$0")/_tool_loop_env_common.sh"
