#!/usr/bin/env bash
# ScienceWorld + GEPA, baseline batch sampling (IndependentSampling / epoch-shuffled).
# Optimized component = initial prompt. Budget/concurrency come from profiles.py
# (max_turns=30, max_tokens=16000 — NOT yet calibrated, see profiles.py).
# Split sizes and budget are overridable via the RUN CONFIG block in
# _tool_loop_env_common.sh, which also starts the env server (:36003, conda env
# agentenv-sciworld).
set -euo pipefail
export TOOL_LOOP_TASK="sciworld"
export GEPA_SAMPLING_STRATEGY="baseline"
TEST_ROOT="/home/jlzeng/code/gepa_new/examples/tool_loop/test/sciworld_gepa_baseline"
# shellcheck source=_tool_loop_env_common.sh
source "$(dirname "$0")/_tool_loop_env_common.sh"
