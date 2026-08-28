#!/usr/bin/env bash
# Reusable start/stop for the AgentGym env servers (textcraft / alfworld).
# Source this, then call `tool_loop_start_server <task>` before launching python.
# An EXIT/INT/TERM trap stops only the server THIS script started (a server that
# was already running is reused and left untouched).
#
# Toggle: TOOL_LOOP_MANAGE_SERVER=1 (default) auto start/stop; 0 = assume it is
# already running (old behavior). Requires conda envs agentenv-textcraft /
# agentenv-alfworld and their `textcraft` / `alfworld` entrypoints.
#
# Relies on the caller having set (already true in the run/eval scripts):
#   TOOL_LOOP_DATA_ROOT   AgentGym checkout root
#   TEST_ROOT             dir for server logs (falls back to /tmp)
# Optional:
#   TOOL_LOOP_ENV_SERVER  full base URL; its port overrides the task default
#   ALFWORLD_DATA         AlfWorld data dir (default ~/.cache/alfworld)

# PGID of the server we launched (empty => we manage nothing).
_TL_SERVER_PGID=""
_TL_SERVER_NAME=""

_tl_conda() {
  # Resolve a conda executable without requiring `conda` on PATH.
  if command -v conda >/dev/null 2>&1; then echo "conda"; return; fi
  for c in "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda"; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
  echo ""  # not found
}

_tl_default_port() {
  case "$1" in
    textcraft) echo 36001 ;;
    alfworld)  echo 36002 ;;
    sciworld)  echo 36003 ;;
    webshop)   echo 36004 ;;
    *) echo "" ;;
  esac
}

_tl_port() {
  # TOOL_LOOP_ENV_SERVER port wins; else the task default.
  local task="$1" default; default="$(_tl_default_port "$task")"
  if [[ -n "${TOOL_LOOP_ENV_SERVER:-}" ]]; then
    local p="${TOOL_LOOP_ENV_SERVER##*:}"; p="${p%%/*}"
    [[ "$p" =~ ^[0-9]+$ ]] && { echo "$p"; return; }
  fi
  echo "$default"
}

_tl_probe() {
  # 0 if a server answers on the port, 1 otherwise.
  curl --noproxy '*' -s -o /dev/null --max-time 3 "http://127.0.0.1:$1/" 2>/dev/null
}

tool_loop_stop_server() {
  # Kill the whole process group we started (uvicorn spawns children).
  [[ -z "$_TL_SERVER_PGID" ]] && return 0
  echo "[server] stopping $_TL_SERVER_NAME (pgid $_TL_SERVER_PGID)"
  kill -TERM -- "-$_TL_SERVER_PGID" 2>/dev/null || true
  # Give it a moment, then hard-kill any survivors.
  for _ in 1 2 3 4 5; do
    kill -0 -- "-$_TL_SERVER_PGID" 2>/dev/null || { _TL_SERVER_PGID=""; return 0; }
    sleep 1
  done
  kill -KILL -- "-$_TL_SERVER_PGID" 2>/dev/null || true
  _TL_SERVER_PGID=""
}

tool_loop_start_server() {
  local task="$1"
  [[ "${TOOL_LOOP_MANAGE_SERVER:-1}" != "1" ]] && { echo "[server] MANAGE_SERVER=0; assuming $task is already running"; return 0; }

  local port; port="$(_tl_port "$task")"
  [[ -z "$port" ]] && { echo "[server] no default port for task '$task'"; return 1; }

  if _tl_probe "$port"; then
    echo "[server] $task already up on :$port — reusing (not managed by this script)"
    return 0
  fi

  local conda; conda="$(_tl_conda)"
  [[ -z "$conda" ]] && { echo "[server] ERROR: conda not found; cannot start $task"; return 1; }

  local root="${TOOL_LOOP_DATA_ROOT:-/home/jlzeng/code/AgentGym}"
  local logdir="${TEST_ROOT:-/tmp}"; mkdir -p "$logdir"
  local log="$logdir/server_${task}.log"

  # Idle keep-alive window. AgentGym's own entrypoints call uvicorn.run() without
  # arguments, so they take uvicorn's default of 5 SECONDS — far shorter than the
  # gap between two /step calls, which spans a full solver generation (up to 20
  # turns of LLM latency). The server then closes its end of the pooled
  # connection; the client only discovers this when it writes, surfacing as
  # ConnectionResetError(104) mid-episode. Raising it removes the cause rather
  # than relying on client-side retries alone.
  local keepalive="${TOOL_LOOP_SERVER_KEEPALIVE:-600}"

  local cmd env_name workdir module
  case "$task" in
    textcraft)
      env_name="agentenv-textcraft"
      workdir="$root/agentenv-textcraft"   # recipes are read via relative paths
      module="agentenv_textcraft"
      ;;
    alfworld)
      env_name="agentenv-alfworld"
      workdir="$root"
      module="agentenv_alfworld"
      export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
      ;;
    sciworld)
      # ScienceWorld ships its own game data inside the pip package (JVM assets
      # via py4j), so there is no data-root env var to set.
      env_name="agentenv-sciworld"
      workdir="$root"
      module="agentenv_sciworld"
      ;;
    webshop)
      # Must run from the webshop/ checkout: web_agent_site resolves its product
      # dump and Lucene index through paths relative to that package.
      env_name="agentenv-webshop"
      workdir="$root/agentenv-webshop"
      module="agentenv_webshop"
      ;;
    *) echo "[server] unknown task '$task'"; return 1 ;;
  esac

  # Resolve the env's interpreter by ABSOLUTE PATH rather than relying on
  # `conda run ... python`. The callers activate the repo's .venv first, and that
  # leaves VIRTUAL_ENV set with .venv/bin ahead of the conda env on PATH — so a
  # bare `python` under `conda run` resolves to the .venv interpreter, which has
  # no agentenv_* package. That is the "Could not import module" failure: the
  # right env was requested, the wrong interpreter ran.
  # Ask conda where the env lives (`conda run python` cannot be trusted here — it
  # would report the shadowing .venv's prefix), then use that env's own bin/python.
  local py env_prefix
  env_prefix="$("$conda" run -n "$env_name" printenv CONDA_PREFIX 2>/dev/null | tr -d '\r')"
  py="$env_prefix/bin/python"
  if [[ -z "$env_prefix" || ! -x "$py" ]]; then
    # Standard conda layout, derived from the conda executable's location.
    py="$(dirname "$(dirname "$conda")")/envs/$env_name/bin/python"
  fi
  if [[ ! -x "$py" ]]; then
    echo "[server] ERROR: no python found for conda env '$env_name' (looked at $py)"; return 1
  fi

  # Invoke uvicorn directly instead of the console script: the AgentGym
  # entrypoints hardcode uvicorn.run() and offer no way to pass
  # --timeout-keep-alive.
  cmd="'$py' -m uvicorn $module:app --host 0.0.0.0 --port $port --timeout-keep-alive $keepalive"

  echo "[server] starting $task on :$port (conda env $env_name), log -> $log"
  # setsid => the server gets its own process group, so the trap can reap uvicorn
  # together with any children. The env's interpreter is exec'd DIRECTLY rather
  # than via `conda run`, which forks a child and exits after its bookkeeping —
  # leaving nothing useful to track or signal.
  #
  # The server's pgid is reported by the server process itself, via a pidfile.
  # $! cannot be used: setsid forks the real process into a NEW process group and
  # the pid bash captures belongs to the short-lived setsid parent. Killing
  # "-$!" then addressed a process group that never existed — silently failing,
  # so tool_loop_stop_server reported success while uvicorn kept running (a real
  # leak, observed), and the liveness check misread a healthy server as dead.
  # Inside setsid the shell IS the session/group leader, so its $$ is the pgid.
  local pidfile; pidfile="$(mktemp)"
  setsid bash -c "echo \$\$ > '$pidfile'; cd '$workdir' && exec $cmd" >"$log" 2>&1 &
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    _TL_SERVER_PGID="$(cat "$pidfile" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "$_TL_SERVER_PGID" ]] && break
    sleep 0.2
  done
  rm -f "$pidfile"
  if [[ -z "$_TL_SERVER_PGID" ]]; then
    echo "[server] ERROR: could not determine $task server pgid; see $log"; return 1
  fi
  _TL_SERVER_PGID=$!
  _TL_SERVER_NAME="$task"
  trap tool_loop_stop_server EXIT INT TERM

  # Wait for readiness. The HTTP app comes up in seconds; AlfWorld's slow part is
  # the first /reset (game compile), which the client's timeout absorbs later.
  local waited=0 timeout="${TOOL_LOOP_SERVER_START_TIMEOUT:-120}"
  while ! _tl_probe "$port"; do
    if ! kill -0 -- "-$_TL_SERVER_PGID" 2>/dev/null; then
      echo "[server] ERROR: $task exited during startup; see $log"; tail -20 "$log" 2>/dev/null; return 1
    fi
    sleep 2; waited=$((waited + 2))
    if (( waited >= timeout )); then
      echo "[server] ERROR: $task not ready after ${timeout}s; see $log"; tail -20 "$log" 2>/dev/null
      tool_loop_stop_server; return 1
    fi
  done
  echo "[server] $task ready on :$port after ${waited}s"
}
