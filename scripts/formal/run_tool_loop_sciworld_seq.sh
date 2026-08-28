#!/usr/bin/env bash
# 依次运行 ScienceWorld 的两个策略脚本：baseline，然后 captransfer。
# 直接调用子脚本，不做任何参数传递/继承；每个子脚本自己读取自己的配置。
#
# 用法:
#   bash scripts/formal/run_tool_loop_sciworld_seq.sh
#
# 注意: 不用 -e，第一个脚本失败时第二个仍然会执行。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
  "$SCRIPT_DIR/run_tool_loop_sciworld_gepa.sh"
  "$SCRIPT_DIR/run_tool_loop_sciworld_gepa_captransfer.sh"
)

FAILURES=0
declare -a RESULTS=()

for script in "${SCRIPTS[@]}"; do
  name="$(basename "$script")"
  echo "======================================================================"
  echo "[seq] running $name"
  echo "======================================================================"

  start=$(date +%s)
  bash "$script"
  code=$?
  elapsed=$(( $(date +%s) - start ))

  if (( code == 0 )); then
    echo "[seq] OK   $name in ${elapsed}s"
  else
    echo "[seq] FAIL $name exit=$code after ${elapsed}s"
    FAILURES=$(( FAILURES + 1 ))
  fi
  RESULTS+=("$name $code ${elapsed}s")
  echo
done

echo "======================================================================"
echo "[seq] 全部结束，$FAILURES 个失败"
printf '%-52s %-6s %s\n' SCRIPT EXIT ELAPSED
for row in "${RESULTS[@]}"; do
  read -r name code elapsed <<< "$row"
  printf '%-52s %-6s %s\n' "$name" "$code" "$elapsed"
done

exit $(( FAILURES > 0 ? 1 : 0 ))
