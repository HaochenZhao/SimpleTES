#!/usr/bin/env bash
# SIGINT the running gpukernel_zero_trimul main.py so it writes a final checkpoint.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

sent=0
while read -r pid; do
  [[ -z "${pid}" ]] && continue
  cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  if echo "${cmd}" | grep -q 'gpukernel_zero_trimul' && echo "${cmd}" | grep -q 'main.py'; then
    echo "SIGINT pid=${pid} cmd=${cmd}"
    kill -INT "${pid}" 2>/dev/null || true
    sent=1
  fi
done < <(pgrep -f 'main.py' || true)

if [[ "${sent}" -eq 0 ]]; then
  echo "no_main_py_found"
  exit 0
fi

for _ in $(seq 1 180); do
  alive=0
  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    if echo "${cmd}" | grep -q 'gpukernel_zero_trimul' && echo "${cmd}" | grep -q 'main.py'; then
      alive=1
    fi
  done < <(pgrep -f 'main.py' || true)
  if [[ "${alive}" -eq 0 ]]; then
    echo "search_stopped"
    exit 0
  fi
  sleep 2
done

echo "search_still_alive_after_wait sending TERM"
while read -r pid; do
  [[ -z "${pid}" ]] && continue
  cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  if echo "${cmd}" | grep -q 'gpukernel_zero_trimul' && echo "${cmd}" | grep -q 'main.py'; then
    kill -TERM "${pid}" 2>/dev/null || true
  fi
done < <(pgrep -f 'main.py' || true)
sleep 10
echo "stop_search_done"
