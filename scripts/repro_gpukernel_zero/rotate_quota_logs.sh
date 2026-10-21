#!/usr/bin/env bash
# Truncate the shared kernel logs that previously hit errno 122.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
cd "${KERNEL_LOG_DIR}"
shopt -s nullglob
for f in kernel_*.log kernel_server.log burn_*.log resume.log align_then_run.log; do
  [[ -f "${f}" ]] || continue
  sz="$(stat -c%s "${f}" 2>/dev/null || echo 0)"
  if [[ "${sz}" -gt 5000000 ]]; then
    ts="$(date +%Y%m%d_%H%M%S)"
    tail -n 80 "${f}" > "${f}.tail.${ts}" || true
    : > "${f}"
    echo "truncated ${f} was_bytes=${sz}"
  fi
done
# Drop leftover prev logs that are huge.
for f in kernel_*.log.prev.* ; do
  [[ -f "${f}" ]] || continue
  sz="$(stat -c%s "${f}" 2>/dev/null || echo 0)"
  if [[ "${sz}" -gt 20000000 ]]; then
    rm -f "${f}"
    echo "removed ${f} was_bytes=${sz}"
  fi
done
echo ROTATE_DONE
