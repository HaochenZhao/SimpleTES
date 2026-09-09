#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/logs"
# Rotate previous failed init log
if [[ -f "${HERE}/logs/simpletes.log" ]]; then
  mv "${HERE}/logs/simpletes.log" "${HERE}/logs/simpletes.log.prev.$(date +%H%M%S)"
fi
nohup bash "${HERE}/run_simpletes.sh" > "${HERE}/logs/simpletes.log" 2>&1 &
echo $! > /tmp/trimul_simpletes.pid
echo "started pid=$(cat /tmp/trimul_simpletes.pid)"
