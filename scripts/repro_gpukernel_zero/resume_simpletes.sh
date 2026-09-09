#!/usr/bin/env bash
# Resume the unguarded paper-scale run from instance-bd60257b.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE="${1:-checkpoints/gpukernel_zero_trimul/2026-09-08/instance-bd60257b}"
exec bash "${HERE}/run_simpletes.sh" --resume "${INSTANCE}"
