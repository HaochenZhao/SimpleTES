#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
cd "${SIMPLES_DIR}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:${FRONTEND_PORT}/v1}"

uv run python main.py \
  --init-program datasets/circle_packing/circle_packing_26/init_program.py \
  --evaluator    datasets/circle_packing/circle_packing_26/evaluator.py \
  --instruction  datasets/circle_packing/circle_packing_26/circle_packing_26.txt \
  --model        openai/gpt-oss-120b \
  --api-base     "${OPENAI_API_BASE}" \
  --api-key      EMPTY \
  --selector     rpucg \
  --num-inspirations 5 \
  --num-chains   16 \
  --k-candidates 8 \
  --max-generations 3200 \
  --early-stop-score 2.63598 \
  --gen-concurrency 64 \
  --eval-concurrency 64 \
  --reasoning-effort medium \
  --output-path  checkpoints/circle_packing_26_repro \
  --skip-preflight \
  "$@"
