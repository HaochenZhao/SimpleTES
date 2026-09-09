#!/usr/bin/env bash
# Unguarded paper-scale TriMul: RPUCG, (C,L,K)=(32,100,16), N=51200.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
cd "${SIMPLES_DIR}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:${FRONTEND_PORT}/v1}"
export no_proxy="${no_proxy},${KERNEL_NODE},localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

mkdir -p "${KERNEL_LOG_DIR}"
uv run python main.py \
  --init-program datasets/gpukernel_zero/trimul/init_program.py \
  --evaluator    datasets/gpukernel_zero/trimul/evaluator.py \
  --instruction  datasets/gpukernel_zero/trimul/trimul_zero.txt \
  --model        openai/gpt-oss-120b \
  --api-base     "${OPENAI_API_BASE}" \
  --api-key      EMPTY \
  --selector     rpucg \
  --num-inspirations 5 \
  --num-chains   32 \
  --k-candidates 16 \
  --max-generations 51200 \
  --init-eval-repeats 1 \
  --gen-concurrency 32 \
  --eval-concurrency 8 \
  --eval-timeout 3000 \
  --reasoning-effort medium \
  --output-path  checkpoints/gpukernel_zero_trimul \
  --skip-preflight \
  "$@"
