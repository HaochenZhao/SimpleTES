#!/usr/bin/env bash
# Paper method, reduced budget: RPUCG + gpt-oss-120b, N = C*L*K = 8*8*4 = 256.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
cd "${SIMPLES_DIR}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:${FRONTEND_PORT}/v1}"
export SIMPLES_KERNEL_ALLOW_TORCH_IMPL=1
# Evaluator HTTP client must not go through the cluster proxy.
export no_proxy="${no_proxy},${KERNEL_NODE},localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

mkdir -p "${KERNEL_LOG_DIR}"
uv run python main.py \
  --init-program datasets/gpukernel/trimul/init_program.py \
  --evaluator    datasets/gpukernel/trimul/evaluator.py \
  --instruction  datasets/gpukernel/trimul/trimul.triton \
  --model        openai/gpt-oss-120b \
  --api-base     "${OPENAI_API_BASE}" \
  --api-key      EMPTY \
  --selector     rpucg \
  --num-inspirations 5 \
  --num-chains   8 \
  --k-candidates 4 \
  --max-generations 256 \
  --init-eval-repeats 1 \
  --gen-concurrency 16 \
  --eval-concurrency 8 \
  --eval-timeout 3000 \
  --reasoning-effort medium \
  --output-path  checkpoints/trimul_repro \
  --skip-preflight \
  "$@"
