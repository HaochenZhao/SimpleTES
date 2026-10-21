#!/usr/bin/env bash
# Paper-aligned guarded TriMul on H20.
# Kernel special case: C=32, K=8, N=51200 (L=200).
# gpt-oss-120b, RPUCG λ=1.0 γ=0.8, reasoning=high, temp=1.0,
# token-forcing context 65536 (program 15536 / prompt+reason 50000).
# 65536 is the paper window when evolved programs no longer fit 49152.
# insp=3 is the paper RPUCG sweet spot so 5 full kernels do not crowd phase-1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
cd "${SIMPLES_DIR}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:${FRONTEND_PORT}/v1}"
export no_proxy="${no_proxy},${KERNEL_NODES},${KERNEL_NODE},localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

mkdir -p "${KERNEL_LOG_DIR}"
uv run python main.py \
  --init-program datasets/gpukernel/trimul/init_program_naive.py \
  --evaluator    datasets/gpukernel/trimul/evaluator.py \
  --instruction  datasets/gpukernel/trimul/trimul.triton \
  --model        openai/gpt-oss-120b \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --api-base     "${OPENAI_API_BASE}" \
  --api-key      EMPTY \
  --llm-backend  vllm_token_forcing \
  --selector     rpucg \
  --puct-c       1.0 \
  --rpucg-gamma  0.8 \
  --num-inspirations 3 \
  --num-chains   32 \
  --k-candidates 8 \
  --max-generations 51200 \
  --init-eval-repeats 1 \
  --gen-concurrency 48 \
  --eval-concurrency 16 \
  --eval-timeout 3000 \
  --temperature  1.0 \
  --reasoning-effort high \
  --context-window 65536 \
  --reasoning-budget 50000 \
  --response-budget 15536 \
  --max-tokens 15536 \
  --output-path  checkpoints/gpukernel_trimul_paper \
  --skip-preflight \
  "$@"
