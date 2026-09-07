#!/usr/bin/env bash
# Stop burn on one node at a time, then launch 8 TP=1 gpt-oss-120b replicas.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

remote() {
  local ip="$1"
  shift
  if [[ "${ip}" == "${HEAD_IP}" ]]; then
    bash -lc "$*"
  else
    # shellcheck disable=SC2086
    ssh ${SSH_OPTS} "root@${ip}" "$*"
  fi
}

launch_replica() {
  local ip="$1"
  local gpu="$2"
  local port=$((BASE_PORT + gpu))
  local logf="${LOG_DIR}/vllm_${ip}_gpu${gpu}.log"
  local cmd
  cmd=$(cat <<EOF
mkdir -p ${LOG_DIR}
if [[ -f /tmp/vllm_gpu_${gpu}.pid ]]; then
  kill "\$(cat /tmp/vllm_gpu_${gpu}.pid)" 2>/dev/null || true
  rm -f /tmp/vllm_gpu_${gpu}.pid
fi
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME=/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/cache/hf
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_HOST_IP=${ip}
export no_proxy=${no_proxy}
export NO_PROXY=${NO_PROXY}
nohup ${VLLM_BIN} serve ${MODEL_PATH} \
  --host 0.0.0.0 \
  --port ${port} \
  --served-model-name ${SERVED_NAME} openai/${SERVED_NAME} \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --distributed-executor-backend mp \
  --disable-log-requests \
  > ${logf} 2>&1 &
echo \$! > /tmp/vllm_gpu_${gpu}.pid
EOF
)
  remote "${ip}" "${cmd}"
  log "vLLM gpu${gpu} launched on ${ip}:${port}"
}

for ip in "${ALL_IPS[@]}"; do
  log "handing ${ip} from burn to vLLM"
  bash "${HERE}/stop_burn.sh" "${ip}"
  for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
    launch_replica "${ip}" "${gpu}"
    sleep 12
  done
done

log "all 32 vLLM processes launched; waiting for /v1/models"
export WAIT_IPS="$(IFS=','; echo "${ALL_IPS[*]}")"
python3 - <<'PY'
import json, os, time, urllib.request
hosts = [x for x in os.environ.get("WAIT_IPS", "").split(",") if x]
base = int(os.environ.get("BASE_PORT", "8100"))
need = []
for h in hosts:
    for g in range(8):
        need.append(f"http://{h}:{base+g}/v1/models")
deadline = time.time() + 45 * 60
ready = set()
while time.time() < deadline:
    for url in need:
        if url in ready:
            continue
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    ready.add(url)
        except Exception:
            pass
    print(f"[wait] {len(ready)}/{len(need)} replicas ready", flush=True)
    if len(ready) == len(need):
        break
    time.sleep(15)
print(json.dumps(sorted(ready), indent=2))
if len(ready) < len(need):
    missing = [u for u in need if u not in ready]
    raise SystemExit("not ready: " + ", ".join(missing[:8]))
PY
log "all replicas serving"
