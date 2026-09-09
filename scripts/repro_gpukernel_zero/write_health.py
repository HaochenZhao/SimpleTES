#!/usr/bin/env python3
import json
import os
import urllib.request

os.environ["no_proxy"] = "*"
out = "/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES/scripts/repro_gpukernel_zero/results/health.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
try:
    with urllib.request.urlopen("http://28.33.243.17:8000/health", timeout=10) as resp:
        body = resp.read().decode()
    payload = {"ok": True, "status": resp.status, "body": body}
except Exception as exc:
    payload = {"ok": False, "error": str(exc)}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
