#!/usr/bin/env python3
"""Exit 0 when instance-bd60257b has a checkpoint with >= 9216 evals."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

INST = Path(
    "/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES"
    "/checkpoints/gpukernel_zero_trimul/2026-09-08/instance-bd60257b"
)
run_log = INST / "run.log"
latest_finished = 0
if run_log.exists():
    try:
        import subprocess
        tail = subprocess.check_output(
            ["tail", "-n", "80", str(run_log)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        tail = ""
    nums = [int(x) for x in re.findall(r"finished_nodes=(\d+)", tail)]
    if nums:
        latest_finished = nums[-1]

best_n = 0
best_meta = None
for meta in sorted(INST.glob("db_state_*/metadata.json"), key=lambda p: p.stat().st_mtime):
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        continue
    n = int(data.get("completed_evaluations") or 0)
    if n >= best_n:
        best_n = n
        best_meta = meta

print(f"finished_nodes={latest_finished} ckpt_evals={best_n} meta={best_meta}")
if best_n >= 9216:
    sys.exit(0)
sys.exit(1)
