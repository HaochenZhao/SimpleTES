#!/usr/bin/env python3
"""Append a one-line snapshot of the latest gpukernel_zero checkpoint."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path("/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES")
CKPT = ROOT / "checkpoints" / "gpukernel_zero_trimul"
OUT = ROOT / "scripts" / "repro_gpukernel_zero" / "results" / "progress.jsonl"


def latest_metadata() -> tuple[Path | None, dict | None]:
    if not CKPT.exists():
        return None, None
    metas = sorted(CKPT.glob("*/instance-*/db_state_*/metadata.json"), key=lambda p: p.stat().st_mtime)
    if not metas:
        return None, None
    path = metas[-1]
    with path.open(encoding="utf-8") as f:
        return path, json.load(f)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    path, meta = latest_metadata()
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata_path": str(path) if path else None,
        "best_score": None if meta is None else meta.get("best_score"),
        "completed_evaluations": None if meta is None else meta.get("completed_evaluations"),
        "generation_attempts": None if meta is None else meta.get("generation_attempts"),
        "generation_failures": None if meta is None else meta.get("generation_failures"),
        "best_node_id": None if meta is None else meta.get("best_node_id"),
    }
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("no_proxy", "*")
    main()
