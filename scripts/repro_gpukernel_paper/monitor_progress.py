#!/usr/bin/env python3
"""Append a one-line snapshot of the latest paper-aligned checkpoint."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path("/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES")
CKPT = ROOT / "checkpoints" / "gpukernel_trimul_paper"
OUT = ROOT / "scripts" / "repro_gpukernel_paper" / "results" / "progress.jsonl"


def latest_meta() -> Path | None:
    metas = sorted(CKPT.glob("*/instance-*/db_state_*/metadata.json"), key=lambda p: p.stat().st_mtime)
    return metas[-1] if metas else None


def main() -> None:
    meta_path = latest_meta()
    rec = {"ts": datetime.now().isoformat(timespec="seconds")}
    if meta_path is None:
        rec["status"] = "no_ckpt"
    else:
        data = json.loads(meta_path.read_text())
        rec.update(
            {
                "ckpt": str(meta_path.parent.relative_to(ROOT)),
                "evals": data.get("completed_evaluations"),
                "gens": data.get("generation_attempts"),
                "best": data.get("best_score"),
                "best_node": data.get("best_node_id"),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
