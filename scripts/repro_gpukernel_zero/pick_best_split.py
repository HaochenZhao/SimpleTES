#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

rows = []
for path in sys.argv[1:]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    rows.append(data)

if not rows:
    raise SystemExit("no bench json")

rows.sort(key=lambda r: (-float(r.get("evals_per_min") or 0), float(r.get("wall_s") or 1e9)))
best = rows[0]
summary = {
    "winner": best["split"],
    "winner_evals_per_min": best.get("evals_per_min"),
    "winner_wall_s": best.get("wall_s"),
    "ranking": [
        {"split": r["split"], "evals_per_min": r.get("evals_per_min"), "wall_s": r.get("wall_s"), "ok": r.get("ok")}
        for r in rows
    ],
}
out = Path(best.get("_out_dir") or Path(sys.argv[1]).parent) / "split_bench_summary.json"
# argv files live in results/split_bench/
out = Path(sys.argv[1]).parent / "split_bench_summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print(best["split"])
