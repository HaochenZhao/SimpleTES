#!/usr/bin/env python3
"""Pick the db_state with the highest completed_evaluations (not name sort)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
best_n = -1
best = None
for meta in root.glob("db_state_*/metadata.json"):
    try:
        n = int(json.loads(meta.read_text(encoding="utf-8")).get("completed_evaluations") or 0)
    except Exception:
        continue
    if n > best_n:
        best_n = n
        best = meta.parent
if best is None:
    raise SystemExit(f"no db_state in {root}")
print(best)
