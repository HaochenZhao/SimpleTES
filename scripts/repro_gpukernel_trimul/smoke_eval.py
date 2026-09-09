#!/usr/bin/env python3
"""POST one program to the dedicated kernel server and print the score."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--task", default="trimul")
    parser.add_argument("--host", default=os.environ.get("KERNEL_NODE", "28.33.243.17"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("KERNEL_PORT", "8000")))
    args = parser.parse_args()

    with open(args.program, encoding="utf-8") as f:
        program_src = f.read()

    url = f"http://{args.host}:{args.port}/evaluate"
    body = json.dumps({"program_src": program_src, "task": args.task}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"[smoke] POST {url} program={args.program} bytes={len(program_src)}", flush=True)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3000) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:
        print(f"[smoke] FAILED after {time.time() - t0:.1f}s: {exc}", file=sys.stderr)
        raise SystemExit(1)
    elapsed = time.time() - t0
    result = payload.get("result") or {}
    print(json.dumps(
        {
            "elapsed_s": round(elapsed, 1),
            "success": payload.get("success"),
            "compiled": result.get("compiled"),
            "correctness": result.get("correctness"),
            "combined_score": result.get("combined_score"),
            "latency_ms": (result.get("metadata") or {}).get("latency_ms"),
            "error": result.get("error"),
            "error_name": result.get("error_name"),
        },
        indent=2,
        default=str,
    ))
    if not result.get("correctness"):
        raise SystemExit(2)


if __name__ == "__main__":
    os.environ.setdefault("no_proxy", "*")
    main()
