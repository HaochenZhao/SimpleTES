#!/usr/bin/env python3
"""Hammer the 3 kernel servers with a fail-heavy mix and report throughput."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOSTS = ["28.33.243.17", "28.33.244.15", "28.33.244.19"]


def load_src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def post_one(host: str, port: int, src: str, tag: str, timeout: int) -> dict:
    url = f"http://{host}:{port}/evaluate"
    body = json.dumps({"program_src": src, "task": "trimul"}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        result = payload.get("result") or {}
        return {
            "tag": tag,
            "host": host,
            "ok": bool(payload.get("success") and result.get("correctness")),
            "elapsed_s": round(time.time() - t0, 2),
            "score": result.get("combined_score"),
            "error_name": result.get("error_name"),
        }
    except Exception as exc:
        return {
            "tag": tag,
            "host": host,
            "ok": False,
            "elapsed_s": round(time.time() - t0, 2),
            "score": None,
            "error_name": type(exc).__name__,
        }


def main() -> None:
    os.environ["no_proxy"] = "*"
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seed = load_src(ROOT / "datasets/gpukernel_zero/trimul/init_program.py")
    fail_raise = load_src(HERE / "bench_programs/fail_raise.py")
    fail_wrong = load_src(HERE / "bench_programs/fail_wrong.py")
    # 36 jobs, ~2/3 fail (compile/correctness) + 1/3 correct seed (needs perf GPUs).
    jobs = (
        [("fail_raise", fail_raise)] * 16
        + [("fail_wrong", fail_wrong)] * 8
        + [("seed", seed)] * 12
    )
    hosts = HOSTS * ((len(jobs) + len(HOSTS) - 1) // len(HOSTS))
    work = [(hosts[i], tag, src) for i, (tag, src) in enumerate(jobs)]

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [
            pool.submit(post_one, host, 8000, src, tag, args.timeout)
            for host, tag, src in work
        ]
        for fut in as_completed(futs):
            rows.append(fut.result())
    wall = time.time() - t0
    n = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    summary = {
        "split": args.split,
        "wall_s": round(wall, 1),
        "n": n,
        "ok": ok,
        "fail": n - ok,
        "evals_per_min": round(n / wall * 60.0, 2),
        "mean_s": round(sum(r["elapsed_s"] for r in rows) / n, 2),
        "by_tag": {},
        "rows": rows,
    }
    for tag in ("fail_raise", "fail_wrong", "seed"):
        part = [r for r in rows if r["tag"] == tag]
        if part:
            summary["by_tag"][tag] = {
                "n": len(part),
                "ok": sum(1 for r in part if r["ok"]),
                "mean_s": round(sum(r["elapsed_s"] for r in part) / len(part), 2),
            }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("split", "wall_s", "n", "ok", "evals_per_min", "by_tag")}, indent=2))


if __name__ == "__main__":
    main()
