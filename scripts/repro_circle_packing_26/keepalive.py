#!/usr/bin/env python3
"""Low-intensity chat loop so idle vLLM replicas stay off the kill list."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("KEEPALIVE_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("KEEPALIVE_MODEL", "gpt-oss-120b")
WORKERS = int(os.environ.get("KEEPALIVE_WORKERS", "4"))
MAX_TOKENS = int(os.environ.get("KEEPALIVE_MAX_TOKENS", "16"))
SLEEP = float(os.environ.get("KEEPALIVE_SLEEP", "0.4"))


def one_call(i: int) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Reply with the single word ok and the integer {i}."}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return f"ok {resp.status}"
    except Exception as exc:
        return f"err {type(exc).__name__}: {exc}"


def main() -> None:
    n = 0
    print(f"[keepalive] {WORKERS} workers -> {URL} model={MODEL}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while True:
            futs = [pool.submit(one_call, n + j) for j in range(WORKERS)]
            results = [f.result() for f in futs]
            n += WORKERS
            ok = sum(1 for r in results if r.startswith("ok"))
            print(f"[keepalive] n={n} ok={ok}/{WORKERS} last={results[0]}", flush=True)
            time.sleep(SLEEP)


if __name__ == "__main__":
    main()
