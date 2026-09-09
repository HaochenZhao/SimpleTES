#!/usr/bin/env python3
"""Keep every vLLM replica busy with long, overlapping decodes."""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

MODEL = os.environ.get("KEEPALIVE_MODEL", "gpt-oss-120b")
PER_REPLICA = int(os.environ.get("KEEPALIVE_PER_REPLICA", "6"))
MAX_TOKENS = int(os.environ.get("KEEPALIVE_MAX_TOKENS", "512"))
TIMEOUT = int(os.environ.get("KEEPALIVE_TIMEOUT", "300"))
PROMPT = (
    "Write a long, continuous technical essay about GPU kernel optimization. "
    "Do not stop early. Use numbered sections and keep generating until the token limit. "
    "Start with section 1 and continue."
)

DEFAULT_BACKENDS = ",".join(
    f"http://{ip}:{port}/v1/chat/completions"
    for ip in ("28.33.242.47", "28.33.244.15", "28.33.244.19")
    for port in range(8100, 8108)
)
URLS = [u.strip() for u in os.environ.get("KEEPALIVE_BACKENDS", DEFAULT_BACKENDS).split(",") if u.strip()]

_ok = 0
_err = 0
_lock = threading.Lock()


def one_call(url: str, i: int) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{PROMPT} id={i}"}],
        "max_tokens": MAX_TOKENS,
        "temperature": 1.0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return f"ok {resp.status}"
    except Exception as exc:
        return f"err {type(exc).__name__}: {exc}"


def worker(url: str, wid: int) -> None:
    global _ok, _err
    n = 0
    while True:
        r = one_call(url, wid * 100000 + n)
        n += 1
        with _lock:
            if r.startswith("ok"):
                _ok += 1
            else:
                _err += 1


def main() -> None:
    os.environ.setdefault("no_proxy", "*")
    n_threads = len(URLS) * PER_REPLICA
    print(
        f"[keepalive_hot] replicas={len(URLS)} per_replica={PER_REPLICA} "
        f"inflight={n_threads} max_tokens={MAX_TOKENS} model={MODEL}",
        flush=True,
    )
    threads = []
    for url in URLS:
        for k in range(PER_REPLICA):
            t = threading.Thread(target=worker, args=(url, len(threads)), daemon=True)
            t.start()
            threads.append(t)
    while True:
        time.sleep(10)
        with _lock:
            print(f"[keepalive_hot] ok={_ok} err={_err} inflight={n_threads}", flush=True)


if __name__ == "__main__":
    main()
