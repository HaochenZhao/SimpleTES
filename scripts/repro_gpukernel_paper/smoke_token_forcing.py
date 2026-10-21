#!/usr/bin/env python3
"""Ping nginx /v1/completions the way SimpleTES token-forcing does."""
from __future__ import annotations

import json
import os
import sys
import urllib.request

API = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
MODEL = os.environ.get("SERVED_NAME", "gpt-oss-120b")


def main() -> None:
    body = {
        "model": MODEL,
        "prompt": "Say the word ok.",
        "max_tokens": 8,
        "temperature": 0.0,
        "return_token_ids": True,
    }
    req = urllib.request.Request(
        f"{API}/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    choice = payload["choices"][0]
    print(json.dumps(
        {
            "model": payload.get("model"),
            "text": choice.get("text", "")[:200],
            "finish_reason": choice.get("finish_reason"),
            "n_token_ids": len(choice.get("token_ids") or []),
            "usage": payload.get("usage"),
        },
        indent=2,
    ))
    if not (choice.get("text") or choice.get("token_ids")):
        raise SystemExit(2)


if __name__ == "__main__":
    os.environ.setdefault("no_proxy", "*")
    try:
        main()
    except Exception as exc:
        print(f"SMOKE_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
