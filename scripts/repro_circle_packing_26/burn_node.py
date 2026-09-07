#!/usr/bin/env python3
"""Occupy every visible GPU on this node with bf16 matmul until SIGTERM."""
from __future__ import annotations

import os
import signal
import time
from multiprocessing import Process

import torch

STOP = False


def _handle_stop(signum, frame):
    global STOP
    STOP = True


def burn_one(gpu: int, hold_gib: float, matrix_size: int) -> None:
    torch.cuda.set_device(gpu)
    n = matrix_size
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(n, n, device="cuda", dtype=torch.bfloat16)
    ballast = None
    if hold_gib > 0:
        ballast = torch.empty(int(hold_gib * (2**30) // 2), device="cuda", dtype=torch.bfloat16)
        ballast.fill_(1.0)
    torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    print(f"[burn] gpu={gpu} pid={os.getpid()} started hold_gib={hold_gib}", flush=True)
    while not STOP:
        for _ in range(30):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
    del ballast
    torch.cuda.empty_cache()


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    n_gpu = torch.cuda.device_count()
    hold_gib = float(os.environ.get("BURN_HOLD_GIB", "40"))
    matrix_size = int(os.environ.get("BURN_MATRIX", "8192"))
    procs = [
        Process(target=burn_one, args=(i, hold_gib, matrix_size), daemon=True)
        for i in range(n_gpu)
    ]
    for p in procs:
        p.start()
    print(f"[burn] launched {n_gpu} workers pid={os.getpid()}", flush=True)
    try:
        while any(p.is_alive() for p in procs) and not STOP:
            time.sleep(2)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=10)


if __name__ == "__main__":
    main()
