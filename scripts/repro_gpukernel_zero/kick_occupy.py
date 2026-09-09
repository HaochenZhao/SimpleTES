#!/usr/bin/env python3
"""Start keepalive locally and launch remote occupy scripts via ssh."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES")
ZERO = ROOT / "scripts" / "repro_gpukernel_zero"
PACK = ROOT / "scripts" / "repro_circle_packing_26"
LOG = ZERO / "logs"
PY = "/opt/conda/envs/torch-base/bin/python"
IPS = ["28.33.244.15", "28.33.244.19", "28.33.243.17"]


def run(cmd: list[str], timeout: int = 20) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return f"ERR {exc}"


def main() -> None:
    os.environ["no_proxy"] = "*"
    os.environ["NO_PROXY"] = "*"
    LOG.mkdir(parents=True, exist_ok=True)
    out = []

    models = run(["curl", "--noproxy", "*", "-fsS", "--max-time", "8", "http://127.0.0.1:8000/v1/models"])
    out.append("frontend=" + models[:200].replace("\n", " "))

    ka = subprocess.Popen(
        ["bash", str(ZERO / "start_keepalive.sh")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ka_out, _ = ka.communicate(timeout=15)
    except Exception as exc:
        ka_out = str(exc)
    out.append("keepalive=" + (ka_out or "").strip())

    occupy = str(ZERO / "occupy_node.sh")
    burn = str(PACK / "burn_node.py")
    for ip in IPS:
        cmd = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=3",
            "-o", "ServerAliveCountMax=2",
            f"root@{ip}",
            f"bash {occupy} {LOG} {PY} {burn} {ip}",
        ]
        out.append(f"{ip}=" + run(cmd, timeout=25).strip()[:300])

    time.sleep(3)
    tail = run(["tail", "-n", "3", str(PACK / "logs" / "keepalive.log")])
    out.append("keepalive_tail=" + tail.replace("\n", " | "))
    for ip in ["28.33.244.15", "28.33.244.19", "28.33.243.17"]:
        smi = LOG / f"smi_{ip}.txt"
        if smi.exists():
            out.append(f"smi_{ip}=\n" + smi.read_text(encoding="utf-8", errors="replace")[-800:])

    (LOG / "occupy_kick.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
