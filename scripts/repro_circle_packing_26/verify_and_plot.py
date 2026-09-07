#!/usr/bin/env python3
"""Independently re-score seed / official best / this run, and plot search curve."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/apdcephfs_sgxy/share_304401417/hunyuan/haochenzhao/SimpleTES")
TASK = ROOT / "datasets/circle_packing/circle_packing_26"
OFFICIAL_BEST = (
    ROOT
    / "best_results/mathematics_discovery/circle_packing_in_a_unit_square_n26"
    / "circle_packing_in_a_unit_square_n26_best.py"
)
PAPER_SCORE = 2.635983
OUT_DIR = ROOT / "scripts/repro_circle_packing_26/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK))
from evaluator import evaluate  # noqa: E402


def score_program(path: Path) -> dict:
    result = evaluate(str(path))
    print(f"{path.name}: {json.dumps({k: result.get(k) for k in ('combined_score', 'validity', 'error')})}")
    return result


def latest_instance(ckpt_root: Path) -> Path | None:
    if not ckpt_root.exists():
        return None
    instances = sorted(ckpt_root.glob("*/instance-*"))
    return instances[-1] if instances else None


def latest_scores_csv(instance: Path) -> Path | None:
    files = sorted(instance.glob("db_state_*/scores_*.csv"))
    return files[-1] if files else None


def plot_curve(csv_path: Path, out_png: Path) -> float | None:
    times = []
    scores = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                scores.append(float(row.get("score", "nan")))
            except ValueError:
                continue
    if not scores:
        return None
    running = []
    best = float("-inf")
    for s in scores:
        if s > best:
            best = s
        running.append(best)
        times.append(len(running))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, running, label="best-so-far")
    ax.axhline(PAPER_SCORE, linestyle="--", label=f"paper {PAPER_SCORE}")
    ax.set_xlabel("evaluated candidates")
    ax.set_ylabel("sum of radii")
    ax.set_title("SimpleTES circle packing n=26")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return best


def main() -> None:
    report = {"paper_score": PAPER_SCORE}
    report["seed"] = score_program(TASK / "init_program.py")
    report["official_best"] = score_program(OFFICIAL_BEST)

    instance = latest_instance(ROOT / "checkpoints/circle_packing_26_repro")
    report["instance"] = str(instance) if instance else None
    if instance is not None:
        best_prog = next(iter(sorted(instance.glob("**/best_program*"))), None)
        if best_prog is not None:
            report["this_run"] = score_program(best_prog)
            report["this_run_program"] = str(best_prog)
        csv_path = latest_scores_csv(instance)
        if csv_path is not None:
            best = plot_curve(csv_path, OUT_DIR / "score_curve.png")
            report["curve_best"] = best
            report["scores_csv"] = str(csv_path)

    this_score = (report.get("this_run") or {}).get("combined_score")
    report["hit_paper_record"] = (
        this_score is not None and float(this_score) >= 2.63598
    )
    out_json = OUT_DIR / "verify_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
