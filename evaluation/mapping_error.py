#!/usr/bin/env python3
"""Measure the homography mapping error.

Reads a CSV of (name, u, v, x_mm_true, y_mm_true) rows — one row per
known reference point (e.g. the field markers or the car at known
stationary positions). Loads the saved homography, computes mapped
(x, y) for each point, and reports the error in mm.

CSV format (header required):
    name,u,v,x_mm,y_mm

Usage:
    python evaluation/mapping_error.py \
        --refs evaluation/ground_truth/reference_points.csv \
        --homography config/homography.yaml \
        --out evaluation/results/mapping_error.png
"""
from __future__ import annotations
import argparse
import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.homography import Homography


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refs", default="evaluation/ground_truth/reference_points.csv")
    p.add_argument("--homography", default="config/homography.yaml")
    p.add_argument("--out", default="evaluation/results/mapping_error.png")
    args = p.parse_args()

    rows = []
    with open(args.refs) as f:
        for row in csv.DictReader(f):
            rows.append((row["name"], float(row["u"]), float(row["v"]),
                         float(row["x_mm"]), float(row["y_mm"])))

    H = Homography.load(args.homography)
    if not H.calibrated:
        print("[warn] No calibrated homography found; using identity.")

    results = []
    for name, u, v, xt, yt in rows:
        x, y = H.pixel_to_world(u, v)
        err = np.hypot(x - xt, y - yt)
        results.append((name, xt, yt, x, y, err))

    errs = np.array([r[5] for r in results])
    print("=" * 60)
    print(f"  {'point':<10}  true (mm)         mapped (mm)        err (mm)")
    print("-" * 60)
    for name, xt, yt, x, y, err in results:
        print(f"  {name:<10}  ({xt:>7.1f},{yt:>7.1f})  ({x:>7.1f},{y:>7.1f})   {err:>6.1f}")
    print("-" * 60)
    print(f"  mean error : {errs.mean():.2f} mm")
    print(f"  max  error : {errs.max():.2f} mm")
    print(f"  std  error : {errs.std():.2f} mm")

    # Scatter: true vs mapped
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for name, xt, yt, x, y, err in results:
        ax.scatter(xt, yt, s=80, marker="o", color="green",
                   label="true" if name == results[0][0] else None)
        ax.scatter(x,  y,  s=80, marker="x", color="red",
                   label="mapped" if name == results[0][0] else None)
        ax.plot([xt, x], [yt, y], color="grey", lw=0.8)
        ax.annotate(name, (xt, yt), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"Mapping error — mean {errs.mean():.1f} mm,  max {errs.max():.1f} mm")
    ax.grid(alpha=0.3); ax.legend(loc="best"); ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(args.out, dpi=150); plt.close()
    print(f"Saved plot: {args.out}")


if __name__ == "__main__":
    main()
