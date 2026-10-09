#!/usr/bin/env python3
"""Generate the detector's ROC curve.

Baltes ML lecture: "Validation". A ROC curve plots the True Positive Rate
(sensitivity) against the False Positive Rate (1 - specificity) as the
classifier's decision threshold is swept across its range. The Area Under
the Curve (AUC) summarises the trade-off: 0.5 = chance, 1.0 = perfect.


Workflow:
  1. Run the pipeline on a video with --mode log, producing an output log.
  2. Hand-label a sample of frames as car-present (1) or car-absent (0)
     into evaluation/ground_truth/labels.csv with columns:  frame_idx,label
  3. Run this script. It replays the video through the detector only, grabs
     the top-detection confidence per frame, sweeps the threshold, and
     plots TPR vs FPR.

Usage:
    python evaluation/roc_curve.py --config config/default.yaml \
        --labels evaluation/ground_truth/labels.csv \
        --out evaluation/results/roc_curve.png
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
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.camera import Capture
from src.detector import HSVDetector
from src.utils import load_yaml


def collect_scores(cfg: dict, labels: dict) -> list[tuple[int, float, int]]:
    """Return list of (frame_idx, max_conf, label) for every labelled frame."""
    inp = cfg["input"]
    cap = Capture(inp["source"],
                  start_sec=inp.get("start_sec"),
                  end_sec=inp.get("end_sec"),
                  resize_width=inp.get("resize_width"),
                  roi=cfg.get("roi"))
    det = HSVDetector(cfg)

    scores = []
    target_frames = set(labels.keys())
    while True:
        frame = cap.read()
        if frame is None: break
        if frame.idx not in target_frames: continue
        roi_mask = frame.roi_mask() if any(v is not None for v in frame.roi.values()) else None
        detections = det.detect(frame.img, roi_mask)
        conf = max((d.confidence for d in detections), default=0.0)
        scores.append((frame.idx, conf, labels[frame.idx]))
    cap.release()
    return scores


def compute_roc(scores) -> dict:
    s = np.array(sorted(scores, key=lambda x: -x[1]))
    conf  = s[:, 1].astype(float)
    label = s[:, 2].astype(int)
    thresholds = np.linspace(0, 1, 101)
    tpr, fpr = [], []
    P = max(1, label.sum()); N = max(1, (1 - label).sum())
    for t in thresholds:
        pred = (conf >= t).astype(int)
        tp = ((pred == 1) & (label == 1)).sum()
        fp = ((pred == 1) & (label == 0)).sum()
        tpr.append(tp / P); fpr.append(fp / N)
    # AUC via trapezoid (reverse-sorted FPR)
    order = np.argsort(fpr)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = trapz(np.array(tpr)[order], np.array(fpr)[order])
    return {"thresholds": thresholds.tolist(), "tpr": tpr, "fpr": fpr, "auc": float(auc)}


def plot_roc(roc: dict, out: str):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot(roc["fpr"], roc["tpr"], lw=2, label=f"AUC = {roc['auc']:.3f}")
    plt.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Detector ROC"); plt.grid(alpha=0.3); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--labels", default="evaluation/ground_truth/labels.csv")
    p.add_argument("--out",    default="evaluation/results/roc_curve.png")
    args = p.parse_args()

    cfg = load_yaml(args.config)

    labels = {}
    with open(args.labels) as f:
        for row in csv.DictReader(f):
            labels[int(row["frame_idx"])] = int(row["label"])
    print(f"Loaded {len(labels)} labelled frames.")

    scores = collect_scores(cfg, labels)
    print(f"Collected scores on {len(scores)} frames.")

    roc = compute_roc(scores)
    print(f"AUC = {roc['auc']:.3f}")
    plot_roc(roc, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
