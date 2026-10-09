#!/usr/bin/env python3
"""Toy Car Vision — run the pipeline.

Usage:
    python run.py                                   # uses config/default.yaml
    python run.py --config config/myrun.yaml
    python run.py --source data/test_videos/x.mp4 --port 5555
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml

from src.pipeline import run


def main():
    p = argparse.ArgumentParser(description="Toy Car Vision tracker.")
    p.add_argument("--config", default="config/default.yaml",
                   help="Path to YAML config file.")
    p.add_argument("--source",  default=None,
                   help="Override input.source (video path or int camera index).")
    p.add_argument("--port", type=int, default=None,
                   help="Override output.udp_port.")
    p.add_argument("--mode", choices=["udp", "log", "both"], default=None,
                   help="Override output.mode.")
    p.add_argument("--no-visual", action="store_true",
                   help="Disable visual annotation output (fastest).")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[error] Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides by rewriting an in-memory copy to a temp file.
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if args.source is not None:
        # Accept integer camera indices as well.
        try:
            cfg["input"]["source"] = int(args.source)
        except ValueError:
            cfg["input"]["source"] = args.source
    if args.port is not None:
        cfg["output"]["udp_port"] = args.port
    if args.mode is not None:
        cfg["output"]["mode"] = args.mode
    if args.no_visual:
        cfg["visualise"]["enabled"] = False

    tmp_path = cfg_path.parent / "_runtime.yaml"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    try:
        stats = run(str(tmp_path))
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    print("\n===== pipeline summary =====")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
