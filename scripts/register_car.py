#!/usr/bin/env python3
"""Register a new car's HSV range + area bounds into the config.

Interactive mode: click a few pixels on the car in a chosen frame. The
script samples the HSV values, computes a tight range (median ± tolerance)
and appends a new car entry to config/default.yaml under detector.cars.

Headless mode: pass --name/--hsv-lower/--hsv-upper directly.

Usage:
    python scripts/register_car.py --source data/test_videos/sample.mp4 --name Red
    python scripts/register_car.py --name Blue --hsv-lower 100,100,50 --hsv-upper 130,255,255
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_yaml, save_yaml


def grab_frame(source, start_sec):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read frame.")
    return frame


def click_mode(frame, radius: int = 3):
    """Collect HSV samples at clicked pixels; return (lower, upper, median_area)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    samples = []
    display = frame.copy()
    window = "register_car — click pixels on the car (ENTER=done, R=reset, Q=quit)"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            patch = hsv[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
            samples.append(patch.reshape(-1, 3))
            cv2.circle(display, (x, y), 5, (0, 255, 255), 2)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, display)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):
            if samples:
                break
            print("Click at least one pixel on the car.")
        elif k == ord("r"):
            samples.clear()
            display = frame.copy()
        elif k == ord("q"):
            samples = []; break
    cv2.destroyWindow(window)

    if not samples:
        return None
    all_pts = np.vstack(samples)
    med = np.median(all_pts, axis=0)
    # tolerance: hue tight (±10), S/V loose (±60)
    lower = [int(max(0,   med[0] - 10)), int(max(0,   med[1] - 60)), int(max(0,   med[2] - 60))]
    upper = [int(min(179, med[0] + 10)), int(min(255, med[1] + 60)), int(min(255, med[2] + 60))]
    return lower, upper


def append_car(cfg_path: Path, car_entry: dict):
    cfg = load_yaml(cfg_path)
    cars = cfg["detector"].setdefault("cars", [])
    # Replace existing entry with same name if present
    cars = [c for c in cars if c["name"] != car_entry["name"]]
    cars.append(car_entry)
    cfg["detector"]["cars"] = cars
    save_yaml(cfg, cfg_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--source", default=None, help="For interactive mode.")
    p.add_argument("--frame-sec", type=float, default=0.0)
    p.add_argument("--hsv-lower", default=None, help='e.g. "100,100,50"')
    p.add_argument("--hsv-upper", default=None, help='e.g. "130,255,255"')
    p.add_argument("--area-min", type=int, default=80)
    p.add_argument("--area-max", type=int, default=2000)
    p.add_argument("--config", default="config/default.yaml")
    args = p.parse_args()

    if args.hsv_lower and args.hsv_upper:
        lower = [int(s) for s in args.hsv_lower.split(",")]
        upper = [int(s) for s in args.hsv_upper.split(",")]
    else:
        if not args.source:
            print("Need either --hsv-lower/--hsv-upper or --source for interactive mode.")
            sys.exit(1)
        frame = grab_frame(args.source, args.frame_sec)
        result = click_mode(frame)
        if result is None:
            print("No samples; aborted."); sys.exit(1)
        lower, upper = result

    entry = {
        "name": args.name,
        "hsv_lower": lower,
        "hsv_upper": upper,
        "area_min": args.area_min,
        "area_max": args.area_max,
    }
    append_car(Path(args.config), entry)
    print(f"Registered car '{args.name}': lower={lower} upper={upper}")
    print(f"  config updated: {args.config}")


if __name__ == "__main__":
    main()
