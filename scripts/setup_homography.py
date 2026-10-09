#!/usr/bin/env python3
"""Interactive homography setup.

Opens a frame from the video, lets you click 4+ reference points on the
field, then prompts for each point's real-world (x, y) in mm. Writes
config/homography.yaml.

Also supports two automated backends:
  --mode aruco     (detect ArUco markers whose world positions are known)
  --mode manual    (click points yourself) [default]

Usage:
    python scripts/setup_homography.py --source data/test_videos/sample.mp4
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# Make src importable from this script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.homography import Homography
from src.utils import save_yaml


def grab_frame(source, start_sec: float):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read frame from source.")
    return frame


# ---------------------------------------------------------------------------
def manual_mode(frame):
    """Let the user click 4+ points; return list of pixel coords."""
    pts = []
    window = "setup_homography - click reference points (ENTER=done, R=reset, Q=quit)"
    display = frame.copy()

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 255, 0), 2)
            cv2.putText(display, str(len(pts)), (x+8, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, display)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):   # Enter
            if len(pts) >= 4:
                break
            print(f"Need at least 4 points, you have {len(pts)}.")
        elif k == ord("r"):
            pts.clear()
            display = frame.copy()
        elif k == ord("q"):
            pts = []
            break
    cv2.destroyWindow(window)
    return pts


def prompt_world_coords(pts):
    """Ask the user for each point's real-world (x, y) in mm."""
    print("\nEnter the real-world coordinates (mm) for each clicked point.")
    print("Use any consistent origin; the field corner is a natural choice.\n")
    world = []
    for i, (u, v) in enumerate(pts, start=1):
        while True:
            raw = input(f"  point {i} at pixel ({u},{v}) -> world x,y mm: ").strip()
            try:
                x, y = [float(s) for s in raw.replace(" ", "").split(",")]
                world.append((x, y))
                break
            except Exception:
                print("    (please type two numbers separated by comma, e.g. 0,0)")
    return world


# ---------------------------------------------------------------------------
def aruco_mode(frame, marker_world_coords: dict):
    """Detect ArUco markers; use known-world mapping to compute homography.

    marker_world_coords: {marker_id: (x_mm, y_mm)} for marker CENTRES.
    """
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(frame)
    except AttributeError:
        # older OpenCV
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

    if ids is None or len(ids) < 4:
        raise RuntimeError(f"ArUco detection failed: found {0 if ids is None else len(ids)} markers (need >=4).")

    img_pts, world_pts = [], []
    for marker_id, corner in zip(ids.flatten(), corners):
        if int(marker_id) not in marker_world_coords:
            continue
        c = corner[0].mean(axis=0)       # centre of the 4 corners
        img_pts.append((float(c[0]), float(c[1])))
        world_pts.append(marker_world_coords[int(marker_id)])
    if len(img_pts) < 4:
        raise RuntimeError("Not enough known markers detected to build H.")
    return img_pts, world_pts


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--mode", choices=["manual", "aruco"], default="manual")
    p.add_argument("--frame-sec", type=float, default=0.0,
                   help="Timestamp (seconds) to grab the setup frame from.")
    p.add_argument("--out", default="config/homography.yaml")
    args = p.parse_args()

    frame = grab_frame(args.source, args.frame_sec)

    if args.mode == "manual":
        pts = manual_mode(frame)
        if len(pts) < 4:
            print("Aborted."); return
        world = prompt_world_coords(pts)
    else:
        # Example coords — users should edit this dict or point to a JSON.
        example = {0: (0, 0), 1: (1500, 0), 2: (1500, 2500), 3: (0, 2500)}
        print("Using example ArUco world coords (edit script to customise):", example)
        pts, world = aruco_mode(frame, example)

    H = Homography.from_points(pts, world)
    err = H.reprojection_error(pts, world)
    print("\n===== homography built =====")
    print(f"  mean reprojection error: {err['mean_mm']:.2f} mm")
    print(f"  max reprojection error:  {err['max_mm']:.2f} mm")
    H.save(args.out)
    print(f"  saved: {args.out}")


if __name__ == "__main__":
    main()
