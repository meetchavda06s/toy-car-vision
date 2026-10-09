"""Pixel <-> real-world (mm) mapping via a 3x3 homography.
"""
from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path
from .utils import load_yaml, save_yaml


class Homography:
    def __init__(self, matrix: np.ndarray | None = None):
        if matrix is None:
            # Identity => world coords == pixel coords (useful before calibration)
            self.H = np.eye(3, dtype=np.float64)
            self.calibrated = False
        else:
            self.H = np.asarray(matrix, dtype=np.float64)
            self.calibrated = True

    # ---------- construction ----------
    @classmethod
    def from_points(cls, img_pts, world_pts) -> "Homography":
        """Compute H from >= 4 point correspondences using RANSAC."""
        img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)
        world_pts = np.asarray(world_pts, dtype=np.float64).reshape(-1, 1, 2)
        if len(img_pts) < 4:
            raise ValueError("Need at least 4 point correspondences for homography.")
        H, _ = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("Homography computation failed.")
        return cls(H)

    @classmethod
    def load(cls, path: str | Path) -> "Homography":
        p = Path(path)
        if not p.exists():
            return cls()  # identity fallback
        data = load_yaml(p)
        if data is None or data.get("matrix") is None:
            return cls()
        return cls(np.array(data["matrix"]))

    def save(self, path: str | Path) -> None:
        save_yaml({"matrix": self.H.tolist()}, path)

    # ---------- transforms ----------
    def pixel_to_world(self, u: float, v: float) -> tuple[float, float]:
        """(u,v) in image -> (x,y) in mm."""
        pt = np.array([u, v, 1.0])
        w = self.H @ pt
        return float(w[0] / w[2]), float(w[1] / w[2])

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Inverse mapping (useful for drawing a world-frame grid overlay)."""
        Hinv = np.linalg.inv(self.H)
        pt = np.array([x, y, 1.0])
        w = Hinv @ pt
        return float(w[0] / w[2]), float(w[1] / w[2])

    def reprojection_error(self, img_pts, world_pts) -> dict:
        """Report per-point and summary errors (mm) for the stored H."""
        errs = []
        for (u, v), (xt, yt) in zip(img_pts, world_pts):
            x, y = self.pixel_to_world(u, v)
            errs.append(np.hypot(x - xt, y - yt))
        errs = np.array(errs)
        return {
            "per_point_mm": errs.tolist(),
            "mean_mm": float(errs.mean()),
            "max_mm": float(errs.max()),
            "std_mm": float(errs.std()),
        }
