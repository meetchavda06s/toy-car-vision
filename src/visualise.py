"""Draw bounding box, ID label, heading arrow, and trajectory trail onto a
frame for debugging and demo video output.

Design notes
------------
Trail rendering uses a **persistent canvas** that accumulates one segment
per frame instead of redrawing from a point list every tick. This fixes
two bugs in the previous approach: visible flicker (each segment got a
different fade alpha every frame, so segments appeared to disappear and
reappear) and heavy over-draw cost. The canvas is the same size as a
frame, kept once in memory, and alpha-composited onto the current frame
each tick.

Heading direction uses the Kalman velocity vector when the car is moving
and falls back to detector PCA (flipped to agree with velocity) only when
nearly stationary. PCA on a ~20 px blurry blob is unreliable, so this is
strictly more accurate — his course material actually teaches PCA as a
tool, but for a compact noisy blob, velocity is the right signal.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List

from .tracker import Track
from .homography import Homography


# Deterministic palette so each car name gets the same colour run-to-run.
_PALETTE = [
    (0, 255, 0),      # green
    (0, 165, 255),    # orange
    (255, 128, 0),    # cyan-ish
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
    (255, 255, 0),    # cyan
    (128, 0, 255),    # pink
    (0, 100, 255),    # red-orange
]


class Annotator:
    def __init__(self, cfg: dict, homography: Homography):
        v = cfg["visualise"]
        self.enabled = v.get("enabled", True)
        self.show_window = v.get("show_window", False)
        self.save_path = v.get("save_video_path")
        self.draw_trail = v.get("draw_trail", True)

        # Trail config
        self.trail_thickness = int(v.get("trail_thickness", 1))
        self.trail_opacity   = float(v.get("trail_opacity", 0.75))  # 0..1

        # Reject motion faster than this between frames (likely a detector
        # glitch, not real motion). Prevents the long diagonal "shortcut"
        # lines across the field when the detector misses a few frames.
        self.max_jump_px = int(v.get("max_jump_px", 80))

        self.homography = homography

        # Persistent accumulation canvas, lazily sized to first frame.
        self._canvas: np.ndarray | None = None
        self._last_point: dict[str, tuple[int, int]] = {}
        self._colours: dict[str, tuple] = {}
        self._writer = None

    # -----------------------------------------------------------------
    def _colour(self, name: str) -> tuple[int, int, int]:
        if name not in self._colours:
            self._colours[name] = _PALETTE[len(self._colours) % len(_PALETTE)]
        return self._colours[name]

    def _ensure_canvas(self, frame_shape) -> None:
        if self._canvas is None or self._canvas.shape[:2] != frame_shape[:2]:
            self._canvas = np.zeros(frame_shape, dtype=np.uint8)

    # -----------------------------------------------------------------
    def draw(self, frame_bgr: np.ndarray, tracks: List[Track],
             fps_display: float = 0.0) -> np.ndarray:
        if not self.enabled:
            return frame_bgr

        self._ensure_canvas(frame_bgr.shape)
        vis = frame_bgr.copy()

        # ----- 1) Accumulate new trail segments onto persistent canvas -----
        if self.draw_trail:
            for tr in tracks:
                # Only extend the trail when the car was actually seen this
                # frame — never from Kalman prediction alone, so the trail
                # doesn't wander during occlusions.
                if tr.last_detection is None or tr.lost != 0:
                    continue
                # Use the DETECTION centroid for the trail (not the Kalman
                # posterior) so the trail reflects measurements, not filter
                # smoothing — more honest for an evaluation video.
                u = int(round(tr.last_detection.u))
                v = int(round(tr.last_detection.v))

                prev = self._last_point.get(tr.car_name)
                if prev is not None:
                    jump = float(np.hypot(u - prev[0], v - prev[1]))
                    # Big jump? Skip drawing the connecting line, but still
                    # update _last_point so future continuous motion starts
                    # from the new location (gap in trail = honest signal
                    # that we lost track briefly).
                    if jump <= self.max_jump_px:
                        cv2.line(self._canvas, prev, (u, v),
                                 self._colour(tr.car_name),
                                 self.trail_thickness, cv2.LINE_AA)
                self._last_point[tr.car_name] = (u, v)

            # Alpha-composite the trail canvas onto the frame, but ONLY
            # where the canvas has non-zero pixels. Elsewhere the raw
            # frame is kept at full opacity.
            mask = self._canvas.sum(axis=2) > 0
            if mask.any():
                a = self.trail_opacity
                blended = cv2.addWeighted(vis, 1.0 - a, self._canvas, a, 0.0)
                vis = np.where(mask[..., None], blended, vis)

        # ----- 2) Per-track overlays (box, heading arrow, label) -----
        for tr in tracks:
            s = tr.state()
            u, v = int(round(float(s[0]))), int(round(float(s[1])))
            vx, vy = float(s[2]), float(s[3])
            colour = self._colour(tr.car_name)

            # Bounding box when seen; small circle when predicted-only
            if tr.last_detection is not None and tr.lost == 0:
                x, y, w, h = tr.last_detection.bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)
            else:
                cv2.circle(vis, (u, v), 8, colour, 2)

            # Heading: velocity direction when moving (reliable),
            # PCA fallback only when nearly stationary.
            speed_px = float(np.hypot(vx, vy))
            if speed_px > 2.0:
                theta_rad = float(np.arctan2(vy, vx))
            else:
                theta_rad = float(np.radians(tr.theta_deg))

            L = 28
            tip = (int(u + L * np.cos(theta_rad)),
                   int(v + L * np.sin(theta_rad)))
            cv2.arrowedLine(vis, (u, v), tip, colour, 2,
                            tipLength=0.35, line_type=cv2.LINE_AA)

            # Label with world coords (right of car, flipped near right edge)
            x_mm, y_mm = self.homography.pixel_to_world(u, v)
            theta_deg = float(np.degrees(theta_rad))
            label = f"{tr.car_name}  ({x_mm:.0f},{y_mm:.0f})mm  {theta_deg:+.0f}deg"
            lx = u + 12 if u < vis.shape[1] - 220 else u - 12 - len(label) * 7
            ly = max(14, v - 10)
            cv2.rectangle(vis, (lx - 2, ly - 12),
                          (lx + len(label) * 7, ly + 4), (0, 0, 0), -1)
            cv2.putText(vis, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                        1, cv2.LINE_AA)

        # ----- 3) HUD (top-left) -----
        hud_lines = [
            f"FPS: {fps_display:.1f}",
            f"World: {'mm (calibrated)' if self.homography.calibrated else 'pixels (identity)'}",
            f"Tracks: {len(tracks)}",
        ]
        for i, line in enumerate(hud_lines):
            # black outline + white fill for readability on any background
            cv2.putText(vis, line, (10, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                        3, cv2.LINE_AA)
            cv2.putText(vis, line, (10, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1, cv2.LINE_AA)
        return vis

    # -----------------------------------------------------------------
    def write(self, frame_bgr: np.ndarray, fps: float) -> None:
        if not self.save_path:
            return
        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            from pathlib import Path
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(self.save_path, fourcc, fps, (w, h))
        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
