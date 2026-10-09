"""Capture abstraction: treats a video file and a live webcam identically.

The pipeline doesn't care which it's reading from — same API for both.
"""
from __future__ import annotations
import cv2
from typing import Optional, Tuple


class Capture:
    """Thin wrapper over cv2.VideoCapture with ROI crop + optional resize."""

    def __init__(
        self,
        source,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        resize_width: Optional[int] = None,
        roi: Optional[dict] = None,
    ):
        # cv2 accepts either an int (camera index) or a filepath string
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

        self.resize_width = resize_width
        self.scale = 1.0
        if resize_width and resize_width < self.width:
            self.scale = resize_width / self.width

        self.roi = roi or {}
        self.end_frame = None

        if start_sec is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * self.fps))
        if end_sec is not None:
            self.end_frame = int(end_sec * self.fps)

    def read(self) -> Optional["Frame"]:
        """Return next frame or None at end of stream."""
        if self.end_frame is not None:
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cur >= self.end_frame:
                return None

        ret, img = self.cap.read()
        if not ret or img is None:
            return None

        if self.scale != 1.0:
            new_w = int(img.shape[1] * self.scale)
            new_h = int(img.shape[0] * self.scale)
            img = cv2.resize(img, (new_w, new_h))

        frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return Frame(img, frame_idx, self.fps, self.roi)

    def release(self):
        self.cap.release()


class Frame:
    """One frame plus metadata and ROI info."""
    def __init__(self, img, idx: int, fps: float, roi: dict):
        self.img = img
        self.idx = idx
        self.fps = fps
        self.timestamp_sec = idx / fps if fps > 0 else 0.0
        self.roi = roi

    @property
    def h(self): return self.img.shape[0]
    @property
    def w(self): return self.img.shape[1]

    def roi_mask(self):
        """Binary mask of the active ROI (255 inside, 0 outside)."""
        import numpy as np
        m = np.full((self.h, self.w), 255, dtype="uint8")
        top    = self.roi.get("top")    or 0
        bottom = self.roi.get("bottom") or self.h
        left   = self.roi.get("left")   or 0
        right  = self.roi.get("right")  or self.w
        if top > 0:         m[:top, :] = 0
        if bottom < self.h: m[bottom:, :] = 0
        if left > 0:        m[:, :left] = 0
        if right < self.w:  m[:, right:] = 0
        return m
