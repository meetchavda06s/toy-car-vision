"""Format and emit per-frame tracking output.

Supports three modes selectable from config:
  - udp:  broadcast UTF-8 lines (assignment spec)
  - log:  append to a text file (useful for offline runs on recorded video)
  - both

Line format matches the assignment:
  <timestamp_us>:"<name>",x,y,theta,vx,vy,omega,u,v
"""
from __future__ import annotations
import socket
from pathlib import Path
from typing import List

from .tracker import Track
from .homography import Homography
from .utils import Timer


class OutputSink:
    def __init__(self, cfg: dict, homography: Homography, fps: float):
        o = cfg["output"]
        self.mode = o.get("mode", "udp")
        self.homography = homography
        self.fps = fps
        self.timer = Timer()

        self.sock = None
        self.addr = None
        if self.mode in ("udp", "both"):
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.addr = (o["udp_host"], int(o["udp_port"]))

        self.log_fp = None
        if self.mode in ("log", "both"):
            Path(o["log_path"]).parent.mkdir(parents=True, exist_ok=True)
            self.log_fp = open(o["log_path"], "w")

    # -----------------------------------------------------------------
    def emit(self, tracks: List[Track], frame_idx: int):
        """Build the string and send/log it. One line per track."""
        # Use frame-based timestamp when running offline on a file (deterministic),
        # otherwise use real monotonic time (what a live camera would want).
        ts_us = int((frame_idx / self.fps) * 1_000_000) if self.fps > 0 else self.timer.us()

        lines = []
        for tr in tracks:
            s = tr.state()
            u, v = float(s[0]), float(s[1])
            vx_px, vy_px = float(s[2]), float(s[3])

            x_mm, y_mm = self.homography.pixel_to_world(u, v)

            if self.homography.calibrated:
                # Convert velocity: approximate using the homography's local Jacobian
                # by mapping (u+vx_px, v+vy_px) and subtracting the mapped origin.
                x1_mm, y1_mm = self.homography.pixel_to_world(u + vx_px, v + vy_px)
                vx_mm_s = (x1_mm - x_mm) * self.fps
                vy_mm_s = (y1_mm - y_mm) * self.fps
            else:
                # No homography: report raw pixel velocities as mm/s stand-in,
                # which is at least internally consistent.
                vx_mm_s = vx_px * self.fps
                vy_mm_s = vy_px * self.fps

            if tr.last_detection is None:
                # Totally missed: assignment spec says report -1000.0, -1000.0.
                line = (f'{ts_us}:"{tr.car_name}",-1000.0,-1000.0,'
                        f'{tr.theta_deg:.1f},0.0,0.0,0.0,-1,-1')
            else:
                line = (f'{ts_us}:"{tr.car_name}",{x_mm:.1f},{y_mm:.1f},'
                        f'{tr.theta_deg:.1f},{vx_mm_s:.1f},{vy_mm_s:.1f},'
                        f'{tr.omega_deg_s:.1f},{int(round(u))},{int(round(v))}')
            lines.append(line)

        if not lines:
            return

        msg = "\n".join(lines) + "\n"
        payload = msg.encode("utf-8")

        if self.sock is not None:
            try:
                self.sock.sendto(payload, self.addr)
            except OSError:
                pass  # silently ignore network errors in offline / sandbox runs
        if self.log_fp is not None:
            self.log_fp.write(msg)

    # -----------------------------------------------------------------
    def close(self):
        if self.sock is not None:
            self.sock.close()
        if self.log_fp is not None:
            self.log_fp.close()
