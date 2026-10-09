"""Multi-object tracker: one Kalman filter per car, Hungarian assignment.

State: [x, y, vx, vy]^T  (pixel space; we convert to mm after tracking)
Measurement: [x, y]

Core techniques (Prof. Baltes course references):
  * Kalman Filter: Baltes SLAM lecture "Kalman Filter". Linear system +
    Gaussian noise assumptions, predict/correct cycle. Process noise Q
    reflects how much we trust the constant-velocity motion model;
    measurement noise R reflects detector pixel-centroid noise.
  * Motion / Sensor Model: Baltes SLAM lectures of the same name. We use
    a constant-velocity motion model (transition matrix F) and a direct
    position measurement model (observation matrix H).
  * Hungarian algorithm for data association: standard graph-matching
    technique; here it minimises total assignment cost between predicted
    track positions and new detections.

Why pixel-space tracking? Pixel noise is roughly isotropic, which the
Kalman filter assumes. World-space tracking would require anisotropic
noise after perspective warping.
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.optimize import linear_sum_assignment

from .detector import Detection
from .utils import wrap_angle_deg


@dataclass
class Track:
    track_id: int
    car_name: str
    kf: cv2.KalmanFilter = field(repr=False)
    last_detection: Optional[Detection] = None
    hits: int = 1          # frames seen
    lost: int = 0          # consecutive frames missed
    theta_deg: float = 0.0      # SMOOTHED, continuity-aware orientation
    omega_deg_s: float = 0.0
    _theta_raw_prev: float = 0.0   # last raw PCA reading (before smoothing)
    _prev_t: float = 0.0
    _locked: bool = False           # True once we've committed to a facing direction

    def state(self) -> np.ndarray:
        """Return [u, v, vx, vy] as a flat 1-D array of 4 floats.

        OpenCV 5.0 exposes statePost as a (4,1) column vector; older
        OpenCVs returned (4,). This helper normalises so every reader
        can do s[0]..s[3] without caring about shape.
        """
        return np.asarray(self.kf.statePost, dtype=np.float32).flatten()


class MultiCarTracker:
    """One Kalman track per car name. Simple nearest-neighbour association
    with a distance gate, resolved globally by the Hungarian algorithm."""

    def __init__(self, cfg: dict):
        tcfg = cfg["tracker"]
        self.max_lost = tcfg.get("max_lost_frames", 15)
        self.max_dist = tcfg.get("max_assignment_dist", 80)
        self.q = tcfg.get("process_noise", 0.01)
        self.r = tcfg.get("measurement_noise", 0.1)
        self.vel_ema = tcfg.get("velocity_ema", 0.3)
        # Theta smoothing: EMA factor for orientation (lower = smoother, more lag)
        self.theta_ema = tcfg.get("theta_ema", 0.25)
        # Minimum speed (px/frame) below which we trust PCA shape over velocity direction
        self.theta_vel_min = tcfg.get("theta_velocity_min_pxpf", 1.5)
        self.tracks: List[Track] = []
        self._next_id = 0

    # -----------------------------------------------------------------
    def _new_kf(self, u: float, v: float, dt: float) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.q
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.r
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        # OpenCV 5.0 requires an explicit column vector (4,1) here; earlier
        # versions silently accepted a (4,) 1-D array. Using reshape keeps
        # the code portable across both.
        kf.statePost = np.array([[u], [v], [0], [0]], dtype=np.float32)
        kf.statePre  = kf.statePost.copy()
        return kf

    # -----------------------------------------------------------------
    def update(self, detections: List[Detection], dt: float, t_now: float) -> List[Track]:
        # 1) Predict all existing tracks.
        for tr in self.tracks:
            tr.kf.transitionMatrix[0, 2] = dt
            tr.kf.transitionMatrix[1, 3] = dt
            tr.kf.predict()

        # 2) Associate detections to tracks (Hungarian, per-car-name).
        matched_pairs, unmatched_dets, unmatched_tracks = self._associate(detections)

        # 3) Update matched tracks.
        for det_i, tr_i in matched_pairs:
            det = detections[det_i]
            tr = self.tracks[tr_i]
            z = np.array([[np.float32(det.u)], [np.float32(det.v)]])
            tr.kf.correct(z)
            tr.hits += 1
            tr.lost = 0
            tr.last_detection = det

            # ---- Orientation: continuity-aware disambiguation + smoothing ----
            theta_resolved = _resolve_theta_continuous(
                theta_pca=det.theta_pca_deg,
                state=tr.state(),
                prev_theta=tr.theta_deg,
                locked=tr._locked,
                vel_min=self.theta_vel_min,
            )
            # First time we see enough motion, lock in the facing direction.
            s = tr.state()
            vx, vy = float(s[2]), float(s[3])
            if not tr._locked and (vx * vx + vy * vy) ** 0.5 >= self.theta_vel_min:
                tr._locked = True

            # EMA-smooth theta to kill single-frame jitter (angle-aware).
            if tr.hits > 1:
                dtheta_smooth = wrap_angle_deg(theta_resolved - tr.theta_deg)
                tr.theta_deg = wrap_angle_deg(tr.theta_deg + self.theta_ema * dtheta_smooth)
            else:
                tr.theta_deg = theta_resolved

            # Angular velocity from the SMOOTHED theta (so it's not spiky).
            if tr.hits > 1 and tr._prev_t > 0:
                dtheta = wrap_angle_deg(tr.theta_deg - tr._theta_raw_prev)
                dt_safe = max(t_now - tr._prev_t, 1e-6)
                tr.omega_deg_s = (1 - self.vel_ema) * tr.omega_deg_s + \
                                 self.vel_ema * (dtheta / dt_safe)
            tr._theta_raw_prev = tr.theta_deg
            tr._prev_t = t_now

        # 4) Age unmatched tracks; drop stale ones.
        for i in unmatched_tracks:
            self.tracks[i].lost += 1
        self.tracks = [tr for tr in self.tracks if tr.lost <= self.max_lost]

        # 5) Spawn new tracks for unmatched detections.
        for di in unmatched_dets:
            det = detections[di]
            # Avoid creating duplicates of the same car name.
            if any(tr.car_name == det.car_name for tr in self.tracks):
                continue
            tr = Track(
                track_id=self._next_id,
                car_name=det.car_name,
                kf=self._new_kf(det.u, det.v, dt),
                last_detection=det,
                theta_deg=det.theta_pca_deg,
                _theta_raw_prev=det.theta_pca_deg,
                _prev_t=t_now,
            )
            self._next_id += 1
            self.tracks.append(tr)

        return self.tracks

    # -----------------------------------------------------------------
    def _associate(self, detections: List[Detection]):
        """Return (matched_pairs, unmatched_dets, unmatched_tracks).

        Matching is done per car-name group so an orange car can't be
        matched to a green car's track.
        """
        matched_pairs = []
        used_tracks, used_dets = set(), set()

        # Group both sides by car name.
        by_name_dets   = _group_by(detections,  lambda d: d.car_name)
        by_name_tracks = _group_by(self.tracks, lambda t: t.car_name)

        for name, det_idxs in by_name_dets.items():
            tr_idxs = by_name_tracks.get(name, [])
            if not tr_idxs:
                continue
            # Build cost matrix.
            cost = np.full((len(det_idxs), len(tr_idxs)), 1e6, dtype=np.float32)
            for i, di in enumerate(det_idxs):
                d = detections[di]
                for j, ti in enumerate(tr_idxs):
                    s = self.tracks[ti].state()
                    pu, pv = float(s[0]), float(s[1])
                    dist = np.hypot(d.u - pu, d.v - pv)
                    if dist <= self.max_dist:
                        cost[i, j] = dist
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < 1e5:
                    matched_pairs.append((det_idxs[r], tr_idxs[c]))
                    used_dets.add(det_idxs[r])
                    used_tracks.add(tr_idxs[c])

        unmatched_dets   = [i for i in range(len(detections))  if i not in used_dets]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        return matched_pairs, unmatched_dets, unmatched_tracks


# ---------------------------------------------------------------------------
def _group_by(items, keyfn):
    out = {}
    for i, it in enumerate(items):
        out.setdefault(keyfn(it), []).append(i)
    return out


def _resolve_theta_continuous(theta_pca: float,
                              state: np.ndarray,
                              prev_theta: float,
                              locked: bool,
                              vel_min: float = 1.5) -> float:
    """Disambiguate PCA's 180° axis-vs-direction, with continuity.

    Strategy (in order of preference):
      1. If the track is already LOCKED, prefer the PCA direction that is
         closest to the PREVIOUS theta — this stops arrow flips mid-motion.
      2. If not locked yet and the Kalman velocity is strong enough, use the
         velocity direction to pick between theta_pca and theta_pca + 180.
      3. Otherwise return theta_pca as-is (we have no basis to flip it).

    PCA returns an axis in [-180, 180). The two candidate facings are theta
    and theta + 180.
    """
    cand_a = wrap_angle_deg(theta_pca)
    cand_b = wrap_angle_deg(theta_pca + 180.0)

    # Case 1: continuity-based (once we've committed to a facing).
    if locked:
        da = abs(wrap_angle_deg(cand_a - prev_theta))
        db = abs(wrap_angle_deg(cand_b - prev_theta))
        return cand_a if da <= db else cand_b

    # Case 2: velocity-based (early frames, before lock-in).
    vx, vy = float(state[2]), float(state[3])
    speed = (vx * vx + vy * vy) ** 0.5
    if speed >= vel_min:
        theta_vel = np.degrees(np.arctan2(vy, vx))
        da = abs(wrap_angle_deg(cand_a - theta_vel))
        db = abs(wrap_angle_deg(cand_b - theta_vel))
        return cand_a if da <= db else cand_b

    # Case 3: no info yet.
    return cand_a
