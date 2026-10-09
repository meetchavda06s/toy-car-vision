"""Car detection via HSV colour segmentation + optional template histogram match.

Pipeline per car (per frame):
  1. Convert BGR -> HSV. (Baltes CV lecture: "Images" / colour spaces)
  2. Binary mask via cv2.inRange on the configured HSV range.
     (Baltes CV lecture: "Image Segmentation")
  3. Morphological OPEN then CLOSE to clean noise.
     (Baltes CV lecture: "Convolution and Edge Detection" discusses the
      underlying filter concept; morphology is the binary cousin.)
  4. findContours, keep blobs whose area is inside [area_min, area_max].
  5. For the largest valid blob, compute centroid via image moments,
     orientation via PCA on contour points (Baltes CV: "PCA / Eigenfaces"),
     and a confidence score.
  6. Confidence can optionally be an HSV-histogram Bhattacharyya distance
     against a stored template per car (Baltes ML: "Bayesian Learning" —
     histogram comparison framed as a Bayesian classifier). This replaces
     the simpler area-based score when `use_hist_confidence: true`.
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional


@dataclass
class Detection:
    car_name: str
    u: float                 # image x (centroid)
    v: float                 # image y (centroid)
    area: float              # mask area in px²
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    theta_pca_deg: float     # orientation from contour PCA, [-180, 180)
    confidence: float        # 0..1, higher = more car-like
    contour: np.ndarray = field(default=None, repr=False)


class HSVDetector:
    """Multi-car HSV detector with optional Bhattacharyya histogram scoring.

    Each configured car entry can optionally carry a `template_path` pointing
    at a BGR image of just the car (e.g. a cropped reference patch). On init
    we build a per-car HSV histogram from it. At detection time, we compute
    the candidate blob's histogram and score it with cv2.compareHist using
    HISTCMP_BHATTACHARYYA, giving a continuous [0..1] distance that is a
    much better ROC-sweep threshold than the simple area-band heuristic.
    """

    def __init__(self, cfg: dict):
        self.cars = cfg["detector"]["cars"]
        self.k = cfg["detector"].get("morph_kernel", 3)
        self._kernel = np.ones((self.k, self.k), np.uint8)
        self.use_hist = bool(cfg["detector"].get("use_hist_confidence", False))

        # Pre-compute reference histograms for cars that have a template.
        # Baltes ML lecture: "Bayesian Learning" — a template histogram is
        # a non-parametric estimate of P(colour | car) that we compare
        # against candidate blobs. Lower Bhattacharyya distance = better match.
        self._templates: Dict[str, np.ndarray] = {}
        for car in self.cars:
            tpl_path = car.get("template_path")
            if tpl_path and Path(tpl_path).exists():
                img = cv2.imread(tpl_path)
                if img is not None:
                    self._templates[car["name"]] = _hsv_hist(img)

    # -----------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray,
               roi_mask: Optional[np.ndarray] = None) -> List[Detection]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        results: List[Detection] = []

        for car in self.cars:
            lo = np.array(car["hsv_lower"], dtype=np.uint8)
            hi = np.array(car["hsv_upper"], dtype=np.uint8)
            amin = car.get("area_min", 50)
            amax = car.get("area_max", 5000)

            # Hue wraps around at 180 — if lower H > upper H we union two
            # ranges to handle red (which crosses the 0/180 wraparound).
            if lo[0] > hi[0]:
                m1 = cv2.inRange(hsv,
                                 np.array([0,       lo[1], lo[2]]),
                                 np.array([hi[0],   hi[1], hi[2]]))
                m2 = cv2.inRange(hsv,
                                 np.array([lo[0],   lo[1], lo[2]]),
                                 np.array([179,     hi[1], hi[2]]))
                mask = m1 | m2
            else:
                mask = cv2.inRange(hsv, lo, hi)

            if roi_mask is not None:
                mask = cv2.bitwise_and(mask, roi_mask)

            # Morphological cleanup (Baltes CV: filter concepts). OPEN
            # removes salt-noise, CLOSE fills small gaps inside the blob.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            valid = [c for c in contours if amin <= cv2.contourArea(c) <= amax]
            if not valid:
                continue
            c = max(valid, key=cv2.contourArea)
            area = float(cv2.contourArea(c))

            # Centroid via image moments. (Standard CV; moments appear in
            # Baltes' "Images" lecture as summary statistics of a region.)
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            u = M["m10"] / M["m00"]
            v = M["m01"] / M["m00"]
            x, y, w, h = cv2.boundingRect(c)

            # Orientation from PCA on contour points. (Baltes CV lecture:
            # "PCA / Eigenfaces" — the first principal component is the
            # long axis of the blob.)
            theta = _pca_angle(c)

            # Confidence: choose scoring method per config.
            if self.use_hist and car["name"] in self._templates:
                # Crop ROI around the blob and compare its HSV histogram
                # to the stored template (Baltes ML: Bayesian / Bhattacharyya).
                pad = 2
                xs = max(0, x - pad); ys = max(0, y - pad)
                xe = min(frame_bgr.shape[1], x + w + pad)
                ye = min(frame_bgr.shape[0], y + h + pad)
                crop = frame_bgr[ys:ye, xs:xe]
                if crop.size > 0:
                    cand_hist = _hsv_hist(crop)
                    # Bhattacharyya returns 0 (identical) .. 1 (disjoint).
                    # Invert so higher = more car-like, matching the ROC
                    # sweep convention.
                    dist = cv2.compareHist(self._templates[car["name"]],
                                           cand_hist,
                                           cv2.HISTCMP_BHATTACHARYYA)
                    conf = float(max(0.0, 1.0 - dist))
                else:
                    conf = 0.0
            else:
                # Fallback: triangle-shaped area band confidence.
                # Peaks when area == geometric mean of (amin, amax).
                ideal = np.sqrt(amin * amax)
                conf = max(0.0, 1.0 - abs(np.log(area + 1e-6) - np.log(ideal))
                                      / np.log(amax / amin + 1e-6))

            results.append(Detection(
                car_name=car["name"],
                u=u, v=v, area=area,
                bbox=(x, y, w, h),
                theta_pca_deg=theta,
                confidence=float(conf),
                contour=c,
            ))

        return results


# ---------------------------------------------------------------------------
def _pca_angle(contour: np.ndarray) -> float:
    """Principal-axis angle (degrees) of a contour via PCA.

    Baltes CV lecture: "PCA / Eigenfaces". We treat the contour's 2D points
    as samples and extract the dominant covariance direction. Returns an
    angle in [-180, 180). Note: 180°-ambiguous — the tracker disambiguates
    using velocity direction when the car is moving.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return 0.0
    _, eigvecs = cv2.PCACompute(pts, mean=None)
    v = eigvecs[0]  # principal axis
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _hsv_hist(bgr_img: np.ndarray) -> np.ndarray:
    """Compute a 2D H-S histogram and L1-normalise it.

    We ignore V (brightness) because it's the most lighting-dependent
    channel. Baltes ML Bayesian Learning: a normalised histogram IS a
    discrete probability distribution P(hue, saturation | car-class).
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_L1)
    return hist
