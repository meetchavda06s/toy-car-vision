"""Top-level pipeline. Reads a config, runs capture -> detect -> track ->
emit -> annotate for every frame."""
from __future__ import annotations
import time
import cv2
from pathlib import Path

from .camera import Capture
from .detector import HSVDetector
from .tracker import MultiCarTracker
from .homography import Homography
from .output import OutputSink
from .visualise import Annotator
from .utils import load_yaml


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)

    # -------- capture --------
    inp = cfg["input"]
    cap = Capture(
        source=inp["source"],
        start_sec=inp.get("start_sec"),
        end_sec=inp.get("end_sec"),
        resize_width=inp.get("resize_width"),
        roi=cfg.get("roi"),
    )

    # -------- homography (from file if present, else identity) --------
    hom_path = Path(config_path).parent / "homography.yaml"
    H = Homography.load(hom_path)

    # -------- modules --------
    detector = HSVDetector(cfg)
    tracker  = MultiCarTracker(cfg)
    sink     = OutputSink(cfg, H, cap.fps)
    annot    = Annotator(cfg, H)

    # -------- stats --------
    stats = {
        "frames_processed": 0,
        "frames_with_detection": 0,
        "fps_measured": 0.0,
        "homography_calibrated": H.calibrated,
    }

    t_wall_start = time.monotonic()
    last_frame_t = None
    fps_display = 0.0

    try:
        while True:
            frame = cap.read()
            if frame is None:
                break

            roi_mask = frame.roi_mask() if any(v is not None for v in frame.roi.values()) else None

            # dt in seconds between consecutive frames
            now = frame.timestamp_sec
            dt = (now - last_frame_t) if last_frame_t is not None else (1.0 / cap.fps)
            last_frame_t = now

            detections = detector.detect(frame.img, roi_mask)
            tracks = tracker.update(detections, dt, now)
            sink.emit(tracks, frame.idx)

            if annot.enabled:
                if stats["frames_processed"] > 0 and stats["frames_processed"] % 10 == 0:
                    fps_display = 10.0 / max(time.monotonic() - t_wall_start, 1e-6) if False else fps_display
                vis = annot.draw(frame.img, tracks, fps_display=fps_display)
                annot.write(vis, cap.fps)
                if annot.show_window:
                    cv2.imshow("ToyCarVision", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            stats["frames_processed"] += 1
            if detections:
                stats["frames_with_detection"] += 1

            # Rolling FPS display, updated every 30 frames
            if stats["frames_processed"] % 30 == 0:
                elapsed = time.monotonic() - t_wall_start
                fps_display = stats["frames_processed"] / max(elapsed, 1e-6)

    finally:
        cap.release()
        sink.close()
        annot.close()
        if cfg["visualise"].get("show_window", False):
            cv2.destroyAllWindows()

    total_elapsed = time.monotonic() - t_wall_start
    stats["fps_measured"] = stats["frames_processed"] / max(total_elapsed, 1e-6)
    stats["elapsed_sec"] = total_elapsed
    return stats
