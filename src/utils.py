"""Small helpers used across modules."""
from __future__ import annotations
import time
import yaml
import numpy as np
from pathlib import Path


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


class Timer:
    """Monotonic timer that returns microseconds since instantiation."""
    def __init__(self):
        self._t0 = time.monotonic()

    def us(self) -> int:
        return int((time.monotonic() - self._t0) * 1_000_000)


def wrap_angle_deg(a: float) -> float:
    """Wrap angle into [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0
