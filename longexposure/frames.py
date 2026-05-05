"""Frame extraction and reference-frame selection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def extract_frames(
    video_path: Path,
    *,
    max_frames: int = 120,
    stride: int = 1,
) -> list[np.ndarray]:
    """Extract RGB frames from a video at a fixed stride."""
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    if stride <= 0:
        raise ValueError("stride must be greater than zero")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    frame_index = 0

    try:
        while len(frames) < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            if frame_index % stride == 0:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            frame_index += 1
    finally:
        capture.release()

    return frames


def choose_reference_frame(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Choose a reference frame for alignment.

    TODO: Replace this first-frame placeholder with a sharper, more deliberate
    reference selection strategy.
    """
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("At least one frame is required")
    return frame_list[0]

