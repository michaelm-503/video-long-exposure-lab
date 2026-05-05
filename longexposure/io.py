"""Input and output helpers for local video and image files."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def save_uploaded_video(uploaded_bytes: bytes, destination: Path) -> Path:
    """Write uploaded video bytes to a local destination path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(uploaded_bytes)
    return destination


def export_image(image: np.ndarray, destination: Path) -> Path:
    """Export an RGB image array as a PNG file."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="RGB").save(destination)
    return destination


def get_video_summary(video_path: Path) -> dict[str, float | int]:
    """Return basic OpenCV metadata for a local video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    duration_seconds = frame_count / fps if fps > 0 else 0.0
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }

