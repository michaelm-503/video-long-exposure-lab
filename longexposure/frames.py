"""Frame extraction and reference-frame selection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from longexposure.io import VideoMetadata, get_video_summary

MAX_DEMO_FRAMES = 300


def _resize_frame(frame: np.ndarray, resize_width: int | None) -> np.ndarray:
    """Resize a BGR frame to a target width while preserving aspect ratio."""
    if resize_width is None or resize_width <= 0:
        return frame

    height, width = frame.shape[:2]
    if width <= 0 or height <= 0 or resize_width >= width:
        return frame

    scale = resize_width / width
    resize_height = max(1, round(height * scale))
    return cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)


def _seek_to_start(capture: cv2.VideoCapture, start_time_seconds: float, fps: float) -> int:
    """Seek to the requested start time and return the frame index to read from."""
    start_time = max(0.0, start_time_seconds)
    start_frame = round(start_time * fps) if fps > 0 else 0

    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    elif start_time > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    return start_frame


def extract_frames(
    video_path: Path,
    *,
    max_frames: int = 90,
    stride: int = 1,
    resize_width: int | None = None,
    start_time_seconds: float = 0.0,
    duration_seconds: float = 3.0,
) -> tuple[list[np.ndarray], VideoMetadata]:
    """Extract BGR frames from a selected video time window."""
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    metadata = get_video_summary(video_path)
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    capped_max_frames = min(max_frames, MAX_DEMO_FRAMES)
    start_frame = _seek_to_start(capture, start_time_seconds, fps)
    end_frame = (
        min(frame_count, start_frame + round(duration_seconds * fps))
        if fps > 0 and frame_count > 0
        else None
    )
    read_count = 0
    failed_reads = 0

    try:
        while len(frames) < capped_max_frames:
            absolute_frame_index = start_frame + read_count
            if end_frame is not None and absolute_frame_index >= end_frame:
                break

            ok, frame_bgr = capture.read()
            if not ok:
                failed_reads += 1
                break

            if read_count % stride == 0:
                frames.append(_resize_frame(frame_bgr, resize_width))

            read_count += 1
    finally:
        capture.release()

    extraction_metadata: VideoMetadata = {
        **metadata,
        "start_time_seconds": max(0.0, start_time_seconds),
        "requested_duration_seconds": duration_seconds,
        "max_frames": capped_max_frames,
        "stride": stride,
        "resize_width": resize_width or 0,
        "start_frame": start_frame,
        "frames_read": read_count,
        "frames_extracted": len(frames),
        "failed_reads": failed_reads,
    }

    if not frames:
        raise ValueError("No frames could be extracted from the selected video window")

    return frames, extraction_metadata


def choose_reference_frame(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Choose a reference frame for alignment.

    TODO: Replace this first-frame placeholder with a sharper, more deliberate
    reference selection strategy.
    """
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("At least one frame is required")
    return frame_list[0]
