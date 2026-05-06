"""Frame rejection, averaging, and crop helpers."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from longexposure.alignment import AlignmentResult

StackingMode = Literal[
    "mean",
    "sigma_clipped_mean",
    "median",
    "lighten",
    "additive",
]
MAX_STACK_CHUNK_BYTES = 128 * 1024 * 1024


def accepted_frames(alignment_results: list[AlignmentResult]) -> list[np.ndarray]:
    """Return frames that passed alignment and quality checks."""
    return [result.frame for result in alignment_results if result.accepted]


def mean_stack(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """Average accepted aligned BGR frames into a photographic stack."""
    if not frames_bgr:
        raise ValueError("At least one accepted frame is required")

    accumulated = np.zeros(frames_bgr[0].shape, dtype=np.float32)
    for frame in frames_bgr:
        accumulated += frame.astype(np.float32)
    averaged = accumulated / len(frames_bgr)
    return np.clip(averaged, 0, 255).astype(np.uint8)


def _chunk_height(frame_shape: tuple[int, ...], frame_count: int) -> int:
    """Return a row chunk size that keeps temporary stack memory bounded."""
    height, width, channels = frame_shape
    bytes_per_row = max(1, frame_count * width * channels * np.dtype(np.float32).itemsize)
    return max(1, min(height, MAX_STACK_CHUNK_BYTES // bytes_per_row))


def median_stack(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """Median-stack BGR frames as an experimental cleanup mode.

    Median stacking is less photographically honest than mean stacking because
    it suppresses transient motion instead of averaging its contribution.
    """
    if not frames_bgr:
        raise ValueError("At least one accepted frame is required")

    output = np.empty_like(frames_bgr[0])
    rows_per_chunk = _chunk_height(frames_bgr[0].shape, len(frames_bgr))
    for row_start in range(0, frames_bgr[0].shape[0], rows_per_chunk):
        row_end = min(frames_bgr[0].shape[0], row_start + rows_per_chunk)
        stack = np.stack(
            [frame[row_start:row_end] for frame in frames_bgr],
        ).astype(np.float32)
        output[row_start:row_end] = np.clip(np.median(stack, axis=0), 0, 255).astype(
            np.uint8
        )
    return output


def sigma_clipped_mean_stack(
    frames_bgr: list[np.ndarray],
    sigma: float = 2.5,
) -> np.ndarray:
    """Robust mean stack with sigma clipping as computational cleanup."""
    if not frames_bgr:
        raise ValueError("At least one accepted frame is required")
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")

    output = np.empty_like(frames_bgr[0])
    rows_per_chunk = _chunk_height(frames_bgr[0].shape, len(frames_bgr))
    for row_start in range(0, frames_bgr[0].shape[0], rows_per_chunk):
        row_end = min(frames_bgr[0].shape[0], row_start + rows_per_chunk)
        stack = np.stack(
            [frame[row_start:row_end] for frame in frames_bgr],
        ).astype(np.float32)
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        keep = np.abs(stack - mean) <= (sigma * std)
        clipped_sum = np.sum(stack * keep, axis=0)
        clipped_count = np.maximum(np.sum(keep, axis=0), 1)
        clipped_mean = clipped_sum / clipped_count
        output[row_start:row_end] = np.clip(clipped_mean, 0, 255).astype(np.uint8)
    return output


def lighten_stack(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """Keep the brightest value at each pixel/channel for trails and fireworks."""
    if not frames_bgr:
        raise ValueError("At least one accepted frame is required")

    lightened = frames_bgr[0].copy()
    for frame in frames_bgr[1:]:
        np.maximum(lightened, frame, out=lightened)
    return lightened


def additive_stack(frames_bgr: list[np.ndarray], gain: float = 1.0) -> np.ndarray:
    """Sum frames in float space, then scale and clip highlights."""
    if not frames_bgr:
        raise ValueError("At least one accepted frame is required")
    if gain < 0:
        raise ValueError("additive gain must be non-negative")

    accumulated = np.zeros(frames_bgr[0].shape, dtype=np.float32)
    for frame in frames_bgr:
        accumulated += frame.astype(np.float32)
    accumulated *= gain
    return np.clip(accumulated, 0, 255).astype(np.uint8)


def stack_frames(
    frames_bgr: list[np.ndarray],
    mode: StackingMode = "mean",
    sigma: float = 2.5,
    additive_gain: float = 1.0,
) -> np.ndarray:
    """Stack BGR frames with the selected stacking mode."""
    if mode == "mean":
        return mean_stack(frames_bgr)
    if mode == "sigma_clipped_mean":
        return sigma_clipped_mean_stack(frames_bgr, sigma)
    if mode == "median":
        return median_stack(frames_bgr)
    if mode == "lighten":
        return lighten_stack(frames_bgr)
    if mode == "additive":
        return additive_stack(frames_bgr, additive_gain)

    raise ValueError(f"Unsupported stacking mode: {mode}")


def average_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Backward-compatible alias for photographic mean stacking."""
    return mean_stack(frames)


def valid_region_mask(
    alignment_results: list[AlignmentResult],
    valid_threshold: float = 0.98,
) -> np.ndarray:
    """Return pixels covered by at least the requested fraction of accepted frames."""
    accepted_results = [result for result in alignment_results if result.accepted]
    if not accepted_results:
        raise ValueError("At least one accepted frame is required")

    height, width = accepted_results[0].frame.shape[:2]
    output_size = (width, height)
    accumulated = np.zeros((height, width), dtype=np.float32)

    for result in accepted_results:
        source_mask = np.full((height, width), 255, dtype=np.uint8)
        if result.matrix is None:
            warped_mask = source_mask
        else:
            warped_mask = cv2.warpAffine(
                source_mask,
                result.matrix,
                output_size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        accumulated += warped_mask > 0

    required_fraction = float(np.clip(valid_threshold, 0.0, 1.0))
    return (accumulated / len(accepted_results)) >= required_fraction


def crop_to_valid_region(
    image_bgr: np.ndarray,
    alignment_results: list[AlignmentResult],
    valid_threshold: float = 0.98,
) -> np.ndarray:
    """Crop an image to the bounding rectangle of the stable valid mask."""
    crop_rect = valid_region_crop_rect(alignment_results, valid_threshold)
    if crop_rect is None:
        return image_bgr

    x_min, y_min, x_max, y_max = crop_rect
    return image_bgr[y_min:y_max, x_min:x_max]


def valid_region_crop_rect(
    alignment_results: list[AlignmentResult],
    valid_threshold: float = 0.98,
) -> tuple[int, int, int, int] | None:
    """Return the stable crop rectangle as x_min, y_min, x_max, y_max."""
    mask = valid_region_mask(alignment_results, valid_threshold)
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]

    if rows.size == 0 or columns.size == 0:
        return None

    y_min, y_max = int(rows[0]), int(rows[-1]) + 1
    x_min, x_max = int(columns[0]), int(columns[-1]) + 1
    return x_min, y_min, x_max, y_max


def crop_unstable_borders(
    image_bgr: np.ndarray,
    alignment_results: list[AlignmentResult] | None = None,
    valid_threshold: float = 0.98,
) -> np.ndarray:
    """Crop unstable borders produced by frame warping."""
    if alignment_results is None:
        return image_bgr

    return crop_to_valid_region(image_bgr, alignment_results, valid_threshold)
