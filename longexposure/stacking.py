"""Frame rejection, averaging, and crop helpers."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from longexposure.alignment import AlignmentResult

StackingMode = Literal["mean", "sigma_clipped_mean", "median"]


def accepted_frames(alignment_results: list[AlignmentResult]) -> list[np.ndarray]:
    """Return frames that passed alignment and quality checks."""
    return [result.frame for result in alignment_results if result.accepted]


def mean_stack(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """Average accepted aligned BGR frames into a photographic stack."""
    if not frames_bgr:
        raise ValueError("At least one accepted frame is required")

    stack = np.stack(frames_bgr).astype(np.float32)
    averaged = np.mean(stack, axis=0)
    return np.clip(averaged, 0, 255).astype(np.uint8)


def median_stack(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """Median-stack BGR frames as an experimental cleanup mode.

    Median stacking is less photographically honest than mean stacking because
    it suppresses transient motion instead of averaging its contribution.
    """
    if not frames_bgr:
        raise ValueError("At least one accepted frame is required")

    stack = np.stack(frames_bgr).astype(np.float32)
    median = np.median(stack, axis=0)
    return np.clip(median, 0, 255).astype(np.uint8)


def sigma_clipped_mean_stack(
    frames_bgr: list[np.ndarray],
    sigma: float = 2.5,
) -> np.ndarray:
    """Robust mean stack with sigma clipping as computational cleanup."""
    if not frames_bgr:
        raise ValueError("At least one accepted frame is required")
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")

    stack = np.stack(frames_bgr).astype(np.float32)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    keep = np.abs(stack - mean) <= (sigma * std)
    clipped_sum = np.sum(np.where(keep, stack, 0.0), axis=0)
    clipped_count = np.maximum(np.sum(keep, axis=0), 1)
    clipped_mean = clipped_sum / clipped_count
    return np.clip(clipped_mean, 0, 255).astype(np.uint8)


def stack_frames(
    frames_bgr: list[np.ndarray],
    mode: StackingMode = "mean",
    sigma: float = 2.5,
) -> np.ndarray:
    """Stack BGR frames with the selected stacking mode."""
    if mode == "mean":
        return mean_stack(frames_bgr)
    if mode == "sigma_clipped_mean":
        return sigma_clipped_mean_stack(frames_bgr, sigma)
    if mode == "median":
        return median_stack(frames_bgr)

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
    mask = valid_region_mask(alignment_results, valid_threshold)
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]

    if rows.size == 0 or columns.size == 0:
        return image_bgr

    y_min, y_max = int(rows[0]), int(rows[-1]) + 1
    x_min, x_max = int(columns[0]), int(columns[-1]) + 1
    return image_bgr[y_min:y_max, x_min:x_max]


def crop_unstable_borders(
    image_bgr: np.ndarray,
    alignment_results: list[AlignmentResult] | None = None,
    valid_threshold: float = 0.98,
) -> np.ndarray:
    """Crop unstable borders produced by frame warping."""
    if alignment_results is None:
        return image_bgr

    return crop_to_valid_region(image_bgr, alignment_results, valid_threshold)
