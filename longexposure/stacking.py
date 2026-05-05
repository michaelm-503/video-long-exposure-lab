"""Frame rejection, averaging, and crop helpers."""

from __future__ import annotations

import numpy as np

from longexposure.alignment import AlignmentResult


def accepted_frames(alignment_results: list[AlignmentResult]) -> list[np.ndarray]:
    """Return frames that passed alignment and quality checks."""
    return [result.frame for result in alignment_results if result.accepted]


def average_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Average accepted RGB frames into a long-exposure-style still image."""
    if not frames:
        raise ValueError("At least one accepted frame is required")

    stack = np.stack(frames).astype(np.float32)
    return np.mean(stack, axis=0)


def crop_unstable_borders(image: np.ndarray) -> np.ndarray:
    """Crop borders that became unstable during alignment.

    TODO: Track valid image masks during alignment and crop to their intersection.
    This placeholder returns the full image unchanged.
    """
    return image

