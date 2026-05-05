"""Frame alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    """A single frame after alignment, with quality information."""

    frame: np.ndarray
    accepted: bool
    score: float
    reason: str


def align_frame(frame: np.ndarray, reference: np.ndarray) -> AlignmentResult:
    """Align one frame to a reference frame.

    TODO: Implement feature- or ECC-based image registration with OpenCV. This
    placeholder keeps the frame unchanged so the app can run end to end.
    """
    if frame.shape != reference.shape:
        return AlignmentResult(
            frame=frame,
            accepted=False,
            score=0.0,
            reason="Frame dimensions do not match the reference",
        )

    return AlignmentResult(
        frame=frame,
        accepted=True,
        score=1.0,
        reason="Alignment placeholder accepted matching frame",
    )


def align_frames(frames: list[np.ndarray], reference: np.ndarray) -> list[AlignmentResult]:
    """Align a collection of frames to the chosen reference frame."""
    return [align_frame(frame, reference) for frame in frames]

