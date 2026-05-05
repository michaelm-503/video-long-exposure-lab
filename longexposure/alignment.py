"""Frame alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AlignmentSettings:
    """Controls for ORB feature matching and frame rejection."""

    max_features: int = 1500
    keep_matches: int = 200
    min_matches: int = 30
    min_inlier_ratio: float = 0.25
    ransac_reproj_threshold: float = 3.0


@dataclass(frozen=True)
class TransformEstimate:
    """Estimated transform and match diagnostics for one moving frame."""

    matrix: np.ndarray | None
    matches: int
    inliers: int
    inlier_ratio: float
    status: str


@dataclass(frozen=True)
class AlignmentResult:
    """A single frame after alignment, with quality information."""

    frame: np.ndarray
    accepted: bool
    score: float
    reason: str
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    status: str = ""
    matrix: np.ndarray | None = None


def estimate_transform_orb(
    reference_bgr: np.ndarray,
    moving_bgr: np.ndarray,
    max_features: int = 1500,
    keep_matches: int = 200,
    ransac_reproj_threshold: float = 3.0,
) -> TransformEstimate:
    """Estimate a partial affine transform from a moving frame to a reference."""
    if reference_bgr.shape[:2] != moving_bgr.shape[:2]:
        return TransformEstimate(None, 0, 0, 0.0, "dimension mismatch")

    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    moving_gray = cv2.cvtColor(moving_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    reference_keypoints, reference_descriptors = orb.detectAndCompute(reference_gray, None)
    moving_keypoints, moving_descriptors = orb.detectAndCompute(moving_gray, None)

    if reference_descriptors is None or moving_descriptors is None:
        return TransformEstimate(None, 0, 0, 0.0, "missing ORB descriptors")
    if len(reference_keypoints) < 3 or len(moving_keypoints) < 3:
        return TransformEstimate(None, 0, 0, 0.0, "not enough keypoints")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(
        matcher.match(moving_descriptors, reference_descriptors),
        key=lambda match: match.distance,
    )
    kept_matches = matches[:keep_matches]
    match_count = len(kept_matches)

    if match_count < 3:
        return TransformEstimate(None, match_count, 0, 0.0, "not enough matches")

    moving_points = np.float32(
        [moving_keypoints[match.queryIdx].pt for match in kept_matches]
    )
    reference_points = np.float32(
        [reference_keypoints[match.trainIdx].pt for match in kept_matches]
    )
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        moving_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
    )

    if matrix is None or inlier_mask is None:
        return TransformEstimate(None, match_count, 0, 0.0, "transform failed")

    inliers = int(inlier_mask.ravel().sum())
    inlier_ratio = inliers / match_count if match_count > 0 else 0.0
    return TransformEstimate(matrix, match_count, inliers, inlier_ratio, "ok")


def warp_frame(
    frame_bgr: np.ndarray,
    matrix: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Warp a BGR frame into the reference frame coordinate system."""
    return cv2.warpAffine(
        frame_bgr,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _identity_matrix() -> np.ndarray:
    """Return a 2x3 identity affine transform matrix."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def align_frames(
    frames: list[np.ndarray],
    reference_index: int,
    settings: AlignmentSettings | None = None,
) -> list[AlignmentResult]:
    """Align a collection of BGR frames to the selected reference frame."""
    if not frames:
        raise ValueError("At least one frame is required")
    if reference_index < 0 or reference_index >= len(frames):
        raise ValueError("reference_index is out of range")

    resolved_settings = settings or AlignmentSettings()
    reference = frames[reference_index]
    height, width = reference.shape[:2]
    output_size = (width, height)
    results: list[AlignmentResult] = []

    for frame_index, frame in enumerate(frames):
        if frame_index == reference_index:
            results.append(
                AlignmentResult(
                    frame=frame,
                    accepted=True,
                    score=1.0,
                    reason="reference frame",
                    status="reference",
                    matrix=_identity_matrix(),
                )
            )
            continue

        estimate = estimate_transform_orb(
            reference,
            frame,
            max_features=resolved_settings.max_features,
            keep_matches=resolved_settings.keep_matches,
            ransac_reproj_threshold=resolved_settings.ransac_reproj_threshold,
        )

        if estimate.matrix is None:
            accepted = False
            status = estimate.status
        elif estimate.matches < resolved_settings.min_matches:
            accepted = False
            status = "too few matches"
        elif estimate.inlier_ratio < resolved_settings.min_inlier_ratio:
            accepted = False
            status = "low inlier ratio"
        else:
            accepted = True
            status = "accepted"

        aligned_frame = (
            warp_frame(frame, estimate.matrix, output_size)
            if accepted and estimate.matrix is not None
            else frame
        )
        results.append(
            AlignmentResult(
                frame=aligned_frame,
                accepted=accepted,
                score=estimate.inlier_ratio,
                reason=status,
                matches=estimate.matches,
                inliers=estimate.inliers,
                inlier_ratio=estimate.inlier_ratio,
                status=status,
                matrix=estimate.matrix,
            )
        )

    return results
