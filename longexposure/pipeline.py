"""End-to-end orchestration for video long-exposure processing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from longexposure.alignment import AlignmentResult, AlignmentSettings, align_frames
from longexposure.diagnostics import alignment_table
from longexposure.frames import (
    ReferenceStrategy,
    extract_frames,
    score_frames_sharpness,
    select_reference_frame,
)
from longexposure.io import VideoMetadata, get_video_summary
from longexposure.stacking import (
    StackingMode,
    accepted_frames,
    crop_unstable_borders,
    stack_frames,
)

INLIER_RATIO_RELAX_STEP = 0.05


@dataclass(frozen=True)
class PipelineSettings:
    """User-controllable settings for one local processing run."""

    start_time_s: float = 0.0
    duration_s: float = 3.0
    max_frames: int = 90
    frame_stride: int = 1
    resize_width: int | None = None
    reference_strategy: ReferenceStrategy = "median"
    orb_max_features: int = 1500
    orb_keep_matches: int = 200
    min_matches: int = 10
    min_inlier_ratio: float = 1.0
    ransac_reproj_threshold: float = 3.0
    stack_mode: StackingMode = "mean"
    sigma: float = 2.5
    crop_borders: bool = True
    valid_border_threshold: float = 0.98


@dataclass(frozen=True)
class PipelineResult:
    """Images and diagnostics from one pipeline run."""

    video_metadata: VideoMetadata
    frames_processed: int
    reference_index: int
    sharpness_scores: list[float]
    alignment_diagnostics: list[dict[str, object]]
    accepted_count: int
    rejected_count: int
    output_image_bgr: np.ndarray
    cropped_output_image_bgr: np.ndarray | None
    warnings: list[str]
    first_frame_bgr: np.ndarray | None = None
    reference_frame_bgr: np.ndarray | None = None
    applied_min_inlier_ratio: float = 0.0


def _apply_inlier_ratio_threshold(
    alignment_results: list[AlignmentResult],
    threshold: float,
) -> list[AlignmentResult]:
    """Apply an inlier-ratio threshold to precomputed alignment results."""
    adjusted_results: list[AlignmentResult] = []
    for result in alignment_results:
        if result.status == "reference":
            adjusted_results.append(result)
        elif result.status != "accepted":
            adjusted_results.append(result)
        elif result.inlier_ratio >= threshold:
            adjusted_results.append(result)
        else:
            adjusted_results.append(
                replace(
                    result,
                    accepted=False,
                    reason="low inlier ratio",
                    status="low inlier ratio",
                )
            )

    return adjusted_results


def _accepted_count(alignment_results: list[AlignmentResult]) -> int:
    """Count accepted alignment results."""
    return sum(1 for result in alignment_results if result.accepted)


def _relax_inlier_ratio(
    alignment_results: list[AlignmentResult],
    start_threshold: float,
    target_accepted_frames: int,
) -> tuple[float, list[AlignmentResult]]:
    """Relax inlier ratio until enough frames are accepted or zero is reached."""
    threshold = float(np.clip(start_threshold, 0.0, 1.0))
    target = max(1, min(target_accepted_frames, len(alignment_results)))

    while threshold > 0:
        adjusted_results = _apply_inlier_ratio_threshold(alignment_results, threshold)
        if _accepted_count(adjusted_results) >= target:
            return threshold, adjusted_results
        threshold = max(0.0, round(threshold - INLIER_RATIO_RELAX_STEP, 2))

    adjusted_results = _apply_inlier_ratio_threshold(alignment_results, threshold)
    return threshold, adjusted_results


def _result_warnings(
    settings: PipelineSettings,
    accepted_count: int,
    applied_min_inlier_ratio: float,
    cropped_output_image_bgr: np.ndarray | None,
) -> list[str]:
    """Build human-readable pipeline warnings."""
    warnings: list[str] = []
    if accepted_count < settings.min_matches:
        warnings.append(
            "Accepted frame count stayed below the minimum target after relaxing "
            "the inlier-ratio threshold."
        )
    if applied_min_inlier_ratio < settings.min_inlier_ratio:
        warnings.append(
            f"Inlier ratio was relaxed from {settings.min_inlier_ratio:.2f} "
            f"to {applied_min_inlier_ratio:.2f}."
        )
    if settings.stack_mode == "median":
        warnings.append(
            "Median stacking is experimental cleanup and less photographically "
            "honest than mean stacking."
        )
    if settings.stack_mode == "sigma_clipped_mean":
        warnings.append(
            "Sigma-clipped mean is computational cleanup, not a pure photographic average."
        )
    if settings.crop_borders and cropped_output_image_bgr is None:
        warnings.append("Border cropping was enabled but no crop was applied.")

    return warnings


def run_pipeline(video_path: str | Path, settings: PipelineSettings) -> PipelineResult:
    """Run video extraction, reference selection, alignment, stacking, and crop."""
    resolved_path = Path(video_path)
    video_metadata = get_video_summary(resolved_path)
    frames_bgr, extraction_metadata = extract_frames(
        resolved_path,
        max_frames=settings.max_frames,
        stride=settings.frame_stride,
        resize_width=settings.resize_width,
        start_time_seconds=settings.start_time_s,
        duration_seconds=settings.duration_s,
    )
    frames_processed = int(extraction_metadata["frames_extracted"])
    sharpness_scores = score_frames_sharpness(frames_bgr)
    reference_index, reference_frame_bgr = select_reference_frame(
        frames_bgr,
        settings.reference_strategy,
    )

    alignment_settings = AlignmentSettings(
        max_features=settings.orb_max_features,
        keep_matches=settings.orb_keep_matches,
        min_matches=settings.min_matches,
        min_inlier_ratio=0.0,
        ransac_reproj_threshold=settings.ransac_reproj_threshold,
    )
    raw_alignment_results = align_frames(frames_bgr, reference_index, alignment_settings)
    applied_min_inlier_ratio, alignment_results = _relax_inlier_ratio(
        raw_alignment_results,
        settings.min_inlier_ratio,
        settings.min_matches,
    )
    accepted_aligned_frames = accepted_frames(alignment_results)
    accepted_count = len(accepted_aligned_frames)
    rejected_count = len(alignment_results) - accepted_count

    output_image_bgr = stack_frames(
        accepted_aligned_frames,
        settings.stack_mode,
        settings.sigma,
    )
    cropped_output_image_bgr: np.ndarray | None = None
    if settings.crop_borders:
        cropped = crop_unstable_borders(
            output_image_bgr,
            alignment_results,
            settings.valid_border_threshold,
        )
        if cropped.shape != output_image_bgr.shape:
            cropped_output_image_bgr = cropped

    warnings = _result_warnings(
        settings,
        accepted_count,
        applied_min_inlier_ratio,
        cropped_output_image_bgr,
    )

    return PipelineResult(
        video_metadata=video_metadata,
        frames_processed=frames_processed,
        reference_index=reference_index,
        sharpness_scores=sharpness_scores,
        alignment_diagnostics=alignment_table(alignment_results).to_dict("records"),
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        output_image_bgr=output_image_bgr,
        cropped_output_image_bgr=cropped_output_image_bgr,
        warnings=warnings,
        first_frame_bgr=frames_bgr[0] if frames_bgr else None,
        reference_frame_bgr=reference_frame_bgr,
        applied_min_inlier_ratio=applied_min_inlier_ratio,
    )
