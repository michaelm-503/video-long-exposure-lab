"""End-to-end orchestration for video long-exposure processing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
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
    stack_frames,
    valid_region_crop_rect,
)

INLIER_RATIO_RELAX_STEP = 0.05
AUTO_INLIER_RATIO_SENTINEL = 0.0


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
    enable_alignment: bool = True
    stack_mode: StackingMode = "mean"
    sigma: float = 2.5
    additive_gain: float = 1.0
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
    frames_bgr: list[np.ndarray] | None = None
    alignment_results: list[AlignmentResult] | None = None
    crop_rect: tuple[int, int, int, int] | None = None
    mask_coverage: float = 0.0


@dataclass(frozen=True)
class PreviewResult:
    """Low-resolution frame/reference data used before the full pipeline run."""

    video_metadata: VideoMetadata
    extraction_metadata: VideoMetadata
    frames_processed: int
    reference_index: int
    sharpness_scores: list[float]
    reference_frame_bgr: np.ndarray
    frames_bgr: list[np.ndarray]


@dataclass(frozen=True)
class StackJobSettings:
    """Settings for rerunning alignment and stacking against extracted frames."""

    orb_max_features: int = 1500
    orb_keep_matches: int = 200
    min_matches: int = 10
    min_inlier_ratio: float = 1.0
    ransac_reproj_threshold: float = 3.0
    enable_alignment: bool = True
    stack_mode: StackingMode = "mean"
    sigma: float = 2.5
    additive_gain: float = 1.0
    crop_borders: bool = True
    valid_border_threshold: float = 0.98


@dataclass(frozen=True)
class StackJobResult:
    """Result from aligning and stacking already-extracted frames."""

    stack_image_bgr: np.ndarray
    cropped_reference_bgr: np.ndarray
    cropped_stack_bgr: np.ndarray
    crop_rect: tuple[int, int, int, int] | None
    accepted_count: int
    rejected_count: int
    diagnostics: list[dict[str, object]]
    alignment_results: list[AlignmentResult]
    applied_min_inlier_ratio: float


@dataclass(frozen=True)
class StackSubsetResult:
    """A restacked subset of accepted aligned frames."""

    output_image_bgr: np.ndarray
    cropped_output_image_bgr: np.ndarray | None
    crop_rect: tuple[int, int, int, int] | None
    selected_indices: list[int]


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


def _resolve_inlier_ratio_threshold(
    alignment_results: list[AlignmentResult],
    requested_threshold: float,
    target_accepted_frames: int,
) -> tuple[float, list[AlignmentResult]]:
    """Apply explicit thresholding or auto-relax from 1.0 when requested."""
    bounded_threshold = float(np.clip(requested_threshold, 0.0, 1.0))
    if bounded_threshold == AUTO_INLIER_RATIO_SENTINEL:
        return _relax_inlier_ratio(
            alignment_results,
            1.0,
            target_accepted_frames,
        )

    return bounded_threshold, _apply_inlier_ratio_threshold(
        alignment_results,
        bounded_threshold,
    )


def _max_transform_movement_px(
    alignment_results: list[AlignmentResult],
    frame_shape: tuple[int, int],
) -> float:
    """Estimate the largest accepted-frame corner movement in pixels."""
    height, width = frame_shape
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(width), 0.0, 1.0],
            [0.0, float(height), 1.0],
            [float(width), float(height), 1.0],
        ],
        dtype=np.float32,
    )
    max_movement = 0.0

    for result in alignment_results:
        if not result.accepted or result.matrix is None or result.status == "reference":
            continue

        transformed = corners @ result.matrix.T
        movement = np.linalg.norm(transformed - corners[:, :2], axis=1)
        max_movement = max(max_movement, float(np.max(movement)))

    return max_movement


def _result_warnings(
    settings: PipelineSettings,
    video_metadata: VideoMetadata,
    sharpness_scores: list[float],
    alignment_results: list[AlignmentResult],
    accepted_count: int,
    applied_min_inlier_ratio: float,
    cropped_output_image_bgr: np.ndarray | None,
) -> list[str]:
    """Build human-readable pipeline warnings."""
    warnings: list[str] = []
    video_width = int(video_metadata.get("width", 0))
    video_height = int(video_metadata.get("height", 0))
    min_dimension = min(video_width, video_height)
    megapixels = (video_width * video_height) / 1_000_000 if video_width and video_height else 0

    if accepted_count < 10:
        warnings.append(
            "Fewer than 10 frames were accepted, so the long-exposure effect may be weak."
        )
    if accepted_count < settings.min_matches:
        if settings.enable_alignment:
            warnings.append(
                "Accepted frame count stayed below the minimum target after relaxing "
                "the inlier-ratio threshold."
            )
        else:
            warnings.append("Fewer frames than the minimum target were available.")
    if min_dimension and (min_dimension < 720 or megapixels < 0.75):
        warnings.append(
            "Source video resolution is low; the final image may look soft or compressed."
        )
    if sharpness_scores and float(np.median(sharpness_scores)) < 80:
        warnings.append(
            "Many source frames appear soft or blurred, so averaging cannot recover crisp detail."
        )
    if settings.enable_alignment and alignment_results and video_width and video_height:
        max_movement_px = _max_transform_movement_px(
            alignment_results,
            alignment_results[0].frame.shape[:2],
        )
        movement_limit = 0.18 * min(alignment_results[0].frame.shape[:2])
        if max_movement_px > movement_limit:
            warnings.append(
                "Estimated frame movement is large; alignment may crop heavily or leave softness."
            )
    if settings.enable_alignment and settings.min_inlier_ratio == AUTO_INLIER_RATIO_SENTINEL:
        warnings.append(
            f"Auto inlier ratio selected {applied_min_inlier_ratio:.2f} "
            "to meet the minimum-match target."
        )
    elif settings.enable_alignment and applied_min_inlier_ratio < settings.min_inlier_ratio:
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
    if settings.stack_mode == "additive":
        warnings.append("Additive stacking can saturate highlights quickly.")
    if settings.stack_mode in {"lighten", "additive"} and settings.enable_alignment:
        warnings.append(
            "For tripod star trails, disable alignment or align only to static foreground."
        )
    if settings.crop_borders and cropped_output_image_bgr is None:
        warnings.append("Border cropping was enabled but no crop was applied.")

    return warnings


def stack_settings_from_pipeline(settings: PipelineSettings) -> StackJobSettings:
    """Build stack-job settings from full pipeline settings."""
    return StackJobSettings(
        orb_max_features=settings.orb_max_features,
        orb_keep_matches=settings.orb_keep_matches,
        min_matches=settings.min_matches,
        min_inlier_ratio=settings.min_inlier_ratio,
        ransac_reproj_threshold=settings.ransac_reproj_threshold,
        enable_alignment=settings.enable_alignment,
        stack_mode=settings.stack_mode,
        sigma=settings.sigma,
        additive_gain=settings.additive_gain,
        crop_borders=settings.crop_borders,
        valid_border_threshold=settings.valid_border_threshold,
    )


def relaxed_stack_settings(strictness: float, settings: PipelineSettings) -> StackJobSettings:
    """Map selective strictness to relaxed alignment settings."""
    bounded = float(np.clip(strictness, 0.0, 1.0))
    return StackJobSettings(
        orb_max_features=2500,
        orb_keep_matches=350,
        min_matches=round(8 + bounded * 22),
        min_inlier_ratio=0.12 + bounded * 0.38,
        ransac_reproj_threshold=6.0 - bounded * 3.0,
        enable_alignment=settings.enable_alignment,
        stack_mode=settings.stack_mode,
        sigma=settings.sigma,
        additive_gain=settings.additive_gain,
        crop_borders=False,
        valid_border_threshold=settings.valid_border_threshold,
    )


def _crop_image(image_bgr: np.ndarray, crop_rect: tuple[int, int, int, int] | None) -> np.ndarray:
    """Crop an image by x_min, y_min, x_max, y_max when a crop rect exists."""
    if crop_rect is None:
        return image_bgr

    x_min, y_min, x_max, y_max = crop_rect
    return image_bgr[y_min:y_max, x_min:x_max]


def _identity_matrix() -> np.ndarray:
    """Return a 2x3 identity affine transform matrix."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _alignment_disabled_results(
    frames: list[np.ndarray],
    reference_index: int,
) -> list[AlignmentResult]:
    """Accept all frames without warping when alignment is disabled."""
    identity = _identity_matrix()
    return [
        AlignmentResult(
            frame=frame,
            accepted=True,
            score=1.0 if index == reference_index else 0.0,
            reason="reference frame" if index == reference_index else "alignment disabled",
            status="reference" if index == reference_index else "alignment disabled",
            matrix=identity.copy(),
        )
        for index, frame in enumerate(frames)
    ]


def accepted_alignment_order(alignment_results: list[AlignmentResult]) -> list[int]:
    """Return accepted result indices in a stable quality order for restacking."""
    accepted_indices = [
        index for index, result in enumerate(alignment_results) if result.accepted
    ]
    reference_indices = [
        index for index in accepted_indices if alignment_results[index].status == "reference"
    ]
    aligned_indices = [
        index for index in accepted_indices if alignment_results[index].status != "reference"
    ]
    aligned_indices.sort(
        key=lambda index: (
            alignment_results[index].inlier_ratio,
            alignment_results[index].inliers,
            alignment_results[index].matches,
        ),
        reverse=True,
    )
    return reference_indices + aligned_indices


def stackable_alignment_order(alignment_results: list[AlignmentResult]) -> list[int]:
    """Return transformable result indices in quality order for the output slider."""
    stackable_indices = [
        index
        for index, result in enumerate(alignment_results)
        if result.status == "reference" or result.matrix is not None
    ]
    reference_indices = [
        index for index in stackable_indices if alignment_results[index].status == "reference"
    ]
    aligned_indices = [
        index
        for index in stackable_indices
        if alignment_results[index].status != "reference"
    ]
    aligned_indices.sort(
        key=lambda index: (
            alignment_results[index].inlier_ratio,
            alignment_results[index].inliers,
            alignment_results[index].matches,
        ),
        reverse=True,
    )
    return reference_indices + aligned_indices


def stack_alignment_subset(
    alignment_results: list[AlignmentResult],
    frame_count: int,
    *,
    stack_mode: StackingMode = "mean",
    sigma: float = 2.5,
    additive_gain: float = 1.0,
    crop_borders: bool = True,
    valid_border_threshold: float = 0.98,
) -> StackSubsetResult:
    """Stack the top transformable aligned frames by inlier quality."""
    ordered_indices = stackable_alignment_order(alignment_results)
    bounded_count = max(1, min(frame_count, len(ordered_indices)))
    selected_indices = ordered_indices[:bounded_count]
    selected_results = [alignment_results[index] for index in selected_indices]
    output_image_bgr = stack_frames(
        [result.frame for result in selected_results],
        stack_mode,
        sigma,
        additive_gain,
    )

    crop_rect = None
    cropped_output_image_bgr: np.ndarray | None = None
    if crop_borders:
        crop_rect = valid_region_crop_rect(selected_results, valid_border_threshold)
        if crop_rect is not None:
            cropped_output_image_bgr = _crop_image(output_image_bgr, crop_rect)

    return StackSubsetResult(
        output_image_bgr=output_image_bgr,
        cropped_output_image_bgr=cropped_output_image_bgr,
        crop_rect=crop_rect,
        selected_indices=selected_indices,
    )


def run_stack_job(
    frames: list[np.ndarray],
    reference_index: int,
    stack_settings: StackJobSettings,
    alignment_allowed_mask: np.ndarray | None = None,
) -> StackJobResult:
    """Align and stack already-extracted frames against the selected reference."""
    if stack_settings.enable_alignment:
        alignment_settings = AlignmentSettings(
            max_features=stack_settings.orb_max_features,
            keep_matches=stack_settings.orb_keep_matches,
            min_matches=stack_settings.min_matches,
            min_inlier_ratio=0.0,
            ransac_reproj_threshold=stack_settings.ransac_reproj_threshold,
        )
        raw_alignment_results = align_frames(
            frames,
            reference_index,
            alignment_settings,
            alignment_allowed_mask=alignment_allowed_mask,
        )
        applied_min_inlier_ratio, alignment_results = _resolve_inlier_ratio_threshold(
            raw_alignment_results,
            stack_settings.min_inlier_ratio,
            stack_settings.min_matches,
        )
    else:
        alignment_results = _alignment_disabled_results(frames, reference_index)
        applied_min_inlier_ratio = 0.0

    accepted_aligned_frames = accepted_frames(alignment_results)
    accepted_count = len(accepted_aligned_frames)
    rejected_count = len(alignment_results) - accepted_count
    stack_image_bgr = stack_frames(
        accepted_aligned_frames,
        stack_settings.stack_mode,
        stack_settings.sigma,
        stack_settings.additive_gain,
    )

    crop_rect = None
    if stack_settings.crop_borders:
        crop_rect = valid_region_crop_rect(
            alignment_results,
            stack_settings.valid_border_threshold,
        )

    reference_bgr = frames[reference_index]
    cropped_reference_bgr = _crop_image(reference_bgr, crop_rect)
    cropped_stack_bgr = _crop_image(stack_image_bgr, crop_rect)

    return StackJobResult(
        stack_image_bgr=stack_image_bgr,
        cropped_reference_bgr=cropped_reference_bgr,
        cropped_stack_bgr=cropped_stack_bgr,
        crop_rect=crop_rect,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        diagnostics=alignment_table(alignment_results).to_dict("records"),
        alignment_results=alignment_results,
        applied_min_inlier_ratio=applied_min_inlier_ratio,
    )


def run_preview(video_path: str | Path, settings: PipelineSettings) -> PreviewResult:
    """Extract frames and select the reference frame for the mask preview."""
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

    return PreviewResult(
        video_metadata=video_metadata,
        extraction_metadata=extraction_metadata,
        frames_processed=frames_processed,
        reference_index=reference_index,
        sharpness_scores=sharpness_scores,
        reference_frame_bgr=reference_frame_bgr,
        frames_bgr=frames_bgr,
    )


def run_pipeline(
    video_path: str | Path,
    settings: PipelineSettings,
    alignment_allowed_mask: np.ndarray | None = None,
) -> PipelineResult:
    """Run video extraction, guided alignment, full-frame stacking, and crop."""
    preview = run_preview(video_path, settings)
    resolved_alignment_allowed_mask = alignment_allowed_mask
    if resolved_alignment_allowed_mask is not None:
        output_height, output_width = preview.reference_frame_bgr.shape[:2]
        if resolved_alignment_allowed_mask.shape != (output_height, output_width):
            resolved_alignment_allowed_mask = cv2.resize(
                resolved_alignment_allowed_mask,
                (output_width, output_height),
                interpolation=cv2.INTER_NEAREST,
            )

    stack_job = run_stack_job(
        preview.frames_bgr,
        preview.reference_index,
        stack_settings_from_pipeline(settings),
        alignment_allowed_mask=resolved_alignment_allowed_mask,
    )
    accepted_count = stack_job.accepted_count
    rejected_count = stack_job.rejected_count
    output_image_bgr = stack_job.stack_image_bgr
    cropped_output_image_bgr: np.ndarray | None = None
    if stack_job.crop_rect is not None:
        cropped_output_image_bgr = stack_job.cropped_stack_bgr

    warnings = _result_warnings(
        settings,
        preview.video_metadata,
        preview.sharpness_scores,
        stack_job.alignment_results,
        accepted_count,
        stack_job.applied_min_inlier_ratio,
        cropped_output_image_bgr,
    )
    mask_coverage = (
        float(np.mean(resolved_alignment_allowed_mask == 0))
        if resolved_alignment_allowed_mask is not None
        else 0.0
    )

    return PipelineResult(
        video_metadata=preview.video_metadata,
        frames_processed=preview.frames_processed,
        reference_index=preview.reference_index,
        sharpness_scores=preview.sharpness_scores,
        alignment_diagnostics=stack_job.diagnostics,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        output_image_bgr=output_image_bgr,
        cropped_output_image_bgr=cropped_output_image_bgr,
        warnings=warnings,
        first_frame_bgr=preview.frames_bgr[0] if preview.frames_bgr else None,
        reference_frame_bgr=preview.reference_frame_bgr,
        applied_min_inlier_ratio=stack_job.applied_min_inlier_ratio,
        frames_bgr=preview.frames_bgr,
        alignment_results=stack_job.alignment_results,
        crop_rect=stack_job.crop_rect,
        mask_coverage=mask_coverage,
    )
