"""End-to-end orchestration for video long-exposure processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from longexposure.alignment import AlignmentResult, AlignmentSettings, align_frames
from longexposure.diagnostics import DiagnosticsSummary, summarize_alignment
from longexposure.frames import extract_frames, select_reference_frame
from longexposure.stacking import (
    StackingMode,
    accepted_frames,
    crop_unstable_borders,
    stack_frames,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configurable limits for a local processing run."""

    max_frames: int = 90
    stride: int = 1
    resize_width: int | None = None
    start_time_seconds: float = 0.0
    duration_seconds: float = 3.0
    alignment_settings: AlignmentSettings | None = None
    stacking_mode: StackingMode = "mean"
    crop_borders: bool = True
    valid_border_threshold: float = 0.98


@dataclass(frozen=True)
class PipelineResult:
    """Result image and diagnostics from a pipeline run."""

    image: np.ndarray
    reference_frame: np.ndarray
    alignment_results: list[AlignmentResult]
    diagnostics: DiagnosticsSummary


def run_pipeline(video_path: Path, config: PipelineConfig | None = None) -> PipelineResult:
    """Run the current long-exposure pipeline.

    The first scaffold intentionally keeps alignment and crop behavior as stubs.
    The averaging step is real, so the app demonstrates the honest photographic
    average mode while the registration logic is still being built.
    """
    resolved_config = config or PipelineConfig()
    frames_bgr, _metadata = extract_frames(
        video_path,
        max_frames=resolved_config.max_frames,
        stride=resolved_config.stride,
        resize_width=resolved_config.resize_width,
        start_time_seconds=resolved_config.start_time_seconds,
        duration_seconds=resolved_config.duration_seconds,
    )
    reference_index, reference_bgr = select_reference_frame(frames_bgr, "median")
    alignment_results = align_frames(
        frames_bgr,
        reference_index,
        resolved_config.alignment_settings,
    )
    frames_to_stack = accepted_frames(alignment_results)
    stacked_bgr = stack_frames(frames_to_stack, resolved_config.stacking_mode)
    if resolved_config.crop_borders:
        stacked_bgr = crop_unstable_borders(
            stacked_bgr,
            alignment_results,
            resolved_config.valid_border_threshold,
        )
    stacked_rgb = cv2.cvtColor(stacked_bgr, cv2.COLOR_BGR2RGB)
    diagnostics = summarize_alignment(alignment_results)

    return PipelineResult(
        image=stacked_rgb,
        reference_frame=cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB),
        alignment_results=alignment_results,
        diagnostics=diagnostics,
    )
