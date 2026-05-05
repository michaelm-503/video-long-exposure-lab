"""End-to-end orchestration for video long-exposure processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from longexposure.alignment import AlignmentResult, align_frames
from longexposure.diagnostics import DiagnosticsSummary, summarize_alignment
from longexposure.frames import choose_reference_frame, extract_frames
from longexposure.stacking import accepted_frames, average_frames, crop_unstable_borders


@dataclass(frozen=True)
class PipelineConfig:
    """Configurable limits for a local processing run."""

    max_frames: int = 90
    stride: int = 1
    resize_width: int | None = None
    start_time_seconds: float = 0.0
    duration_seconds: float = 3.0


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
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    reference = choose_reference_frame(frames)
    alignment_results = align_frames(frames, reference)
    frames_to_stack = accepted_frames(alignment_results)
    averaged = average_frames(frames_to_stack)
    cropped = crop_unstable_borders(averaged)
    diagnostics = summarize_alignment(alignment_results)

    return PipelineResult(
        image=cropped,
        reference_frame=reference,
        alignment_results=alignment_results,
        diagnostics=diagnostics,
    )
