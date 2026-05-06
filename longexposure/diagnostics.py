"""Diagnostics for explaining what happened during processing."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "video-long-exposure-lab-matplotlib"),
)

from matplotlib.figure import Figure
import pandas as pd

from longexposure.alignment import AlignmentResult


@dataclass(frozen=True)
class DiagnosticsSummary:
    """High-level counts and scores from a pipeline run."""

    total_frames: int
    accepted_frames: int
    rejected_frames: int
    average_score: float


def summarize_alignment(results: list[AlignmentResult]) -> DiagnosticsSummary:
    """Summarize alignment results for display in the app."""
    total = len(results)
    accepted = sum(1 for result in results if result.accepted)
    rejected = total - accepted
    average_score = (
        sum(result.score for result in results) / total
        if total > 0
        else 0.0
    )

    return DiagnosticsSummary(
        total_frames=total,
        accepted_frames=accepted,
        rejected_frames=rejected,
        average_score=average_score,
    )


def alignment_table(results: list[AlignmentResult]) -> pd.DataFrame:
    """Build a tabular view of frame-level alignment diagnostics."""
    rows = [
        {
            "frame index": index,
            "accepted/rejected": "accepted" if result.accepted else "rejected",
            "matches": result.matches,
            "inliers": result.inliers,
            "inlier_ratio": round(result.inlier_ratio, 3),
            "status": result.status or result.reason,
        }
        for index, result in enumerate(results)
    ]
    return pd.DataFrame(rows)


def sharpness_figure(scores: list[float], reference_index: int) -> Figure:
    """Build a simple sharpness plot with the selected reference frame marked."""
    figure = Figure(figsize=(5.4, 2.4), layout="constrained")
    axis = figure.subplots()
    frame_numbers = list(range(len(scores)))

    axis.plot(frame_numbers, scores, color="#2f6fed", linewidth=1.8)
    axis.scatter(frame_numbers, scores, color="#2f6fed", s=18)

    if scores:
        axis.scatter(
            [reference_index],
            [scores[reference_index]],
            color="#d44f24",
            s=48,
            zorder=3,
            label=f"Reference frame {reference_index}",
        )
        axis.legend(loc="best")

    axis.set_title("Frame sharpness")
    axis.set_xlabel("Extracted frame index")
    axis.set_ylabel("Laplacian variance")
    axis.grid(True, alpha=0.25)

    return figure


def accepted_rejected_figure(results: list[AlignmentResult]) -> Figure:
    """Build a compact accepted/rejected count chart."""
    accepted = sum(1 for result in results if result.accepted)
    rejected = len(results) - accepted
    figure = Figure(figsize=(4.2, 2.2), layout="constrained")
    axis = figure.subplots()
    axis.bar(
        ["Accepted", "Rejected"],
        [accepted, rejected],
        color=["#2f8f5b", "#b84a4a"],
    )
    axis.set_title("Frame acceptance")
    axis.set_ylabel("Frames")
    axis.grid(True, axis="y", alpha=0.25)
    return figure


def inlier_ratio_figure(results: list[AlignmentResult]) -> Figure:
    """Build a frame-index plot of alignment inlier ratios."""
    figure = Figure(figsize=(5.4, 2.4), layout="constrained")
    axis = figure.subplots()
    frame_numbers = list(range(len(results)))
    ratios = [result.inlier_ratio for result in results]
    colors = ["#2f8f5b" if result.accepted else "#b84a4a" for result in results]

    axis.plot(frame_numbers, ratios, color="#6d7685", linewidth=1.3, alpha=0.8)
    axis.scatter(frame_numbers, ratios, color=colors, s=22, zorder=3)
    axis.set_title("Alignment inlier ratio")
    axis.set_xlabel("Extracted frame index")
    axis.set_ylabel("Inlier ratio")
    axis.set_ylim(0, 1.02)
    axis.grid(True, alpha=0.25)
    return figure
