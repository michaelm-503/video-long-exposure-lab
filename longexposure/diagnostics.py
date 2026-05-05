"""Diagnostics for explaining what happened during processing."""

from __future__ import annotations

from dataclasses import dataclass

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
            "frame": index,
            "accepted": result.accepted,
            "score": result.score,
            "reason": result.reason,
        }
        for index, result in enumerate(results)
    ]
    return pd.DataFrame(rows)

