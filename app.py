"""Streamlit app for experimenting with video-based long exposure images."""

from __future__ import annotations

from dataclasses import replace
from html import escape
from pathlib import Path
from typing import cast

import cv2
import pandas as pd
import streamlit as st

from longexposure.alignment import AlignmentSettings, align_frames
from longexposure.diagnostics import alignment_table, sharpness_figure
from longexposure.frames import (
    ReferenceStrategy,
    extract_frames,
    score_frames_sharpness,
    select_reference_frame,
)
from longexposure.io import get_video_summary, save_uploaded_video_to_temp
from longexposure.stacking import accepted_frames, average_frames


APP_TITLE = "Video Long Exposure Lab"
INLIER_RATIO_RELAX_STEP = 0.05


def _natural_media_width(width: float | int) -> int:
    """Return a safe pixel width for media displayed at natural size."""
    return max(1, int(width))


def _metadata_table(metadata: dict[str, float | int]) -> pd.DataFrame:
    """Format video metadata for Streamlit display."""
    return pd.DataFrame(
        {
            "Metric": ["FPS", "Frame count", "Duration seconds", "Width", "Height"],
            "Value": [
                round(float(metadata["fps"]), 2),
                metadata["frame_count"],
                round(float(metadata["duration_seconds"]), 2),
                metadata["width"],
                metadata["height"],
            ],
        }
    )


def _extraction_table(metadata: dict[str, float | int]) -> pd.DataFrame:
    """Format extraction metadata for Streamlit display."""
    return pd.DataFrame(
        {
            "Metric": [
                "Frames extracted",
                "Frames read",
                "Start frame",
                "Frame stride",
                "Resize width",
                "Failed reads",
            ],
            "Value": [
                metadata["frames_extracted"],
                metadata["frames_read"],
                metadata["start_frame"],
                metadata["stride"],
                metadata["resize_width"],
                metadata["failed_reads"],
            ],
        }
    )


def _render_summary_table(table: pd.DataFrame) -> None:
    """Render a compact static two-column table without dataframe scrollbars."""
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row.Metric))}</td>"
        f"<td>{escape(str(row.Value))}</td>"
        "</tr>"
        for row in table.itertuples(index=False)
    )
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e6e8ef;
            border-radius: 6px;
            display: inline-block;
            margin: 0.25rem 0 1rem 0;
            overflow: hidden;
        ">
        <table class="summary-table" style="
            border-collapse: collapse;
            font-size: 0.875rem;
            line-height: 1.35;
            min-width: 250px;
        ">
            <thead>
                <tr style="background: #f7f8fb;">
                    <th style="border-bottom: 1px solid #e6e8ef; padding: 0.5rem 0.75rem; text-align: left;">Metric</th>
                    <th style="border-bottom: 1px solid #e6e8ef; padding: 0.5rem 0.75rem; text-align: left;">Value</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
        <style>
            table.summary-table td {{
                border-top: 1px solid #eef0f4;
                padding: 0.45rem 0.75rem;
                white-space: nowrap;
            }}
            table.summary-table td:first-child {{
                min-width: 135px;
            }}
            table.summary-table td:last-child {{
                min-width: 70px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _apply_inlier_ratio_threshold(
    alignment_results: list,
    threshold: float,
) -> list:
    """Apply an inlier-ratio threshold to precomputed alignment results."""
    adjusted_results = []
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


def _accepted_count(alignment_results: list) -> int:
    """Count accepted alignment results."""
    return sum(1 for result in alignment_results if result.accepted)


def _relax_inlier_ratio(
    alignment_results: list,
    start_threshold: float,
    target_accepted_frames: int,
) -> tuple[float, list]:
    """Relax inlier ratio until enough frames are accepted or zero is reached."""
    threshold = start_threshold
    target = max(1, min(target_accepted_frames, len(alignment_results)))

    while threshold > 0:
        adjusted_results = _apply_inlier_ratio_threshold(alignment_results, threshold)
        if _accepted_count(adjusted_results) >= target:
            return threshold, adjusted_results
        threshold = max(0.0, round(threshold - INLIER_RATIO_RELAX_STEP, 2))

    adjusted_results = _apply_inlier_ratio_threshold(alignment_results, threshold)
    return threshold, adjusted_results


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("A local portfolio lab for honest frame averaging from video.")

    uploaded_file = st.file_uploader(
        "Upload a short video",
        type=["mov", "mp4", "m4v"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload a short local video to inspect metadata and extracted frames.")
        return

    uploaded_bytes = uploaded_file.getvalue()

    suffix = Path(uploaded_file.name).suffix
    video_path = save_uploaded_video_to_temp(uploaded_bytes, suffix)

    try:
        summary = get_video_summary(video_path)
        video_width = int(summary["width"])
        video_duration = float(summary["duration_seconds"])
        default_resize_width = round(video_width * 0.95) if video_width > 0 else 0
        st.video(uploaded_bytes, width=_natural_media_width(video_width))

        with st.sidebar:
            st.header("Processing")
            max_frames = st.slider(
                "Maximum frames",
                min_value=5,
                max_value=300,
                value=90,
                step=5,
            )
            stride = st.number_input(
                "Frame stride",
                min_value=1,
                max_value=60,
                value=1,
                step=1,
            )
            resize_width = st.number_input(
                "Resize width",
                min_value=1,
                max_value=max(1, video_width),
                value=max(1, default_resize_width),
                step=1,
            )
            start_time_seconds = st.number_input(
                "Start time seconds",
                min_value=0.0,
                value=0.0,
                step=0.5,
            )
            process_duration_seconds = st.number_input(
                "Duration seconds to process",
                min_value=0.1,
                value=min(3.0, video_duration) if video_duration > 0 else 3.0,
                step=0.5,
            )
            reference_strategy = cast(
                ReferenceStrategy,
                st.selectbox(
                    "Reference strategy",
                    options=["median", "sharpest", "middle", "first"],
                    index=0,
                ),
            )
            st.header("Alignment")
            max_orb_features = st.number_input(
                "Max ORB features",
                min_value=100,
                max_value=5000,
                value=1500,
                step=100,
            )
            keep_top_matches = st.number_input(
                "Keep top matches",
                min_value=10,
                max_value=1000,
                value=200,
                step=10,
            )
            minimum_matches = st.number_input(
                "Minimum matches",
                min_value=3,
                max_value=500,
                value=10,
                step=1,
            )
            minimum_inlier_ratio = st.slider(
                "Minimum inlier ratio",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.01,
            )
            ransac_reproj_threshold = st.number_input(
                "RANSAC reprojection threshold",
                min_value=0.5,
                max_value=20.0,
                value=3.0,
                step=0.5,
            )

        st.subheader("Video Metadata")
        _render_summary_table(_metadata_table(summary))

        with st.spinner("Extracting frames from the selected time window..."):
            frames, extraction_metadata = extract_frames(
                video_path,
                max_frames=max_frames,
                stride=int(stride),
                resize_width=int(resize_width),
                start_time_seconds=float(start_time_seconds),
                duration_seconds=float(process_duration_seconds),
            )

        st.subheader("Extraction")
        _render_summary_table(_extraction_table(extraction_metadata))

        first_frame_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        st.image(
            first_frame_rgb,
            caption="First extracted frame",
            width="content",
        )

        sharpness_scores = score_frames_sharpness(frames)
        reference_index, reference_frame = select_reference_frame(
            frames,
            reference_strategy,
        )
        reference_frame_rgb = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2RGB)

        st.subheader("Reference Frame")
        st.metric("Selected reference index", reference_index)
        st.image(
            reference_frame_rgb,
            caption=f"Reference frame ({reference_strategy})",
            width="content",
        )
        st.pyplot(
            sharpness_figure(sharpness_scores, reference_index),
            width="content",
        )

        alignment_settings = AlignmentSettings(
            max_features=int(max_orb_features),
            keep_matches=int(keep_top_matches),
            min_matches=int(minimum_matches),
            min_inlier_ratio=0.0,
            ransac_reproj_threshold=float(ransac_reproj_threshold),
        )

        with st.spinner("Aligning frames to the selected reference..."):
            raw_alignment_results = align_frames(
                frames,
                reference_index,
                alignment_settings,
            )
            applied_inlier_ratio, alignment_results = _relax_inlier_ratio(
                raw_alignment_results,
                float(minimum_inlier_ratio),
                int(minimum_matches),
            )
            accepted_aligned_frames = accepted_frames(alignment_results)
            averaged_bgr = average_frames(accepted_aligned_frames)

        st.subheader("Alignment")
        metric_columns = st.columns(2)
        metric_columns[0].metric(
            "Accepted frames",
            f"{len(accepted_aligned_frames)} / {len(frames)}",
        )
        metric_columns[1].metric(
            "Applied minimum inlier ratio",
            f"{applied_inlier_ratio:.2f}",
        )
        st.dataframe(
            alignment_table(alignment_results),
            hide_index=True,
            width="content",
        )

        averaged_rgb = cv2.cvtColor(averaged_bgr.astype("uint8"), cv2.COLOR_BGR2RGB)
        st.subheader("Aligned Average")
        st.image(
            averaged_rgb,
            caption="Accepted aligned frames averaged full-frame",
            width="content",
        )
    except ValueError as error:
        st.error(str(error))
    finally:
        video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
