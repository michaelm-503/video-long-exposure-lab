"""Streamlit app for experimenting with video-based long exposure images."""

from __future__ import annotations

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
                value=30,
                step=1,
            )
            minimum_inlier_ratio = st.slider(
                "Minimum inlier ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
            )
            ransac_reproj_threshold = st.number_input(
                "RANSAC reprojection threshold",
                min_value=0.5,
                max_value=20.0,
                value=3.0,
                step=0.5,
            )

        st.subheader("Video Metadata")
        st.dataframe(_metadata_table(summary), hide_index=True, width="content")

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
        st.dataframe(
            _extraction_table(extraction_metadata),
            hide_index=True,
            width="content",
        )

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
            min_inlier_ratio=float(minimum_inlier_ratio),
            ransac_reproj_threshold=float(ransac_reproj_threshold),
        )

        with st.spinner("Aligning frames to the selected reference..."):
            alignment_results = align_frames(frames, reference_index, alignment_settings)
            accepted_aligned_frames = accepted_frames(alignment_results)
            averaged_bgr = average_frames(accepted_aligned_frames)

        st.subheader("Alignment")
        st.metric("Accepted frames", f"{len(accepted_aligned_frames)} / {len(frames)}")
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
