"""Streamlit app for experimenting with video-based long exposure images."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from longexposure.frames import extract_frames
from longexposure.io import get_video_summary, save_uploaded_video_to_temp


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

        st.subheader("Video Metadata")
        st.dataframe(_metadata_table(summary), hide_index=True, width="stretch")

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
            width="stretch",
        )

        first_frame_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        st.image(
            first_frame_rgb,
            caption="First extracted frame",
            width="content",
        )
    except ValueError as error:
        st.error(str(error))
    finally:
        video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
