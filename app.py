"""Streamlit app for experimenting with video-based long exposure images."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st

from longexposure.diagnostics import alignment_table
from longexposure.io import export_image, get_video_summary, save_uploaded_video
from longexposure.pipeline import PipelineConfig, run_pipeline


APP_TITLE = "Video Long Exposure Lab"
OUTPUT_DIR = Path("outputs")


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("A local portfolio lab for honest frame averaging from video.")

    uploaded_file = st.file_uploader(
        "Upload a short video",
        type=["mp4", "mov", "m4v", "avi"],
        accept_multiple_files=False,
    )

    with st.sidebar:
        st.header("Processing")
        max_frames = st.slider(
            "Maximum frames",
            min_value=5,
            max_value=300,
            value=120,
            step=5,
        )
        stride = st.number_input(
            "Frame stride",
            min_value=1,
            max_value=60,
            value=1,
            step=1,
        )
        run_requested = st.button(
            "Build image",
            type="primary",
            disabled=uploaded_file is None,
        )

    if uploaded_file is None:
        st.info("Upload a short local video to create a still image from averaged frames.")
        return

    with NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
        video_path = save_uploaded_video(uploaded_file.getvalue(), Path(temp_file.name))

    summary = get_video_summary(video_path)
    st.subheader("Video")
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": ["Frames", "FPS", "Width", "Height", "Duration seconds"],
                "Value": [
                    summary["frame_count"],
                    round(float(summary["fps"]), 2),
                    summary["width"],
                    summary["height"],
                    round(float(summary["duration_seconds"]), 2),
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    if not run_requested:
        st.info("Choose processing settings, then build the image.")
        return

    config = PipelineConfig(max_frames=max_frames, stride=int(stride))

    with st.spinner("Extracting, aligning, filtering, averaging, and cropping frames..."):
        result = run_pipeline(video_path, config)
        output_path = export_image(result.image, OUTPUT_DIR / "long_exposure.png")

    st.subheader("Result")
    st.image(result.image.astype("uint8"), caption="Long-exposure average", use_container_width=True)

    with output_path.open("rb") as image_file:
        st.download_button(
            "Download PNG",
            data=image_file,
            file_name=output_path.name,
            mime="image/png",
        )

    diagnostics = result.diagnostics
    metric_columns = st.columns(4)
    metric_columns[0].metric("Frames", diagnostics.total_frames)
    metric_columns[1].metric("Accepted", diagnostics.accepted_frames)
    metric_columns[2].metric("Rejected", diagnostics.rejected_frames)
    metric_columns[3].metric("Avg score", f"{diagnostics.average_score:.2f}")

    with st.expander("Frame diagnostics"):
        st.dataframe(alignment_table(result.alignment_results), use_container_width=True)


if __name__ == "__main__":
    main()
