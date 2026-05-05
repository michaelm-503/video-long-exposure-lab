"""Streamlit app for experimenting with video-based long exposure images."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from longexposure.diagnostics import sharpness_figure
from longexposure.frames import ReferenceStrategy
from longexposure.io import get_video_summary, save_uploaded_video_to_temp
from longexposure.pipeline import PipelineResult, PipelineSettings, run_pipeline
from longexposure.stacking import StackingMode


APP_TITLE = "Video Long Exposure Lab"
OUTPUT_DIR = Path("outputs")
STACKING_MODE_LABELS: dict[str, StackingMode] = {
    "Mean": "mean",
    "Sigma-clipped mean (cleanup)": "sigma_clipped_mean",
    "Median (experimental)": "median",
}


def _natural_media_width(width: float | int) -> int:
    """Return a safe pixel width for media displayed at natural size."""
    return max(1, int(width))


def _encode_image(image_bgr: np.ndarray, extension: str) -> bytes:
    """Encode a BGR image for downloads."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension == ".jpg" else []
    ok, encoded = cv2.imencode(extension, image_bgr, encode_params)
    if not ok:
        raise ValueError(f"Could not encode image as {extension}")
    return encoded.tobytes()


def _save_output_images(image_bgr: np.ndarray) -> tuple[Path, Path]:
    """Save PNG and JPEG outputs locally for portfolio/demo runs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "long_exposure.png"
    jpg_path = OUTPUT_DIR / "long_exposure.jpg"
    cv2.imwrite(str(png_path), image_bgr)
    cv2.imwrite(str(jpg_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return png_path, jpg_path


@st.cache_data(show_spinner=False)
def _cached_video_summary(video_path: str) -> dict[str, float | int]:
    """Return cached video metadata for a local temporary file path."""
    return get_video_summary(Path(video_path))


def _metadata_table(metadata: dict[str, float | int]) -> pd.DataFrame:
    """Format video metadata for display."""
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


def _settings_from_sidebar(
    *,
    video_width: int,
    video_duration: float,
) -> PipelineSettings:
    """Collect Streamlit sidebar controls into pipeline settings."""
    default_resize_width = round(video_width * 0.95) if video_width > 0 else 1

    with st.sidebar:
        st.header("Processing")
        max_frames = st.slider(
            "Maximum frames",
            min_value=5,
            max_value=300,
            value=90,
            step=5,
        )
        frame_stride = st.number_input(
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
        start_time_s = st.number_input(
            "Start time seconds",
            min_value=0.0,
            value=0.0,
            step=0.5,
        )
        duration_s = st.number_input(
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
        orb_max_features = st.number_input(
            "Max ORB features",
            min_value=100,
            max_value=5000,
            value=1500,
            step=100,
        )
        orb_keep_matches = st.number_input(
            "Keep top matches",
            min_value=10,
            max_value=1000,
            value=200,
            step=10,
        )
        min_matches = st.number_input(
            "Minimum matches",
            min_value=3,
            max_value=500,
            value=10,
            step=1,
        )
        min_inlier_ratio = st.slider(
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

        st.header("Stacking")
        stacking_label = st.selectbox(
            "Stacking mode",
            options=list(STACKING_MODE_LABELS),
            index=0,
        )
        stack_mode = STACKING_MODE_LABELS[stacking_label]
        sigma = st.number_input(
            "Sigma clipping",
            min_value=0.5,
            max_value=10.0,
            value=2.5,
            step=0.1,
        )
        crop_borders = st.checkbox("Crop borders", value=True)
        valid_border_threshold = st.slider(
            "Valid border threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.98,
            step=0.01,
        )

    return PipelineSettings(
        start_time_s=float(start_time_s),
        duration_s=float(duration_s),
        max_frames=int(max_frames),
        frame_stride=int(frame_stride),
        resize_width=int(resize_width),
        reference_strategy=reference_strategy,
        orb_max_features=int(orb_max_features),
        orb_keep_matches=int(orb_keep_matches),
        min_matches=int(min_matches),
        min_inlier_ratio=float(min_inlier_ratio),
        ransac_reproj_threshold=float(ransac_reproj_threshold),
        stack_mode=stack_mode,
        sigma=float(sigma),
        crop_borders=bool(crop_borders),
        valid_border_threshold=float(valid_border_threshold),
    )


def _render_pipeline_result(result: PipelineResult) -> None:
    """Render pipeline output and diagnostics."""
    st.subheader("Extraction")
    _render_summary_table(
        pd.DataFrame({"Metric": ["Frames processed"], "Value": [result.frames_processed]})
    )

    if result.first_frame_bgr is not None:
        first_frame_rgb = cv2.cvtColor(result.first_frame_bgr, cv2.COLOR_BGR2RGB)
        st.image(first_frame_rgb, caption="First extracted frame", width="content")

    st.subheader("Reference Frame")
    st.metric("Selected reference index", result.reference_index)
    if result.reference_frame_bgr is not None:
        reference_frame_rgb = cv2.cvtColor(
            result.reference_frame_bgr,
            cv2.COLOR_BGR2RGB,
        )
        st.image(reference_frame_rgb, caption="Reference frame", width="content")
    st.pyplot(
        sharpness_figure(result.sharpness_scores, result.reference_index),
        width="content",
    )

    st.subheader("Alignment")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Accepted frames", result.accepted_count)
    metric_columns[1].metric("Rejected frames", result.rejected_count)
    metric_columns[2].metric(
        "Applied minimum inlier ratio",
        f"{result.applied_min_inlier_ratio:.2f}",
    )
    st.dataframe(
        pd.DataFrame(result.alignment_diagnostics),
        hide_index=True,
        width="content",
    )

    for warning in result.warnings:
        st.warning(warning)

    final_bgr = (
        result.cropped_output_image_bgr
        if result.cropped_output_image_bgr is not None
        else result.output_image_bgr
    )
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    _save_output_images(final_bgr)
    png_bytes = _encode_image(final_bgr, ".png")
    jpg_bytes = _encode_image(final_bgr, ".jpg")

    st.subheader("Final Output")
    st.image(final_rgb, caption="Long-exposure still", width="content")
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download PNG",
        data=png_bytes,
        file_name="long_exposure.png",
        mime="image/png",
        on_click="ignore",
    )
    download_columns[1].download_button(
        "Download JPEG",
        data=jpg_bytes,
        file_name="long_exposure.jpg",
        mime="image/jpeg",
        on_click="ignore",
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
        summary = _cached_video_summary(str(video_path))
        video_width = int(summary["width"])
        video_duration = float(summary["duration_seconds"])
        st.video(uploaded_bytes, width=_natural_media_width(video_width))

        settings = _settings_from_sidebar(
            video_width=video_width,
            video_duration=video_duration,
        )

        st.subheader("Video Metadata")
        _render_summary_table(_metadata_table(summary))

        if not st.button("Run pipeline", type="primary"):
            st.info("Adjust settings, then run the pipeline to produce the final image.")
            return

        with st.status("Processing video", expanded=True) as status:
            st.write("Extracting frames and scoring sharpness.")
            st.write("Selecting a reference frame and aligning accepted frames.")
            st.write("Stacking frames and cropping unstable borders.")
            result = run_pipeline(video_path, settings)
            status.update(label="Processing complete", state="complete")

        _render_pipeline_result(result)
    except ValueError as error:
        st.error(str(error))
    finally:
        video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
