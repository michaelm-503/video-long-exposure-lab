"""Streamlit app for experimenting with video-based long exposure images."""

from __future__ import annotations

import base64
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:  # pragma: no cover - depends on local environment install
    st_canvas = None

from longexposure.blending import (
    extract_paint_mask_from_canvas,
    make_alignment_allowed_mask,
)
from longexposure.diagnostics import sharpness_figure
from longexposure.frames import ReferenceStrategy
from longexposure.io import get_video_summary, save_uploaded_video_to_temp
from longexposure.pipeline import (
    PipelineResult,
    PipelineSettings,
    PreviewResult,
    run_preview,
    run_pipeline,
    stack_alignment_subset,
    stackable_alignment_order,
)
from longexposure.stacking import StackingMode


APP_TITLE = "Video Long Exposure Lab"
OUTPUT_DIR = Path("outputs")
PREVIEW_RESIZE_WIDTH = 800
PREVIEW_FRAME_REFRESH_THRESHOLD = 120
VIDEO_MIME_TYPES = {
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
}
STACKING_MODE_LABELS: dict[str, StackingMode] = {
    "Mean": "mean",
    "Sigma-clipped mean (cleanup)": "sigma_clipped_mean",
    "Median (experimental)": "median",
}


def _inject_ui_styles() -> None:
    """Keep media previews content-sized in Streamlit 1.40."""
    st.markdown(
        """
        <style>
            video.content-sized-video {
                display: block;
                max-width: 100%;
                max-height: 70vh;
                height: auto;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _encode_image(image_bgr: np.ndarray, extension: str) -> bytes:
    """Encode a BGR image for downloads."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension == ".jpg" else []
    ok, encoded = cv2.imencode(extension, image_bgr, encode_params)
    if not ok:
        raise ValueError(f"Could not encode image as {extension}")
    return encoded.tobytes()


def _encode_gray_png(mask: np.ndarray) -> bytes:
    """Encode a 2D mask as PNG bytes."""
    ok, encoded = cv2.imencode(".png", np.clip(mask, 0, 255).astype(np.uint8))
    if not ok:
        raise ValueError("Could not encode alpha mask as PNG")
    return encoded.tobytes()


def _save_output_images(image_bgr: np.ndarray) -> tuple[Path, Path]:
    """Save PNG and JPEG outputs locally for portfolio/demo runs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "long_exposure.png"
    jpg_path = OUTPUT_DIR / "long_exposure.jpg"
    cv2.imwrite(str(png_path), image_bgr)
    cv2.imwrite(str(jpg_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return png_path, jpg_path


def _render_video(
    video_bytes: bytes,
    *,
    suffix: str,
    video_width: int,
) -> None:
    """Render video at its intrinsic width or shrink to fit."""
    encoded_video = base64.b64encode(video_bytes).decode("ascii")
    mime_type = VIDEO_MIME_TYPES.get(suffix.lower(), "video/mp4")
    max_width = min(PREVIEW_RESIZE_WIDTH, max(1, video_width))
    st.markdown(
        f"""
        <video class="content-sized-video" controls preload="metadata"
            style="width: min({max_width}px, 100%);">
            <source src="data:{mime_type};base64,{encoded_video}" type="{mime_type}">
        </video>
        """,
        unsafe_allow_html=True,
    )


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


def _target_min_matches(frame_count_hint: int | None) -> int:
    """Return the default alignment target: half the frames, capped at 60."""
    estimated_frames = frame_count_hint or 20
    return max(3, min(60, round(estimated_frames * 0.5)))


def _settings_from_sidebar(
    *,
    video_duration: float,
    frame_count_hint: int | None = None,
) -> tuple[PipelineSettings, bool]:
    """Collect Streamlit sidebar controls into pipeline settings."""
    default_min_matches = _target_min_matches(frame_count_hint)

    with st.sidebar:
        st.header("Processing")
        max_frames = st.slider(
            "Maximum frames",
            min_value=5,
            max_value=300,
            value=300,
            step=5,
        )
        frame_stride = st.number_input(
            "Frame stride",
            min_value=1,
            max_value=60,
            value=1,
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
            value=video_duration if video_duration > 0 else 3.0,
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
        update_preview = st.button(
            "Update preview",
            type="secondary",
            use_container_width=True,
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
        if (
            st.session_state.get("min_matches_target") != default_min_matches
            or st.session_state.get("min_matches", default_min_matches) > 60
        ):
            st.session_state.min_matches = default_min_matches
            st.session_state.min_matches_target = default_min_matches
        min_matches = st.number_input(
            "Minimum matches",
            min_value=3,
            max_value=60,
            key="min_matches",
            step=1,
        )
        min_inlier_ratio = st.slider(
            "Minimum inlier ratio (0.00 = Auto)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help=(
                "Set to 0.00 for auto mode. Auto starts at 1.00 and steps down "
                "until the minimum-match target is met. Any other value is a "
                "fixed cutoff."
            ),
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

    settings = PipelineSettings(
        start_time_s=float(start_time_s),
        duration_s=float(duration_s),
        max_frames=int(max_frames),
        frame_stride=int(frame_stride),
        resize_width=None,
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
    return settings, update_preview


def _preview_settings(settings: PipelineSettings, video_width: int) -> PipelineSettings:
    """Use the same processing window at a smaller width for mask preview."""
    preview_width = min(PREVIEW_RESIZE_WIDTH, video_width) if video_width > 0 else PREVIEW_RESIZE_WIDTH
    return replace(settings, resize_width=preview_width)


def _preview_signature(
    upload_signature: tuple[str, int],
    settings: PipelineSettings,
    video_width: int,
) -> tuple[object, ...]:
    """Return only the values that affect preview extraction/reference selection."""
    preview_width = min(PREVIEW_RESIZE_WIDTH, video_width) if video_width > 0 else PREVIEW_RESIZE_WIDTH
    return (
        upload_signature,
        settings.start_time_s,
        settings.duration_s,
        settings.max_frames,
        settings.frame_stride,
        settings.reference_strategy,
        preview_width,
    )


def _output_size_from_metadata(
    metadata: dict[str, float | int],
    resize_width: int | None,
) -> tuple[int, int]:
    """Return the expected final frame size as height, width."""
    source_width = int(metadata["width"])
    source_height = int(metadata["height"])
    if resize_width is None or resize_width <= 0 or resize_width >= source_width:
        return source_height, source_width

    output_height = max(1, round(source_height * resize_width / source_width))
    return output_height, int(resize_width)


def _extract_guidance_mask(
    canvas_result,
    background_rgb_display: np.ndarray,
    preview_size: tuple[int, int],
    output_size: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Extract painted alpha and convert it to an ORB allowed-region mask."""
    paint_alpha = extract_paint_mask_from_canvas(
        canvas_result,
        background_rgb_display,
        preview_size,
    )
    coverage = float(np.mean(paint_alpha > 0.05))
    if coverage <= 0.0005:
        return None, None, 0.0

    if paint_alpha.shape != output_size:
        output_height, output_width = output_size
        paint_alpha = cv2.resize(
            paint_alpha,
            (output_width, output_height),
            interpolation=cv2.INTER_LINEAR,
        )

    allowed_mask = make_alignment_allowed_mask(paint_alpha)
    return paint_alpha, allowed_mask, float(np.mean(paint_alpha > 0.05))


def _render_preview_and_mask(
    preview: PreviewResult,
    *,
    can_process: bool,
) -> tuple[object | None, np.ndarray | None, bool]:
    """Render reference preview, sharpness plot, and mask canvas."""
    st.subheader("Reference and Alignment Guide")
    st.metric("Selected reference index", preview.reference_index)
    st.pyplot(
        sharpness_figure(preview.sharpness_scores, preview.reference_index),
        use_container_width=False,
    )

    if st_canvas is None:
        reference_rgb = cv2.cvtColor(preview.reference_frame_bgr, cv2.COLOR_BGR2RGB)
        st.image(reference_rgb, caption="Reference frame")
        st.error(
            "Install streamlit-drawable-canvas, then restart Streamlit to paint an alignment guide."
        )
        process_clicked = st.button(
            "Process image",
            type="primary",
            disabled=not can_process,
        )
        return None, None, process_clicked

    if "canvas_nonce" not in st.session_state:
        st.session_state.canvas_nonce = 0

    background_rgb_display, canvas_width, canvas_height = _canvas_background(
        preview.reference_frame_bgr,
        max_width=PREVIEW_RESIZE_WIDTH,
    )
    canvas_column, controls_column = st.columns([4, 1], gap="medium")
    with controls_column:
        brush_size = st.slider("Brush size", min_value=4, max_value=128, value=64, step=4)
        if st.button("Reset mask"):
            st.session_state.canvas_nonce += 1
            st.session_state.pop("guided_result", None)
        process_clicked = st.button(
            "Process image",
            type="primary",
            disabled=not can_process,
        )

    with canvas_column:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=brush_size,
            stroke_color="#FFFFFF",
            background_image=Image.fromarray(background_rgb_display),
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            display_toolbar=False,
            key=f"alignment_guide_canvas_{st.session_state.canvas_nonce}",
        )
    return canvas_result, background_rgb_display, process_clicked


def _render_pipeline_result(result: PipelineResult, settings: PipelineSettings) -> None:
    """Render guided pipeline output and diagnostics."""
    st.subheader("Extraction")
    _render_summary_table(
        pd.DataFrame({"Metric": ["Frames processed"], "Value": [result.frames_processed]})
    )

    st.subheader("Alignment")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Accepted frames", result.accepted_count)
    metric_columns[1].metric("Rejected frames", result.rejected_count)
    metric_columns[2].metric(
        "Applied minimum inlier ratio",
        f"{result.applied_min_inlier_ratio:.2f}",
    )
    metric_columns[3].metric("Guided mask coverage", f"{result.mask_coverage:.1%}")
    st.dataframe(
        pd.DataFrame(result.alignment_diagnostics),
        hide_index=True,
    )

    for warning in result.warnings:
        st.warning(warning)

    selected_count = result.accepted_count
    final_bgr = (
        result.cropped_output_image_bgr
        if result.cropped_output_image_bgr is not None
        else result.output_image_bgr
    )
    if result.alignment_results is not None and result.accepted_count > 0:
        stackable_count = len(stackable_alignment_order(result.alignment_results))
        initial_count = min(result.accepted_count, stackable_count)
        selected_count = st.slider(
            "# averaged frames",
            min_value=1,
            max_value=stackable_count,
            value=initial_count,
            step=1,
        )
        subset = stack_alignment_subset(
            result.alignment_results,
            selected_count,
            stack_mode=settings.stack_mode,
            sigma=settings.sigma,
            crop_borders=settings.crop_borders,
            valid_border_threshold=settings.valid_border_threshold,
        )
        final_bgr = (
            subset.cropped_output_image_bgr
            if subset.cropped_output_image_bgr is not None
            else subset.output_image_bgr
        )

    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    _save_output_images(final_bgr)
    png_bytes = _encode_image(final_bgr, ".png")
    jpg_bytes = _encode_image(final_bgr, ".jpg")

    st.subheader("Photographic Full-Frame Average")
    st.image(
        final_rgb,
        caption=f"Photographic full-frame average ({selected_count} frames)",
    )
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download PNG",
        data=png_bytes,
        file_name="long_exposure.png",
        mime="image/png",
    )
    download_columns[1].download_button(
        "Download JPEG",
        data=jpg_bytes,
        file_name="long_exposure.jpg",
        mime="image/jpeg",
    )


def _canvas_background(reference_bgr: np.ndarray, max_width: int = 700) -> tuple[np.ndarray, int, int]:
    """Build a display-sized RGB canvas background."""
    reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
    height, width = reference_rgb.shape[:2]
    display_width = min(max_width, width)
    display_height = max(1, round(height * display_width / width))
    background_rgb_display = cv2.resize(
        reference_rgb,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )
    return background_rgb_display, display_width, display_height


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _inject_ui_styles()
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
    upload_signature = (uploaded_file.name, len(uploaded_bytes))
    if st.session_state.get("upload_signature") != upload_signature:
        st.session_state.upload_signature = upload_signature
        st.session_state.pop("preview_result", None)
        st.session_state.pop("preview_signature", None)
        st.session_state.pop("guided_result", None)
        st.session_state.pop("guided_settings", None)
        st.session_state.pop("canvas_nonce", None)

    suffix = Path(uploaded_file.name).suffix
    video_path = save_uploaded_video_to_temp(uploaded_bytes, suffix)

    try:
        summary = _cached_video_summary(str(video_path))
        video_width = int(summary["width"])
        video_duration = float(summary["duration_seconds"])
        suffix = Path(uploaded_file.name).suffix
        _render_video(uploaded_bytes, suffix=suffix, video_width=video_width)

        preview_result = st.session_state.get("preview_result")
        frame_count_hint = (
            preview_result.frames_processed
            if isinstance(preview_result, PreviewResult)
            else None
        )
        settings, update_preview = _settings_from_sidebar(
            video_duration=video_duration,
            frame_count_hint=frame_count_hint,
        )
        preview_settings = _preview_settings(settings, video_width)
        preview_signature = _preview_signature(
            upload_signature,
            settings,
            video_width,
        )
        preview_is_stale = st.session_state.get("preview_signature") != preview_signature

        st.subheader("Video Metadata")
        _render_summary_table(_metadata_table(summary))

        if update_preview or preview_result is None:
            with st.status("Preparing mask preview", expanded=True) as status:
                st.write("Extracting preview frames and selecting the reference.")
                preview_result = run_preview(video_path, preview_settings)
                st.session_state.preview_result = preview_result
                st.session_state.preview_signature = preview_signature
                st.session_state.canvas_nonce = st.session_state.get("canvas_nonce", 0) + 1
                st.session_state.pop("guided_result", None)
                status.update(label="Preview ready", state="complete")
            preview_is_stale = False
        elif preview_is_stale:
            st.warning("Preview settings changed. Update the preview before processing the image.")

        if preview_result is None:
            st.info("Update the preview to choose the reference frame and paint an alignment guide.")
            return

        preview_result = cast(PreviewResult, preview_result)
        if preview_result.frames_processed > PREVIEW_FRAME_REFRESH_THRESHOLD and preview_is_stale:
            st.warning("This clip uses more than 120 preview frames, so the preview needs to be refreshed before processing.")

        can_run_pipeline = not (
            preview_is_stale and preview_result.frames_processed > PREVIEW_FRAME_REFRESH_THRESHOLD
        )
        canvas_result, background_rgb_display, process_clicked = _render_preview_and_mask(
            preview_result,
            can_process=can_run_pipeline,
        )

        if process_clicked:
            alignment_allowed_mask = None
            mask_coverage = 0.0
            if canvas_result is not None and background_rgb_display is not None:
                _paint_alpha, alignment_allowed_mask, mask_coverage = _extract_guidance_mask(
                    canvas_result,
                    background_rgb_display,
                    preview_result.reference_frame_bgr.shape[:2],
                    _output_size_from_metadata(summary, settings.resize_width),
                )

            with st.status("Processing video", expanded=True) as status:
                st.write("Extracting full-resolution frames.")
                if alignment_allowed_mask is not None:
                    st.write(f"Using painted alignment guide ({mask_coverage:.1%} masked).")
                st.write("Aligning accepted frames and building the full-frame average.")
                result = run_pipeline(video_path, settings, alignment_allowed_mask)
                st.session_state.guided_result = result
                st.session_state.guided_settings = settings
                status.update(label="Processing complete", state="complete")

        result = st.session_state.get("guided_result")
        result_settings = st.session_state.get("guided_settings", settings)
        if result is None:
            st.info("Process the image when the reference and mask are ready.")
            return

        _render_pipeline_result(cast(PipelineResult, result), cast(PipelineSettings, result_settings))
    except ValueError as error:
        st.error(str(error))
    finally:
        video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
