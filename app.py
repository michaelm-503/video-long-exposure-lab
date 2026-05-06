"""Streamlit app for experimenting with video-based long exposure images."""

from __future__ import annotations

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
    blend_with_alpha,
    extract_paint_mask_from_canvas,
    feather_mask,
    make_alignment_allowed_mask,
)
from longexposure.diagnostics import sharpness_figure
from longexposure.frames import ReferenceStrategy
from longexposure.io import get_video_summary, save_uploaded_video_to_temp
from longexposure.pipeline import (
    PipelineResult,
    PipelineSettings,
    relaxed_stack_settings,
    run_pipeline,
    run_stack_job,
)
from longexposure.stacking import StackingMode


APP_TITLE = "Video Long Exposure Lab"
OUTPUT_DIR = Path("outputs")
STACKING_MODE_LABELS: dict[str, StackingMode] = {
    "Mean": "mean",
    "Sigma-clipped mean (cleanup)": "sigma_clipped_mean",
    "Median (experimental)": "median",
}


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
        st.image(first_frame_rgb, caption="First extracted frame")

    st.subheader("Reference Frame")
    st.metric("Selected reference index", result.reference_index)
    if result.reference_frame_bgr is not None:
        reference_frame_rgb = cv2.cvtColor(
            result.reference_frame_bgr,
            cv2.COLOR_BGR2RGB,
        )
        st.image(reference_frame_rgb, caption="Reference frame")
    st.pyplot(
        sharpness_figure(result.sharpness_scores, result.reference_index),
        use_container_width=False,
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

    st.subheader("Strict Photographic Full-Frame Average")
    st.image(
        final_rgb,
        caption="Strict photographic full-frame average",
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


def _run_selective_stack(
    strict_result: PipelineResult,
    strict_settings: PipelineSettings,
    canvas_result,
    background_rgb_display: np.ndarray,
    strictness: float,
    feather_radius_px: int,
    blend_strength: float,
) -> dict[str, object]:
    """Run relaxed masked stacking and return display/download payloads."""
    if strict_result.frames_bgr is None or strict_result.reference_frame_bgr is None:
        raise ValueError("Strict pipeline result does not include frames for selective mode")

    output_size = strict_result.reference_frame_bgr.shape[:2]
    paint_alpha = extract_paint_mask_from_canvas(
        canvas_result,
        background_rgb_display,
        output_size,
    )
    allowed_mask = make_alignment_allowed_mask(paint_alpha)
    paint_coverage = float(np.mean(paint_alpha > 0.05))
    allowed_coverage = float(np.mean(allowed_mask > 0))
    relaxed_settings = relaxed_stack_settings(strictness, strict_settings)
    selective_stack = run_stack_job(
        strict_result.frames_bgr,
        strict_result.reference_index,
        relaxed_settings,
        alignment_allowed_mask=allowed_mask,
    )

    reference_bgr = strict_result.reference_frame_bgr
    motion_stack_bgr = selective_stack.cropped_stack_bgr
    alpha = paint_alpha
    if selective_stack.crop_rect is not None:
        x_min, y_min, x_max, y_max = selective_stack.crop_rect
        reference_bgr = reference_bgr[y_min:y_max, x_min:x_max]
        alpha = alpha[y_min:y_max, x_min:x_max]

    feathered_alpha = feather_mask(alpha, feather_radius_px)
    blended_bgr = blend_with_alpha(
        reference_bgr,
        motion_stack_bgr,
        feathered_alpha,
        blend_strength,
    )

    return {
        "reference_bgr": reference_bgr,
        "motion_stack_bgr": motion_stack_bgr,
        "alpha": feathered_alpha,
        "blend_bgr": blended_bgr,
        "accepted_count": selective_stack.accepted_count,
        "diagnostics": selective_stack.diagnostics,
        "paint_coverage": paint_coverage,
        "allowed_coverage": allowed_coverage,
    }


def _render_selective_workflow(
    strict_result: PipelineResult,
    strict_settings: PipelineSettings,
) -> None:
    """Render the optional selective motion blend workflow."""
    st.subheader("Selective Motion Blend")
    st.write(
        "Paint over the regions where you want the long-exposure motion effect. "
        "The app will use the unpainted regions to estimate camera alignment, "
        "then blend the relaxed stack only into the painted region."
    )

    if st_canvas is None:
        st.error(
            "Install streamlit-drawable-canvas, then restart Streamlit to use mask painting."
        )
        return
    if strict_result.reference_frame_bgr is None:
        st.error("No reference frame is available for selective blending.")
        return

    if "canvas_nonce" not in st.session_state:
        st.session_state.canvas_nonce = 0

    background_rgb_display, canvas_width, canvas_height = _canvas_background(
        strict_result.reference_frame_bgr,
    )
    brush_size = st.slider("Brush size", min_value=4, max_value=80, value=24, step=2)
    if st.button("Reset mask"):
        st.session_state.canvas_nonce += 1
        st.session_state.pop("selective_payload", None)

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=brush_size,
        stroke_color="#FFFFFF",
        background_image=Image.fromarray(background_rgb_display),
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="freedraw",
        key=f"motion_mask_canvas_{st.session_state.canvas_nonce}",
    )

    strictness = st.slider(
        "Frame acceptance strictness",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
    )
    feather_radius_px = st.slider(
        "Feather radius",
        min_value=0,
        max_value=80,
        value=18,
        step=2,
    )
    blend_strength = st.slider(
        "Blend strength",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
    )

    if st.button("Generate / update selective stack", type="primary"):
        selective_payload = _run_selective_stack(
            strict_result,
            strict_settings,
            canvas_result,
            background_rgb_display,
            strictness,
            feather_radius_px,
            blend_strength,
        )
        st.session_state.selective_payload = selective_payload

    selective_payload = st.session_state.get("selective_payload")
    if not selective_payload:
        return

    reference_rgb = cv2.cvtColor(
        selective_payload["reference_bgr"],
        cv2.COLOR_BGR2RGB,
    )
    motion_rgb = cv2.cvtColor(
        selective_payload["motion_stack_bgr"],
        cv2.COLOR_BGR2RGB,
    )
    alpha_display = (selective_payload["alpha"] * 255).astype(np.uint8)
    blend_bgr = selective_payload["blend_bgr"]
    blend_rgb = cv2.cvtColor(blend_bgr, cv2.COLOR_BGR2RGB)

    st.image(reference_rgb, caption="Reference image")
    st.image(motion_rgb, caption="Relaxed motion stack")
    st.image(alpha_display, caption="Alpha mask")
    st.image(blend_rgb, caption="Selective motion blend")

    strict_ratio = (
        selective_payload["accepted_count"] / strict_result.accepted_count
        if strict_result.accepted_count > 0
        else 0.0
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Strict accepted frames", strict_result.accepted_count)
    metric_columns[1].metric(
        "Selective accepted frames",
        selective_payload["accepted_count"],
    )
    metric_columns[2].metric("Selective / strict", f"{strict_ratio:.2f}x")

    mask_columns = st.columns(2)
    mask_columns[0].metric(
        "Painted mask coverage",
        f"{selective_payload['paint_coverage']:.1%}",
    )
    mask_columns[1].metric(
        "ORB allowed area",
        f"{selective_payload['allowed_coverage']:.1%}",
    )

    st.dataframe(
        pd.DataFrame(selective_payload["diagnostics"]),
        hide_index=True,
    )

    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download selective PNG",
        data=_encode_image(blend_bgr, ".png"),
        file_name="selective_motion_blend.png",
        mime="image/png",
    )
    download_columns[1].download_button(
        "Download selective JPEG",
        data=_encode_image(blend_bgr, ".jpg"),
        file_name="selective_motion_blend.jpg",
        mime="image/jpeg",
    )
    download_columns[2].download_button(
        "Download alpha mask",
        data=_encode_gray_png(alpha_display),
        file_name="selective_motion_alpha.png",
        mime="image/png",
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
    upload_signature = (uploaded_file.name, len(uploaded_bytes))
    if st.session_state.get("upload_signature") != upload_signature:
        st.session_state.upload_signature = upload_signature
        st.session_state.pop("strict_result", None)
        st.session_state.pop("strict_settings", None)
        st.session_state.pop("selective_payload", None)

    suffix = Path(uploaded_file.name).suffix
    video_path = save_uploaded_video_to_temp(uploaded_bytes, suffix)

    try:
        summary = _cached_video_summary(str(video_path))
        video_width = int(summary["width"])
        video_duration = float(summary["duration_seconds"])
        st.video(uploaded_bytes)

        settings = _settings_from_sidebar(
            video_width=video_width,
            video_duration=video_duration,
        )

        st.subheader("Video Metadata")
        _render_summary_table(_metadata_table(summary))

        if not st.button("Run pipeline", type="primary"):
            result = st.session_state.get("strict_result")
        else:
            with st.status("Processing video", expanded=True) as status:
                st.write("Extracting frames and scoring sharpness.")
                st.write("Selecting a reference frame and aligning accepted frames.")
                st.write("Stacking frames and cropping unstable borders.")
                result = run_pipeline(video_path, settings)
                st.session_state.strict_result = result
                st.session_state.strict_settings = settings
                st.session_state.pop("selective_payload", None)
                status.update(label="Processing complete", state="complete")

        if result is None:
            st.info("Adjust settings, then run the pipeline to produce the final image.")
            return

        _render_pipeline_result(result)
        enable_selective = st.toggle("Enable selective motion blend", value=False)
        if enable_selective:
            strict_settings = st.session_state.get("strict_settings", settings)
            _render_selective_workflow(result, strict_settings)
    except ValueError as error:
        st.error(str(error))
    finally:
        video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
