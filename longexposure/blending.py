"""Selective motion blending helpers."""

from __future__ import annotations

import cv2
import numpy as np


def extract_paint_mask_from_canvas(
    canvas_result,
    background_rgb_display: np.ndarray | None,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Extract a painted alpha mask from a drawable-canvas result."""
    if canvas_result.image_data is None:
        height, width = output_size
        return np.zeros((height, width), dtype=np.float32)

    canvas_rgba = np.asarray(canvas_result.image_data).astype(np.uint8)
    canvas_rgb = canvas_rgba[:, :, :3]
    canvas_alpha = canvas_rgba[:, :, 3].astype(np.float32) / 255.0

    # With background_image, drawable-canvas returns the drawing layer, not the
    # reference image. The alpha channel is therefore the cleanest source of
    # truth for painted strokes. Diffing against the reference is only a fallback
    # for environments that return fully opaque canvas data.
    if 0.0 < float(canvas_alpha.max()) and float(canvas_alpha.min()) < 1.0:
        alpha = canvas_alpha
    elif background_rgb_display is None:
        alpha = np.max(canvas_rgb, axis=2).astype(np.float32) / 255.0
    else:
        background_rgb = np.asarray(background_rgb_display).astype(np.uint8)
        if background_rgb.shape[:2] != canvas_rgb.shape[:2]:
            background_rgb = cv2.resize(
                background_rgb,
                (canvas_rgb.shape[1], canvas_rgb.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        difference = cv2.absdiff(canvas_rgb, background_rgb)
        alpha = np.max(difference, axis=2).astype(np.float32) / 255.0

    alpha = (alpha > 0.05).astype(np.float32)

    output_height, output_width = output_size
    if alpha.shape != (output_height, output_width):
        alpha = cv2.resize(
            alpha,
            (output_width, output_height),
            interpolation=cv2.INTER_LINEAR,
        )

    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def feather_mask(alpha: np.ndarray, feather_radius_px: int) -> np.ndarray:
    """Feather a float alpha mask with Gaussian blur."""
    clipped = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    if feather_radius_px <= 0:
        return clipped

    kernel_size = max(3, feather_radius_px * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(clipped, (kernel_size, kernel_size), 0)
    max_value = float(blurred.max())
    if max_value > 0:
        blurred = blurred / max_value
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def make_alignment_allowed_mask(
    paint_alpha: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """Return an inverted uint8 mask where unpainted pixels allow ORB features."""
    allowed = paint_alpha <= threshold
    return (allowed.astype(np.uint8) * 255)


def blend_with_alpha(
    reference_bgr: np.ndarray,
    motion_stack_bgr: np.ndarray,
    alpha: np.ndarray,
    blend_strength: float,
) -> np.ndarray:
    """Blend a motion stack into a reference image using a feathered alpha mask."""
    if reference_bgr.shape != motion_stack_bgr.shape:
        raise ValueError("reference_bgr and motion_stack_bgr must have matching shapes")
    if alpha.shape != reference_bgr.shape[:2]:
        raise ValueError("alpha shape must match image height and width")

    scaled_alpha = np.clip(alpha.astype(np.float32) * blend_strength, 0.0, 1.0)
    alpha_3d = scaled_alpha[:, :, None]
    reference = reference_bgr.astype(np.float32)
    motion = motion_stack_bgr.astype(np.float32)
    blended = reference * (1.0 - alpha_3d) + motion * alpha_3d
    return np.clip(blended, 0, 255).astype(np.uint8)
