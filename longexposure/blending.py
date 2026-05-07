"""Selective motion blending helpers."""

from __future__ import annotations

import cv2
import numpy as np


def _empty_mask(output_size: tuple[int, int]) -> np.ndarray:
    """Return a zero alpha mask for a height, width output size."""
    height, width = output_size
    return np.zeros((height, width), dtype=np.float32)


def _canvas_has_drawn_objects(canvas_result) -> bool | None:
    """Return whether drawable-canvas JSON contains user-drawn objects."""
    json_data = getattr(canvas_result, "json_data", None)
    if not isinstance(json_data, dict):
        return None

    objects = json_data.get("objects")
    if objects is None:
        return None

    return bool(objects)


def _path_point(command: list[object]) -> tuple[int, int] | None:
    """Return the endpoint from a Fabric.js path command."""
    if len(command) < 3:
        return None

    try:
        return int(round(float(command[-2]))), int(round(float(command[-1])))
    except (TypeError, ValueError):
        return None


def _object_mask_from_canvas_json(canvas_result, output_size: tuple[int, int]) -> np.ndarray | None:
    """Build a conservative mask from drawable-canvas object metadata."""
    json_data = getattr(canvas_result, "json_data", None)
    if not isinstance(json_data, dict):
        return None

    objects = json_data.get("objects")
    if not isinstance(objects, list) or not objects:
        return None

    height, width = output_size
    mask = np.zeros((height, width), dtype=np.uint8)
    drew_anything = False
    for item in objects:
        if not isinstance(item, dict):
            continue

        drew_object = False
        stroke_width = max(1, int(round(float(item.get("strokeWidth", 8) or 8))))
        path = item.get("path")
        if isinstance(path, list):
            previous_point = None
            for command in path:
                if not isinstance(command, list):
                    continue
                point = _path_point(command)
                if point is None:
                    continue
                if previous_point is not None:
                    cv2.line(mask, previous_point, point, 255, stroke_width)
                    drew_anything = True
                    drew_object = True
                previous_point = point

        if drew_object:
            continue

        try:
            left = float(item.get("left", 0.0) or 0.0)
            top = float(item.get("top", 0.0) or 0.0)
            item_width = float(item.get("width", 0.0) or 0.0)
            item_height = float(item.get("height", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue

        padding = max(2, stroke_width)
        x_min = max(0, int(round(left - padding)))
        y_min = max(0, int(round(top - padding)))
        x_max = min(width, int(round(left + item_width + padding)))
        y_max = min(height, int(round(top + item_height + padding)))
        if x_max > x_min and y_max > y_min:
            mask[y_min:y_max, x_min:x_max] = 255
            drew_anything = True

    if not drew_anything:
        return None

    return (mask.astype(np.float32) / 255.0).astype(np.float32)


def extract_paint_mask_from_canvas(
    canvas_result,
    background_rgb_display: np.ndarray | None,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Extract a painted alpha mask from a drawable-canvas result."""
    has_drawn_objects = _canvas_has_drawn_objects(canvas_result)
    if has_drawn_objects is False:
        return _empty_mask(output_size)

    object_mask = _object_mask_from_canvas_json(canvas_result, output_size)
    if canvas_result.image_data is None:
        return object_mask if object_mask is not None else _empty_mask(output_size)

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

    coverage = float(np.mean(alpha > 0.05))
    object_coverage = float(np.mean(object_mask > 0.05)) if object_mask is not None else 0.0
    if coverage > 0.95:
        if 0.0 < object_coverage < 0.95:
            return object_mask
        return _empty_mask(output_size)

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
