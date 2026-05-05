# Codex Prompt 03 — Implement sharpness scoring and reference frame selection

Implement frame sharpness scoring and reference frame selection.

Files to modify:
- longexposure/frames.py
- longexposure/diagnostics.py
- app.py

Functional requirements:
1. Add a function `laplacian_sharpness(frame_bgr: np.ndarray) -> float`.
   - Convert to grayscale.
   - Use variance of Laplacian as a simple focus/sharpness metric.
2. Add `score_frames_sharpness(frames: list[np.ndarray]) -> list[float]`.
3. Add `select_reference_frame(frames, strategy)`.
   Supported strategies:
   - "sharpest": frame with highest Laplacian variance
   - "middle": middle extracted frame
   - "first": first extracted frame
4. Add Streamlit control for reference strategy.
   Default: "sharpest".
5. Show:
   - selected reference index
   - selected reference frame
   - sharpness plot as a simple matplotlib figure
6. Put plotting helper in `diagnostics.py`.

Acceptance criteria:
- After upload/extraction, app scores all frames.
- User can select reference strategy.
- App displays reference frame and sharpness plot.