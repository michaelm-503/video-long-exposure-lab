# Codex Prompt 05 — Implement average stacking and border cropping

Implement photographic averaging and unstable-border cropping.

Files to modify:
- longexposure/stacking.py
- longexposure/pipeline.py
- app.py

Functional requirements:
1. Implement `mean_stack(frames_bgr: list[np.ndarray]) -> np.ndarray`.
   - Convert frames to float32 or float64.
   - Average all accepted aligned frames.
   - Clip to 0-255.
   - Return uint8 BGR.
2. Implement optional `median_stack(frames_bgr)` as an experimental mode, clearly labeled as less photographically honest.
3. Implement optional `sigma_clipped_mean_stack(frames_bgr, sigma=2.5)` as robust mode.
   - Clearly label this as computational cleanup.
4. Implement border crop:
   - For each accepted aligned frame, also warp a white mask through the same transform.
   - Accumulate valid masks.
   - Find region where all or most accepted frames are valid.
   - Crop final image to remove unstable black/reflect borders.
5. Add Streamlit controls:
   - stacking mode: "mean", "sigma_clipped_mean", "median"
   - crop borders: checkbox default true
   - valid border threshold, default 0.98
6. Show final output image.
7. Add download button for JPEG and PNG.
8. Save output to local `outputs/` when running locally, but do not require persistence for hosted demo.

Acceptance criteria:
- User can produce a final long-exposure still.
- Mean stacking is default.
- Final image can be downloaded.
- Cropping removes warped borders.