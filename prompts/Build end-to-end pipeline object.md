# Codex Prompt 06 — Build end-to-end pipeline object

Refactor the app so processing is orchestrated through a single pipeline function.

Files to modify:
- longexposure/pipeline.py
- app.py

Functional requirements:
1. Create a dataclass `PipelineSettings` with:
   - start_time_s
   - duration_s
   - max_frames
   - frame_stride
   - resize_width
   - reference_strategy
   - orb_max_features
   - orb_keep_matches
   - min_matches
   - min_inlier_ratio
   - ransac_reproj_threshold
   - stack_mode
   - sigma
   - crop_borders
   - valid_border_threshold
2. Create a dataclass `PipelineResult` with:
   - video_metadata
   - frames_processed
   - reference_index
   - sharpness_scores
   - alignment_diagnostics dataframe-compatible records
   - accepted_count
   - rejected_count
   - output_image_bgr
   - optional cropped_output_image_bgr
   - warnings list
3. Implement `run_pipeline(video_path: str, settings: PipelineSettings) -> PipelineResult`.
4. Update `app.py` to:
   - collect settings from sidebar
   - run pipeline from a button
   - show progress with `st.progress` or `st.status`
   - show warnings
   - show diagnostics
   - show final output
5. Add `st.cache_data` only where safe.
   - Do not cache huge video frame arrays blindly.
   - Avoid caching uploaded raw bytes.
6. Keep the app responsive and readable.

Acceptance criteria:
- Main app code is thin.
- Pipeline code is testable outside Streamlit.
- Processing behavior is captured by settings and result objects.