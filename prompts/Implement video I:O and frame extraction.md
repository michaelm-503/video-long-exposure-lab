# Codex Prompt 02 — Implement video I/O and frame extraction

Implement video loading and frame extraction.

Files to modify:
- longexposure/io.py
- longexposure/frames.py
- app.py

Functional requirements:
1. In Streamlit, allow upload of `.mov`, `.mp4`, and `.m4v`.
2. Save the uploaded file to a temporary file because OpenCV `VideoCapture` works best from a file path.
3. Extract basic metadata:
   - fps
   - frame count
   - duration seconds
   - width
   - height
4. Add user controls:
   - max frames to process, default 90
   - frame stride, default 1
   - resize width for processing, default 95% of original width
   - start time seconds, default 0
   - duration seconds to process, default 3
5. Extract frames as BGR NumPy arrays using OpenCV.
6. Downscale frames for processing while preserving aspect ratio.
7. Return both:
   - extracted frames
   - metadata dictionary
8. Show a video preview using `st.video`.
9. Show extracted metadata in the UI.
10. Show the first extracted frame as a preview.

Implementation notes:
- Use `cv2.VideoCapture`.
- Be defensive: handle unreadable videos, zero frames, missing fps, and failed frame reads.
- Cap processed frames to avoid slow hosted demos.
- Keep original video file temporary; do not persist uploads permanently.
- Use `tempfile.NamedTemporaryFile(delete=False, suffix=...)`.
- Clean up temp files where reasonable.

Acceptance criteria:
- User can upload a video.
- App displays metadata.
- App extracts frames from selected time window.
- App displays first extracted frame.
- No alignment or stacking yet.