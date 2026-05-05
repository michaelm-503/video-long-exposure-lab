# Codex Prompt 04 — Implement ORB-based global alignment

Implement CPU-friendly frame alignment using OpenCV ORB feature matching.

Files to modify:
- longexposure/alignment.py
- longexposure/diagnostics.py
- app.py

Context:
OpenCV ORB is a fast feature detector/descriptor suitable for local CPU processing. OpenCV documentation describes ORB as combining FAST keypoint detection with BRIEF-style descriptors and multiscale/orientation handling. We want a lightweight alignment approach suitable for short videos.

Functional requirements:
1. Implement `estimate_transform_orb(reference_bgr, moving_bgr, max_features=1500, keep_matches=200, ransac_reproj_threshold=3.0)`.
2. Convert frames to grayscale.
3. Detect ORB features and descriptors.
4. Match descriptors with `cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)`.
5. Sort matches by distance and keep top `keep_matches`.
6. Estimate a transform using `cv2.estimateAffinePartial2D`.
   - Use RANSAC.
   - This gives similarity/limited affine behavior: translation, rotation, scale, limited shear.
7. Return:
   - affine matrix, shape 2x3
   - number of matches
   - number of inliers
   - inlier ratio
   - status string
8. Implement `warp_frame(frame_bgr, matrix, output_size)`.
9. Implement `align_frames(frames, reference_index, settings)`.
   - Reference frame should use identity transform.
   - If alignment fails, mark the frame rejected.
   - If inlier ratio is below threshold, mark rejected.
   - If matches are below threshold, mark rejected.
10. Add Streamlit controls:
   - max ORB features
   - keep top matches
   - minimum matches
   - minimum inlier ratio
   - RANSAC reprojection threshold
11. Show diagnostics table:
   - frame index
   - accepted/rejected
   - matches
   - inliers
   - inlier ratio
   - status
12. Show accepted frame count.

Important:
- Do not use optical flow for this MVP.
- Do not use final image masking.
- This app should demonstrate an honest aligned full-frame average.

Acceptance criteria:
- Frames are aligned to selected reference.
- Poorly aligned frames are rejected.
- Diagnostics table appears in Streamlit.