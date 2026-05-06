# Codex Prompt 12 — Write final README as portfolio case study

Rewrite README.md as a polished portfolio case study.

Sections:
1. Project title:
   Video Long Exposure Lab

2. One-sentence summary:
   A local-first Streamlit app that converts short video clips into long-exposure still images using transparent frame alignment and photographic averaging.

3. Motivation:
   iOS Photos app has a simple setting to turn Live photos into a long exposure, but blurs handheld shots because it does not always perform robust motion compensation. Paid apps allowed long exposure generation at the time of capture. Traditional photography requires a tripod and expensive filters for daytime use. This project explores a reproducible alternative.

4. What it does:
   - Upload .mov/.mp4
   - Extract frames
   - Select reference frame
   - Align frames using ORB + RANSAC affine transform
   - Reject poorly aligned frames
   - Mean-average accepted frames
   - Crop unstable borders
   - Export final image
   - Show diagnostics

5. Honest photographic mode:
   Explain that the default output is full-frame aligned averaging, not selective water replacement.

6. Optional computational modes:
   - sigma-clipped mean
   - median stack
   - lighten for night photography
   - alignment exclusion mask
   Make clear these are optional.

7. Why results may be soft:
   - source video compression
   - rolling shutter
   - autofocus/exposure shifts
   - hand motion inside individual frames
   - parallax
   - wind/moving foliage

8. Technical stack:
   - Streamlit
   - OpenCV
   - NumPy
   - Matplotlib
   - Pandas

9. How to run:
   pip install -r requirements.txt
   streamlit run app.py

10. Hosted demo caveats:
   - file size limits
   - processing time
   - short clips recommended
   - no uploaded files intentionally persisted

11. Portfolio framing:
   Connect this to reproducible computer vision pipelines:
   - deterministic processing
   - transparent diagnostics
   - constrained modes
   - explainable artifacts