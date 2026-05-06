# Codex Prompt 10 — Add quality safeguards and user-facing warnings

Improve robustness and make the app portfolio-polished.

Files to modify:
- app.py
- longexposure/pipeline.py
- longexposure/diagnostics.py
- README.md

Functional requirements:
1. Add warnings when:
   - fewer than 10 frames accepted
   - video resolution is very low
   - estimated transforms imply very large frame movement
   - output may be soft due to source video blur
2. Add a small "How to get best results" section:
   - use full-resolution video
   - keep camera as still as possible
   - include static rocks/trees/buildings for alignment
   - avoid extreme parallax
   - avoid heavy wind in foliage
   - use 2-5 seconds for waterfalls
3. Add diagnostic visuals:
   - accepted/rejected bar chart
   - inlier ratio plot over frame index
   - sharpness plot
4. Add select above final output for:
   - reference frame vs final output side by side
   - slider viewer

Acceptance criteria:
- The app is understandable to a reviewer.
- Failures explain themselves.
- Diagnostics reinforce the portfolio story.