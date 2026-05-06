# Codex Prompt 8 — Promote mask to main guided-alignment pipeline

I tested the current workflow and found:
- strict stack accepted only 21/84 frames and preserves static detail but under-smooths water (strobe patterns)
- relaxed motion stack looks visually strong
- selective blend appears similar to the reference image (but is indeed changed) despite a broad alpha mask, suggesting a blend bug. Bug will be eliminated by refactor
- product direction should change: the painted mask should become part of the main alignment pipeline, not only a post-processing add-on

Goal:
Refactor the app around a simpler primary workflow:

1. Present video selection screen
2. Load low res preview and alpha mask interface
    - load video file, extract frames, and proceed all the way to reference frame using basic settings:
      - maximum frames: 300
      - start time seconds: 0.00
      - resize width: 800
      - reference strategy
3. Present frame sharpness plot and reference frame in alpha mask canvas interface (skip first frame view)
    - Resize width setting will automatically update to 100% for the pipeline run. 
    - (optional) User can set processing settings. Draw an 'update preview' button. Required if frame count exceeds 120.
    - (optional) User can paint in the mask using the low res version. Mask should re-res automatically. User can use the update preview button to force a canvas at full resolution.
    - (optional) User can adjust alignment and stacking settings
      - Set minimum matches to 50% of frame count
    - Run pipeline if frame requirement met.
4. Run pipeline
   - if mask exists, use inverse mask as ORB allowed feature region
   - final averaging remains full-frame
5. Produce primary output:
   - slider for # averaged frames. Order frames by inlier_ratio (after alignment) to determine how to add/remove frames per user request.
   - present download buttons
   - present update processing button to enable the user to review alignment/stacking/mask settings.

Important product principle:
The painted mask is an alignment guide. It tells the app where NOT to estimate camera motion. The full-frame stack remains a photographic average of accepted aligned frames.
