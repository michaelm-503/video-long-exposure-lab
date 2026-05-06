# Codex Prompt 7 — Add toggleable selective motion blend workflow

We have an MVP Streamlit app that produces a strict photographic full-frame long-exposure stack from video:
- frame extraction
- reference selection
- ORB alignment
- frame rejection
- mean stacking
- crop/export

Current issue:
Strict inlier ratio preserves static fidelity but rejects many frames, reducing waterfall smoothing. We want an optional add-on workflow where the user paints a motion-effect mask, then reruns alignment/stacking with relaxed acceptance and blends the relaxed stack into the reference frame only inside the painted region.

Do not remove or replace the strict photographic full-frame output.

Desired UX:
1. Show the strict photographic full-frame stack first.
2. Provide a download button for the strict result.
3. Add a persistent toggle:
   "Enable selective motion blend"
4. When toggle is off:
   - hide mask painting UI
   - show strict output
   - keep strict download visible
5. When toggle is on:
   - show mask painting UI
   - user paints the moving water/mist region
   - user clicks "Generate / update selective stack"
   - rerun alignment and stacking using relaxed settings
   - use inverse painted mask as the ORB alignment feature mask
   - use painted mask as the final blend alpha mask
   - show selective blend result
   - provide download button for selective result
6. The toggle should remain visible so the user can revert to the original strict output and hide the mask UI.

Important simplification for MVP:
Use one painted mask for two purposes:
- painted region = where long-exposure effect is blended into the final image
- inverse painted region = where ORB is allowed to find alignment features

We understand there is a third category, such as wind-moving vegetation, that is neither desired motion effect nor good alignment area. Do not implement a second mask yet. Keep the MVP simple.

Implementation requirements:

A. Add dependency
- Add `streamlit-drawable-canvas` to requirements.txt.
- If compatibility requires it, pin a working Streamlit version.

B. Add new module
Create:
`longexposure/blending.py`

Functions:
1. `extract_paint_mask_from_canvas(canvas_result, background_rgb_display, output_size) -> np.ndarray`
   - Return float alpha mask in range 0..1.
   - Mask shape should match output image height/width.
   - Use canvas image difference vs background if needed.
   - Painted white strokes should become alpha = 1.

2. `feather_mask(alpha: np.ndarray, feather_radius_px: int) -> np.ndarray`
   - Gaussian blur.
   - Normalize/clip to 0..1.

3. `make_alignment_allowed_mask(paint_alpha: np.ndarray, threshold: float = 0.05) -> np.ndarray`
   - Return uint8 mask where 255 = allowed ORB features and 0 = excluded.
   - Invert the painted region.
   - Shape must match processing frame size.

4. `blend_with_alpha(reference_bgr, motion_stack_bgr, alpha, blend_strength: float) -> np.ndarray`
   - alpha should be feathered and scaled by blend_strength.
   - final = reference * (1 - alpha) + motion_stack * alpha
   - Return uint8 BGR.

C. Refactor pipeline minimally
Add or expose a function:
`run_stack_job(frames, reference_index, stack_settings, alignment_allowed_mask=None)`

This should:
- reuse already extracted frames
- reuse already selected reference frame
- rerun ORB alignment/rejection/stacking
- accept an optional ORB feature mask
- return stack image, cropped reference, cropped stack, crop rect, accepted count, diagnostics

Do not reload video for selective stack updates.

D. Add selective stack settings
Create a relaxed preset derived from a user-facing slider:

UI label:
"Frame acceptance strictness"

Slider range:
0.0 to 1.0
- 0.0 = most relaxed, more smoothing, higher artifact risk
- 1.0 = more conservative, less smoothing, lower artifact risk
Default: 0.35

Map internally:
- min_inlier_ratio = 0.12 + strictness * 0.38
- min_matches = round(8 + strictness * 22)
- ransac_reproj_threshold = 6.0 - strictness * 3.0
- orb_max_features = 2500
- orb_keep_matches = 350

E. Streamlit UI details
After strict output section:

Add:
`enable_selective = st.toggle("Enable selective motion blend", value=False)`

If enabled:
1. Display explanation:
   "Paint over the regions where you want the long-exposure motion effect. The app will use the unpainted regions to estimate camera alignment, then blend the relaxed stack only into the painted region."

2. Canvas:
   - Use the cropped reference image or the same reference image geometry used for strict output.
   - Drawing mode: freedraw.
   - Brush color: white.
   - Brush size slider.
   - Display width capped, e.g. 700 px.
   - Add reset mask button.

3. Controls:
   - Frame acceptance strictness slider.
   - Feather radius slider.
   - Blend strength slider.
   - Generate/update selective stack button.

4. On button click:
   - convert painted canvas to alpha mask
   - create inverse allowed alignment mask
   - rerun stack job with relaxed settings and alignment mask
   - store selective stack result in `st.session_state`

5. After selective stack exists:
   - feather alpha mask
   - blend reference and selective stack
   - display:
     a. reference image
     b. relaxed motion stack
     c. alpha mask
     d. final selective blend
   - show diagnostics:
     - strict accepted frames
     - selective accepted frames
     - strict vs selective accepted ratio
   - download buttons:
     - selective blend JPEG
     - selective blend PNG
     - alpha mask PNG

F. Geometry requirements
- The reference image, motion stack, and alpha mask must be the same shape before blending.
- If the selective stack job crops borders, apply the same crop to:
  - reference
  - motion stack
  - alpha mask
- If this is too complicated for MVP, disable cropping for the selective stack or use the same crop rect as the strict output. But do not silently blend mismatched geometries.

G. Labels
Use clear mode labels:
- "Strict photographic full-frame average"
- "Selective motion blend"
- "Relaxed motion stack"

H. Acceptance criteria
- Existing strict output still works and can be downloaded.
- Toggle hides/shows the selective workflow.
- User can paint a mask.
- User can rerun relaxed alignment/stacking using the painted mask.
- Relaxed stack accepts more frames when strictness is reduced.
- Final selective blend keeps unpainted areas from the reference and paints smoother water from the relaxed stack.
- User can toggle off to return to the strict output.