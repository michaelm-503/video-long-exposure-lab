# Codex Prompt 9 — Add star-trail stacking modes

Add additional stacking modes for star trails.

Current stacking modes include mean / median / sigma-clipped mean. Add:

1. `lighten_stack`
   - User-facing label: "Lighten / star trails"
   - For each pixel/channel, keep the maximum value across accepted aligned frames.
   - This should preserve bright moving stars as trails while keeping dark sky mostly dark.
   - Implement with `np.maximum.reduce` or iterative `np.maximum`.

2. `additive_stack`
   - User-facing label: "Additive / sum"
   - Sum frames in float space.
   - Add user control `additive_gain`, default 1.0.
   - Clip final output to 0..255.
   - Warn that additive stacking can saturate highlights quickly.

3. Update stack mode selector:
   - Mean photographic average
   - Median
   - Sigma-clipped mean
   - Lighten / star trails
   - Additive / sum

4. Add mode guidance in UI:
   - Mean: best for waterfalls, streams, clouds, smoothing motion
   - Lighten: best for star trails, light trails, fireworks
   - Additive: experimental; can brighten/saturate quickly

5. Alignment guidance:
   - For star trails, users usually want the stars to move, not be aligned away.
   - If using alignment, align to foreground/static landscape, not stars.
   - Add a checkbox or help text: "For tripod star trails, disable alignment or use very light foreground alignment."

Acceptance criteria:
- User can select "Lighten / star trails".
- Star/light trails accumulate instead of averaging to gray.
- Existing mean stack behavior is unchanged.