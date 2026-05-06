# Video Long Exposure Lab

Video Long Exposure Lab is a local Streamlit portfolio project for turning a
short video into a long-exposure-style still image.

The app is designed around an honest photographic process:

1. Extract frames from an uploaded video.
2. Choose a reference frame.
3. Align frames to that reference.
4. Reject frames that do not align well.
5. Average accepted aligned frames.
6. Crop unstable borders.
7. Export the final image.

## Honest Photographic Average Mode

The core visual idea is a real average of video frames, not a synthetic blur or
AI-generated image. Each accepted frame contributes light and color to the final
still after being aligned to the chosen reference view.

This approach can create water, sky, traffic, crowd, and handheld motion effects
that resemble long-exposure photography while preserving a clear relationship to
the source video.

## Stacking Modes

- Mean photographic average: best for waterfalls, streams, clouds, and smoothing
  motion.
- Median: useful for cleanup, but it suppresses transient motion.
- Sigma-clipped mean: cleanup-oriented averaging for noisy or inconsistent
  frames.
- Lighten / star trails: preserves the brightest pixel/channel values for star
  trails, light trails, and fireworks.
- Additive / sum: experimental brightening mode that can saturate highlights
  quickly.

## How To Get Best Results

- Use full-resolution video.
- Keep the camera as still as possible.
- Include static rocks, trees, buildings, or landscape for alignment.
- Avoid extreme parallax between foreground and background.
- Avoid heavy wind in foliage when it is needed for alignment.
- Use 2-5 seconds for waterfalls and streams.
- For tripod star trails, disable alignment so the stars can trail naturally.
  If alignment is needed for a night landscape, guide it with static foreground
  rather than stars.

## Limitations

- Video frames are compressed, often noisy, and usually lower quality than still
  photos.
- Moving subjects can become transparent or smeared when averaged.
- Large camera movement, rolling shutter, parallax, and scene changes can break
  alignment.
- Alignment can stabilize the camera view, but it cannot invent detail that was
  not captured in the original frames.
- Low-resolution or blurred source video will still look soft after averaging.
- Additive stacking can clip bright highlights quickly.

## Why Video May Still Be Soft After Alignment

Even good alignment cannot fully undo motion blur, compression artifacts,
rolling shutter distortion, missed focus, low shutter speed, atmospheric haze, or
the lower per-frame resolution of many videos. Averaging can reduce random noise,
but it can also reveal small alignment errors as softness. The goal is a
photographically honest result, not a super-resolution reconstruction.

## Run Locally

Create the conda environment:

```bash
conda create -n long-exposure-lab -c conda-forge python=3.12 streamlit opencv numpy pillow imageio imageio-ffmpeg pandas matplotlib
conda activate long-exposure-lab
```

Alternatively, create the environment with Python 3.12 and install the pinned
application requirements with `pip install -r requirements.txt`.

Start the app:

```bash
streamlit run app.py
```

All processing happens locally in the Streamlit runtime. The app does not use
deep learning, cloud APIs, or remote processing services.

## Portfolio Framing

This project is intentionally scoped as a portfolio lab rather than a production
SaaS app. The interesting work is in making the photographic pipeline visible:
frame extraction, reference choice, alignment quality, rejection decisions,
averaging, border handling, and export.

Good future improvements include sample videos with documented tradeoffs,
presets for common shooting scenarios, and richer export controls.
