# Codex Prompt 01 — Create project scaffold

Build a new Python project called `video-long-exposure-lab`.

Goal:
Create a Streamlit web app that converts an uploaded video into a long-exposure-style still image using an honest photographic process:
- extract frames
- choose a reference frame
- align frames to the reference
- reject bad frames
- average accepted aligned frames
- crop unstable borders
- export image

This is a portfolio project, not a production SaaS app.

Create the following structure:

video-long-exposure-lab/
  README.md
  requirements.txt
  .gitignore
  app.py
  longexposure/
    __init__.py
    io.py
    frames.py
    alignment.py
    stacking.py
    diagnostics.py
    pipeline.py
  sample_data/
    README.md
  outputs/
    .gitkeep
  .streamlit/
    config.toml

Requirements:
- Python 3.12 compatible.
- Use Streamlit for UI.
- Use OpenCV and NumPy for processing.
- Keep code modular and readable.
- Add docstrings and type hints.
- Do not use deep learning.
- Do not use cloud APIs.
- All processing should happen locally in the Streamlit runtime.

requirements.txt should include:
streamlit
opencv-python-headless
numpy
pillow
imageio
imageio-ffmpeg
pandas
matplotlib

Create a new conda env with these packages: `long-exposure-lab`

.streamlit/config.toml:
Set a reasonable upload limit for demo use, e.g. 300 MB.
Use headless-safe settings.

README.md should explain:
- What the app does
- The honest photographic average mode
- Limitations
- How to run locally
- Why video may still be soft after alignment
- Portfolio framing

Do not implement the full algorithm yet. Create clean stubs with TODOs where needed.