# SwapMeMaybe — Submission Draft

## Project overview

<!-- Maximum 100 words. Current draft: 68 words. -->

SwapMeMaybe is a prototype for changing a creator's on-camera top from one reference image. Changing outfits interrupts a stream; editing afterward loses the live experience. My approach combines garment masking, GPU video editing, and delayed audio/video output designed for OBS. The interesting experiment is a deliberate, fixed delay: process short overlapping video windows while keeping the playback clock steady and preserving the original image outside the clothing mask.

## The build

<!-- Maximum 100 words. Current draft: 81 words. -->

Built in Python with MediaPipe, PySide6, FastAPI, ComfyUI/Wan2.1 VACE, RIFE, FFmpeg, and Nosana deployment tooling. Local validation covers the desktop UI, signed transport, scheduling, compositing, and encoded audio/video: 46 tests passed, and a five-minute synthetic run played 7,200 frames. I experimented with overlapping windows, garment masks, fixed-delay playback, and missed-deadline fallbacks. The GPU backend is implemented but untested on hardware. Next, I would deploy a Nosana worker, assess clothing quality and throughput, then test a real camera, microphone, and OBS stream.

## Project links & demo

### Live demo / deployed project

Local desktop prototype; use the setup instructions below. No public deployment yet.

### Demo video

[TODO: Add a publicly accessible walkthrough, maximum two minutes. Clearly label synthetic transport footage.]

### GitHub / source code

[LiveSwap source code on GitHub](https://github.com/HappyCool121/liveswap-submission)

### Figma / prototype (optional)

Not provided; optional.

**“This project is not based on code” checkbox:** Leave unchecked.

### Other supporting materials or try-it instructions

Prerequisites: macOS, `uv`, and FFmpeg. Download the source, open a terminal in its root directory, and run:

```sh
uv sync --extra client --extra worker --extra dev
uv run --all-extras SwapMeMaybe demo --duration 6 --out artifacts/demo.mp4
```

Open `artifacts/demo.mp4`. This generates a six-second synthetic transport demo with silent audio. It exercises local processing and output without accessing a camera, microphone, or rented GPU; it does not demonstrate clothing replacement.

Full setup: [instructions.md](instructions.md). Recorded validation results: [artifacts/validation.json](artifacts/validation.json).

## Team & collaboration

Solo project by [TODO: Your name]. I developed SwapMeMaybe with AI coding assistance from Codex.

## AI & tool attribution

Codex provided AI coding assistance and helped draft this submission. SwapMeMaybe uses MediaPipe for pose tracking and garment segmentation. The implemented GPU path uses Wan2.1 VACE through ComfyUI for video editing and RIFE for interpolation. Supporting tools include Python, uv, PySide6, FastAPI, FFmpeg, and Nosana deployment tooling.

[TODO: Confirm any additional AI tools, collaborators, or third-party assets requiring credit.]

## Before final submission

- Replace the name and video placeholders, verify the source-code link, and confirm the attribution details.
- Verify judge access to every supplied link and keep the demo video at two minutes or less.
- Keep the project overview and build responses below 100 words each after revisions.
- Preserve the distinction between verified local transport and pending GPU validation; update claims only when new evidence is available.
