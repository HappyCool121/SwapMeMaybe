# SwapMeMaybe

A macOS client and persistent Nosana GPU worker for delayed upper-body clothing replacement. The client preserves the original camera outside the garment mask and sends synchronized video and microphone audio to OBS.

The Python package and command-line tool use the name `liveswap`.

See the [submission draft](submission.md) for the project overview, build summary, attribution, and judge try-it instructions. Recorded local validation results are in [artifacts/validation.json](artifacts/validation.json).

The real backend builds a native Wan2.1 VACE ComfyUI graph and performs RIFE interpolation on the worker. A separately labeled **mock backend performs no clothing replacement** and exists to exercise transport, timing, and output locally.

## Quick local check

```sh
uv sync --extra client --extra worker --extra dev
uv run --all-extras liveswap prepare-models
uv run --all-extras liveswap doctor
uv run --all-extras liveswap demo --duration 6 --out artifacts/demo.mp4
uv run --all-extras pytest
```

The demo uses generated test content, a real local signed HTTP worker, the window scheduler, stitching, fixed-delay playback, and FFmpeg. It writes video and an adjacent JSON report. It does not access the camera or microphone.

## Deployment

See [instructions.md](instructions.md) for client installation, GPU weights, Docker, Nosana provisioning, OBS setup, benchmarking, qualification, teardown, and troubleshooting.

```sh
uv run --all-extras liveswap keygen
uv run --all-extras liveswap gui
```

The default OBS transport is localhost TCP/MPEG-TS because the available FFmpeg build does not include SRT. SRT is supported when FFmpeg has libsrt. Both carry video and audio together.

## Targets and qualification

- One mostly stationary person; one similar-fit top and reference image per session.
- 12 fps generated clothing, interpolated to 24 fps; 720p output.
- Fixed 5–10 second processing delay; five-minute initial test, then two-hour qualification.
- Brief edited-frame freeze, then a slate if output is missing. Delayed audio continues.
- Start and stop workers per session. A managed deployment file enables automatic shutdown and recovery.

`profiles/vace.json` is a **20-step quality baseline**, not a claimed real-time profile. Low-step profiles require a verified compatible acceleration adapter. No genuine GPU speed, hourly price, or clothing quality is certified by the local tests. Production mode requires a matching benchmark and an operator-recorded visual review; `--experimental` is for initial GPU qualification.

## Layout

| Location | Purpose |
| --- | --- |
| `src/liveswap/client/` | Capture, MediaPipe masks, audio, scheduling, output, desktop controls |
| `src/liveswap/worker/` | Signed single-flight API, native VACE graph, warmup, RIFE |
| `comfy_nodes/liveswap_nodes/` | Explicit frame/mask input and temporary frame output nodes |
| `deploy/` | GPU image and process supervisor |
| `profiles/` | Immutable processing configurations |
| `tests/` | Contract, transport, lifecycle, overlap, masking, and deadline checks |
| `PLAN.md` | Agreed stages and validation record |

Existing `downloader/` files are unrelated and are not used or modified by SwapMeMaybe.
