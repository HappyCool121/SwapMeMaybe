# Plan: Nosana livestream clothing replacement

## Goal

Implement the agreed single-person, similar-fit upper-body replacement MVP and detailed root-level `instructions.md` deployment documentation.

## Context

The workspace contains an unrelated downloader, which must remain untouched. The approved specification targets Apple Silicon, camera plus microphone, 720p/24 fps output, 12 fps garment generation, a fixed 5–10 second delay, a five-minute initial run, and two-hour qualification. One garment is fixed per session. VACE is the first backend; performance is not yet verified.

## Research Summary

- ComfyUI native VACE accepts frame batches, masks, reference images, and lengths of 4n+1. Masks do not imply sparse inference.
- MediaPipe multiclass segmentation distinguishes clothes and protected skin/hair; timestamp matching is required.
- Nosana deployments run persistent GPU containers. Live window requests belong to the worker API rather than separate platform jobs.
- RIFE interpolation runs on the GPU worker. OBS accepts local SRT/MPEG-TS with video and audio together.
- Relevant sources: https://docs.comfy.org/tutorials/video/wan/vace, https://learn.nosana.com/, https://github.com/hzwer/ECCV2022-RIFE, https://obsproject.com/kb/srt-protocol-streaming-guide.
- Context7 is unavailable. Official source code and documentation are used instead.

## Constraints

- Preserve existing files and behavior in `downloader/`.
- Keep changes additive and reversible; no real cloud spending or webcam access is needed for implementation verification.
- Explicitly allow Python dependencies for MediaPipe, OpenCV, PySide6, sounddevice, HTTPX, FastAPI, NumPy, Pillow, cryptography, pytest, and Ruff.
- Keep reference/model versions and worker profiles explicit. Never claim synthetic passthrough tests demonstrate garment replacement or GPU performance.
- Fixed playback clock, bounded buffers, protected original pixels, and brief freeze followed by a slate on missing output.
- User explicitly authorized implementation after planning; the plan-writing phase is limited to this file, followed by the requested implementation phase.

## Out of Scope

Full-body editing, multiple users/people, live garment changes, major silhouette changes, public account/billing services, and synchronizing unrelated OBS sources.

## Reversibility

New application, tests, deployment assets, and documentation are separate from the existing downloader. Cloud commands are explicit CLI operations; development tests use local synthetic workers.

---

## Step A: Core contracts, timing, and authenticated transport

### Status

`done`

### Objective

Establish deterministic frame/window contracts and bounded timeline behavior.

### Tasks

- [x] Create package/configuration, archive validation, signed HTTP transport, timing, interpolation timestamps, and masks/compositing primitives.
- [x] Cover overlap, out-of-order completion, deadlines, signature replay, and malformed archives.

### Relevant Files

- `src/liveswap/`, `tests/`, `pyproject.toml`

### Expected Changes

- create: package, examples, tests, dependency manifest

### Do Not Modify

- `downloader/`

### Commands

```bash
uv run --extra dev pytest
```

### Acceptance Criteria

- [x] Useful-frame accounting includes overlap.
- [x] Invalid or stale requests/results cannot enter playback.

### Validation Results

- Contract, archive, signature, session, timing, and compositing tests passed in the local suite.

### Findings / Notes

- Remote worker clocks do not determine playback; source presentation timestamps and client deadlines do.

---

## Step B: Worker, ComfyUI graph, and interpolation

### Status

`review`

### Objective

Provide a real GPU backend and a distinctly labeled development backend.

### Tasks

- [x] Implement single-flight worker API, readiness/warmup, asset cache, result expiry, and session leases.
- [x] Supply native VACE graph generation and frame input/output custom nodes.
- [x] Integrate persistent RIFE and validate exact output frame counts at the transport boundary.
- [x] Supply worker image and Nosana job/deployment tools without publishing private signing keys.

### Relevant Files

- `src/liveswap/worker/`, `deploy/`, `comfy_nodes/`

### Expected Changes

- create: worker, graph/nodes, Docker assets

### Do Not Modify

- `downloader/`

### Commands

```bash
uv run --extra worker --extra dev pytest tests
```

### Acceptance Criteria

- [x] Worker rejects concurrent inference and unauthenticated media requests.
- [x] Test backend cannot be mistaken for a qualified diffusion backend.

### Validation Results

- Worker roundtrip, replay, cancellation, session ownership, and data-directory protection tests passed.
- Public job and deployment-create payloads validated against https://api.nosana.com/openapi.json without cloud mutation; evidence in `artifacts/nosana-schema-validation.json`.
- Confirmed the Docker base image tag exists for amd64 and pinned upstream ComfyUI requirements/source.
- NVIDIA inference and Docker image runtime: not run. Docker daemon is offline, no NVIDIA GPU is available locally, and RIFE/GPU weights and a funded Nosana session remain deployment prerequisites.

### Findings / Notes

- ComfyUI and RIFE repositories must be pinned; user-provided weights are required for GPU qualification.

---

## Step C: Client, audio/video output, and benchmarking

### Status

`review`

### Objective

Run the complete timestamped stream and expose useful diagnostics.

### Tasks

- [x] Implement capture, multiclass masking, window scheduling, timeline stitching, original-frame compositing, and fixed-delay audio playback.
- [x] Add desktop controls, CLI operation, local TCP/SRT/file output, and deterministic offline demo.
- [x] Add benchmark reports for p95 turnaround, useful fps, pool capacity, and delay feasibility.

### Relevant Files

- `src/liveswap/client/`, `src/liveswap/benchmark.py`

### Expected Changes

- create: client and benchmark tools

### Do Not Modify

- `downloader/`

### Commands

```bash
uv run --extra dev pytest
uv run liveswap --help
```

### Acceptance Criteria

- [x] Local smoke test produces a playable video/audio artifact through the actual client and worker transport.
- [x] Missing output produces a brief freeze then slate without growing delay.

### Validation Results

- Five-minute synthetic HTTP/client/encoder run: 7,200 captured/played frames, 7,199 available processed frames, one fallback/tracking miss, 277 completed windows, no failed/dropped windows, peak source buffer 120 frames, max playback scheduling lateness 0.106 s. Evidence: `artifacts/five-minute-transport.json` and `.mp4`. This is not GPU clothing inference.
- Slow synthetic worker run: 192 played frames, 57 missing frames handled by fallback, three dropped windows, max playback scheduling lateness 0.0234 s. Evidence: `artifacts/deadline-fallback.json` and `.mp4`.
- Final short synthetic run: 96/96 available frames with no failed or dropped windows; `artifacts/final-smoke.json` and `.mp4`.
- ffprobe confirmed H.264 1280x720 at 24 fps and AAC mono 48 kHz with matching four-second durations in the short file test.
- Actual localhost TCP FFmpeg sender/receiver test passed, including start-time alignment within 50 ms.
- MediaPipe versioned/checksummed models loaded and executed on synthetic input; no person was expected or detected.
- Desktop UI rendered successfully offscreen; preview/status mailboxes retain only their newest item.
- Live camera/microphone, OBS decoder latency, and real GPU appearance/performance: not run; manual deployment qualification.

### Findings / Notes

- The five-minute real GPU run and two-hour soak remain deployment acceptance checks, not synthetic performance claims.

---

## Step D: Final verification and cleanup

### Status

`done`

### Objective

Deliver tested code and reproducible deployment instructions.

### Tasks

- [x] Run the relevant test suite, lint checks, offline integration run, and media inspection.
- [x] Review added files for unintended changes and remove temporary debugging code.
- [x] Write `instructions.md` covering local setup, weights, image build/push, Nosana provisioning, OBS, benchmarking, troubleshooting, and teardown.
- [x] Record risks, remaining qualification work, and actual validation results.

### Relevant Files

- `instructions.md`, `README.md`, `tests/`, `PLAN.md`

### Expected Changes

- create: instructions and readme
- modify: this plan's validation results

### Do Not Modify

- `downloader/`

### Commands

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

### Acceptance Criteria

- [x] Required local tests pass and documentation commands match the implementation.
- [x] Hardware-dependent checks are explicitly identified as unrun.

### Validation Results

- Relevant local suite, including the opt-in TCP sender/receiver integration test: passed. Final count is recorded in `artifacts/validation.json`.
- Ruff check: passed for application, custom nodes, deployment supervisor, and tests.
- Python wheel build: passed (`dist/liveswap-0.1.0-py3-none-any.whl`).
- Reviewed new-file inventory; `downloader/` was not edited. No real Nosana deployment, webcam capture, or microphone capture was performed.

### Findings / Notes

- This workspace is not a Git repository; inspect the added-file inventory instead of relying on git diff.

## Follow-up Work

- Measure genuine warm Nosana inference and visual quality; select a passing profile before production qualification.
- Execute the five-minute real stream and two-hour soak with physical camera, microphone, and OBS.

## Decision Log

| Date | Decision | Rationale | Impact |
| --- | --- | --- | --- |
| 2026-10-11 | Implement the approved MVP in the current directory | Explicit user request | Add application and deployment instructions |
| 2026-10-11 | Use signed worker requests with public verification keys | Public deployment definitions must not expose worker access secrets | Private signing keys remain on the client |
| 2026-10-11 | Keep GPU and hardware qualification separate from local verification | No Nosana GPU session or physical-device test has been run | Report measured evidence honestly |
| 2026-10-11 | Use localhost TCP/MPEG-TS by default; retain optional SRT | The installed FFmpeg lacks libsrt; OBS accepts TCP/MPEG-TS | Working local transport without replacing system FFmpeg |
