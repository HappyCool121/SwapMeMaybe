# LiveSwap installation and deployment

This guide deploys the client on an Apple Silicon Mac and the diffusion/interpolation worker on Nosana NVIDIA GPUs. Begin with one GPU and measure it before provisioning a pool.

The included 20-step VACE profile is a quality baseline. **A runnable deployment does not establish that it meets the 5–10 second target.** The benchmark and stream qualification steps below determine that. Synthetic tests perform no clothing replacement.

## 1. Components and requirements

The Mac performs camera capture, pose/segmentation, window assembly, compositing, buffering, microphone synchronization, and final encoding. GPU workers perform VACE and RIFE. Each worker has its own single-replica Nosana deployment so result requests have predictable worker affinity.

You need:

- An Apple Silicon Mac, a camera, microphone, and OBS Studio. This implementation was developed on an M4 with 16 GB RAM.
- Python 3.11, managed by `uv`; the project contains `.python-version` and `uv.lock`.
- FFmpeg and ffprobe. Local TCP streaming is sufficient. SRT additionally requires an FFmpeg build with libsrt.
- Docker Desktop or another Docker daemon for building Linux/amd64 images.
- A container registry namespace you can push to, preferably publishing the image by immutable digest.
- A Nosana account, API key, and funded compute/credit balance usable by deployments.
- A Nosana NVIDIA GPU market with enough VRAM. Start by testing a 24 GB or larger GPU; profile qualification decides actual suitability.
- The RIFE **paper model** weights described below. Multi-gigabyte Wan/T5/VAE downloads happen on the GPU host and can take several minutes.

Startup time is separate from camera-to-OBS delay. The latter applies once models are warm and the session buffer is running.

## 2. Install the local client

Clone the submission repository and open its directory:

```sh
git clone https://github.com/HappyCool121/liveswap-submission.git
cd liveswap-submission
```

If the tools are missing, install `uv` and FFmpeg with your package manager. On macOS with Homebrew:

```sh
brew install uv ffmpeg
```

Install the application and local development/worker dependencies:

```sh
uv sync --extra client --extra worker --extra dev
uv run --all-extras liveswap prepare-models
uv run --all-extras liveswap doctor
```

Use `--all-extras` in the commands in this guide. A plain `uv run` can synchronize away optional client or worker dependencies.

`prepare-models` downloads version 1 of Pose Landmarker Lite and Selfie Multiclass segmentation to `models/mediapipe/`, verifies known SHA-256 values, and records a manifest. It does not download Wan or RIFE.

Run the local tests and a short encoded demo:

```sh
uv run --all-extras pytest -q
uv run --all-extras liveswap demo --duration 6 --out artifacts/demo.mp4
ffprobe -v error -show_streams -show_format artifacts/demo.mp4
```

The real localhost TCP encoder/receiver test is opt-in:

```sh
LIVESWAP_INTEGRATION=1 uv run --all-extras pytest -q
```

The demo waits for its fixed delay before writing the generated test scene. The resulting video contains the requested capture duration. Its AAC track is silent because no microphone is opened. Every frame is labeled as a transport test.

For a longer local transport run:

```sh
uv run --all-extras liveswap demo --duration 300 --out artifacts/five-minute-transport.mp4
```

To exercise missed deadlines and the slate fallback:

```sh
uv run --all-extras liveswap demo --duration 8 --mock-delay 2 --out artifacts/deadline-fallback.mp4
```

To check your actual camera/microphone and MediaPipe locally before renting a GPU, use the explicitly unedited passthrough mode:

```sh
uv run --all-extras liveswap run --passthrough --delay 5 --duration 30 --output artifacts/camera-passthrough.mp4
```

This command intentionally records the original clothing and labels it as a transport test. It is separate from the runtime failure fallback, which never switches to the unprocessed camera.

Reports are written next to the demo videos. Mock throughput numbers are useful only for debugging the local pipeline.

## 3. Create the client signing key

```sh
uv run --all-extras liveswap keygen --out .liveswap/client.pem
```

This creates an Ed25519 private key with owner-only permissions and prints the public key. Keep the private key on the Mac. Only the public verification key goes into the worker's environment and public Nosana job definition.

If a key already exists, print its public half instead of overwriting it:

```sh
uv run --all-extras liveswap public-key --key .liveswap/client.pem
```

Worker requests are signed over method, path, timestamp, nonce, and body hash. The worker rejects stale, modified, and replayed requests. Remote worker URLs require HTTPS. Localhost HTTP is allowed for development.

## 4. Prepare the garment and footage

Use one JPG or PNG image of a single top. Crop out unnecessary background and any model's face. Prefer a front view with visible neckline, color, and fabric. For the first run, use similar fit and sleeve coverage to the clothing already worn.

Place it somewhere convenient, for example:

```text
assets/my-top.png
```

The client validates it locally, flattens transparency to a neutral background, strips metadata, and uploads the prepared image once to each worker. Subsequent windows reference the prepared image's content hash. Restart the session to change it.

Also prepare at least a minute of representative footage for benchmarking: one person, stable framing, moderate torso turns, and some hand crossings. This is an input to the benchmark, not an automatic quality certification.

## 5. Prepare RIFE weights

The GPU image contains pinned RIFE code, but the large pretrained weights are supplied separately.

1. Open the official [ECCV2022-RIFE repository](https://github.com/hzwer/ECCV2022-RIFE).
2. In its **Evaluation** section, download the paper's **RIFE model**, rather than `RIFE_m` or a third-party/newer v4/v5 training package.
3. Extract the checkpoint named `flownet.pkl`.
4. Put it at this exact local path:

```text
models/rife/flownet.pkl
```

The adapter instantiates the repository's `model.RIFE.Model` and strictly loads that model's state dictionary. A checkpoint for a different architecture will fail warmup instead of silently producing a different interpolation implementation.

Record the file checksum:

```sh
shasum -a 256 models/rife/flownet.pkl
```

The worker exposes the loaded checksum in its provenance information. All workers in a session must have identical provenance.

## 6. Understand and select the worker profile

`profiles/vace.json` initially uses:

| Setting | Value |
| --- | --- |
| Model | Wan2.1 VACE 1.3B FP16 |
| Generated crop | 384 × 384 |
| Window / overlap | 17 / 4 source frames |
| Sampling | 20-step quality baseline |
| Generated / returned rate | 12 / 24 fps |
| Reference | Fixed per session |

The included profile is deliberately not presented as a verified four-step configuration. An arbitrary LoRA named “Lightning” is not proof of VACE compatibility.

To use a proven acceleration adapter:

1. Validate that specific adapter against the chosen VACE checkpoint and sampler in ComfyUI first.
2. Copy its `.safetensors` file into `deploy/loras/` before building the image.
3. Copy the baseline profile to a new JSON file.
4. Set `lora` to the adapter's basename, and set its verified `lora_strength`, `sampler`, `scheduler`, `cfg`, and `steps`.
5. Set `acceleration_verified` to `true` only after that compatibility check. Profiles at six steps or fewer are rejected without this explicit setting and an adapter name.
6. Rebuild the image and benchmark the new profile. Changing any profile field changes its ID; previous results no longer qualify it.

The image can also load the official 14B FP16 file with a selected weight dtype, but VRAM and speed must be measured. Do not infer memory residency from the file name or GPU model alone.

Generate all eight resolution/window candidates from an existing profile:

```sh
uv run --all-extras liveswap matrix-profiles --profile profiles/vace.json --out artifacts/profiles
```

These cover 384/512 crops and 17/25/33/81 frame windows. A running worker has an immutable profile; use a separate deployment or restart/reconfigure it for each candidate.

## 7. Build and publish the GPU image

Start Docker Desktop and verify the daemon is reachable:

```sh
docker version
```

Set your registry image name:

```sh
export LIVESWAP_IMAGE='ghcr.io/YOUR_ACCOUNT/liveswap:0.1.0'
```

Build for the architecture used by Nosana NVIDIA hosts, even when building on Apple Silicon:

```sh
docker buildx build --platform linux/amd64 -f deploy/Dockerfile -t "$LIVESWAP_IMAGE" --load .
```

The Docker context excludes the downloader, private keys, local recordings, and unrelated files. RIFE weights and optional adapters are intentionally included.

The image pins these upstream source revisions:

- ComfyUI: `7f7fd918ed0815c5d5ef2ccfe44bfc8e72cfd217`
- RIFE: `5d8adbdd40e12c2c8f91930eff838aebe561c086`
- Wan model bundle: `123acf1cc74bccbb9bfff8ac1ee72edc08c2341d`

Application worker dependencies are installed from the hashed `deploy/requirements-worker.txt`, exported from `uv.lock`. If you intentionally change application dependencies, regenerate it before building:

```sh
uv export --frozen --extra worker --no-dev --no-emit-project --output-file deploy/requirements-worker.txt
```

Wan, T5, and VAE files are mounted as pinned Hugging Face resources by Nosana. The entrypoint also has a pinned download fallback if those resource paths are unavailable. ComfyUI listens only inside the container on `127.0.0.1:8188`; the authenticated adapter is exposed on port 8000.

Publish using your registry's normal login process:

```sh
docker login ghcr.io
docker push "$LIVESWAP_IMAGE"
```

For a real deployment, prefer the resulting digest rather than a mutable tag:

```sh
export LIVESWAP_IMAGE='ghcr.io/YOUR_ACCOUNT/liveswap@sha256:REPLACE_WITH_PUSHED_DIGEST'
```

On a separate Linux NVIDIA machine, the image can be tested with Docker's `--gpus all`. An Apple Silicon Mac can build the image but cannot execute its CUDA inference backend.

## 8. Configure Nosana and inspect the public job

Create a Nosana API key using your account/dashboard. Set it in the terminal that will provision workers and launch managed sessions. For zsh, this prompts without echoing the key:

```sh
read -rs 'NOSANA_API_KEY?Nosana API key: '
export NOSANA_API_KEY
```

If your API/account configuration also needs an explicit owner identifier, set your public account address as `NOSANA_USER_ID`. The application uses bearer authentication against `https://api.nosana.com`.

Use [Nosana Explore](https://explore.nosana.com/) to choose an available GPU market. Copy its actual public-key market address; do not use a GPU marketing name as the address:

```sh
export LIVESWAP_MARKET='REPLACE_WITH_NOSANA_MARKET_ADDRESS'
```

Render the job locally before spending credits:

```sh
uv run --all-extras liveswap nosana job --image "$LIVESWAP_IMAGE" --profile profiles/vace.json --out artifacts/nosana-job.json
```

Inspect that file. It contains public model/resource configuration and the client's public verification key. It must not contain the private signing key or Nosana API key. Live frames travel through the signed HTTPS worker API; they are not embedded in the platform job definition.

## 9. Provision one worker and benchmark it

The following command starts a **paid** GPU deployment:

```sh
uv run --all-extras liveswap nosana deploy --image "$LIVESWAP_IMAGE" --market "$LIVESWAP_MARKET" --count 1 --profile profiles/vace.json --out .liveswap/benchmark-deployment.json
```

Every created deployment ID is saved immediately. If provisioning fails after partial success, the tool attempts to stop created deployments and retains their IDs for recovery.

Read deployment status and its port-8000 HTTPS endpoint:

```sh
uv run --all-extras liveswap nosana status --deployment .liveswap/benchmark-deployment.json
```

Copy the worker URL into a variable:

```sh
export LIVESWAP_WORKER_URL='https://REPLACE_WITH_WORKER_ENDPOINT'
```

Run at least ten warm representative windows:

```sh
uv run --all-extras liveswap benchmark --worker "$LIVESWAP_WORKER_URL" --key .liveswap/client.pem --profile profiles/vace.json --garment assets/my-top.png --video assets/representative.mp4 --samples 10 --report artifacts/benchmark.json
```

The benchmark waits for readiness. Readiness requires an actual diffusion/VAE/RIFE warmup, not just a listening web server. Upload, inference, interpolation, download, and validation are included in turnaround measurements.

Optionally add `--hourly-price YOUR_CURRENT_USD_RATE` to include an estimated worker and pool compute rate. Startup/idle time and actual billing must be checked separately in Nosana.

Inspect:

- `turnaround_p95_seconds`
- `capacity.required_fixed_delay_seconds`
- `capacity.workers_with_25_percent_headroom`
- `capacity.delay_pass`
- `worker_provenance`

Stop the benchmark deployment when it is no longer needed:

```sh
uv run --all-extras liveswap nosana stop --deployment .liveswap/benchmark-deployment.json
```

The timing rule is:

```text
useful frames per window = length − overlap
useful fps per worker = useful frames / measured p95 turnaround
workers with headroom = ceil(15 × measured p95 turnaround / useful frames)
fixed delay = round up to 0.5 s [max(5, (length − 1)/12 + p95 turnaround + 1)]
```

A required delay above ten seconds fails the selected product target. Additional independent workers cannot fix that window latency. Try a shorter window, smaller crop, proven acceleration, or a different backend before scaling.

## 10. Configure OBS

The working default uses TCP/MPEG-TS. In OBS:

1. Add a **Media Source** to the desired scene.
2. Uncheck **Local File**.
3. Set Input to `tcp://127.0.0.1:10000`.
4. Set Input Format to `mpegts` if the dialog exposes it.
5. Set **Network Buffering** to **0 MB** for this localhost connection. OBS otherwise defaults to 2 MB, which can add delay outside the application's buffer.
6. In **FFmpeg Options**, use `fflags=nobuffer analyzeduration=100000 probesize=32768` to limit startup probing and buffering. The values use FFmpeg's units; options are separated by spaces.
7. Set the scene/output to 1280 × 720 and 24 fps for this MVP.
8. Use the audio carried by this Media Source. Mute any separately captured copy of the microphone in the scene to avoid an undelayed duplicate.

The app sends to `tcp://127.0.0.1:10000?listen=1`. OBS is the connecting receiver and must omit `listen=1`. The application waits up to 30 seconds for the receiver before starting its capture clock, then sends a slate and silence while its fixed-delay buffer fills.

If OBS tried to connect before the listener opened, restart playback or toggle the source after the application displays “Waiting for OBS to connect.”

For optional SRT, first confirm `liveswap doctor` reports `srt: true`. Use:

```text
App output: srt://127.0.0.1:10000?mode=listener&latency=20000
OBS input:  srt://127.0.0.1:10000
```

Camera and microphone are synchronized by this program. Gameplay, desktop audio, music, and other OBS sources are outside the MVP's synchronization contract. The streaming platform adds its own downstream delay.

## 11. Run the first five-minute GPU stream

After one-worker measurements, provision the reported pool size using a new metadata file. Replace the example count with the actual measured requirement:

```sh
uv run --all-extras liveswap nosana deploy --image "$LIVESWAP_IMAGE" --market "$LIVESWAP_MARKET" --count 6 --profile profiles/vace.json --out .liveswap/stream-deployment.json
```

The number six is an example, not a measured requirement for this implementation.

Choose a delay of 5–10 seconds that is at least the report's required fixed delay. For a first qualification run, use `--experimental` because appearance has not yet been reviewed:

```sh
uv run --all-extras liveswap run --deployment .liveswap/stream-deployment.json --profile profiles/vace.json --garment assets/my-top.png --benchmark artifacts/benchmark.json --experimental --delay 8 --duration 300 --metrics artifacts/first-gpu-session.json
```

Grant macOS Camera and Microphone permissions to the terminal or app when prompted. List audio devices if needed:

```sh
uv run --all-extras liveswap devices
```

Use `--camera INDEX` and `--microphone INDEX` for nondefault devices. `--no-microphone` intentionally sends silence.

For recorded-video qualification without opening devices:

```sh
uv run --all-extras liveswap run --deployment .liveswap/stream-deployment.json --profile profiles/vace.json --garment assets/my-top.png --video assets/representative.mp4 --benchmark artifacts/benchmark.json --experimental --delay 8 --duration 300 --output artifacts/gpu-review.mp4
```

The file-input mode uses silent audio; it does not extract the video's original soundtrack. A sufficiently long input is required for the requested capture duration.

A managed session starts stopped deployments as needed, monitors workers, and requests pool shutdown in `finally` on success, failure, or normal interruption. It never adds delay to hide a growing queue. Worker connection failures trigger stop/start recovery of the affected single-replica deployment.

`--worker URL` can be repeated instead of using `--deployment`. This mode assumes an externally managed pool; the client cannot stop Nosana deployments whose IDs it does not own. Prefer managed deployment files for normal use.

## 12. Use the desktop controls

Launch:

```sh
uv run --all-extras liveswap gui
```

Select the garment, matching profile, signing key, and benchmark report. For automatic lifecycle management, select the pool JSON file and clear the Worker URLs field. Otherwise enter comma-separated HTTPS worker URLs.

Set camera, microphone, delay, duration, and output URL. Start with 300 seconds. Zero duration means unlimited. Enable Experimental for initial GPU qualification. Start OBS's Media Source and click Start Session.

Stop Session also works while waiting for workers or OBS. Closing the window waits for session cleanup rather than leaving an active capture thread behind.

The preview is the processed delayed output. Status shows effective generation rate, missed output, and estimated hourly compute rate when supplied in the benchmark.

For a local desktop test, start a mock worker in another terminal:

```sh
export LIVESWAP_PUBLIC_KEY="$(uv run --all-extras liveswap public-key --key .liveswap/client.pem)"
uv run --all-extras liveswap worker --profile profiles/mock.json --public-key "$LIVESWAP_PUBLIC_KEY" --port 8000 --data-dir .liveswap/local-worker
```

In the desktop client, choose `profiles/mock.json`, enter `http://127.0.0.1:8000`, select a valid reference image, enable Synthetic transport test, and choose a short duration with `artifacts/desktop-test.mp4` as the output. The scene is generated locally and labeled as unedited test content. Stop the local worker with Ctrl-C when finished.

## 13. Review appearance and qualify full streams

Review 20 representative ten-second clips at normal playback speed. Include hand crossings, moderate turns, lighting changes, mask boundaries, and window seams. At least 18 must be acceptable. Verify that faces/hands are retained and no conspicuous garment jump or seam flash appears.

Record the results only after actually performing that review:

```sh
uv run --all-extras liveswap record-visual-review artifacts/benchmark.json --accepted-clips 18 --reviewed-clips 20
```

This command records an operator assessment; it does not inspect or rate video automatically.

For production mode, omit `--experimental`. It requires at least ten benchmark samples, a passing recorded visual review, sufficient measured pool capacity, a matching profile/provenance, and a suitable fixed delay.

Run the two-hour qualification:

```sh
uv run --all-extras liveswap run --deployment .liveswap/stream-deployment.json --profile profiles/vace.json --garment assets/my-top.png --benchmark artifacts/benchmark.json --delay 8 --duration 7200 --metrics artifacts/two-hour-session.json
```

Check these release criteria:

- Stable 720p/24 fps delivery and no increasing backlog.
- Actual camera-to-OBS delay remains within 5–10 seconds. Use a visible clock or a recorded timing reference; internal playback scheduling alone does not measure OBS's decoder latency.
- Camera/microphone sync stays within 50 ms. Record a clap at the beginning, middle, and end and compare visible contact to the waveform.
- Nominal edited-frame availability is at least 99%; inspect `edited_frames / played_frames` for genuine GPU runs.
- Source buffer stays bounded; observe memory during the entire run.
- Killing/disconnecting a worker produces a brief freeze then slate, with delayed audio continuing. Recovery resumes at current scheduled timestamps.
- Leaving/returning and unreliable tracking invoke the specified fallback.
- Managed worker shutdown is reflected in Nosana status and billing.

Only genuine GPU/device runs can establish those criteria. Once qualified, `--duration 0` removes the application's time limit.

If a device introduces a consistent audio/video offset, use `--microphone-delay-ms` or the desktop Microphone sync adjustment. A positive value delays audio; if the voice leads the image by 70 ms, try `--microphone-delay-ms 70` and repeat the clap check. A negative value advances audio. This adjustment does not change the session's fixed video delay.

## 14. Stop and recover workers

After any test, especially a killed terminal or machine shutdown:

```sh
uv run --all-extras liveswap nosana stop --deployment .liveswap/stream-deployment.json
uv run --all-extras liveswap nosana status --deployment .liveswap/stream-deployment.json
```

Confirm every deployment is stopped in Nosana. A worker has a 120-second authenticated-contact lease and its supervisor exits when the lease expires, but do not use that as a substitute for checking platform state and billing after abnormal shutdown.

Keep the pool metadata files until teardown has been confirmed. They contain the IDs needed to stop partially provisioned pools. Model loading and endpoint startup time can still be billable.

Temporary window data/results expire and are removed on acknowledgement or worker shutdown. The client writes recordings only when a file output path is selected.

## 15. Troubleshooting

| Symptom | Action |
| --- | --- |
| Docker cannot connect to its daemon | Start Docker Desktop, then rerun `docker version` |
| Docker build cannot find `flownet.pkl` | Complete the paper-model download and place it at `models/rife/flownet.pkl` |
| RIFE state-dictionary mismatch | Use the Evaluation section's original RIFE paper checkpoint for `model.RIFE.Model`, not RIFE_m/newer training variants |
| Worker remains unready | Inspect private worker logs for missing models, CUDA/VRAM problems, node imports, or an incompatible checkpoint |
| HTTP 401 from a worker | Match the private key to the deployed public key; check host/client wall clocks |
| Profile/provenance mismatch | Use the exact deployed JSON and one immutable image; rerun benchmarks after changes |
| HTTP 409 on job submission | One window is already active; the scheduler handles normal single-flight operation |
| No edited output | Check tracking, garment mask, worker errors, and deadline feasibility; output intentionally becomes a slate |
| “Output is not consuming frames” | Connect/restart OBS's Media Source, check port 10000, or try file output |
| VideoToolbox error | Add `--codec libx264`; repeat the local output/latency check |
| SRT protocol not found | Use the default TCP transport or install an FFmpeg build with libsrt |
| Mask misses clothing | Improve framing/lighting and use a similar-fit top; inspect camera positioning before changing the mask model |
| Delay/capacity gate fails | Reduce window/resolution, validate acceleration, or change backend; do not simply add GPUs to fix window latency |
| Echo or early voice in OBS | Mute the separate direct microphone source and use the Media Source's delayed audio |
| Startup consumes time before broadcast | Expected cold download/model warmup; start ahead of the stream |

## 16. Reference documentation

- [Nosana API and authentication](https://learn.nosana.com/api/intro.html)
- [Nosana live API schema](https://api.nosana.com/openapi.json)
- [Nosana deployment strategies](https://learn.nosana.com/deployments/strategies.html)
- [Nosana resource caching](https://learn.nosana.com/deployments/jobs/job-definition/cached-resources.html)
- [ComfyUI VACE examples](https://docs.comfy.org/tutorials/video/wan/vace)
- [ComfyUI server API](https://docs.comfy.org/development/comfyui-server/comms_routes)
- [MediaPipe image segmentation](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter)
- [RIFE implementation and weight links](https://github.com/hzwer/ECCV2022-RIFE)
- [OBS MPEG-TS/SRT protocol guide](https://obsproject.com/kb/srt-protocol-streaming-guide)
- [OBS Media Source buffering/options implementation](https://github.com/obsproject/obs-studio/blob/master/plugins/obs-ffmpeg/obs-ffmpeg-source.c)

## Implementation verification status

Local API/contract tests, MediaPipe model loading, encoded synthetic playback, deadline fallback, and desktop rendering can be verified on this Mac. The validation record in `PLAN.md` describes the runs actually completed. GPU image execution, real Nosana inference/cost, physical camera/microphone sync, OBS decoder latency, and the two-hour live soak require the deployment environment and remain separate qualification steps.
