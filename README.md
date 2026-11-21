# Bunny Tracker

Bunny Tracker is a self-hosted rabbit monitor. A Linux camera agent detects
motion and saves clips. The owner can watch the live camera, label behaviors
such as grooming, feeding, or rearing, and choose clips to share on the
[public demo](https://bunny-tracker.app).

The default install runs in motion-only mode. Live behavior predictions and
email alerts require a promoted seven-class model. Camera streaming, clip
collection, labeling, and public highlights work without one.

---

## Project structure

| Location | Contents |
|---|---|
| [agent/](agent/) | Capture agent (C++20), CMake builds, tests, and vendored dependencies |
| [client/](client/) | React dashboard and Vite build |
| [server/](server/) | FastAPI backend, inference checks, training, and tests |
| [data/](data/) | Runtime stores, recordings, and shared `.env` |
| [deploy/pi/](deploy/pi/) | Raspberry Pi services, installation, and data sync |
| [docs/](docs/) | Development, labeling, training, and research guides |
| [run.sh](run.sh) | Local launcher for the server and agent |

## Guides

| Task | Guide |
|---|---|
| Find code or run checks | [Development](docs/development.md) |
| Build or run the agent | [Capture agent](agent/README.md) |
| Collect clips | [Clipping Mode](docs/clipping.md) |
| Label behavior | [Ethogram](docs/ethogram.md) |
| Train and promote a model | [Training pipeline](docs/pipeline.md) |
| Install on a Raspberry Pi | [Deployment](deploy/pi/README.md) |
| Read the research context | [References](docs/references.md) |
| See planned work | [Stretch goals](docs/stretch-goals.md) |

## How it works

- The C++ agent uses one `ffmpeg` process to read the camera, produce a
  private MJPEG stream, and record rolling video segments. By default, normal
  monitoring samples motion every 1.5 seconds and records 12-second segments.
  Clipping Mode samples every 0.5 seconds and saves 8-second clips for labeling.
- With a promoted model, ONNX Runtime runs a MobileNetV4 backbone and a GRU
  over successive frames. The agent posts behavior changes to the public
  text feed. When motion and the score for a non-resting behavior stay above
  their thresholds for enough frames, the server can email the clip. A
  per-behavior cooldown and a minimum gap between emails limit repeat alerts.

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | React + Vite |
| Backend | Python (FastAPI + uvicorn) |
| Capture/ML | C++20, ffmpeg, ONNX Runtime, pixel-diff motion detection |
| Training | PyTorch + `timm`, offline Python commands |
| Alerts | `smtplib` (Gmail SMTP) |
| Process management | systemd on the Raspberry Pi, `run.sh` for development |
| Deployment | Cloudflare named tunnel (HTTPS, custom domain) |

---

## Setup

### 1. Install dependencies

Run these commands from the repository root with Node.js, npm, and uv installed.
The client build produces the dashboard that FastAPI serves. `--group dev`
adds pytest and Ruff to the Python 3.14 runtime.

```bash
(cd client && npm ci && npm run build)
(cd server && uv sync --frozen --python 3.14 --group dev)
```

To train a model or run the Python test suite, install the separate Python
3.12 environment with CPU PyTorch. Both environments use `server/uv.lock`.

```bash
(cd server && UV_PROJECT_ENVIRONMENT=.venv-train uv sync --frozen --python 3.12 --group train)
```

For camera capture, install `ffmpeg` and the Linux build dependencies in the
[agent guide](agent/README.md#build-and-test). Build the C++ agent from the
repository root. You can skip this step for a dashboard-only setup.

```bash
(cd agent && scripts/fetch-onnxruntime.sh && \
  cmake --preset default && cmake --build --preset default -j2)
```

### 2. Configure the environment

Create `data/.env` in the repository root. The server and agent both read it.
Choose a dashboard password and generate a separate agent token with
`python3 -c "import secrets; print(secrets.token_hex(32))"`.

```ini
# Required. Avoid embedded double quotes and backslashes.
ADMIN_PASSWORD="your-password"
AGENT_TOKEN="paste-your-generated-token"

# Camera. v4l2 for a real camera, or lavfi with
# CAMERA_INPUT=testsrc2=size=640x480:rate=15 for a synthetic source.
CAMERA_FORMAT=v4l2
CAMERA_INPUT=/dev/v4l/by-id/usb-Your_Camera-video-index0

# Everything below is optional. Commented values are the defaults.

# Behavior alerts
# AGENT_INTEREST_THRESHOLD=60        # minimum score for a non-resting behavior (0-100)
# AGENT_MOTION_FLOOR=1               # minimum motion level for an alert
# AGENT_ALERT_STREAK=3               # frames over threshold before an alert

# Clipping Mode, read by the agent at startup
# AGENT_CLIPPING_FRAME_SECONDS=0.5   # motion sampling cadence while harvesting
# AGENT_CLIPPING_SEGMENT_SECONDS=8   # length of a harvested clip
# AGENT_CLIPPING_STREAK=1            # frames over threshold before a clip fires
# AGENT_CLIPPING_RUN_MODEL=0         # 1 keeps inference on (diagnostic only)

# Initial storage limits. The dashboard saves overrides in data/monitor.json.
# CLIPPING_MAX_UNLABELED=500         # auto-stop at this many unlabeled clips
# CLIPPING_MIN_FREE_GB=10            # auto-stop below this much free disk

# Gmail alerts. Email is off until all three of these are set.
# EMAIL_USER=you@gmail.com
# EMAIL_PASS="your Gmail app password"
# NOTIFY_TO=you@example.com
# NOTIFY_COOLDOWN_MINUTES=10
```

Find your camera with `ls -l /dev/v4l/by-id/` and use the `video-index0`
entry.

### 3. Run

From the repository root, run `./run.sh` to start the server and agent, or
`./run.sh server` for the dashboard alone. Open http://localhost:3001, select
Owner login, and enter `ADMIN_PASSWORD`. Ctrl+C stops the launcher.

---

## Using it

1. Open Camera to watch the private live view. The header controls
   monitoring, email alerts, and Clipping Mode.
2. Open Label Studio to review captured or imported clips. Keys `1` to
   `7` assign a behavior and `0` marks a clip as `out_of_view`. The training
   pipeline uses human-confirmed behavior labels. Predictions on alert clips
   appear as suggestions for review. See the [labeling guide](docs/ethogram.md).
3. Open Highlights and select Show on an eligible labeled clip to
   publish it on the demo. Resting and out-of-view clips stay private.
4. Train offline in the `.venv-train` environment. The pipeline extracts
   frames, runs cross-validation, fits a head, and exports a candidate for
   evaluation.
   Promotion installs the ONNX files and writes `server/models/promoted.json`.
   Restart the agent to load the model. The Training tab displays its metadata.
   See the [training commands](docs/pipeline.md#the-order).

### Clipping Mode

Clipping Mode collects unlabeled footage whenever motion crosses a threshold.
It works without a trained model. See the [Clipping Mode guide](docs/clipping.md)
for sensitivity, cooldown, dry runs, and storage limits.

---

## Development

The [development guide](docs/development.md) maps the source directories and
covers Python tests, C++ tests, manual camera checks, and generated files.

Run the Python checks from `server/` after building the client and installing
both Python environments. Run them in a Git checkout because the pose-feature
test reads the current commit ID.

```bash
.venv/bin/python -m pytest -m "not local_data"
.venv/bin/ruff check .
```

For the C++ build and tests, see [agent/README.md](agent/README.md#build-and-test).

---

## Deployment

### Raspberry Pi

[deploy/pi/](deploy/pi/) runs Bunny Tracker on a Raspberry Pi 5 with three
systemd services. The C++ camera agent (`bunnycam-agent`) sends frames and
clips to FastAPI (`bunnycam-server`). The server listens on loopback behind a
Cloudflare HTTPS tunnel (`bunnycam-tunnel`). Owner login uses a session
cookie. The agent authenticates separately with `AGENT_TOKEN`.

The agent writes rolling segments to tmpfs under `/run/bunnycam/segments`.
FastAPI holds the latest live frame in memory. Saved clips and labels stay
on disk. The Pi neither builds the client nor trains. Build `client/dist` on
another machine and copy it over, and use `deploy/pi/pull-data.sh` to copy
labeled clips to the training machine. The [Pi guide](deploy/pi/README.md)
covers installation, updates, and transferring a promoted model.

---

## Roadmap

Planned additions include a load cell for water intake and a BME280 sensor
for room temperature and humidity. See the [roadmap](docs/stretch-goals.md).

---

## Notes

- Git ignores recordings (`data/recordings/`), current labels
  (`data/labels.json`), backups, and `data/.env`.
- The server serializes label updates and writes them atomically. It keeps
  a `.bak` copy and daily backups.
