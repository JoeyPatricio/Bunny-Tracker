# Bunny Tracker

A self-hosted machine-learning pet monitor. A camera watches your rabbits, a
classifier recognizes what they're doing, zoomies, grooming, rearing, feeding,
or resting, and you get an email with a clip when something noteworthy happens.

Live demo: **https://bunny-tracker.app** (public demo view; the live stream is
private to the owner).

---

## Project structure

| Location | Contents |
|---|---|
| [agent/](agent/) | C++20 capture agent, CMake builds, tests, and vendored dependencies |
| [client/](client/) | React dashboard and Vite build |
| [server/](server/) | FastAPI backend, Python reference agent, inference, training, and tests |
| [data/](data/) | Runtime stores, recordings, and shared `.env` |
| [deploy/pi/](deploy/pi/) | Raspberry Pi services, installation, and data sync |
| [docs/](docs/) | Development, labeling, training, and research guides |
| [run.sh](run.sh) | Local launcher for the server and Python agent |

The C++ agent in `agent/` is being prepared for cutover. `run.sh` and the base
Pi service still launch the Python reference agent in `server/agent/`.

## Guides

| Task | Guide |
|---|---|
| Find code or run checks | [Development](docs/development.md) |
| Build or run the C++ agent | [Capture agent](agent/README.md) |
| Collect clips | [Clipping Mode](docs/clipping.md) |
| Label behavior | [Ethogram](docs/ethogram.md) |
| Train and promote a model | [Training pipeline](docs/pipeline.md) |
| Install on a Raspberry Pi | [Deployment](deploy/pi/README.md) |
| Read the research context | [References](docs/references.md) |
| See planned work | [Stretch goals](docs/stretch-goals.md) |

## How it works

```
[camera] ──► capture agent (ffmpeg) ──► ONNX backbone embedding ──► temporal head ──► behavior
                  │                                                            │
                  ├──► rolling 12s clips                                       ├──► email alert + clip
                  └──► MJPEG live stream ──► dashboard                         └──► public text feed
```

- **Capture agent**: the Python reference (`server/agent/capture.py`) is a service
  that reads the camera with one `ffmpeg` process, sampling a frame every few
  seconds for inference, recording rolling clips, and pushing a live view to
  the dashboard. Runs the same on a PC (`dshow`) or a Raspberry Pi (`v4l2`),
  only two `.env` lines change. The [C++ port](agent/README.md) uses the same
  configuration and models on Linux.
- **Classifier**: a frozen `timm` edge backbone (`mobilenetv4_conv_small`) plus
  a small learned temporal head, exported to ONNX and run via ONNX Runtime.
  Retraining is now an offline CLI script (`server/training/train.py`), not
  a browser tab; see the note below.
- **Alerts**: a non-normal behavior emails you the clip (Python `smtplib` +
  Gmail), with motion gating, a multi-frame debounce, and a global cooldown to
  suppress false positives.
- **Web app**: Camera (live feed + predictions), Label Studio (label clips),
  and Training (a read-only panel showing the deployed model's metadata). A
  separate public demo page shows highlights and a live behavior feed without
  exposing the camera. Labeling a clip does not publish it. Select Show in
  Highlights to make it public. After upgrading, previously auto-shown clips
  also require Show before they appear publicly.

**Model status:** there is currently no usable classifier. The behavior
vocabulary moved to ethogram v2 (7 classes, see
[docs/ethogram.md](docs/ethogram.md)) and the deployed head still predicts the
5 v1 classes, so the agent refuses to load it and runs motion-only until a
retrain lands. Clipping Mode is unaffected, which is the point: it never uses
the model, so footage can still be harvested while the dataset is rebuilt.

The two accuracy numbers this README used to quote (61.5% for the deployed
model, 74.4% for the one it replaced) were validation-set numbers, and that
same split also drove early stopping, checkpoint selection, and the window
sweep. They were selection scores, not held-out estimates, and should not be
cited as accuracy.

## Tech stack

| Layer       | Tech                                                             |
|-------------|--------------------------------------------------------------------|
| Frontend    | React + Vite                                                       |
| Backend     | Python (FastAPI + uvicorn)                                          |
| Capture/ML  | ffmpeg (`imageio-ffmpeg`), ONNX Runtime, pixel-diff motion detection |
| Training    | PyTorch + `timm`, offline CLI script                                |
| Alerts      | `smtplib` (Gmail SMTP)                                              |
| Process mgmt| systemd on the Raspberry Pi (server, agent, tunnel), `run.sh` for dev |
| Deployment  | Cloudflare named tunnel (HTTPS, custom domain)                      |

---

## Setup

### 1. Install dependencies

The runtime environment stays torch-free for Raspberry Pi deployment. The
optional training environment uses CPU PyTorch. Both resolve from
`server/uv.lock`. See [docs/pipeline.md](docs/pipeline.md).

```bash
cd client && npm install && npm run build
cd ../server

# Runtime (server + agent), plus the test runner and linter. No torch.
uv sync --frozen --python 3.14 --group dev

# Only if you're retraining the model. CPU-only, also used by CI.
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --frozen --python 3.12 --group train
```

Production runtime installs can omit `--group dev`. Update dependencies in
`pyproject.toml` and regenerate with `uv lock`. On Windows, set
`UV_PROJECT_ENVIRONMENT` using your shell's environment syntax, and use
`.venv\Scripts\python.exe` rather than `.venv/bin/python`.

### 2. Configure the environment

Create `data/.env` (shared by both the server and the agent):

```ini
# Email alerts (Gmail App Password, not your normal password)
EMAIL_USER=you@gmail.com
EMAIL_PASS=xxxx xxxx xxxx xxxx
NOTIFY_TO=you@example.com
NOTIFY_COOLDOWN_MINUTES=10

# Dashboard login. Double quotes, and no " or \ inside: python-dotenv skips a
# line it can't parse, and the login then reports the password as not set.
ADMIN_PASSWORD="your-password"

# Camera agent's own service credential (not the human login) — any random
# 64-char hex string, e.g. `python -c "import secrets; print(secrets.token_hex(32))"`
AGENT_TOKEN="generate-your-own-random-token"

# Camera agent, Linux / Raspberry Pi shown; on Windows use dshow and
# video=Your Webcam Name
CAMERA_FORMAT=v4l2
CAMERA_INPUT=/dev/v4l/by-id/usb-Your_Camera-video-index0
AGENT_INTEREST_THRESHOLD=60         # 0-100; how far from "resting" before it's worth an email
AGENT_MOTION_FLOOR=4
AGENT_ALERT_STREAK=3

# Clipping Mode (bulk clip harvesting). All optional - these are first-run
# defaults only; once you touch the dashboard controls, monitor.json wins.
AGENT_CLIPPING_FRAME_SECONDS=0.5    # motion sampling cadence while harvesting
AGENT_CLIPPING_SEGMENT_SECONDS=8    # length of a harvested clip
AGENT_CLIPPING_STREAK=1             # frames over threshold before a clip fires
AGENT_CLIPPING_RUN_MODEL=0          # 1 keeps inference on (diagnostic only)
CLIPPING_MAX_UNLABELED=500          # auto-stop at this many unlabeled clips
CLIPPING_MIN_FREE_GB=10             # auto-stop below this much free disk
```

Find your camera on Linux with `ls -l /dev/v4l/by-id/` (use the
`video-index0` entry), or on Windows with
`ffmpeg -f dshow -list_devices true -i dummy`.

### 3. Run

Production runs 24/7 on a Raspberry Pi under the systemd units in
`deploy/pi/`. See [Raspberry Pi](#raspberry-pi).

For local development, `./run.sh` starts the server and agent in the
foreground (`./run.sh server` for the dashboard without a camera).

The dashboard is at http://localhost:3001. Log in with `ADMIN_PASSWORD` to reach
the camera, labeling, and training tabs. Without login you see the public demo.

---

## Using it

1. **Label**: import clips, then tag them in **Label Studio**. Keys `1`-`7`
   assign a behavior, `0` marks a clip with no rabbit in view. The vocabulary
   is the ethogram in [docs/ethogram.md](docs/ethogram.md), not an ad-hoc list:
   `feeding`, `grooming`, `locomotion`, `locomotion_rapid`, `rearing`,
   `resting_lying`, `resting_sitting`, plus `out_of_view`.
2. **Train**: an offline CLI pipeline, extract -> cross-validate -> fit ->
   export a candidate -> evaluate -> promote; see
   [docs/pipeline.md](docs/pipeline.md#the-order). Export writes to
   `server/models/candidate/`, and only `training/promote.py`, which refuses
   a candidate that failed its gate, replaces the model the agent loads. The
   Training tab in the dashboard is a read-only view of whatever is deployed.
3. **Monitor**: the agent loads the exported ONNX model and runs live. Toggle
   **Monitoring** and **Email** from the dashboard header.

Clips the agent records during an alert carry the predicted behavior as a
**suggestion** shown in Label Studio, confirm or correct it with a normal
hand label. Only human-confirmed labels are ever used for training, so the
model's own predictions can't feed back into its training data.

### Clipping Mode

Clipping Mode collects unlabeled footage whenever motion crosses a threshold.
It works without a trained model. See the [Clipping Mode guide](docs/clipping.md)
for sensitivity, cooldown, dry runs, and storage limits.

---

## Development

The [development guide](docs/development.md) maps the source directories and
covers Python tests, C++ tests, manual camera checks, and generated files.

From `server/`, run the Python checks with:

```bash
.venv/bin/python -m pytest -m "not local_data"
.venv/bin/ruff check .
```

For the C++ build and tests, see [agent/README.md](agent/README.md#build-and-test).

---

## Deployment

Production is a Raspberry Pi 5, where systemd runs three services:
`bunnycam-server`, `bunnycam-agent` (both Python), and `bunnycam-tunnel`
(Cloudflare). The tunnel serves the custom domain over public HTTPS without
any router port forwarding. The server listens on loopback only, behind the
tunnel. A Linux PC holds the clip archive and does the training, and the
finished model goes back to the Pi.

Security: cookie session auth (Secure over HTTPS) for the human login, a
separate service-token (`AGENT_TOKEN`) for the camera agent so it can never
lock out the human admin, rate-limited login, timing-safe password check, and
a server-enforced private live stream.

### Raspberry Pi

[deploy/pi/README.md](deploy/pi/README.md) covers the Pi install, updating
it, and the PC side. The runtime is the same `uv sync --frozen --python 3.14`
as above, not the OS Python. Rolling segments and the live frame go to tmpfs
under `/run/bunnycam`. The Pi has no Node, so `client/dist` is built on the
PC and copied over. Labeling happens on the Pi in the dashboard, and
`deploy/pi/pull-data.sh` copies clips and labels to the PC before training.
Email goes straight through SMTP, n8n is no longer used. The Pi has no
classifier until a seven-class model is promoted (see Model status above).

---

## Roadmap

Planned stretch goals (see [docs/stretch-goals.md](docs/stretch-goals.md) for
the full plan):

- **Water intake tracking**: a load cell under the water bowl measures
  drinking in ml, with low-level and low-intake alerts.
- **Room environment**: a BME280 on the same Pi tracks temperature and
  humidity, with heat-stress alerts (rabbits overheat above ~28C).

**Relabel, then retrain.** All 112 clips were unlabeled during the v2 ethogram
migration and need recoding against the new definitions; v1 labels are archived
in `data/labels.v1.json` so the v1-to-v2 recoding can still be reported. The
classifier cannot run until a 7-class head is trained and exported.

---

## Notes

- Recordings (`data/recordings/`), labels (`data/labels.json`), backups,
  and `.env` are all git-ignored. The ONNX artifacts in `server/models/` are
  also used by the C++ parity tests. Live classification requires a model
  promoted for the current ethogram.
- Labels are written atomically and serialized, with `.bak` plus daily backups,
  so concurrent edits can't corrupt or wipe them.
- No third-party ML service, inference runs locally via ONNX Runtime.
