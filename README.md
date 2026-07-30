# 🐇 Bunny Tracker

A self-hosted machine-learning pet monitor. A camera watches your rabbits, a
classifier recognizes what they're doing, zoomies, yawning, grooming, standing,
or resting, and you get an email with a clip when something noteworthy happens.

Live demo: **https://bunny-tracker.app** (public demo view; the live stream is
private to the owner).

---

## How it works

```
[camera] ──► capture agent (ffmpeg) ──► ONNX backbone embedding ──► temporal head ──► behavior
                  │                                                            │
                  ├──► rolling 12s clips                                       ├──► email alert + clip
                  └──► MJPEG live stream ──► dashboard                         └──► public text feed
```

- **Capture agent** (`server_py/agent/capture.py`): a headless Python service
  that reads the camera with one `ffmpeg` process, sampling a frame every few
  seconds for inference, recording rolling clips, and pushing a live view to
  the dashboard. Runs the same on a PC (`dshow`) or a Raspberry Pi (`v4l2`),
  only two `.env` lines change.
- **Classifier**: a frozen `timm` edge backbone (`mobilenetv4_conv_small`) plus
  a small learned temporal head, exported to ONNX and run via ONNX Runtime -
  replaced the original browser-trained TensorFlow.js/MobileNetV2 classifier.
  Retraining is now an offline CLI script (`server_py/training/train.py`), not
  a browser tab; see the note below.
- **Alerts**: a non-normal behavior emails you the clip (Python `smtplib` +
  Gmail), with motion gating, a multi-frame debounce, and a global cooldown to
  suppress false positives.
- **Web app**: Camera (live feed + predictions), Label Studio (label clips),
  and Training (a read-only panel showing the deployed model's metadata). A
  separate public demo page shows highlights and a live behavior feed without
  exposing the camera.

**Model accuracy note:** the currently deployed classifier is measurably weaker
than the one it replaced (61.5% vs. 74.4% held-out accuracy) - the retrain
didn't have enough labeled data yet to beat the original. The rest of the
system (capture, motion detection, alerting, dashboard) was ported and verified
independently of this; it's a known, accepted gap pending more labeled clips.

## Tech stack

| Layer       | Tech                                                             |
|-------------|--------------------------------------------------------------------|
| Frontend    | React + Vite                                                       |
| Backend     | Python (FastAPI + uvicorn)                                          |
| Capture/ML  | ffmpeg (`imageio-ffmpeg`), ONNX Runtime, pixel-diff motion detection |
| Training    | PyTorch + `timm`, offline CLI script                                |
| Alerts      | `smtplib` (Gmail SMTP)                                              |
| Process mgmt| pm2 (server, agent, tunnel, n8n)                                    |
| Deployment  | Cloudflare named tunnel (HTTPS, custom domain)                      |

---

## Setup

### 1. Install dependencies

```bash
cd client && npm install && npm run build
cd ../server_py
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# Only if you're retraining the model:
python -m venv .venv-train && .venv-train/Scripts/pip install -r requirements-train.txt
```

### 2. Configure the environment

Create `server/.env` (shared by both the server and the agent):

```ini
# Email alerts (Gmail App Password, not your normal password)
EMAIL_USER=you@gmail.com
EMAIL_PASS=xxxx xxxx xxxx xxxx
NOTIFY_TO=you@example.com
NOTIFY_COOLDOWN_MINUTES=10

# Dashboard login (wrap in quotes if it contains a # )
ADMIN_PASSWORD="your-password"

# Camera agent's own service credential (not the human login) — any random
# 64-char hex string, e.g. `python -c "import secrets; print(secrets.token_hex(32))"`
AGENT_TOKEN="generate-your-own-random-token"

# Camera agent, PC defaults shown; for a Raspberry Pi use v4l2 / /dev/video0
CAMERA_FORMAT=dshow
CAMERA_INPUT=video=Your Webcam Name
AGENT_CONFIDENCE_THRESHOLD=70
AGENT_MOTION_FLOOR=4
AGENT_ALERT_STREAK=3
```

Find your camera name (Windows): `ffmpeg -f dshow -list_devices true -i dummy`.

### 3. Run

Production / 24-7 (server, capture agent, tunnel, and n8n under pm2):

```bash
pm2 start ecosystem.config.cjs
pm2 save
```

The dashboard is at http://localhost:3001. Log in with `ADMIN_PASSWORD` to reach
the camera, labeling, and training tabs. Without login you see the public demo.

---

## Using it

1. **Label**: import clips, then tag them in **Label Studio**
   (`Z` zoomies, `Y` yawn, `N` normal, `G` grooming, `S` standing).
2. **Train**: run `server_py/training/train.py` (see that file's docstring for
   the full pipeline: extract -> split -> fit -> export to ONNX). The Training
   tab in the dashboard is now a read-only view of whatever model is deployed.
3. **Monitor**: the agent loads the exported ONNX model and runs live. Toggle
   **Monitoring** and **Email** from the dashboard header.

Clips the agent records during an alert carry the predicted behavior as a
**suggestion** shown in Label Studio, confirm or correct it with a normal
hand label. Only human-confirmed labels are ever used for training, so the
model's own predictions can't feed back into its training data.

---

## Tests

Plain scripts, not pytest. Run them from `server_py/` with the runtime venv:

```bash
.venv/Scripts/python.exe tests/test_motion_parity.py
.venv/Scripts/python.exe tests/test_security_regressions.py
.venv/Scripts/python.exe tests/test_server_robustness.py
.venv/Scripts/python.exe tests/test_capture_loop.py
.venv/Scripts/python.exe agent/test_capture_offline.py
```

`training/test_train_guards.py` needs torch, so run that one with
`.venv-train/Scripts/python.exe`.

---

## Project structure

```
BunnyTracker/
├── ecosystem.config.cjs        # pm2 process definitions (server, agent, tunnel, n8n)
├── client/                     # React + Vite frontend
│   └── src/
│       └── components/
│           ├── AgentFeed.jsx        # Live view from the agent + record button
│           ├── LabelingStudio.jsx   # Clip labeling UI
│           ├── TrainingStudio.jsx   # Read-only deployed-model metadata panel
│           ├── ActivityLog.jsx      # Recent behavior predictions (polls /api/predictions)
│           ├── RecordingGallery.jsx # Recordings, filterable by label
│           └── DemoView.jsx         # Public demo (highlights + live feed)
│
├── server/                     # Data only - no server code lives here anymore
│   ├── recordings/             # Saved .webm clips (git-ignored)
│   ├── model/                  # Retired tfjs artifacts. No Python code path
│   │                           #   can run them; still served read-only at
│   │                           #   /model/* and covered by the traversal tests
│   ├── labels.json, hidden.json, monitor.json, activity-log.json, ...
│   └── agent/segments/         # Rolling capture segments
│
└── server_py/                   # Python backend + agent + training
    ├── app/                      # FastAPI server (routers, auth, storage)
    ├── agent/                    # Headless capture agent (capture.py, ffmpeg_io.py, client.py)
    ├── inference/                 # ONNX Runtime wrappers (backbone, temporal head, predictor)
    ├── shared/                    # Code shared by the trainer and the agent (frames, motion, labels)
    ├── training/                  # Offline PyTorch training CLI
    ├── models/                    # Deployed ONNX artifacts (backbone, head)
    ├── .venv/                     # Runtime deps (no torch) - what actually runs in production
    └── .venv-train/               # Training deps (torch, timm) - dev box only
```

---

## Deployment

Four processes run under pm2 and auto-start at login: `bunnycam-server`,
`bunnycam-agent` (both Python), `bunnycam-tunnel` (Cloudflare), and
`bunnycam-n8n`. The tunnel maps a custom domain to the local server over HTTPS,
so the camera PC never exposes a port.

Security: cookie session auth (Secure over HTTPS) for the human login, a
separate service-token (`AGENT_TOKEN`) for the camera agent so it can never
lock out the human admin, rate-limited login, timing-safe password check, and
a server-enforced private live stream.

### Raspberry Pi

The capture agent is platform-agnostic in principle - swap two `.env` lines
(`CAMERA_FORMAT=v4l2`, `CAMERA_INPUT=/dev/video0`) and point `ecosystem.config.cjs`
at the Pi's own venv/cloudflared/n8n paths.

---

## Roadmap

Planned stretch goals (see [docs/stretch-goals.md](docs/stretch-goals.md) for
the full plan):

- **Permanent hardware**: move the whole stack (server, agent, tunnel, n8n)
  to a Raspberry Pi 5 so it runs 24/7 without the PC.
- **Water intake tracking**: a load cell under the water bowl measures
  drinking in ml, with low-level and low-intake alerts.
- **Room environment**: a BME280 on the same Pi tracks temperature and
  humidity, with heat-stress alerts (rabbits overheat above ~28C).

**Retrain once more labeled data exists** - the current classifier is a known
regression from the one it replaced; see the model accuracy note above.

---

## Notes

- Recordings (`server/recordings/`), labels (`server/labels.json`), backups,
  and `.env` are all git-ignored. The deployed ONNX artifacts in
  `server_py/models/` are committed, so a clone can run inference without
  retraining.
- Labels are written atomically and serialized, with `.bak` plus daily backups,
  so concurrent edits can't corrupt or wipe them.
- No third-party ML service, inference runs locally via ONNX Runtime.