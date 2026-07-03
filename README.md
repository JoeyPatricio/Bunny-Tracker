# 🐇 PetCam (Bunny Tracker)

A self-hosted machine-learning pet monitor. A camera watches your rabbits, a
classifier recognizes what they're doing — zoomies, yawning, grooming, standing,
or resting — and you get an email with a clip when something noteworthy happens.

Live demo: **https://bunny-tracker.app** (public demo view; the live stream is
private to the owner).

---

## How it works

```
[camera] ──► capture agent (ffmpeg) ──► MobileNetV2 embedding ──► classifier ──► behavior
                  │                                                        │
                  ├──► rolling 12s clips                                   ├──► email alert + clip
                  └──► MJPEG live stream ──► dashboard                     └──► public text feed
```

- **Capture agent** (`server/agent/capture.mjs`) — a headless Node service that
  reads the camera with one `ffmpeg` process, sampling a frame every few seconds
  for inference, recording rolling clips, and pushing a live MJPEG stream. Runs
  the same on a PC (`dshow`) or a Raspberry Pi (`v4l2`) — only two `.env` lines change.
- **Classifier** — MobileNetV2 transfer learning in TensorFlow.js. A dense head
  trained on 150+ hand-labeled clips maps the mean ‖ std of 8 frame embeddings
  (2×1280 dims — the std across frames is the motion signal) to five behaviors.
  Train it from the browser in the Training tab; validation is split by source
  video so the accuracy number reflects real generalization.
- **Alerts** — a non-normal behavior emails you the clip (Nodemailer + Gmail),
  with motion gating, a multi-frame debounce, and a global cooldown to suppress
  false positives.
- **Web app** — Camera (live feed + predictions), Label Studio (label clips),
  and Training (extract features + train the model). A separate public demo page
  shows highlights and a live behavior feed without exposing the camera.

## Tech stack

| Layer       | Tech                                                        |
|-------------|-------------------------------------------------------------|
| Frontend    | React + Vite, TensorFlow.js, MobileNetV2                     |
| Backend     | Node.js + Express                                           |
| Capture/ML  | ffmpeg (bundled), TensorFlow.js (Node), pixel-diff motion   |
| Alerts      | Nodemailer (Gmail SMTP)                                     |
| Process mgmt| pm2 (server, agent, tunnel)                                 |
| Deployment  | Cloudflare named tunnel (HTTPS, custom domain)             |

---

## Setup

### 1. Install dependencies

```bash
npm run install:all
```

Installs packages for the root, client, and server.

### 2. Configure the environment

Create `server/.env`:

```ini
# Email alerts (Gmail App Password — not your normal password)
EMAIL_USER=you@gmail.com
EMAIL_PASS=xxxx xxxx xxxx xxxx
NOTIFY_TO=you@example.com
NOTIFY_COOLDOWN_MINUTES=10

# Dashboard login (wrap in quotes if it contains a # )
ADMIN_PASSWORD="your-password"

# Camera agent — PC defaults shown; for a Raspberry Pi use v4l2 / /dev/video0
CAMERA_FORMAT=dshow
CAMERA_INPUT=video=Your Webcam Name
AGENT_CONFIDENCE_THRESHOLD=70
AGENT_MOTION_FLOOR=4
AGENT_ALERT_STREAK=3
```

Find your camera name (Windows): `ffmpeg -f dshow -list_devices true -i dummy`.

### 3. Build the client

```bash
cd client && npm run build
```

The server serves the built app, so everything runs on one port.

### 4. Run

Development (server + client dev server):

```bash
npm run dev
```

Production / 24-7 (server, capture agent, and tunnel under pm2):

```bash
pm2 start ecosystem.config.cjs
pm2 save
```

The dashboard is at http://localhost:3001. Log in with `ADMIN_PASSWORD` to reach
the camera, labeling, and training tabs. Without login you see the public demo.

---

## Using it

1. **Label** — record or import clips, then tag them in **Label Studio**
   (`Z` zoomies, `Y` yawn, `N` normal, `G` grooming, `S` standing).
2. **Train** — open **Training** and hit Start. It extracts MobileNet features,
   trains the classifier, shows a confusion matrix, and saves the model.
3. **Monitor** — the agent loads the saved model and runs live. Toggle
   **Monitoring** and **Email** from the dashboard header.

Clips the agent records during an alert carry the predicted behavior as a
**suggestion** shown in Label Studio — confirm or correct it with a normal
hand label. Only human-confirmed labels are ever used for training, so the
model's own predictions can't feed back into its training data.

---

## Project structure

```
PetCam/
├── ecosystem.config.cjs        # pm2 process definitions (server, agent, tunnel)
├── client/                     # React + Vite frontend
│   └── src/
│       ├── components/
│       │   ├── VideoFeed.jsx        # Browser webcam view (fallback)
│       │   ├── AgentFeed.jsx        # Live MJPEG stream from the agent + record
│       │   ├── LabelingStudio.jsx   # Clip labeling UI
│       │   ├── TrainingStudio.jsx   # Feature extraction + model training
│       │   ├── RecordingGallery.jsx # Recordings, filterable by label
│       │   └── DemoView.jsx         # Public demo (highlights + live feed)
│       └── hooks/                   # useWebcam, useMotion, useInference
│
└── server/                     # Express backend
    ├── agent/capture.mjs       # Headless capture + inference service
    ├── lib/labelStore.js       # Atomic, serialized label storage
    ├── lib/backupLabels.js     # Daily label backups (7-day retention)
    ├── routes/                 # recordings, labels, model, sms, auth,
    │                           #   stream, monitor, predictions
    ├── model/                  # Saved TensorFlow.js model + label map
    ├── recordings/             # Saved .webm clips (git-ignored)
    └── index.js                # Server entry point
```

---

## Deployment

Three processes run under pm2 and auto-start at login: `bunnycam-server`,
`bunnycam-agent`, and `bunnycam-tunnel` (Cloudflare). The tunnel maps a custom
domain to the local server over HTTPS, so the camera PC never exposes a port.

Security: cookie session auth (Secure over HTTPS), rate-limited login,
timing-safe password check, and a server-enforced private live stream.

### Raspberry Pi

The capture agent is platform-agnostic. To move from a PC to a Pi: clone, run
`npm install`, swap two `.env` lines (`CAMERA_FORMAT=v4l2`,
`CAMERA_INPUT=/dev/video0`), and start the same pm2 processes.

---

## Notes

- Recordings (`server/recordings/`), labels (`server/labels.json`), the model,
  backups, and `.env` are all git-ignored.
- Labels are written atomically and serialized, with `.bak` plus daily backups,
  so concurrent edits can't corrupt or wipe them.
- No third-party ML service — inference runs locally in TensorFlow.js.
```
