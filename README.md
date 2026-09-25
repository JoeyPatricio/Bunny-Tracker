# 🐇 Bunny Tracker

A self-hosted machine-learning pet monitor. A camera watches your rabbits, a
classifier recognizes what they're doing, zoomies, grooming, rearing, feeding,
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
| Process mgmt| pm2 (server, agent, tunnel, n8n)                                    |
| Deployment  | Cloudflare named tunnel (HTTPS, custom domain)                      |

---

## Setup

### 1. Install dependencies

Two Python environments, on purpose: the runtime venv never contains torch
because it is what ships to the Raspberry Pi. See
[docs/pipeline.md](docs/pipeline.md) for the full rule.

```bash
cd client && npm install && npm run build
cd ../server_py

# Runtime (server + agent). No torch.
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Only if you're retraining the model. Pulls a CUDA build of torch.
uv venv .venv-train --python 3.12
uv pip install --python .venv-train/bin/python -r requirements-train.txt
```

Both venvs here were created with `uv`, so **neither has `pip` inside it** and
`.venv/bin/pip install ...` will not work. Use `uv pip install --python <venv>`
as above. On Windows the interpreter is `.venv\Scripts\python.exe` rather than
`.venv/bin/python`.

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

1. **Label**: import clips, then tag them in **Label Studio**. Keys `1`-`7`
   assign a behavior, `0` marks a clip with no rabbit in view. The vocabulary
   is the ethogram in [docs/ethogram.md](docs/ethogram.md), not an ad-hoc list:
   `feeding`, `grooming`, `locomotion`, `locomotion_rapid`, `rearing`,
   `resting_lying`, `resting_sitting`, plus `out_of_view`.
2. **Train**: run `server_py/training/train.py` (see that file's docstring for
   the full pipeline: extract -> split -> fit -> export to ONNX). The Training
   tab in the dashboard is now a read-only view of whatever model is deployed.
3. **Monitor**: the agent loads the exported ONNX model and runs live. Toggle
   **Monitoring** and **Email** from the dashboard header.

Clips the agent records during an alert carry the predicted behavior as a
**suggestion** shown in Label Studio, confirm or correct it with a normal
hand label. Only human-confirmed labels are ever used for training, so the
model's own predictions can't feed back into its training data.

### Clipping Mode

Normal monitoring only records when the classifier sees something noteworthy,
which is the wrong shape for building a training set: it needs a working model
to decide what to keep, and it keeps very little. **Clipping Mode** inverts
that - it saves a clip every time motion crosses a threshold, so an unattended
run produces a large pile of unlabeled footage to hand-label later.

Toggle it with **✂ Clipping** in the dashboard header. While it's on:

- **The classifier does not run at all.** No predictions, no email alerts, no
  label suggestions - a quiet Activity Log is expected, not a bug. This is also
  why Clipping Mode works with no trained model on disk, which is exactly the
  situation you're in when you're collecting data to train one.
- Clips are 8s (vs. 12s), sampled every 0.5s (vs. 1.5s), and the clip you get
  is the segment the motion happened in rather than the one before it.
- They land in `server/recordings/` as `recording-clip-*.webm`, unlabeled, so
  Label Studio's **Unlabeled** filter is your labeling queue.

The panel under the live feed has the controls:

| Control | What it does |
|---|---|
| Sensitivity (1-10) | Maps to a motion threshold (8.0 down to 0.4). Higher catches more, including more empty footage. |
| Cooldown | Minimum gap between clips. At or above the clip length you get at most one clip per segment. |
| Dry run | Reports what *would* be captured without saving anything. |
| Live motion trace | The agent's real motion level with your threshold drawn across it. |

**Tuning it.** Don't guess at the threshold - turn on dry run and watch the
trace for ten minutes. Rabbit movement shows up as clear spikes; put the line
just under them. A threshold slightly too low will fill the entire clip budget
overnight with footage of nothing.

**Guardrails.** Harvesting stops automatically at `clippingMaxUnlabeled`
(default 500) *unlabeled* clips, or when free disk drops below
`clippingMinFreeGb` (default 10). Labeled clips don't count toward the cap, so
working through Label Studio frees room. **Nothing is ever deleted** - when a
limit is hit the dashboard shows why and waits for you to re-arm it.

**Spread the harvest over several days.** `training/dataset.py::group_key`
groups agent clips by capture hour so clips from one session can't straddle the
train/val split. 500 clips from a single overnight run is only a handful of
groups, which makes the split coarse and the validation number noisy; the same
clip count gathered across several days is a far more useful dataset.

---

## Tests

Plain scripts, not pytest. Run them from `server_py/` with the runtime venv:

```bash
.venv/bin/python tests/test_motion_parity.py
.venv/bin/python tests/test_security_regressions.py
.venv/bin/python tests/test_server_robustness.py
.venv/bin/python tests/test_capture_loop.py
.venv/bin/python tests/test_clipping_mode.py
.venv/bin/python agent/test_capture_offline.py
```

The two training tests need torch, so they run from the training venv:

```bash
.venv-train/bin/python training/test_train_guards.py
.venv-train/bin/python training/test_export_guards.py
```

There is no aggregate runner yet, so a failure in one script does not stop the
others and you have to read all of them. One is added in Phase 1 of
[docs/upgrade-plan.md](docs/upgrade-plan.md).

`tests/test_server_robustness.py` has a timing-sensitive check that fails
spuriously on fast machines (it asserts a deliberately-blocking control really
does stall the event loop, and on fast hardware the control finishes too
quickly to measure). Re-run before investigating; Phase 1 fixes it properly.

`tests/test_clipping_mode.py` covers Clipping Mode end to end (model-free
decision loop, segment selection, budget) and needs no camera or model.

To exercise the whole capture pipeline with no webcam, point the agent at a
synthetic source - `_build_args` paces `lavfi` with `-re` so the frame and
segment timings stay realistic:

```ini
CAMERA_FORMAT=lavfi
CAMERA_INPUT=testsrc2=size=640x480:rate=15
```

---

## Project structure

```
BunnyTracker/
├── ecosystem.config.cjs        # pm2 process definitions (server, agent, tunnel, n8n)
├── client/                     # React + Vite frontend
│   └── src/
│       └── components/
│           ├── AgentFeed.jsx        # Live view from the agent + record button
│           ├── ClippingPanel.jsx    # Clipping Mode tuning (sensitivity, trace, budget)
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

**Relabel, then retrain.** All 112 clips were unlabeled during the v2 ethogram
migration and need recoding against the new definitions; v1 labels are archived
in `server/labels.v1.json` so the v1-to-v2 recoding can still be reported. The
classifier cannot run until a 7-class head is trained and exported.

---

## Notes

- Recordings (`server/recordings/`), labels (`server/labels.json`), backups,
  and `.env` are all git-ignored. The deployed ONNX artifacts in
  `server_py/models/` are committed, so a clone can run inference without
  retraining.
- Labels are written atomically and serialized, with `.bak` plus daily backups,
  so concurrent edits can't corrupt or wipe them.
- No third-party ML service, inference runs locally via ONNX Runtime.
---

## References

The behaviour vocabulary this project is moving toward is defined in
[docs/ethogram.md](docs/ethogram.md). Everything below is either a source for
those definitions, directly comparable prior work, or a tool the project uses
or should be measured against.

### Ethogram and behavioural definitions

- NC3Rs / IAT / RSPCA. **Laboratory rabbit ethogram**, from *Refining rabbit
  care: A resource for those working with rabbits in research*, UFAW/RSPCA
  Rabbit Behaviour and Welfare Group (2008).
  https://nc3rs.org.uk/sites/default/files/2022-01/Laboratory%20rabbit%20ethogram.pdf
  Primary source for the v2 state and event definitions. It is itself a
  compilation of the three papers below.
- Morton DB et al. (2003). Refinements in rabbit husbandry: Second report of
  the BVAAWF/FRAME/RSPCA/UFAW joint working group on refinement.
  *Laboratory Animals* 27: 301-329 (Appendix 1, pp. 325-7).
- Held SDE, Turner RJ, Wootton RJ (2001). The behavioural repertoire of
  non-breeding group-housed female laboratory rabbits (*Oryctolagus
  cuniculus*). *Animal Welfare* 10(4): 437-443.
- Gunn D, Morton DB (1995). Inventory of the behaviour of New Zealand White
  rabbits in laboratory cages. *Applied Animal Behaviour Science* 45(3-4):
  277-292. doi:10.1016/0168-1591(95)00627-5

### Prior work on automated rabbit behaviour classification

- Adedeji OJ, Abayomi-Alli A, Arogundade O'T, Abayomi-Alli O, Omoyiola BO
  (2023). **Deep Transfer Learning for Classification of Rabbit Behaviour
  Using Publicly Available Datasets.** *First International Conference on the
  Advancements of Artificial Intelligence in African Context (AAIAC)*, IEEE.
  doi:10.1109/AAIAC60008.2023.10465471

  The closest published work to this project and the natural baseline to
  compare against. YOLOv8 + ByteTrack for rabbit detection and tracking
  (mAP50 0.96), and a fine-tuned ViT-B/16 for behaviour classification over
  seven classes (digging, humping and mating, eating, periscoping, jumping and
  binkying, laying, flopping and loafing), reaching 86% accuracy against 80%
  for EfficientNet and 76% for ResNet18, on roughly 1100 images.

  Differences that matter for this project: their classifier operates on
  **single still images** scraped from YouTube, Roboflow, GitHub and Google
  Images, not on video, so temporal behaviours are inferred from one frame.
  The split is 70:30 train/validation with no test set and, as far as the paper
  describes it, no grouping by source video, so frames from the same clip can
  appear on both sides. Their class list is vernacular and carries no
  operational definitions or inter-observer reliability statistics, and the
  paper reports confusion between the visually similar classes (laying and
  periscoping). The paper is candid that its dataset is "not robust enough to
  cover the numerous ranges of behavioural variations" and that comparative
  benchmarks in this domain barely exist.

### Computational ethology tools

Established tooling in this field. Cited here because any paper coming out of
this project will be expected to position itself against them.

- Friard O, Gamba M (2016). BORIS: a free, versatile open-source event-logging
  software for video/audio coding and live observations. *Methods in Ecology
  and Evolution* 7(11): 1325-1330. https://www.boris.unito.it/
  The standard tool for interval-coded ethogram annotation.
- Bohnslav JP et al. (2021). DeepEthogram, a machine learning pipeline for
  supervised behavior classification from raw pixels. *eLife* 10:e63377.
  doi:10.7554/eLife.63377
  Closest prior art methodologically: supervised behaviour classification
  directly from video, which is what the classifier here does.
- Mathis A et al. (2018). DeepLabCut: markerless pose estimation of
  user-defined body parts with deep learning. *Nature Neuroscience* 21:
  1281-1289.
- Pereira TD et al. (2022). SLEAP: A deep learning system for multi-animal
  pose tracking. *Nature Methods* 19: 486-495.
- Goodwin NL, Nilsson SRO et al. (2024). Simple Behavioral Analysis (SimBA) as
  a platform for explainable machine learning in behavioral neuroscience.
  *Nature Neuroscience* 27: 1411-1424. doi:10.1038/s41593-024-01649-9
- Hsu AI, Yttri EA (2021). B-SOiD, an open-source unsupervised algorithm for
  identification and fast prediction of behaviors. *Nature Communications*
  12: 5188. doi:10.1038/s41467-021-25420-x

### Models and runtime used here

- Qin D et al. (2024). MobileNetV4: Universal Models for the Mobile Ecosystem.
  *European Conference on Computer Vision (ECCV)*. The deployed backbone is
  `mobilenetv4_conv_small`.
- Wightman R (2019). **PyTorch Image Models** (`timm`).
  https://github.com/huggingface/pytorch-image-models
  Source of the frozen backbone and its pretrained weights.
- **ONNX Runtime.** https://onnxruntime.ai/ Inference for the exported
  backbone and temporal head, including the int8 quantized backbone.
- **FFmpeg.** https://ffmpeg.org/ All video capture, segmenting, and frame
  decode, via `imageio-ffmpeg`.
