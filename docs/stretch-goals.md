# Stretch goals: Pi migration, water tracking, and the paper

Status (2026-10-19): the Phase A deployment files are in `deploy/pi/` and the
agent's transient paths are configurable. None of it has run on a Pi yet.
[deploy/pi/README.md](../deploy/pi/README.md) is the install procedure. If
this plan and that guide disagree, the guide wins.

## Goals

1. **Pi migration.** The whole stack (server, agent, Cloudflare tunnel, and
   maybe n8n) runs 24/7 on a Raspberry Pi 5. Done means the Windows PC is off,
   https://bunny-tracker.app loads, the live stream works, overnight motion
   saves a clip, and an alert email arrives.
2. **Water and room tracking.** The dashboard shows bowl level, today's intake
   in ml with a 7-day trend, and room temperature and humidity. Emails go out
   when the bowl is low, when intake drops well below the rabbits' own
   baseline (an early GI stasis sign), or when the room gets hot (heat stress
   starts around 28C).
3. **A paper.** A preprint whose figures and tables come from scripts in this
   repo and a frozen set of artifacts, plus a released dataset that doesn't
   include video from inside the house.

Phase A comes first because the water sensor needs the Pi's GPIO. Phase C can
start any time.

## Non-goals

- Improving the classifier. There's no seven-class model yet, so the agent
  runs motion-only. Capture, Clipping Mode and labeling don't need a model.
  The Pi gets one once a model is promoted on the PC.
- Training on the Pi. `.venv-train` stays on the dev machine.
- A vision-based drinking class. The load cell covers water.
- Multiple cameras, a database, or porting the Windows autostart `.bat`.

## Decisions

**Hardware** (one order, about $170-200):

- Raspberry Pi 5, 8GB, with the official active cooler and 27W PSU
- A USB3 SSD or NVMe HAT. If it has to be microSD, a 128GB high-endurance A2
  card.
- 5kg load cell, HX711 board and jumper wires
- BME280 I2C sensor with 30cm+ of wire so it sits away from the Pi's heat
- A heavy ceramic bowl, two rigid plates and M4/M5 standoffs for the load cell
- The existing onn 4K USB webcam

**Storage writes are the real constraint.** The rolling 12s segments write
about 10GB/day and the live-view JPEG gets rewritten 12 times a second (about
30GB/day). Neither is kept. `deploy/pi/bunnycam.env` points
`AGENT_SEGMENT_DIR` and `AGENT_LIVE_FRAME_PATH` at `/run/bunnycam/`, a tmpfs
created by `deploy/pi/bunnycam.tmpfiles`.

**OS and Python.** Raspberry Pi OS Lite 64-bit. onnxruntime has no 32-bit ARM
wheels. Don't use the system Python (3.11). The lock has no cp311 wheels, so
uv installs a managed 3.14 instead. `uv.lock` lists aarch64 wheels for NumPy
and ONNX Runtime, but nobody has installed them on hardware yet.

**Inference** is ONNX Runtime on CPU with `backbone_int8.onnx` and
`head.onnx`. No torch on the Pi. Latency on the Pi hasn't been measured.

**systemd instead of pm2.** pm2 would mean installing Node just to supervise
two Python processes and cloudflared. systemd handles restarts, boot order and
stopping the agent's ffmpeg child. Node is only needed if n8n stays, and the
client can be built on the PC.

**ffmpeg** comes from apt. `agent/capture.py` already checks `FFMPEG_PATH`
first.

**Camera.** `agent/ffmpeg_io.py` now pins MJPEG, size and framerate under v4l2
the same way it does under dshow. Without pinning, the webcam defaults to
uncompressed 4K and motion detection sees a frozen scene.

**Water sensor design:**

- A separate process, `server/sensor/water_sensor.py`, that posts to the
  server's HTTP API like the agent does.
- GPIO and I2C libraries stay out of the runtime dependencies since they don't
  install on the dev machines.
- Its own secret, `WATER_SENSOR_SECRET`, checked with `hmac.compare_digest`
  next to `is_agent_authed()` in `app/auth.py`. A failing sensor can never
  lock out the admin login.
- Readings go to append-only monthly JSONL (`data/water/YYYY-MM.jsonl`).
  `json_store.py` rewrites the whole file on every write, which is wrong for
  2,880 readings a day.
- Calibration is 1g = 1ml, with tare and scale set once against a known
  weight.

**Tunnel** is migrated, not recreated, so DNS and the tunnel ID don't change.

**n8n is optional.** With `ALERT_WEBHOOK_URL` unset, the agent sends mail
through the server's `POST /api/notify` using `smtplib`.

## Phase A: Pi migration

1. **Order hardware.** Nothing else on the Pi can start until it arrives.
2. **Base setup.** Flash Pi OS Lite 64-bit and follow guide steps 1 to 4.
   Check: `uname -m` prints `aarch64`, the venv lists
   `CPUExecutionProvider`, and the camera offers `mjpeg` at `1280x720`.
3. **Portability changes.** v4l2 pinning, the path overrides and the systemd
   units are already in the repo. Check: on Windows, `pm2 restart all` still
   works and motion values haven't changed.
4. **Performance check on the Pi.**
   - Once a model is promoted and copied over, run
     `.venv/bin/python -m scripts.bench_pi` and compare with the 1500ms frame
     cadence.
   - Run the Pi agent for 30 minutes against the PC server. Start it by hand
     with `AGENT_SERVER_URL=http://<pc-ip>:3001`, the PC's `AGENT_TOKEN`,
     `CAMERA_FORMAT=v4l2`, `CAMERA_INPUT=/dev/video0`,
     `FFMPEG_PATH=/usr/bin/ffmpeg` and the tmpfs paths. Watch frame spacing
     and drop counts in the log, CPU in `top` (libvpx is the likely hog), and
     `vcgencmd measure_temp` under 80C.
   - If it falls behind, find which stage is slow before changing anything.
     Changing `AGENT_FRAME_SECONDS` changes the model's window and means
     revalidating it.
   - Check: steady cadence, and a hand wave saves a clip to the PC gallery.
5. **Move the server and data.** Guide steps 5 to 7. The archive keeps the
   durable `data/` tree and all label sidecars, and skips segments,
   `data/tmp/` and `sessions.json`. Build `client/dist/` on the PC. Move the
   model only after promotion, checked with `deploy/pi/model_bundle.py`.
   Check: `curl -sI http://127.0.0.1:3001/` answers and a hand wave saves a
   clip.
6. **Move the tunnel.** Guide step 8. Stop the PC tunnel, then
   `systemctl enable --now bunnycam-tunnel`. Check: the site loads and login
   works from a phone on cellular.
7. **Drop n8n (or keep it).** Dropping it means removing `ALERT_WEBHOOK_URL`
   from the Pi's `.env`. Keeping it means installing Node 22 and n8n, and
   copying the workflow, credentials and the `encryptionKey` from
   `~/.n8n/config`. Check: a forced motion event sends mail.
8. **Autostart and burn-in.** Reboot and confirm everything comes back (guide
   step 10). Stop the PC apps with `pm2 stop bunnycam-server bunnycam-agent
   bunnycam-tunnel bunnycam-n8n` and remove `start-bunnycam.bat` from startup.
   Keep the PC checkout for two weeks as a rollback. Burn in for a week,
   checking `systemctl status`, `journalctl -u bunnycam-agent`, temps and
   disk use. Check: with the PC off, the site is up, an overnight clip gets
   captured, and persistent storage only grows by saved clips.

## Phase B: Water and environment tracking

9. **Wire the sensor and log raw data for 24h.** This is the riskiest step.
   Load cell between the plates, HX711 on 3.3V and two GPIO pins. A throwaway
   `read_hx711.py` tares, calibrates and logs grams once a second to CSV.
   Check: drift under a few grams, refills show as clean jumps, drinks as
   small drops, and bumps recover within seconds.
10. **Sensor service.** Sample at 1Hz, take a 30-sample rolling median, read
    the BME280 over I2C, and POST
    `{ "grams", "tempC", "humidity", "at" }` every 30s to
    `/api/water/reading` with an `X-Water-Secret` header. Runs as
    `deploy/pi/bunnycam-water.service`. Mount the BME280 at rabbit height,
    out of the sun, 30cm+ from the Pi. Check: readings flow, a wrong secret
    gets 401, and the temperature matches a household thermometer.
11. **Server route.** `app/routers/water.py`, with the prefix added to
    `GUARDED_PREFIXES` in `app/auth.py` (otherwise `AdminGuardMiddleware`
    returns 401 before the route runs). `POST /api/water/reading` appends a
    line. `GET /api/water/summary` returns current level, today's intake,
    refills, 7-day totals and today's temperature and humidity range. Intake
    is the sum of drops. A rise over 100g is a refill. Tune the constants from
    the step 9 data. Check: a synthetic day replayed through the endpoint
    gives intake within 10% and exactly one refill.
12. **Alerts.** Reuse the existing notify path, with one cooldown gate like
    `alert_gate()` in `notify.py`.
    - Low level: under `LOW_LEVEL_G` (150), at most once per 12h.
    - Low intake: last 24h under 50% of the 7-day average, after 3+ days of
      data, at most once a day.
    - Heat: warning at `TEMP_WARN_C` (27), urgent at `TEMP_DANGER_C` (29),
      each at most once per 2h. Optionally a cold warning at 10.
    - Check: with thresholds forced, exactly one email per alert and none
      during the cooldown.
13. **Dashboard panel.** `client/src/components/WaterPanel.jsx` on the Camera
    tab, admin only. Shows level as a percent of the last refill, today's
    intake, a 7-day bar chart in plain CSS (no chart library), and
    temperature and humidity. Check: numbers match the summary endpoint and
    update after a refill.

## Phase C: The paper

The paper describes an open-source, edge-deployable home-monitoring system for
rabbits: a published-ethogram vocabulary, measured label reliability, nested
grouped CV, and paired comparisons against baselines. Whether a temporal head
beats static aggregation is an open question. The old v1 numbers can't answer
it.

14. **Write the claim first.** One paragraph and a target venue, before
    fitting any v2 models. State the setting, clip-level weak labels, targeted
    collection, class imbalance and single camera. Call the blind re-code
    intra-observer reliability unless a second person codes a sample.
15. **Run the experiments.** Finish the v2 recode, the blind re-code and the
    holdout cut first. Then:
    - `training.crossval` for majority, logistic regression, mean/std MLP and
      GRU on shared folds. Augmentation and crop arms once their caches exist.
    - Pose only as an offline comparison with independently annotated
      keypoints. No pose claim without grouped evaluation.
    - `training.learning_curve` for performance vs. training groups, keeping
      the outer eval sets fixed.
    - `training.figures` turns the reports in `server/models/` into
      SVG/PDF figures with JSON sidecars that record source hashes.

    Repeat SDs aren't confidence intervals. Repeated predictions aren't new
    clips. Self-agreement isn't a ceiling on model performance. The fp32 vs.
    int8 check is in-sample. Only report latency for hardware that was
    actually measured. Every plotted number has to trace back to a report and
    its hash.
16. **Freeze the artifacts.** Code revision, lockfile, ethogram version, label
    hash, manifests, seeds, CV and OOF files, all reports, ONNX files and
    figure sidecars, plus a freeze date. The paper describes that snapshot
    even if the deployed model changes later. Check: frozen reports regenerate
    every figure without refitting.
17. **Write it.** Methods cover the motion gate, two-stage inference, how
    clips get labeled, how the blind sample was drawn, and what nested
    grouped CV protects against. Limitations cover one room and camera, few
    animals, one observer, small classes, targeted collection, drift over
    time, and clip labels vs. short training windows.
18. **Release.** Only artifacts I've reviewed for privacy. Embeddings and the
    manifest reproduce head training without footage. They don't reproduce
    backbone, crop or pose experiments. Include scripts, lockfile, commands,
    hashes and sidecars. Leave out recordings, credentials and notes. Check:
    a clean environment reproduces the head experiments and figures.

## Risks

1. **onnxruntime won't install on the Pi.** Step 2 checks it first. If a
   wheel is missing, try uv with `--python 3.12` or `3.13` before building
   from source.
2. **The Pi is too slow.** libvpx encoding is more likely to be the
   bottleneck than inference. Step 4 measures it before migrating.
3. **Storage wears out.** Handled by tmpfs, a journald cap
   (`deploy/pi/journald-bunnycam.conf`) and an SSD. Ignored, this shows up as
   corruption months later.
4. **Load cell data is too noisy.** Step 9 runs before any service code. The
   intake alert compares against the rabbits' own baseline, so absolute
   accuracy matters less. If it's unusable, fall back to level-only refill
   reminders.
5. **Credential migration goes wrong.** The PC stack stays startable until
   burn-in passes. n8n can be dropped.
6. **Camera negotiates the wrong format under v4l2.** It would show up in step
   4 as stretched frame cadence, not an error.
7. **Scope creep.** More backbones and datasets turn this into a benchmark
   paper. If an experiment isn't in the step 14 claim, it's out.
8. **The system changes mid-draft.** Step 16's freeze means the paper
   describes a dated snapshot.
