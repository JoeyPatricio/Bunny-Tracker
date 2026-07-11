# Stretch Goals: Pi Migration and Water Tracking

## Goal

Two deliverables.

1. The whole BunnyCam stack (server, capture agent, Cloudflare tunnel, n8n)
   runs 24/7 on a Raspberry Pi 5. Done looks like: the Windows PC is shut
   down, https://bunny-tracker.app loads from anywhere, the live stream works,
   a bunny moving overnight produces a clip in the gallery, and an alert email
   arrives.
2. Water intake and room environment tracking. Done looks like: the dashboard
   shows the current bowl level, today's intake in ml with a 7-day trend, and
   the room's temperature and humidity with daily min/max. An email arrives
   when the bowl runs low, when daily intake drops well below the rabbits'
   own baseline (an early GI stasis signal), or when the room gets hot
   (rabbits cannot sweat; heat stress starts around 28C).

Phase A (migration) comes first: the water sensor wires into the Pi's GPIO,
which the PC does not have.

## Non-goals

- No classifier work (temporal modeling, other/absent class, confidence
  calibration). Those stay on the research paper's future-work list.
- No vision-based "drinking" behavior class in this round. The load cell
  covers water; a drinking class can be layered on later.
- No multi-camera support.
- No database. Water data uses the project's existing file-based storage
  pattern.
- No port of the Windows autostart .bat. The Pi uses pm2's systemd startup.

## Assumptions and decisions

- **Hardware to buy** (one order, ~$150-170 total):
  - Raspberry Pi 5, 8GB (n8n alone idles at ~400MB; 8GB leaves headroom)
  - Official active cooler and 27W USB-C PSU (sustained encode + inference
    will throttle a passively cooled Pi)
  - Case, 64GB+ A2-rated microSD (high endurance; recordings churn is modest)
  - 5kg strain-gauge load cell + HX711 breakout board + female-female jumper
    wires (~$10)
  - BME280 temperature/humidity sensor breakout, I2C version (~$5-8), plus
    30cm+ of wire so it can sit at rabbit level away from the Pi's own heat
  - Heavy ceramic bowl if the current one is light, plus two small rigid
    plates (plywood or acrylic) and M4/M5 standoffs to sandwich the load cell
  - The onn 4K USB webcam moves to the Pi; no Pi camera module needed.
- **OS**: Raspberry Pi OS Lite 64-bit (Bookworm). Node 22 LTS via NodeSource.
  arm64 is required for n8n and the tfjs wasm backend.
- **Inference stays pure-JS** `@tensorflow/tfjs`. Do not attempt
  `@tensorflow/tfjs-node` on ARM; there are no prebuilt binaries and source
  builds are a time sink. If the CPU backend is too slow, add
  `@tensorflow/tfjs-backend-wasm` (SIMD + threads) before touching anything
  else.
- **Remove orphaned deps** `@tensorflow/tfjs-node` and `canvas` from
  `server/package.json`. Verified unimported anywhere in server source; on
  ARM they would drag `npm install` through hours of native builds for
  nothing.
- **ffmpeg**: use Debian's ffmpeg on the Pi. `capture.mjs` currently hardcodes
  the `@ffmpeg-installer` binary path; add an `FFMPEG_PATH` env override with
  the installer as fallback. Keep libvpx/webm output so nothing downstream
  changes.
- **Sensor reader is a Python sidecar**, not Node. Python HX711 libraries are
  mature; Node bindings are not. It runs as a fifth pm2 process and talks to
  the server over the same HTTP API everything else uses.
- **Water data storage**: append-only JSONL, one file per month
  (`server/water/YYYY-MM.jsonl`), matching the project's file-store pattern.
- **Tunnel migrates, not recreated**: copy the existing cert and tunnel
  credentials to the Pi so DNS and the tunnel ID stay untouched.
- **Calibration**: 1g of water = 1ml. Tare and scale factor set once with a
  known weight.
- **n8n on the Pi is optional.** It is one email fan-out workflow, and
  `capture.mjs` already falls back to the built-in Nodemailer path when
  `ALERT_WEBHOOK_URL` is unset. If n8n misbehaves on ARM, drop it and lose
  nothing but the fan-out flexibility.

## Steps

### Phase A: Pi migration

#### 1. Order hardware

Everything in the hardware list above, one order. Nothing else can start
until the Pi arrives, but step 4 (portability commits) can happen meanwhile.

#### 2. Base Pi setup

Flash Pi OS Lite 64-bit with SSH enabled (Raspberry Pi Imager sets user +
wifi). Then:

```bash
sudo apt update && sudo apt install -y ffmpeg git python3-pip
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt install -y nodejs
sudo npm i -g pm2
```

Plug in the webcam and confirm it exposes MJPEG at 720p:

```bash
ffmpeg -f v4l2 -list_formats all -i /dev/video0
```

**Verify:** output lists `mjpeg` at `1280x720` (30fps).

#### 3. Portability commits (on the PC, can run in parallel with shipping)

Three small changes, committed and verified on Windows before the Pi work
depends on them:

- Remove `@tensorflow/tfjs-node` and `canvas` from `server/package.json`.
- In `capture.mjs`, resolve the ffmpeg binary as
  `process.env.FFMPEG_PATH ?? ffmpegInstaller.path`.
- Add `ecosystem.pi.config.cjs`: same four apps with Linux paths
  (`/usr/local/bin/cloudflared`, global `n8n`), plus a fifth entry
  `bunnycam-water` (Python interpreter, `server/sensor/water_sensor.py`,
  initially not started).

**Verify:** on Windows, `npm install` in `server/`, `pm2 restart all`,
dashboard loads, agent log shows normal frames. Nothing regressed.

#### 4. Performance spike on the Pi (the step most likely to sink Phase A)

Clone the repo on the Pi, `npm install` in `server/`. Point the agent at the
still-running PC server (`AGENT_SERVER_URL=http://<pc-ip>:3001` in a Pi-local
`.env`, camera lines swapped to `CAMERA_FORMAT=v4l2`,
`CAMERA_INPUT=/dev/video0`, `FFMPEG_PATH=/usr/bin/ffmpeg`). Run
`node agent/capture.mjs` for 30 minutes.

Measure:

- Frame log lines hold the 1.5s cadence (the `busy` flag silently drops
  frames when inference is slow, so stretched spacing = too slow).
- `top`: ffmpeg + node combined comfortably inside the 4 cores.
- `vcgencmd measure_temp` stays under 80C sustained.

Fallback ladder if inference is too slow, in order: add
`@tensorflow/tfjs-backend-wasm`; raise `AGENT_FRAME_SECONDS` to 2-3 (note the
8-frame window then spans 16-24s instead of the ~12s the classifier was
trained on; acceptable degradation); worst case run motion-capture-only with
the classifier disabled and behavior alerts lost.

**Verify:** steady cadence in logs, and a hand wave in front of the Pi's
camera saves a clip that appears in the PC server's gallery.

#### 5. Move the server and its data

Copy from PC to Pi: `server/recordings/`, `server/labels.json`,
`server/backups/`, `server/model/`, `server/hidden.json`,
`server/autolabels.json`, and `.env` (with the three Pi camera/ffmpeg lines
from step 4). Build the client on the Pi (`cd client && npm install && npm
run build`) or copy `client/dist/`. Start server + agent under pm2 with
`ecosystem.pi.config.cjs`.

**Verify:** `http://<pi-ip>:3001` shows the dashboard with the full old
recording gallery, live feed streams, login works, agent uploads a motion
clip end to end on the Pi alone.

#### 6. Move the Cloudflare tunnel

Install arm64 cloudflared from Cloudflare's apt repo. Copy
`C:\Users\joepa\.cloudflared\cert.pem` and the tunnel credentials JSON
(`<tunnel-id>.json`) to `~/.cloudflared/` on the Pi. Stop the tunnel process
on the PC, start `bunnycam-tunnel` on the Pi.

**Verify:** https://bunny-tracker.app loads from a phone on cellular while
the PC's tunnel process is stopped.

#### 7. Move n8n (or drop it)

`sudo npm i -g n8n` on the Pi. Export the workflow and credentials from the
PC's n8n UI; also copy the `encryptionKey` from the PC's `~/.n8n/config`
(without it, imported credentials cannot decrypt). Import on the Pi, run
under pm2 with the same env block from the ecosystem file. If this fights
back for more than an hour, unset `ALERT_WEBHOOK_URL` in `.env` instead and
let the built-in email path take over; revisit later.

**Verify:** force a motion event, n8n execution log shows a run, email lands
at the hotmail address.

#### 8. Autostart, burn-in, decommission

`pm2 save && pm2 startup systemd` (run the printed command). Reboot the Pi:
everything comes back. Kill the PC stack (`pm2 kill`) and remove
`start-bunnycam.bat` from Windows startup, but keep the PC checkout intact
for two weeks as rollback. Burn in for a week: check `pm2 list` uptimes,
temps, and disk usage a few times.

**Verify:** after a Pi reboot with the PC off, the public site is live and a
real bunny clip is captured overnight without intervention.

### Phase B: Water and environment tracking

#### 9. Wire the sensor and spike the data (riskiest step of Phase B)

Sandwich the load cell between the two plates, bowl on top. HX711 to GPIO:
VCC to 3.3V, GND to GND, DT and SCK to two free GPIO pins. Write
`server/sensor/read_hx711.py` as a throwaway: tare, calibrate the scale
factor against a known weight (weigh a full water bottle on a kitchen scale
first), then print grams once per second to a CSV. Let it log for 24 hours
with the rabbits living normally.

This spike answers the questions the real service depends on: baseline noise
and drift magnitude, what a drink looks like in the data, what a refill looks
like, and what a rabbit bumping or leaning on the platform looks like.

**Verify:** the 24h CSV shows drift under a few grams, at least one refill as
a clean positive step, drinking as small negative steps, and bumps as short
spikes that return to baseline within seconds.

#### 10. Sensor service

`server/sensor/water_sensor.py`: sample the load cell at 1Hz, keep a
30-sample rolling median (kills bump spikes), read the BME280 over I2C
(enable I2C via `raspi-config`; wire VCC/GND/SDA/SCL; no calibration needed),
and POST once per 30s to `POST /api/water/reading` as
`{ "grams": <n>, "tempC": <n>, "humidity": <n>, "at": <iso> }` with an
`X-Water-Secret` header. Add `WATER_SENSOR_SECRET` to `.env`; a shared secret
is simpler and more appropriate for a headless GPIO client than cookie auth.
Enable the `bunnycam-water` pm2 entry from step 3. Mount the BME280 near the
rabbits' level, away from direct sun and at least 30cm from the Pi, which
runs warm enough to skew a sensor sitting next to it.

**Verify:** `pm2 logs bunnycam-water` shows readings flowing with plausible
grams, tempC, and humidity; server log shows 200s; a wrong secret gets 401;
tempC roughly matches a household thermometer in the same spot.

#### 11. Server route and storage

`server/routes/water.js`, mounted in `index.js` like the existing routes:

- `POST /api/water/reading` (secret-guarded): append to
  `server/water/YYYY-MM.jsonl`.
- `GET /api/water/summary` (admin-guarded, same `adminGuard` as other
  routes): current grams, today's intake ml, refill count, 7 daily totals,
  plus current tempC/humidity and today's min/max of each. Intake = sum of
  negative steps between readings, ignoring steps larger than a refill
  threshold in either direction. Refill = positive step over 100g. Tunable
  constants at the top of the file, informed by the step 9 spike data.

**Verify:** a small script replays a synthetic day (baseline, 8 drinks of
10-20g, one 300g refill, a few bump spikes) through the POST endpoint;
summary reports intake within 10% of the known total and exactly one refill.

#### 12. Alerts

Checked in the reading handler, reusing the existing notification paths (the
n8n webhook with a `type: "water"` field, or the Nodemailer fallback,
mirroring the routing in `capture.mjs`):

- Low level: current grams below `LOW_LEVEL_G` (default 150). At most one
  alert per 12h.
- Low intake: trailing 24h intake below 50% of the trailing 7-day daily
  average (needs 3+ days of data before arming). At most one per day. This is
  the health alert, so it is relative to the rabbits' own baseline, not an
  absolute number.
- High temperature: warning at `TEMP_WARN_C` (default 27), urgent at
  `TEMP_DANGER_C` (default 29), each at most once per 2h. Unlike the intake
  alert these are absolute thresholds; heat stress does not care what last
  week looked like. Optional low-temp warning (default 10) for winter.

**Verify:** temporarily set `LOW_LEVEL_G` above the current level and
`TEMP_WARN_C` below room temperature; exactly one email arrives for each,
and a second does not within the cooldown.

#### 13. Dashboard panel

`client/src/components/WaterPanel.jsx` on the Camera tab (admin view only,
consistent with keeping live data off the public demo): current level as a
percentage of the last refill, today's intake in ml, a 7-day bar chart using
plain CSS bars in the existing style system (the project has no chart
library; do not add one for seven bars), and current temperature/humidity
with today's min/max.

**Verify:** panel numbers match `GET /api/water/summary`; refill the bowl and
the level updates within a minute; temperature updates after warming the
sensor in your hand.

## Risks

1. **Inference or encoding too slow on the Pi.** Confronted first (step 4)
   before anything is migrated, with a written fallback ladder. Worst case
   preserves motion capture, which is the system's core.
2. **Load-cell data too noisy for meaningful intake numbers.** Rabbits step
   on platforms and bump bowls; cheap load cells drift with temperature. The
   24h raw-data spike (step 9) runs before any service code is written, the
   median filter and step-detection are designed from that real data, and the
   health alert compares the rabbits against their own baseline so absolute
   accuracy matters less. If the spike shows the data is unusable, the
   fallback is level-only tracking (refill reminders, no intake), which the
   same hardware still supports.
3. **Credential migration snags (tunnel cert, n8n encryption key).** The PC
   stack stays intact and startable until the Pi passes burn-in, so rollback
   is `pm2 start` on the PC. n8n is explicitly droppable (step 7) since the
   email fallback is already wired into the agent.
