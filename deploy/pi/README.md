# Raspberry Pi deployment

This guide installs the server, capture agent, and Cloudflare tunnel on a
Raspberry Pi 5 running 64-bit Raspberry Pi OS. systemd runs all three
services. The Windows PC keeps its PM2 setup (`ecosystem.config.cjs`) as the
rollback path. Nothing here has been run on Pi hardware yet.

There is no usable classifier yet. No seven-class model has been promoted, so
the agent runs motion-only. Clipping Mode, motion clips, labeling, the live
view, and manual recording all work without one. Step 6 applies once a model
has been promoted on the PC. The Pi never trains.

| File | Installed as |
|---|---|
| `bunnycam-server.service`, `bunnycam-agent.service`, `bunnycam-tunnel.service` | `/etc/systemd/system/` |
| `bunnycam.env` | read in place by the server and agent units |
| `bunnycam.tmpfiles` | `/etc/tmpfiles.d/bunnycam.conf` |
| `journald-bunnycam.conf` | `/etc/systemd/journald.conf.d/bunnycam.conf` |
| `cloudflared-config.example.yml` | `/etc/cloudflared/config.yml`, edited |
| `model_bundle.py` | run from the repo on the PC and on the Pi |

The units assume a `bunnycam` system user and a checkout at
`/opt/bunny-tracker`. If you choose other values, change them in every unit
and in `bunnycam.env`.

## 1. Base system

```bash
uname -m    # must print aarch64, onnxruntime has no 32-bit ARM wheel
sudo apt update && sudo apt install -y ffmpeg git curl v4l-utils
sudo useradd --system --create-home --home-dir /var/lib/bunnycam \
    --user-group --shell /usr/sbin/nologin --groups video bunnycam
sudo git clone <repo> /opt/bunny-tracker
sudo chown -R bunnycam:bunnycam /opt/bunny-tracker
```

Install cloudflared for arm64 from [Cloudflare's package
repository](https://pkg.cloudflare.com/index.html), which puts it at
`/usr/bin/cloudflared`.

## 2. Python runtime

The project requires Python 3.12 to 3.14, and the lock pins NumPy and ONNX
Runtime builds that have no cp311 wheels. Don't build the venv from the OS
`python3`. Install uv for the service user and let it fetch the same
interpreter the README uses:

```bash
sudo -u bunnycam -H sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
cd /opt/bunny-tracker/server
sudo -u bunnycam -H /var/lib/bunnycam/.local/bin/uv sync --frozen --python 3.14
sudo -u bunnycam .venv/bin/python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

The last line must list `CPUExecutionProvider`. uv installs the interpreter
under `/var/lib/bunnycam/.local/share/uv/python/`, and the venv links to it,
so don't delete that directory. Runtime installs leave out the `dev` and
`train` groups. After a `git pull` that changes `uv.lock`, rerun the `uv sync`
line and restart both Python services.

## 3. Camera

```bash
ls -l /dev/v4l/by-id/
ffmpeg -hide_banner -f v4l2 -list_formats all -i /dev/v4l/by-id/<camera>-video-index0
```

The second command should list `mjpeg` at `1280x720`. Use the by-id path for
`CAMERA_INPUT` in step 5. `/dev/video0` can point at a different device after
a USB re-enumeration. The agent unit adds the `video` group, which is what
grants access to the device.

## 4. System configuration

```bash
cd /opt/bunny-tracker
sudo cp deploy/pi/bunnycam.tmpfiles /etc/tmpfiles.d/bunnycam.conf
sudo systemd-tmpfiles --create /etc/tmpfiles.d/bunnycam.conf
sudo mkdir -p /etc/systemd/journald.conf.d
sudo cp deploy/pi/journald-bunnycam.conf /etc/systemd/journald.conf.d/bunnycam.conf
sudo systemctl restart systemd-journald
sudo cp deploy/pi/bunnycam-*.service /etc/systemd/system/
sudo systemctl daemon-reload
```

`/run` is tmpfs. tmpfiles recreates `/run/bunnycam/segments` at every boot,
before any service starts. `bunnycam.env` points both `AGENT_SEGMENT_DIR` and
`AGENT_LIVE_FRAME_PATH` there. The server reads the same file, so the manual
Record endpoint finds the segments the agent wrote. The agent prunes segments
older than 10 minutes once a minute, so the directory holds up to about 11
minutes of video. That's roughly 80 MB at the nominal 1 Mbit/s, which fits
in `/run`'s default tmpfs on a Pi 5. Check with `df -h /run` during burn-in.

Logs go to the journal (`journalctl -u bunnycam-agent`). The journald drop-in
caps the entire system journal at 200 MB and one month, not only the bunnycam
units. That fits a Pi dedicated to this stack. Change it if the Pi runs
other services whose logs need different retention. `run.sh` is a foreground
launcher for the dev box and appends to `logs/` with no rotation, so don't use
it on the Pi.

Copy the units again and run `daemon-reload` whenever a pull changes them.

## 5. Move the data

Stop the Bunny Tracker apps on the PC first so labels and recordings don't
change during the copy. Name them, so other PM2 apps on the PC keep running:

```powershell
pm2 stop bunnycam-server bunnycam-agent bunnycam-tunnel bunnycam-n8n
```

Leave out `bunnycam-n8n` if it isn't installed. Remove
`start-bunnycam.bat` from Windows startup as well, or the next login starts
them again. Then, from the repo root on the PC in PowerShell (Windows ships
`tar` and `scp`):

```powershell
tar -czf data.tgz --exclude data/.env --exclude data/agent/segments `
    --exclude data/tmp --exclude "data/sessions.json*" data
cd client; npm ci; npm run build; cd ..
tar -czf client-dist.tgz client/dist
scp data.tgz client-dist.tgz data/.env <user>@<pi>:~/
```

The archive carries the whole durable `data/` tree: recordings, `labels.json`
and its `labels.meta.json`, `labels.reliability.json`,
`labels.reliability_sample.json`, and `labels.holdout.json` sidecars, autolabels,
hidden, monitor state, the activity log, and backups. Rolling segments are
temporary and stay behind. `sessions.json` holds bearer-equivalent session
tokens and also stays behind, so you log in again on the Pi.

On the Pi:

```bash
cd /opt/bunny-tracker
sudo tar --no-same-owner -xzf ~/data.tgz -C /opt/bunny-tracker
sudo tar --no-same-owner -xzf ~/client-dist.tgz -C /opt/bunny-tracker
sudo chown -R bunnycam:bunnycam data client/dist
sudo install -o bunnycam -g bunnycam -m 600 ~/.env data/.env
rm ~/data.tgz ~/client-dist.tgz ~/.env
ls -l --time-style=full-iso data/recordings | head
```

The archives sit in your login home, which the `bunnycam` user can't read,
so root extracts them and `chown` hands the result to the service user.
`tar` keeps file modification times on both ends. Some capture-time fallbacks
read a recording's mtime, so check that the listing shows the original dates
and not today's. Any copy method that resets mtimes loses that information.

Edit `data/.env` for the Pi:

- `CAMERA_FORMAT=v4l2` and `CAMERA_INPUT=/dev/v4l/by-id/...` from step 3.
  `CAMERA_SIZE` and `CAMERA_FPS` keep their meaning.
- Email: to send alerts directly over SMTP with no n8n, delete
  `ALERT_WEBHOOK_URL`. The agent then posts to `/api/notify`, which sends
  through `smtplib` with `EMAIL_USER`, `EMAIL_PASS`, and `NOTIFY_TO`. A
  webhook URL copied from the PC still points at the PC's n8n.
- `bunnycam.env` sets `PY_PORT`, `AGENT_SERVER_URL`, `FFMPEG_PATH`, and the two
  tmpfs paths. Process environment wins over `data/.env`, so remove those
  keys from `data/.env` to keep one source.

## 6. Move a promoted model

Skip this until a seven-class model has passed its gate and been promoted on
the PC. The agent loads nothing without `promoted.json`, and a graph
exported with external data fails to load without its `.onnx.data` sidecar.
`model_bundle.py` packs `promoted.json` plus every artifact its manifest lists,
after checking each file's sha256. From the repo root on the PC:

```powershell
server\.venv\Scripts\python.exe deploy\pi\model_bundle.py pack server\models model-bundle.tar.gz
scp model-bundle.tar.gz <user>@<pi>:~/
```

On the Pi:

```bash
cd /opt/bunny-tracker
sudo systemctl stop bunnycam-agent
sudo tar --no-same-owner -xzf ~/model-bundle.tar.gz -C server/models
sudo chown -R bunnycam:bunnycam server/models
sudo -u bunnycam server/.venv/bin/python deploy/pi/model_bundle.py verify server/models \
    && sudo systemctl start bunnycam-agent
rm ~/model-bundle.tar.gz
journalctl -u bunnycam-agent -n 20    # expect "Models ready"
```

The agent stays stopped until the new files verify. If `verify` fails, copy
the bundle again before starting the agent.

## 7. Start the server and agent

```bash
sudo systemctl enable --now bunnycam-server bunnycam-agent
journalctl -u bunnycam-agent -f
```

The agent exits if the server isn't answering yet or the camera isn't
enumerated, and systemd restarts it every 5 seconds with no retry limit. A
few connection errors right after boot are expected. On stop, the agent gets
SIGTERM and shuts down its ffmpeg child. Anything left in the
unit after that, or after 20 seconds, gets SIGKILL. A restart never meets an
old ffmpeg still holding the camera.

## 8. Move the tunnel

The server listens on `127.0.0.1:3001` only, and uvicorn applies
`X-Forwarded-For` and `X-Forwarded-Proto` only from loopback peers. cloudflared
on the same host is the one trusted proxy.

From the PC, copy the tunnel's credentials file
(`%USERPROFILE%\.cloudflared\<tunnel-id>.json`) and note the ingress rules in
its `config.yml`. `cert.pem` is needed only to create, route, or delete
tunnels, not to run one. Then on the Pi:

```bash
sudo install -d -o root -g bunnycam -m 750 /etc/cloudflared
sudo install -o root -g bunnycam -m 640 ~/<tunnel-id>.json /etc/cloudflared/
sudo install -o root -g bunnycam -m 640 deploy/pi/cloudflared-config.example.yml /etc/cloudflared/config.yml
sudoedit /etc/cloudflared/config.yml    # tunnel ID, credentials path, PC's hostnames
rm ~/<tunnel-id>.json
```

Point each ingress `service` at `http://127.0.0.1:3001`, not `localhost`. On
the Pi, `localhost` can resolve to `::1`, where nothing listens.

Two connectors for one tunnel both receive traffic, so a live PC connector
would hide a broken Pi. Make sure the PC's `bunnycam-tunnel` is still
stopped from step 5 (`pm2 list`), then:

```bash
sudo systemctl enable --now bunnycam-tunnel
```

Check https://bunny-tracker.app from a phone on cellular, and log in through
it.

## 9. Optional direct LAN access

Skip this unless you need `http://<pi-ip>:3001` on the home network.
Run `sudo systemctl edit bunnycam-server` and add:

```ini
[Service]
ExecStart=
ExecStart=/opt/bunny-tracker/server/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 3001 --proxy-headers --forwarded-allow-ips=127.0.0.1,::1
```

Keep `--forwarded-allow-ips` at loopback. LAN clients then can't spoof their
address to the login rate limiter. The server adds the CSP
`upgrade-insecure-requests` directive only to requests it sees as HTTPS, so
plain-HTTP LAN pages should load their subresources over HTTP. Nobody has
tested that in a browser against the Pi yet. Check it before you rely on LAN
access.

## 10. Reboot test and burn-in

```bash
sudo reboot
# after it comes back, with no login:
systemctl status bunnycam-server bunnycam-agent bunnycam-tunnel
ls /run/bunnycam/segments
journalctl -u bunnycam-agent --since -1h | grep -ci restart
vcgencmd measure_temp
df -h /opt/bunny-tracker
```

The public site should be live with the PC off, segments should appear under
`/run/bunnycam`, and persistent storage should grow only by saved clips. Keep
the PC checkout intact for two weeks as rollback. To roll back, stop and
disable the three Pi units, then on the PC run `pm2 start ecosystem.config.cjs`
and put `start-bunnycam.bat` back in Windows startup.
