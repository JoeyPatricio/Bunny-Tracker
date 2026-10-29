# Raspberry Pi deployment

This guide installs the server, capture agent, and Cloudflare tunnel on a
Raspberry Pi 5 running 64-bit Raspberry Pi OS. systemd runs all three
services. It was first run on 2026-10-29, on Raspberry Pi OS Lite (Debian 13)
with an Arducam B0590 USB Day/Night camera, and the steps below are the ones
that worked. The Pi is production. The Linux PC is the clip archive and the
training machine (see [Pulling data to the PC](#12-pulling-data-to-the-pc)).
There is no second production machine to roll back to.

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
| `pull-data.sh` | run from the repo on the PC |

The units assume a `bunnycam` system user and a checkout at
`/opt/bunny-tracker`. If you choose other values, change them in every unit
and in `bunnycam.env`.

Run the commands one at a time. Pasting a block lets the first command that
reads the terminal (a `sudo` password prompt, `curl | sh`) swallow the lines
after it, and they never run.

## 1. Base system

Flash Raspberry Pi OS Lite (64-bit) with Raspberry Pi Imager, and set the
hostname, user, Wi-Fi, and SSH in its OS customisation. If
`<hostname>.local` doesn't answer, the PC probably has no mDNS resolver
(Arch ships without one), so find the Pi by IP. Give it a DHCP reservation in
the router.

```bash
uname -m    # must print aarch64, onnxruntime has no 32-bit ARM wheel
sudo apt update && sudo apt install -y ffmpeg git curl v4l-utils rsync
sudo useradd --system --create-home --home-dir /var/lib/bunnycam \
    --user-group --shell /usr/sbin/nologin --groups video bunnycam
sudo git clone https://github.com/JoeyPatricio/Bunny-Tracker.git /opt/bunny-tracker
sudo chown -R bunnycam:bunnycam /opt/bunny-tracker
```

The repository is public, so the clone needs no credentials. Use the HTTPS
URL. The Pi has no GitHub SSH key. `rsync` is for `pull-data.sh`.

If you SSH in from a terminal the Pi has no terminfo for (kitty, for one),
editors and pagers fail with `Error opening terminal`. Install its terminfo
(`sudo apt install -y kitty-terminfo`) or prefix the command with
`TERM=xterm-256color`.

Install cloudflared for arm64 from [Cloudflare's package
repository](https://pkg.cloudflare.com/index.html), which puts it at
`/usr/bin/cloudflared`:

```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install -y cloudflared
```

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

The last line must list `CPUExecutionProvider`. `AzureExecutionProvider`
ships with the package and is unused. uv installs the interpreter under
`/var/lib/bunnycam/.local/share/uv/python/`, and the venv links to it, so
don't delete that directory. Runtime installs leave out the `dev` and `train`
groups. The installer's advice to add `~/.local/bin` to `PATH` doesn't apply:
the commands call uv by its full path, and `bunnycam` has no login shell.

## 3. Camera

```bash
ls -l /dev/v4l/by-id/
ffmpeg -hide_banner -f v4l2 -list_formats all -i /dev/v4l/by-id/<camera>-video-index0
```

The second command should list `mjpeg` at `1280x720`. A UVC camera exposes
several indexes. `video-index0` is the picture, the others are metadata. Use
the by-id path for `CAMERA_INPUT` in step 5. `/dev/video0` can point at a
different device after a USB re-enumeration. The agent unit adds the `video`
group, which is what grants access to the device.

To see a frame before anything else is installed:

```bash
ffmpeg -hide_banner -f v4l2 -input_format mjpeg -video_size 1280x720 \
    -i /dev/v4l/by-id/<camera>-video-index0 -frames:v 1 -update 1 ~/test.jpg
```

Don't point an IR camera through glass or clear plastic. The IR LEDs reflect
back and wash out the night image.

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

Check with `sudo ls -ld /run/bunnycam /run/bunnycam/segments`. Both are
`drwxr-x--- bunnycam bunnycam`, so without `sudo` your login user gets
`Permission denied`, which is intended.

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

## 5. Data, web app, and .env

`client/dist` is gitignored, so the clone has no web app. Without it every
page is a 500. The Pi has no Node, so build it on the PC. The PC's `data/` is
the archive `pull-data.sh` keeps, which is what a reinstall restores from.
From the repo root on the PC:

```bash
tar -czf data.tgz --exclude data/.env --exclude data/agent/segments \
    --exclude data/tmp --exclude "data/sessions.json*" data
(cd client && npm ci && npm run build)
tar -czf client-dist.tgz client/dist
scp data.tgz client-dist.tgz <user>@<pi>:~/
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
rm ~/data.tgz ~/client-dist.tgz
sudo ls -l --time-style=full-iso data/recordings | head
```

The archives sit in your login home, which the `bunnycam` user can't read,
so root extracts them and `chown` hands the result to the service user.
`tar` keeps file modification times on both ends. Some capture-time fallbacks
read a recording's mtime, so check that the listing shows the original dates
and not today's. Any copy method that resets mtimes loses that information.

The secrets live only on the Pi, so a fresh install writes a new `data/.env`
from the template in the top-level README:

```bash
sudo install -o bunnycam -g bunnycam -m 600 /dev/null data/.env
sudoedit data/.env
```

- `CAMERA_FORMAT=v4l2` and `CAMERA_INPUT=/dev/v4l/by-id/...` from step 3.
- `ADMIN_PASSWORD` in double quotes, with no `"` or `\` inside. A new
  `AGENT_TOKEN` from `python3 -c "import secrets; print(secrets.token_hex(32))"`.
- `EMAIL_USER`, `EMAIL_PASS`, and `NOTIFY_TO` for alerts. With
  `ALERT_WEBHOOK_URL` unset, the agent posts to `/api/notify`, which sends
  through `smtplib`.
- Leave out `PY_PORT`, `AGENT_SERVER_URL`, `FFMPEG_PATH`, and the two tmpfs
  paths. `bunnycam.env` sets them, and process environment wins over
  `data/.env`.

python-dotenv skips a line it can't parse with only a warning, so a broken
`ADMIN_PASSWORD` line shows up as `ADMIN_PASSWORD not set in data/.env` at
login. Parse the file the way the server does, printing lengths, not values:

```bash
sudo -u bunnycam server/.venv/bin/python -c "from dotenv import dotenv_values as d; print({k: (len(v) if v else repr(v)) for k, v in d('data/.env').items()})"
```

Every key should appear with a length, and nothing should print `could not
parse statement`.

## 6. Move a promoted model

Skip this until a seven-class model has passed its gate and been promoted on
the PC. The agent loads nothing without `promoted.json`, and a graph
exported with external data fails to load without its `.onnx.data` sidecar.
`model_bundle.py` packs `promoted.json` plus every artifact its manifest lists,
after checking each file's sha256. From the repo root on the PC:

```bash
server/.venv/bin/python deploy/pi/model_bundle.py pack server/models model-bundle.tar.gz
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
restart counter of 1 right after enabling is expected. On stop, the agent gets
SIGTERM and shuts down its ffmpeg child. Anything left in the
unit after that, or after 20 seconds, gets SIGKILL. A restart never meets an
old ffmpeg still holding the camera.

If `monitor.json` says monitoring is off, the agent logs `Monitoring is OFF`
and leaves the camera idle until you switch it on in the dashboard.

Before the tunnel is up, reach the dashboard through SSH from the PC:

```bash
ssh -L 3001:127.0.0.1:3001 <user>@<pi>
```

and open http://localhost:3001. Make sure nothing on the PC already listens
on port 3001. If something does, the forward binds only `[::1]` and the
browser alternates between the two servers.

## 8. Tunnel

The server listens on `127.0.0.1:3001` only, and uvicorn applies
`X-Forwarded-For` and `X-Forwarded-Proto` only from loopback peers. cloudflared
on the same host is the one trusted proxy.

There are two ways to get the tunnel credentials. Reusing an existing tunnel
keeps DNS untouched. Cloudflare can't reissue a credentials file, so if it's
lost, create a new tunnel and repoint DNS.

**Reuse an existing tunnel.** Copy its `<tunnel-id>.json` to your home on the
Pi. `cert.pem` is needed only to create, route, or delete tunnels, not to run
one.

**Create a new tunnel.** On the Pi, as your login user:

```bash
cloudflared tunnel login        # open the URL on any machine, pick the zone
cloudflared tunnel create bunnycam-pi
cloudflared tunnel route dns --overwrite-dns bunnycam-pi bunny-tracker.app
```

`create` prints the tunnel ID and writes `~/.cloudflared/<tunnel-id>.json`.
`--overwrite-dns` replaces the CNAME that points at the old tunnel. Keep
`~/.cloudflared/cert.pem` for managing tunnels later. Delete the old tunnel
with `cloudflared tunnel list` and `cloudflared tunnel delete <name>`.

Either way, install the credentials and config:

```bash
sudo install -d -o root -g bunnycam -m 750 /etc/cloudflared
sudo install -o root -g bunnycam -m 640 ~/<path>/<tunnel-id>.json /etc/cloudflared/
sed 's/<TUNNEL-ID>/<tunnel-id>/g' deploy/pi/cloudflared-config.example.yml | sudo tee /etc/cloudflared/config.yml >/dev/null
sudo chown root:bunnycam /etc/cloudflared/config.yml && sudo chmod 640 /etc/cloudflared/config.yml
sudo cat /etc/cloudflared/config.yml
```

Add any other hostnames to the ingress list. Point each ingress `service` at
`http://127.0.0.1:3001`, not `localhost`. On the Pi, `localhost` can resolve
to `::1`, where nothing listens. Remove the extra credentials copy from your
home once the tunnel runs.

Two connectors for one tunnel both receive traffic, so another live connector
would hide a broken Pi. Stop any other connector for the same tunnel, then:

```bash
sudo systemctl enable --now bunnycam-tunnel
systemctl --no-pager status bunnycam-tunnel    # expect "Registered tunnel connection"
```

Check https://bunny-tracker.app from a phone on cellular, and log in through
it. A Cloudflare 530 means the tunnel has no connector.

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
systemctl --no-pager is-active bunnycam-server bunnycam-agent bunnycam-tunnel
sudo ls /run/bunnycam/segments
journalctl -u bunnycam-agent --since -1h | grep -ci restart
vcgencmd measure_temp
df -h /opt/bunny-tracker
```

The public site should be live, segments should appear under `/run/bunnycam`
while monitoring is on, and persistent storage should grow only by saved
clips. The Clipping panel in the dashboard shows free disk and the unlabeled
count.

## 11. Updating the Pi

```bash
cd /opt/bunny-tracker
sudo -u bunnycam git pull
```

Then, depending on what the pull changed:

- `server/uv.lock`: rerun the `uv sync` line from step 2.
- `deploy/pi/*.service`: copy the units again and run `sudo systemctl daemon-reload`.
- Anything under `server/`: `sudo systemctl restart bunnycam-server bunnycam-agent`.
- Anything under `client/`: rebuild `client/dist` on the PC and copy it over
  as in step 5. The server reads it per request, so no restart is needed.

## 12. Pulling data to the PC

The PC keeps the clip archive and trains from it. Labeling happens on the
Pi, in the dashboard. Before a training run, from the repo root on the PC:

```bash
deploy/pi/pull-data.sh                 # from jp@10.0.0.150
deploy/pi/pull-data.sh <user>@<pi>     # or BUNNYCAM_PI=<user>@<pi>
```

It copies the Pi's `data/` into the PC's `data/` one way, keeping
modification times. The Pi's labels replace the PC's. Nothing is deleted on
the PC, so a clip later removed from the Pi stays in the archive. `data/.env`,
`sessions.json`, `tmp/`, and rolling segments stay on the Pi. It needs
`rsync` on both machines.
