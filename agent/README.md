# Capture agent

Back to the [project overview](../README.md) or the
[source map](../docs/development.md#source-map).

`bunnycam-agent` reads the camera through one ffmpeg process, runs motion
detection and the promoted ONNX model on sampled frames, and talks to the
FastAPI server: live view, predictions, clip uploads and alerts. It reads
`data/.env` and the process environment, and loads `server/models/`. Linux
only (`CAMERA_FORMAT` is `v4l2` or `lavfi`).

| Module | Does |
|---|---|
| `src/motion` | Pixel-diff motion level, hot cells and lighting changes |
| `src/inference` | Frame normalize, ONNX backbone and temporal head, `promoted.json` checks |
| `src/decision` | Per-frame decision: alert, motion capture or clip, with streaks and cooldowns |
| `src/ffmpeg` | ffmpeg arguments, process lifecycle, raw frames and the MJPEG live view on fd 3 |
| `src/server` | HTTP client, uploads, the pending-clip queue and segment pruning |
| `src/main.cpp` | Configuration, the monitor state machine, threads and signals |

The agent replaced a Python agent that used to live in `server/agent/`. The
fixtures in `tests/fixtures/` were recorded from it, and the tests still
require the same behaviour: motion levels to 1e-9, every decision exactly,
embeddings and probabilities to 1e-4, and ffmpeg arguments exactly. The
recorder is gone, so a deliberate behaviour change means editing the
fixtures by hand alongside the code.

The inference fixture uses `backbone_fp32.onnx`, not the deployed int8
backbone. ONNX Runtime's int8 kernels saturate on x86 CPUs without VNNI, so
int8 output depends on the machine. The tests only check that the int8 model
loads and gives a finite 1280-wide embedding.

## Build and test

Packages: `cmake g++ libssl-dev nlohmann-json3-dev catch2 unzip` on Debian
(`cmake nlohmann-json catch2 unzip` on Arch). cpp-httplib is vendored in
`third_party/`. ONNX Runtime 1.29.0 comes from Microsoft's NuGet package:

```bash
cd agent
scripts/fetch-onnxruntime.sh          # into ~/.cache/bunnycam/onnxruntime-1.29.0
cmake --preset default && cmake --build --preset default -j && ctest --preset default
```

The `asan` (ASan+UBSan) and `tsan` presets build into `build-asan/` and
`build-tsan/`. CI runs all three. Run `clang-format -i` on anything you touch.

The binary finds the repo from its own path (`<repo>/agent/<build>/`), so
it must stay inside the checkout. `./run.sh` and the Pi's
`bunnycam-agent.service` both run `agent/build/bunnycam-agent`.

## Latency

Each frame log line carries `t=<ms>`, measured from the moment the reader
has a whole raw frame to the moment the decision returns. That covers motion,
normalize, backbone, head and decision, not uploads.

```bash
journalctl -u bunnycam-agent --since "1 hour ago" | agent/scripts/latency_summary.sh
```

With no promoted model the agent runs motion-only, so `t=` measures motion
alone. Rerun after a 7-class model is promoted to get the full
capture-to-prediction figure.

## Switching the Pi over

This is a one-time step while the Pi still runs the Python agent from
`main`. The Pi is production and there is no second machine, so take the
baseline first, and keep `main` as the rollback.

### 1. Baseline the Python agent

Record the C++ run later over the same hours of day.

```bash
PID=$(systemctl show -p MainPID --value bunnycam-agent)
# RSS (kB) and cumulative CPU seconds, once a minute for 24 h.
for i in $(seq 1440); do
  echo "$(date +%s) $(ps -o rss=,cputimes= -p "$PID")"; sleep 60
done > ~/baseline-python.txt
```

Average CPU is the difference in CPU seconds over the difference in wall
seconds. Also record:

- Live-view latency. Put a phone stopwatch in front of the camera and
  photograph it beside the dashboard's live view a few times. The delay is
  the difference between the two readings.
- Clips per hour: `journalctl -u bunnycam-agent --since today | grep -c 'saved unlabeled'`,
  divided by the hours covered.

### 2. Build and swap

The branch has to be on GitHub for the Pi to fetch it.

```bash
cd /opt/bunny-tracker
sudo -H -u bunnycam git fetch origin capture-agent
sudo -H -u bunnycam git checkout capture-agent
```

Build and test as in step 2 of the [Pi guide](../deploy/pi/README.md), then
install the new unit. The server restarts too, to pick up its side of the
branch:

```bash
sudo cp deploy/pi/bunnycam-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart bunnycam-server bunnycam-agent
journalctl -u bunnycam-agent -f       # expect "BunnyCam Agent starting..."
```

The Python agent logged `BunnyCam Python Agent starting...` instead.

Rollback:

```bash
cd /opt/bunny-tracker
sudo -H -u bunnycam git checkout main
sudo cp deploy/pi/bunnycam-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart bunnycam-server bunnycam-agent
```

### 3. Burn-in, 72 h

- No unrecovered crash. `systemctl show -p NRestarts bunnycam-agent` stays
  at 0, and `journalctl -u bunnycam-agent | grep -E 'failed to start|loop failed'`
  finds nothing.
- No orphaned ffmpeg. Restart the agent 5 times
  (`sudo systemctl restart bunnycam-agent`), and after each one
  `pgrep -c -x ffmpeg` prints 1.
- Toggle monitoring and Clipping Mode from the dashboard a few times. The
  live view comes back each time, and clips land in Label Studio.
- Record RSS and CPU with the step 1 loop (into `~/burnin-agent.txt`), plus
  live-view latency, clips per hour, and `latency_summary.sh` p50 and p95.
  Put them next to the baseline.

Merge `capture-agent` into `main` once the burn-in passes.
