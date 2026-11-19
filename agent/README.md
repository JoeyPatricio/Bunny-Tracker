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

## Pi switch-over and burn-in

The Pi switched from the Python agent to this one on 2026-11-18 at 16:16
PST, at 323369d. The build and `ctest` ran on the Pi as in step 2 of the
[Pi guide](../deploy/pi/README.md), and all 49 tests passed. No baseline of
the Python agent was recorded before the switch.

`main` no longer has the Python agent. The last commit the Pi ran with it is
a39ca21, so roll back to that, along with its unit file:

```bash
cd /opt/bunny-tracker
sudo -H -u bunnycam git checkout a39ca21
sudo cp deploy/pi/bunnycam-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart bunnycam-server bunnycam-agent
journalctl -u bunnycam-agent -f       # expect "BunnyCam Python Agent starting..."
```

The C++ agent logs `BunnyCam Agent starting...` instead.

### Burn-in, 72 h

Restarting the server and agent together makes the agent fail once with
`Agent failed to start: Could not establish connection`, because it starts
before uvicorn is listening. systemd restarts it 5 seconds later, and
`NRestarts` reads 1 from then on. That is expected and not a crash.

- No unrecovered crash. `systemctl show -p NRestarts bunnycam-agent` stays
  where it was at the start of the burn-in, and
  `journalctl -u bunnycam-agent --since "2026-11-18 16:17" | grep -E 'failed to start|loop failed'`
  finds nothing.
- No orphaned ffmpeg. Restart the agent alone 5 times
  (`sudo systemctl restart bunnycam-agent`), and after each one
  `pgrep -c -x ffmpeg` prints 1.
- Toggle monitoring and Clipping Mode from the dashboard a few times. The
  live view comes back each time, and clips land in Label Studio.
- Record RSS and CPU once a minute for 24 h:

  ```bash
  PID=$(systemctl show -p MainPID --value bunnycam-agent)
  for i in $(seq 1440); do
    echo "$(date +%s) $(ps -o rss=,cputimes= -p "$PID")"; sleep 60
  done > ~/burnin-agent.txt
  ```

  Average CPU is the difference in CPU seconds over the difference in wall
  seconds.
- Record live-view latency. Put a phone stopwatch in front of the camera and
  photograph it beside the dashboard's live view a few times. The delay is
  the difference between the two readings.
- Record clips per hour:
  `journalctl -u bunnycam-agent --since today | grep -c 'saved unlabeled'`,
  divided by the hours covered. Also record `latency_summary.sh` p50 and p95.
