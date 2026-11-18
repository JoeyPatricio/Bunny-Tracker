# C++ capture agent

`bunnycam-agent` is a C++20 port of `server/agent/` (the Python capture
agent). It reads the same `data/.env` and environment variables, loads the
same `server/models/` files, and talks to the same FastAPI endpoints. The
difference you can see is the live view: ffmpeg writes MJPEG to fd 3 and the
agent POSTs each JPEG as it arrives, instead of polling a JPEG file. Linux
only (`CAMERA_FORMAT` is `v4l2` or `lavfi`).

| Module | Ports |
|---|---|
| `src/motion` | `shared/motion.py` |
| `src/decision` | `AgentDecisionLoop`, `interest_score` |
| `src/inference` | `inference/*.py`, `shared/frames.normalize` |
| `src/ffmpeg` | `agent/ffmpeg_io.py` |
| `src/server` | `agent/client.py`, uploads and the clip queue |
| `src/main.cpp` | `Agent`, `main()` |

The Python agent is the reference until cutover.
`server/scripts/dump_agent_golden.py` records its behaviour into
`tests/fixtures/`, and the tests require the C++ to reproduce it: motion
levels to 1e-9, every decision exactly, embeddings and probabilities to 1e-5,
ffmpeg arguments exactly.

## Build and test

Packages: `cmake g++ libssl-dev nlohmann-json3-dev catch2 unzip` on Debian
(`cmake nlohmann-json catch2 unzip` on Arch). cpp-httplib is vendored in
`third_party/`. ONNX Runtime 1.29.0 comes from Microsoft's NuGet package:

```bash
cd agent-cpp
scripts/fetch-onnxruntime.sh          # into ~/.cache/bunnycam/onnxruntime-1.29.0
cmake --preset default && cmake --build --preset default -j && ctest --preset default
```

The `asan` (ASan+UBSan) and `tsan` presets build into `build-asan/` and
`build-tsan/`. CI runs all three. Run `clang-format -i` on anything you touch.

The binary finds the repo from its own path (`<repo>/agent-cpp/<build>/`), so
it must stay inside the checkout.

## Latency

Each frame log line carries `t=<ms>`, measured from the moment the reader
has a whole raw frame to the moment the decision returns. That covers motion,
normalize, backbone, head and decision, not uploads.

```bash
journalctl -u bunnycam-agent --since "1 hour ago" | agent-cpp/scripts/latency_summary.sh
```

With no promoted model the agent runs motion-only, so `t=` measures motion
alone. Rerun after a 7-class model is promoted to get the full
capture-to-prediction figure.

## Pi burn-in

The Pi is production and there is no second machine, so the swap is a
systemd drop-in and rollback is one command. The Python agent stays in the
checkout until cutover.

### 1. Baseline the Python agent

Do this before the swap, and record the C++ run over the same hours of day.

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

### 2. Build on the Pi

The branch has to be on GitHub for the Pi to fetch it.

```bash
sudo apt install -y cmake g++ libssl-dev nlohmann-json3-dev catch2 unzip
cd /opt/bunny-tracker
sudo -H -u bunnycam git fetch origin cpp/capture-agent
sudo -H -u bunnycam git checkout cpp/capture-agent
cd agent-cpp
sudo -H -u bunnycam scripts/fetch-onnxruntime.sh
sudo -H -u bunnycam cmake --preset default
sudo -H -u bunnycam cmake --build --preset default -j4
sudo -H -u bunnycam ctest --preset default
```

All tests must pass on the Pi before the swap, including the real-ffmpeg
lavfi test and the ONNX parity tests on aarch64.

### 3. Swap

```bash
sudo systemctl edit bunnycam-agent
```

Add:

```ini
[Service]
ExecStart=
ExecStart=/opt/bunny-tracker/agent-cpp/build/bunnycam-agent
```

Then `sudo systemctl restart bunnycam-agent` and check
`journalctl -u bunnycam-agent -f` for `🐇 BunnyCam C++ Agent starting...`.

Rollback:

```bash
sudo systemctl revert bunnycam-agent && sudo systemctl restart bunnycam-agent
```

### 4. Burn-in, 72 h

- No unrecovered crash. `systemctl show -p NRestarts bunnycam-agent` stays
  at 0, and `journalctl -u bunnycam-agent | grep -E 'failed to start|loop failed'`
  finds nothing.
- No orphaned ffmpeg. Restart the agent 5 times
  (`sudo systemctl restart bunnycam-agent`), and after each one
  `pgrep -c -x ffmpeg` prints 1.
- Toggle monitoring and Clipping Mode from the dashboard a few times. The
  live view comes back each time, and clips land in Label Studio.
- Record RSS and CPU with the step 1 loop (into `~/burnin-cpp.txt`), plus
  live-view latency, clips per hour, and `latency_summary.sh` p50 and p95.
  Put them next to the baseline.

Cutover to `main` (deleting the Python agent, updating `deploy/pi/`, `run.sh`
and the docs) happens after the burn-in passes.
