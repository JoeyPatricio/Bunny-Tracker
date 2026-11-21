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

The fixtures in `tests/fixtures/` pin the agent's behaviour: motion levels to
1e-9, every decision exactly, embeddings and probabilities to 1e-4, and
ffmpeg arguments exactly. They have no generator, so a deliberate behaviour
change means editing the fixtures by hand alongside the code.

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
alone.
