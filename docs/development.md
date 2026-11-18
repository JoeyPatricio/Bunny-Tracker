# Development

Back to the [project overview](../README.md).

## Source map

```text
Bunny-Tracker/
├── agent/                         C++20 capture agent
│   ├── src/                       Capture, motion, inference, decisions, HTTP
│   ├── tests/                     Catch2 tests, fixtures, fake ffmpeg commands
│   ├── scripts/                   ONNX Runtime download and latency summary
│   ├── third_party/               Vendored cpp-httplib and its license
│   └── CMakePresets.json           Default, ASan/UBSan, and TSan builds
├── client/
│   └── src/
│       ├── App.jsx                Dashboard tabs, login, and shared controls
│       ├── components/            Live feed, labeling, recordings, and training UI
│       ├── labels.js              Frontend behavior vocabulary
│       └── styles/                Global styles
├── server/
│   ├── app/
│   │   ├── main.py                FastAPI entry point and route registration
│   │   ├── routers/               HTTP endpoints grouped by resource
│   │   └── lib/                   Storage, labels, clip budgets, and time helpers
│   ├── agent/                     Python reference capture agent
│   ├── inference/                 ONNX backbone, temporal head, and predictor
│   ├── shared/                    Frame processing, motion, labels, and paths
│   ├── training/                  Offline training and model promotion commands
│   ├── tests/
│   │   ├── test_scripts.py         Pytest runner for the standalone check scripts
│   │   ├── test_*.py               Backend and Python agent checks
│   │   ├── training/              Training checks, including decode parity
│   │   ├── checks.py              Shared PASS/FAIL/SKIP reporting
│   │   └── fixtures/              Motion frames and expected results
│   ├── scripts/                   Benchmarks, fixture generators, camera smoke tools
│   ├── models/                    ONNX artifacts, candidates, and archived models
│   └── pyproject.toml             Dependencies, pytest selection, and Ruff rules
├── data/                          Runtime stores, recordings, and shared .env
├── deploy/pi/                     systemd units, install guide, and data sync
├── docs/                          Development and user guides, research references
└── run.sh                         Local Python server and agent launcher
```

The C++ agent in `agent/` and the Python reference in `server/agent/` share
the server API, `data/.env`, and `server/models/`. The launcher and base Pi
service use Python until cutover. See the [agent guide](../agent/README.md)
for C++ builds, parity fixtures, and Pi cutover steps.

## Where to make changes

| Change | Start here |
|---|---|
| Dashboard or public demo | [App.jsx](../client/src/App.jsx) and [components/](../client/src/components/) |
| API endpoint | [app/routers/](../server/app/routers/), registered in [app/main.py](../server/app/main.py) |
| Authentication or configuration | [auth.py](../server/app/auth.py), [security.py](../server/app/security.py), and [config.py](../server/app/config.py) |
| Label storage or clip selection | [app/lib/](../server/app/lib/) |
| C++ capture behavior | [agent/src/](../agent/src/) and [agent/tests/](../agent/tests/) |
| Python reference behavior | [server/agent/](../server/agent/) and [server/tests/](../server/tests/) |
| Behavior vocabulary | [shared/labels.py](../server/shared/labels.py), [client/src/labels.js](../client/src/labels.js), and [ethogram](ethogram.md#where-the-vocabulary-lives) |
| Training or evaluation | [server/training/](../server/training/) and [server/tests/training/](../server/tests/training/) |

## Tests

Install dependencies as described in [setup](../README.md#setup). Python
checks need the runtime venv with `--group dev` and the training venv. Build
`client/dist/` first because the security tests exercise the real SPA fallback.
Run these commands from `server/`:

```bash
.venv/bin/python -m pytest -m "not local_data"
.venv/bin/ruff check .
```

Pytest collects only `tests/test_scripts.py`. That runner launches each check
script in the appropriate venv. The scripts have their own async drivers,
so running pytest directly on the entire `tests/` directory bypasses those
drivers. Each script can also run on its own:

```bash
.venv/bin/python tests/test_capture_loop.py
.venv/bin/python tests/test_security_regressions.py
.venv/bin/python tests/training/test_promote_reports.py
.venv-train/bin/python tests/training/test_train_guards.py
.venv-train/bin/python tests/training/test_evaluate_gates.py
```

Scripts report `PASS`, `FAIL`, or `SKIP` per check. A `SKIP` means the machine
couldn't make a measurement. The runner fails if a script exits nonzero or
its interpreter is missing. `BT_RUNTIME_PYTHON` and `BT_TRAIN_PYTHON` can
override the interpreter paths.

`tests/training/check_decode_parity.py` needs real recordings and is marked
`local_data`. Running `.venv/bin/python -m pytest` includes that check.

The [CI workflow](../.github/workflows/ci.yml) builds the client and both Python
venvs, runs the checks that don't need local data, and runs Ruff. The C++ job
checks formatting and runs Catch2 under the default, ASan/UBSan, and TSan
presets. See [C++ build and test commands](../agent/README.md#build-and-test).

## Manual camera checks

These tools run from `server/` and aren't part of the automated suite:

```bash
# Replay fixture frames through a promoted ONNX model.
.venv/bin/python scripts/smoke_capture_offline.py

# Run the real Python agent for 60 seconds, then shut down ffmpeg.
.venv/bin/python scripts/smoke_capture_live.py 60
```

The live tool needs a running server and the agent configuration in
`data/.env`. It uses the normal capture, recording, and notification behavior.
To exercise capture without a webcam, use a synthetic source:

```ini
CAMERA_FORMAT=lavfi
CAMERA_INPUT=testsrc2=size=640x480:rate=15
```

## Data and generated files

`data/` holds recordings, labels, monitor state, and secrets. These are local
runtime files. The tracked `data/labels.v1.json` supplies the label migration
report in `training/recoding_report.py`.

The active inference paths use `server/models/`. Its ONNX graphs are also
inputs to the C++ parity tests. Live classification requires a promoted model
whose labels match the current ethogram. Model files are read from disk.
The dashboard reads their metadata through `/api/model`.

Model candidates, promoted artifacts, and archives have paths shared by the
server, both agents, training commands, and deployment tools. Use the
[training pipeline](pipeline.md) to export and promote models. Its
[dependency instructions](pipeline.md#the-two-venv-rule) also cover regenerating
`uv.lock` and the requirements files from `pyproject.toml`.

`server/scripts/dump_agent_frames.py` generates Python motion fixtures, and
`server/scripts/dump_agent_golden.py` generates C++ parity fixtures. Update the
generator when the expected behavior changes.

Builds, Python caches, `client-dist.tgz`, recordings, and runtime stores are
ignored by Git.
The frontend build lives in `client/dist/`, C++ builds live in `agent/build*/`,
and Python environments live in `server/.venv/` and `server/.venv-train/`.
