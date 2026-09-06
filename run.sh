#!/usr/bin/env bash
# Start Bunny Tracker without pm2: the FastAPI server and the capture agent,
# both on the runtime venv's own interpreter (server_py/.venv), the same two
# processes ecosystem.config.cjs defines. Ctrl+C stops both.
#
#   ./run.sh          server + agent   (dashboard on http://localhost:3001)
#   ./run.sh server   server only      (dashboard, no camera)
#   ./run.sh agent    agent only       (a server must already be running)
#
# Output is prefixed [server]/[agent] on screen and appended to logs/.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_PY="$ROOT/server_py"
PYTHON="$SERVER_PY/.venv/bin/python"
PORT="${PY_PORT:-3001}"
LOGS="$ROOT/logs"
WHAT="${1:-all}"

if [ ! -x "$PYTHON" ]; then
  echo "No runtime venv at $PYTHON" >&2
  echo "Create it with:  python -m venv $SERVER_PY/.venv && $SERVER_PY/.venv/bin/pip install -r $SERVER_PY/requirements.txt" >&2
  exit 1
fi
if [ ! -f "$ROOT/server/.env" ]; then
  echo "Missing $ROOT/server/.env (camera, admin password, agent token) — see README." >&2
  exit 1
fi

mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PY_PORT="$PORT"

pids=()
cleanup() {
  trap - INT TERM EXIT
  [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null
  wait 2>/dev/null
  echo "[run.sh] stopped"
}
trap cleanup INT TERM EXIT

# Launch $1 (label) running the rest of the args under the venv python, from
# server_py. Piping through a process substitution keeps $! as the python PID
# itself, so cleanup signals the process and not a tee in front of it.
start() {
  local name="$1"; shift
  ( cd "$SERVER_PY" && exec "$PYTHON" "$@" ) \
    > >(sed -u "s/^/[$name] /" | tee -a "$LOGS/$name.log") 2>&1 &
  pids+=($!)
  echo "[run.sh] $name started (pid ${pids[-1]}, log $LOGS/$name.log)"
}

if [ "$WHAT" = all ] || [ "$WHAT" = server ]; then
  if (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
    echo "Port $PORT is already in use — is Bunny Tracker already running?" >&2
    exit 1
  fi
  start server -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
  # The agent's first call fails outright if the server isn't listening yet.
  for _ in $(seq 40); do
    (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null && break
    sleep 0.25
  done
  echo "[run.sh] dashboard: http://localhost:$PORT"
fi

if [ "$WHAT" = all ] || [ "$WHAT" = agent ]; then
  start agent -m agent.capture
fi

if [ ${#pids[@]} -eq 0 ]; then
  echo "Usage: ./run.sh [all|server|agent]" >&2
  exit 2
fi

# Any process exiting on its own takes the other down with it, so a dead
# agent can't sit unnoticed behind a healthy dashboard.
wait -n
