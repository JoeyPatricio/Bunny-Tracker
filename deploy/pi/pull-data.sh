#!/usr/bin/env bash
# Copy the Pi's data/ (recordings, labels and their sidecars, autolabels,
# hidden, monitor state, activity log, label backups) into this checkout's
# data/, so the PC holds the archive and training reads current labels.
# Run it before a training run.
#
#   deploy/pi/pull-data.sh                  from jp@10.0.0.150
#   deploy/pi/pull-data.sh user@host        from another Pi
#   BUNNYCAM_PI=user@host deploy/pi/pull-data.sh
#
# One way, Pi to PC. The Pi is where labelling happens, so its labels.json
# replaces the PC's. Nothing is deleted on the PC: a clip removed or pruned on
# the Pi stays here. data/.env and sessions.json stay behind, they are the
# Pi's secrets. Rolling segments and tmp/ are not worth copying.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PI="${1:-${BUNNYCAM_PI:-jp@10.0.0.150}}"
REMOTE_DATA="/opt/bunny-tracker/data/"

if ! command -v rsync >/dev/null; then
  echo "rsync is not installed on this machine." >&2
  exit 1
fi

# -rlt, not -a: keep modification times (some capture-time fallbacks read a
# recording's mtime), but not the bunnycam owner or the Pi's modes.
rsync -rlt --human-readable --info=stats1,progress2 \
  --exclude '/.env' --exclude '/sessions.json*' \
  --exclude '/tmp/' --exclude '/agent/segments/' \
  "$PI:$REMOTE_DATA" "$ROOT/data/"

echo "Pulled $PI:$REMOTE_DATA into $ROOT/data/"
echo "Recordings here: $(find "$ROOT/data/recordings" -type f | wc -l)"
