#!/usr/bin/env bash
# Copy the Pi's data/ (recordings, labels and their sidecars, autolabels,
# hidden, monitor state, activity log, label backups) into this checkout's
# data/, so this machine holds the archive and training reads current labels.
# Run it before a training run.
#
#   deploy/pi/pull-data.sh user@host
#   BUNNYCAM_PI=user@host deploy/pi/pull-data.sh
#
# One way, Pi to here. The Pi is where labelling happens, so its labels.json
# replaces the local one. Nothing is deleted here: a clip removed or pruned on
# the Pi stays. data/.env and sessions.json stay behind, they are the
# Pi's secrets. Rolling segments and tmp/ are not worth copying.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PI="${1:-${BUNNYCAM_PI:-}}"
if [ -z "$PI" ]; then
  echo "Usage: $0 user@host   (or set BUNNYCAM_PI=user@host)" >&2
  exit 2
fi
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
