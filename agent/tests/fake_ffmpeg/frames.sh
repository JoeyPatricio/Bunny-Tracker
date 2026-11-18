#!/bin/sh
# Writes 20 raw frames as fast as the pipe takes them, then idles. Each frame
# is kFrameBytes (224*224*3) and its first byte is its number, 1-20.
frame() { printf "\\$(printf '%03o' "$1")"; head -c 150527 /dev/zero; }
i=1
while [ "$i" -le 20 ]; do frame "$i"; i=$((i + 1)); done
exec sleep 30
