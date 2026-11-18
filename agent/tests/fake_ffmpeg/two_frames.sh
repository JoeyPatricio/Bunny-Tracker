#!/bin/sh
# Frame 1, then frame 2 once the consumer has had time to take frame 1.
frame() { printf "\\$(printf '%03o' "$1")"; head -c 150527 /dev/zero; }
frame 1
sleep 0.3
frame 2
exec sleep 30
