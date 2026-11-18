#!/bin/sh
# MJPEG on fd 3: a whole JPEG, one split across reads at its SOI and EOI,
# junk with no SOI, a JPEG with no EOI past the 2 MiB cap, then a whole one.
printf '\377\330AAAA\377\331' >&3
sleep 0.2
printf '\377' >&3
sleep 0.2
printf '\330BB\377' >&3
sleep 0.2
printf '\331' >&3
printf 'junk' >&3
printf '\377\330' >&3
head -c 2621440 /dev/zero >&3
printf '\377\330CC\377\331' >&3
exec sleep 30
