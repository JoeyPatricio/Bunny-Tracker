#!/bin/sh
# Reports its open fds as one JPEG on fd 3, e.g. "\377\330 0 1 2 3 255 \377\331".
# The glob runs in this shell, so /proc/self is this process.
fds=
for f in /proc/self/fd/*; do fds="$fds ${f##*/}"; done
printf '\377\330%s \377\331' "$fds" >&3
exec sleep 30
