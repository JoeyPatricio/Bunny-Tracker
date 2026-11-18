#!/bin/sh
# 10000 bytes of stderr with no newline, then ready on fd 3.
head -c 10000 /dev/zero | tr '\0' x >&2
printf '\377\330R\377\331' >&3
exec sleep 30
