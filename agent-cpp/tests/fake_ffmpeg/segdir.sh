#!/bin/sh
# The one fake that reads its args: it reports on fd 3 whether the segment
# directory in the output pattern existed when ffmpeg started.
for arg; do
    case $arg in */seg-%05d.webm) dir=${arg%/*} ;; esac
done
if [ -d "$dir" ]; then printf '\377\330ok\377\331' >&3; else printf '\377\330missing\377\331' >&3; fi
exec sleep 30
