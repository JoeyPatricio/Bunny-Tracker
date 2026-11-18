#!/bin/sh
# Reports ready with one JPEG on fd 3, then idles until signalled.
printf '\377\330R\377\331' >&3
exec sleep 30
