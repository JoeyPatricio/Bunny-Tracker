#!/bin/sh
# Ignores SIGTERM (the disposition survives exec), then reports ready with one
# JPEG on fd 3.
trap '' TERM
printf '\377\330R\377\331' >&3
exec sleep 30
