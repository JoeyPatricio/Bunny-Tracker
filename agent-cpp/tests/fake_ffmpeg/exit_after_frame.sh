#!/bin/sh
# One frame, then an unexpected exit. The pause lets the reader take the frame
# before the supervisor tears it down.
printf '\001'
head -c 150527 /dev/zero
sleep 0.2
exit 1
