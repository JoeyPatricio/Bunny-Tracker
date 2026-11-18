#!/usr/bin/env bash
# Frame-to-decision latency from the agent's "frame  t=<ms>ms" log lines:
# from the reader finishing a raw frame to the decision returning (motion,
# normalize, backbone, head, decision; uploads excluded).
#
#   journalctl -u bunnycam-agent --since "1 hour ago" | agent-cpp/scripts/latency_summary.sh
#   agent-cpp/scripts/latency_summary.sh < agent.log
set -euo pipefail

grep -o 'frame  t=[0-9.]*ms' | sed 's/.*t=//; s/ms$//' | sort -n | awk '
  { v[NR] = $1 }
  END {
    if (NR == 0) { print "no frame lines" > "/dev/stderr"; exit 1 }
    p50 = v[int((NR - 1) * 0.50) + 1]
    p95 = v[int((NR - 1) * 0.95) + 1]
    printf "frames=%d  p50=%.1fms  p95=%.1fms  max=%.1fms\n", NR, p50, p95, v[NR]
  }'
