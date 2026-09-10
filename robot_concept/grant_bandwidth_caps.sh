#!/usr/bin/env bash
#
# grant_bandwidth_caps.sh — allow the "mind" node network view to capture loopback bandwidth.
#
# The mind dashboard (DSR graph viewer → mind node → View data) sniffs `lo` with a raw AF_PACKET
# socket to show per-edge bandwidth. That needs the CAP_NET_RAW capability. robot_concept runs with
# Agent.graph=true, so it hosts the graph viewer and therefore the sniffer — grant the capability to
# its binary here (once per rebuild) instead of running the whole agent as root.
#
# Usage:  ./grant_bandwidth_caps.sh
#
# NOTE: setcap marks the binary as "secure-execution" (AT_SECURE=1), so the dynamic loader ignores
# LD_LIBRARY_PATH and $ORIGIN-less RPATHs. If robot_concept then fails to find libdsr_api.so /
# libdsr_gui.so, either install those libs to a standard path (ldconfig) or drop the capability with
# `sudo setcap -r bin/robot_concept` and run the viewer with sudo instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/bin/robot_concept"

if [[ ! -x "$BIN" ]]; then
    echo "error: robot_concept binary not found (or not executable): $BIN" >&2
    echo "       build it first, then re-run this script." >&2
    exit 1
fi

echo "Granting cap_net_raw+ep to: $BIN"
echo "(sudo required)"
sudo setcap cap_net_raw+ep "$BIN"

printf 'getcap -> '
getcap "$BIN"
echo "Done. Reopen the mind node's 'View data' — the status line should read 'capturing on lo'."
