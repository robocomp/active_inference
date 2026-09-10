#!/usr/bin/env bash
# Restart-supervisor for a CORTEX concept agent. Keeps the agent alive across the residual
# DDS-emit-from-raw-thread crash (see dsr-graphviewer-crash-fixes memory) until the cortex
# emit-after-unlock fix lands. Peers re-acquire the agent via the presence protocol on restart.
#
#   ./run_agent_supervised.sh retina [etc/config.toml]
#   ./run_agent_supervised.sh bottle_concept
#
# Stop with Ctrl+C (the trap kills the child and exits the loop).
set -u

AGENT="${1:?usage: run_agent_supervised.sh <agent-dir-name> [config]}"
CONFIG="${2:-etc/config.toml}"
DIR="$(cd "$(dirname "$0")/$AGENT" && pwd)"
BIN="$DIR/bin/$AGENT"

[ -x "$BIN" ] || { echo "no executable: $BIN"; exit 1; }

restarts=0
child=0
cleanup() { echo; echo "[supervisor] stopping $AGENT (was restarted $restarts time(s))"; kill -TERM "$child" 2>/dev/null; wait "$child" 2>/dev/null; exit 0; }
trap cleanup INT TERM

echo "[supervisor] $AGENT — restart-on-exit; Ctrl+C to stop"
while true; do
    cd "$DIR"
    "$BIN" "$CONFIG" &
    child=$!
    wait "$child"
    rc=$?
    # rc 0 / 143(SIGTERM) / 130(SIGINT) = intentional stop -> do NOT restart.
    if [ "$rc" = "0" ] || [ "$rc" = "143" ] || [ "$rc" = "130" ]; then
        echo "[supervisor] $AGENT exited cleanly (rc=$rc) — not restarting"
        break
    fi
    restarts=$((restarts + 1))
    echo "[supervisor] $AGENT crashed (rc=$rc) — restart #$restarts in 2s"
    sleep 2
done
