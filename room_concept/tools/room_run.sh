#!/usr/bin/env bash
# room_run.sh — run room_concept in the BACKGROUND and its terminal viewer in the FOREGROUND.
#
#   tools/room_run.sh [config]          (default etc/config.toml; run from anywhere)
#
# · The agent's raw stdout+stderr go to tmp/agent_<YYYY-mm-dd_HH-MM-SS>.log (the viewer's Raw log tab).
# · It is started in its OWN SESSION (setsid), so closing this terminal or killing the viewer does NOT
#   kill it. Reattach any time with:  python3 tools/room_tui.py
# · Quitting the viewer with `q` offers "stop agent" (SIGTERM — graceful, its graph cleanup runs) or
#   "detach". Nothing here ever sends SIGKILL: a -9 leaks the agent's nodes into the shared graph.
# · If THIS script is interrupted before the viewer starts, the agent gets SIGTERM.
#
# Running  bin/room_concept etc/config.toml  directly still works exactly as before.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here" || exit 1
cfg="${1:-etc/config.toml}"
[ -x bin/room_concept ] || { echo "room_run: bin/room_concept not built (make -C build -j8)"; exit 1; }
[ -f "$cfg" ] || { echo "room_run: no config $cfg"; exit 1; }
python3 -c 'import textual' 2>/dev/null || { echo "room_run: Textual missing — python3 -m pip install textual"; exit 1; }

if pgrep -x room_concept >/dev/null; then
    echo "room_run: a room_concept is already running (pid $(pgrep -x room_concept | tr '\n' ' '))."
    echo "          attach to it with:  python3 tools/room_tui.py"
    exit 1
fi

mkdir -p tmp
ts="$(date +%Y-%m-%d_%H-%M-%S)"
log="tmp/agent_${ts}.log"
sockdir="${XDG_RUNTIME_DIR:-/tmp}/rc_status"
[ -n "${XDG_RUNTIME_DIR:-}" ] || sockdir="/tmp/rc_status_$(id -u)"

setsid bin/room_concept "$cfg" </dev/null >"$log" 2>&1 &
pid=$!
echo "room_run: room_concept pid $pid, log $log"

stop_agent() {
    trap - INT TERM HUP
    if kill -0 "$pid" 2>/dev/null; then
        echo "room_run: interrupted — sending SIGTERM to $pid (graceful)"
        kill -TERM "$pid"
    fi
    exit 130
}
trap stop_agent INT TERM HUP

# Wait for the agent's status socket (it is created first thing in initialize()).
start=$(date +%s)
sock=""
while :; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "room_run: room_concept exited before its status socket appeared. Last lines of $log:"
        tail -n 25 "$log"
        exit 1
    fi
    sock="$(ls -t "$sockdir"/room_concept_*.sock 2>/dev/null | head -n 1)"
    if [ -n "$sock" ] && [ "$(stat -c %Y "$sock")" -ge "$start" ]; then break; fi
    if [ $(( $(date +%s) - start )) -ge 60 ]; then
        echo "room_run: no status socket after 60 s ([Status] Enable = false?). The agent keeps running;"
        echo "          follow it with:  tail -f $log    stop it with:  kill $pid"
        exit 1
    fi
    sleep 0.2
done

trap - INT TERM HUP
exec python3 tools/room_tui.py --log "$log" --pid "$pid" --sock "$sock"
