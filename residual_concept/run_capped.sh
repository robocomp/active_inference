#!/usr/bin/env bash
#
# run_capped.sh — launch residual_concept inside a memory-capped cgroup.
#
# WHY: on 2026-08-08 this agent grew to 33.8 GB on a 62 GB machine with NO swap. The kernel's global
# OOM killer picked it (correctly — it was the largest), but because it had been D-Bus activated the
# kill landed in the session's dbus.service cgroup and took the whole GNOME session down with it:
# gvfs unmounted, polkit agent gone, portals dead, Brave and Mathpix core-dumped. A per-agent cap turns
# "one agent leaks" into "one agent dies" instead of "the desktop dies".
#
# TWO LIMITS, on purpose:
#   MemoryHigh — soft. The kernel throttles the process and reclaims aggressively above this. Nothing is
#                killed; the agent just gets visibly slow. That is the WARNING, and with the leak-watch
#                rows in etc/perf_diag.csv you can see it coming.
#   MemoryMax  — hard. Exceed it and the cgroup OOM killer SIGKILLs this agent and nothing else.
#
# CAVEAT — a cgroup OOM kill is SIGKILL, which CANNOT be caught, so request_shutdown()/
# cleanup_owned_nodes() never run and any node this agent owns LEAKS into the shared persistent graph
# (CLAUDE.md, "Stopping an agent"). That is survivable here: remove_owned_residual_nodes() sweeps
# leftover residual_* nodes both in initialize() and once on the first Operating cycle. It is still a
# dirtier exit than SIGTERM, so treat a MemoryMax kill as an incident to investigate, not as a routine
# recycle. To stop the agent normally use Ctrl-C or `kill` (SIGTERM) — never `kill -9`.
#
# USAGE:
#   ./run_capped.sh                        # caps at MemoryHigh=3G / MemoryMax=4G
#   MEM_HIGH=6G MEM_MAX=8G ./run_capped.sh # override the caps
#   ./run_capped.sh --heaptrack            # same caps, under heaptrack (allocation-site profile)
#   ./run_capped.sh -- etc/other.toml      # pass a different config through
#
# Steady state for this agent is ~220 MB, so the default 4 GB is ~18x headroom: generous enough that a
# legitimately heavy cycle is never killed, small enough that a runaway cannot reach the desktop.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"      # etc/ paths in the config are relative to the component dir

MEM_HIGH="${MEM_HIGH:-3G}"
MEM_MAX="${MEM_MAX:-4G}"
USE_HEAPTRACK=0

args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --heaptrack) USE_HEAPTRACK=1; shift ;;
        --)          shift; args+=("$@"); break ;;
        *)           args+=("$1"); shift ;;
    esac
done
[[ ${#args[@]} -eq 0 ]] && args=(etc/config.toml)

if [[ ! -x bin/residual_concept ]]; then
    echo "run_capped.sh: bin/residual_concept not found or not executable — build it first (cbuild)" >&2
    exit 1
fi

cmd=(bin/residual_concept "${args[@]}")

if [[ $USE_HEAPTRACK -eq 1 ]]; then
    command -v heaptrack >/dev/null || { echo "run_capped.sh: heaptrack not installed" >&2; exit 1; }
    # heaptrack itself needs room for its own bookkeeping on top of the agent's heap; without this a
    # profiling run would hit MemoryMax earlier than an unprofiled one and you would be measuring the cap.
    MEM_HIGH="${HEAPTRACK_MEM_HIGH:-6G}"
    MEM_MAX="${HEAPTRACK_MEM_MAX:-8G}"
    mkdir -p etc/heaptrack
    cmd=(heaptrack -o "etc/heaptrack/residual_$(date +%Y%m%d-%H%M%S)" "${cmd[@]}")
    echo "run_capped.sh: heaptrack enabled → analyse with 'heaptrack_print --print-leaks <file>.zst'"
fi

echo "run_capped.sh: MemoryHigh=$MEM_HIGH MemoryMax=$MEM_MAX  cmd: ${cmd[*]}"

# --scope runs the command as a transient cgroup child of THIS shell's session, so Ctrl-C and job control
# behave normally and stdout/stderr stay on this terminal (unlike --service, which would detach it into
# the journal and swallow the [perf] lines).
exec systemd-run --user --scope --quiet \
    --unit="residual-concept-$$" \
    -p "MemoryHigh=$MEM_HIGH" \
    -p "MemoryMax=$MEM_MAX" \
    -- "${cmd[@]}"
