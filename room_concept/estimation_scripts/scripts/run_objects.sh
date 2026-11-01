#!/bin/bash
# OBJECT AGENTS, separate from the door/room tour fleet (user 2026-11-01: table_concept left the tour scripts; the object
# agents run from here, alongside a fleet that is already up).
#   run_objects.sh start [out_dir]   start every object agent (logs in <out_dir>/logs/<agent>.log, default crossing/objects)
#   run_objects.sh stop              stop them with SIGTERM (never kill -9: they own nodes in the shared graph)
#   run_objects.sh status            which of them are running
# Agents: table, chair, bottle, hood, refrigerator. MICROWAVE: there is no microwave agent (kitchen_metaconcept's config:
# "microwave ✗ no node, but retina's YOLO set already has 'microwave'") — it is listed and reported missing.
# A binary in the worktree ($WT/<agent>/bin) is preferred over the main checkout's; each agent runs from the main agent
# directory (its data files live there), line-buffered (stdbuf -oL, as run_crossing.sh).
B=/home/pbustos/robocomp/components/active_inference
WT=${WT:-/home/pbustos/robocomp/components/active_inference.worktrees/rt-build}
OBJECTS=${OBJECTS:-"table_concept chair_concept bottle_concept hood_concept refrigerator_concept microwave_concept"}
comm_of() { echo "${1:0:15}"; }   # the kernel's process name is the first 15 characters (pkill/pgrep -x match it)

case "${1:-}" in
  start)
    OUT=$(readlink -f "${2:-$B/room_concept/estimation_scripts/crossing/objects}"); mkdir -p $OUT/logs
    for a in $OBJECTS; do
      if pgrep -x "$(comm_of $a)" > /dev/null; then echo "$a: already running"; continue; fi
      exe=""; cfg=""
      if [ -x "$WT/$a/bin/$a" ]; then exe=$WT/$a/bin/$a; cfg=$WT/$a/etc/config.toml
      elif [ -x "$B/$a/bin/$a" ]; then exe=bin/$a; cfg=etc/config.toml; fi
      if [ -z "$exe" ]; then echo "$a: NO AGENT (no binary in $WT/$a/bin or $B/$a/bin) — skipped"; continue; fi
      (cd $B/$a && DISPLAY=:0 setsid nohup stdbuf -oL -eL $exe $cfg > $OUT/logs/$a.log 2>&1 < /dev/null &)
      echo "$a: started ($exe) -> $OUT/logs/$a.log"; sleep 3
    done;;
  stop)
    for a in $OBJECTS; do pkill -TERM -x "$(comm_of $a)" && echo "$a: SIGTERM sent"; done
    sleep 8
    for a in $OBJECTS; do pgrep -x "$(comm_of $a)" > /dev/null && echo "$a: STILL RUNNING (wait, or SIGTERM again — never -9)"; done
    true;;
  status)
    for a in $OBJECTS; do
      if pgrep -x "$(comm_of $a)" > /dev/null; then echo "$a: running"; else echo "$a: not running"; fi
    done;;
  *) echo "usage: $0 start [out_dir] | stop | status"; exit 2;;
esac
