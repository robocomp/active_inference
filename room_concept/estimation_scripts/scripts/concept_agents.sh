#!/bin/bash
# ALL CONCEPT AGENTS together, on top of a fleet that is already up (room_concept, controller, retina, ltsm, …).
#   concept_agents.sh start [out_dir]   start every concept agent (+ ltom_agent with LTOM=1); logs in <out_dir>/logs/<agent>.log
#   concept_agents.sh stop              SIGTERM, then wait (never kill -9: they own nodes in the shared graph, and
#                                       ltom_agent saves its repertory only on a graceful stop)
#   concept_agents.sh status            which are running
# THE LIST is $CONCEPT_AGENTS — a NEW concept agent (nightstand, toilet, bed, …) is one word added to the default below.
# Binary: $WT/<agent>/bin/<agent> when present (default WT = the phase0 worktree), else the main checkout's. Every agent runs
# from the MAIN agent directory (its data files live there), line-buffered. <out_dir>/agents.txt records which binary and
# which commit each one ran, so a run can be traced back to the code.
B=/home/pbustos/robocomp/components/active_inference
WT=${WT:-/home/pbustos/robocomp/components/active_inference.worktrees/phase0}
CONCEPT_AGENTS=${CONCEPT_AGENTS:-"chair_concept table_concept bottle_concept hood_concept refrigerator_concept cabinet_concept"}
ALL="$CONCEPT_AGENTS"; [ -n "$LTOM" ] && ALL="$ALL ltom_agent"
comm_of() { echo "${1:0:15}"; }   # the kernel's process name is the first 15 characters (pgrep/pkill -x match it)

case "${1:-}" in
  start)
    OUT=$(readlink -f "${2:-$B/room_concept/estimation_scripts/crossing/concept_agents}"); mkdir -p $OUT/logs
    : > $OUT/agents.txt
    for a in $ALL; do
      if pgrep -x "$(comm_of $a)" > /dev/null; then echo "$a: already running"; continue; fi
      exe=""; cfg=""; src=""
      if [ -x "$WT/$a/bin/$a" ]; then exe=$WT/$a/bin/$a; cfg=$WT/$a/etc/config.toml; src=$WT
      elif [ -x "$B/$a/bin/$a" ]; then exe=$B/$a/bin/$a; cfg=$B/$a/etc/config.toml; src=$B; fi
      if [ -z "$exe" ]; then echo "$a: NO BINARY (not in $WT/$a/bin nor $B/$a/bin) — skipped"; continue; fi
      (cd $B/$a && DISPLAY=:0 setsid nohup stdbuf -oL -eL $exe $cfg > $OUT/logs/$a.log 2>&1 < /dev/null &)
      printf '%s\t%s\t%s\n' $a $exe "$(git -C $src rev-parse --short HEAD 2>/dev/null)" >> $OUT/agents.txt
      sleep 2
    done
    # a crash at load (bottle 2026-11-13: SIGSEGV in libIceStorm static init) leaves an empty log and no process: say so
    sleep 5
    for a in $ALL; do
      if pgrep -x "$(comm_of $a)" > /dev/null; then echo "$a: running"; else echo "$a: NOT RUNNING (see $OUT/logs/$a.log)"; fi
    done;;
  stop)
    for a in $ALL; do pkill -TERM -x "$(comm_of $a)" && echo "$a: SIGTERM sent"; done
    for i in $(seq 1 30); do
      left=""; for a in $ALL; do pgrep -x "$(comm_of $a)" > /dev/null && left="$left $a"; done
      [ -z "$left" ] && { echo "all stopped"; exit 0; }
      sleep 1
    done
    echo "STILL RUNNING after 30 s:$left (wait, or SIGTERM again — never -9)"; exit 1;;
  status)
    for a in $ALL; do
      if pgrep -x "$(comm_of $a)" > /dev/null; then echo "$a: running"; else echo "$a: not running"; fi
    done;;
  *) echo "usage: [LTOM=1] [WT=<worktree>] [CONCEPT_AGENTS=\"…\"] $0 start [out_dir] | stop | status"; exit 2;;
esac
