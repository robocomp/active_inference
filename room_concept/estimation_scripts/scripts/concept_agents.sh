#!/bin/bash
# ALL CONCEPT AGENTS together, on top of a fleet that is already up (room_concept, controller, retina, ltsm, …).
#   concept_agents.sh start [out_dir]   start every concept agent (+ ltom_agent with LTOM=1); logs in <out_dir>/logs/<agent>.log
#   concept_agents.sh stop              SIGTERM, then wait (never kill -9: they own nodes in the shared graph, and
#                                       ltom_agent saves its repertory only on a graceful stop)
#   concept_agents.sh status            which are running
# THE LIST is $CONCEPT_AGENTS — a NEW concept agent (or countable class: folder + bin link, see CONCEPT_AGENT_RECIPE.md) is one word added to the default below.
# Binary: $WT/<agent>/bin/<agent> when present (default WT = the phase0 worktree), else the main checkout's; a SYMLINK there
# (a countable class) pins the agent to the worktree — a dangling one is reported and skipped, never replaced by main's.
# Every agent runs from the agent directory of the checkout its binary came from ($src/<agent>: etc/ and the manifest path
# ../common/concept_manifest are relative to it), line-buffered. <out_dir>/agents.txt records which binary and
# which commit each one ran, and the link target (readlink -f) the link path resolves to, so a run can be traced back to the code.
B=/home/pbustos/robocomp/components/active_inference
# ★CONCEPT_WT, not WT (2026-11-14): the tour scripts EXPORT WT (= the fleet's rt-build worktree), which silently
# overrode this default — objects_live1 ran with NO concept agent ("NO BINARY … skipped") and nobody saw them.
WT=${CONCEPT_WT:-/home/pbustos/robocomp/components/active_inference.worktrees/phase0}
CONCEPT_AGENTS=${CONCEPT_AGENTS:-"chair_concept table_concept bottle_concept hood_concept refrigerator_concept cabinet_concept nightstand_concept bed_concept toilet_concept"}
ALL="$CONCEPT_AGENTS"; [ -n "$LTOM" ] && ALL="$ALL ltom_agent"
comm_of() { echo "${1:0:15}"; }   # the kernel's process name is the first 15 characters (pgrep/pkill -x match it)

case "${1:-}" in
  start)
    OUT=$(readlink -f "${2:-$B/room_concept/estimation_scripts/crossing/concept_agents}"); mkdir -p $OUT/logs
    : > $OUT/agents.txt
    for a in $ALL; do
      if pgrep -x "$(comm_of $a)" > /dev/null; then echo "$a: already running"; continue; fi
      exe=""; cfg=""; src=""
      # countable classes (nightstand, bed, toilet): bin/<a> is a SYMLINK to countable_concept/bin/countable_concept. Execute the LINK
      # path (never its target) so comm = <a> and pgrep -x stays per class. A link in a checkout decides for that checkout: a
      # dangling worktree link means countable_concept is not built THERE, and falling back to main's binary would silently run
      # other code against the worktree's class folders — so it is reported and skipped.
      for r in $WT $B; do
        l=$r/$a/bin/$a
        if [ -L "$l" ]; then
          if [ -e "$l" ] && [ -x "$l" ]; then exe=$l; cfg=$r/$a/etc/config.toml; src=$r
          elif [ ! -e "$l" ]; then echo "$a: link target missing — build countable_concept ($l -> $(readlink "$l"))"; exe=missing
          else echo "$a: link target not executable ($l -> $(readlink -f "$l")) — skipped"; exe=missing; fi
          break
        elif [ -x "$l" ]; then exe=$l; cfg=$r/$a/etc/config.toml; src=$r; break; fi
      done
      [ "$exe" = missing ] && continue
      if [ -z "$exe" ]; then echo "$a: NO BINARY (not in $WT/$a/bin nor $B/$a/bin) — skipped"; continue; fi
      # Every line is prefixed with wall-clock seconds (ms), so recall / birth TIMING can be read off the logs (ltom_agent
      # prints no time of its own). perl, not `ts` (moreutils is not installed). SIGTERM goes to the agent (pkill -x); perl
      # ends on its EOF.
      # run from the agent's OWN checkout: the manifest path (../common/concept_manifest) and etc/ are relative to it
      (cd $src/$a && DISPLAY=:0 setsid nohup bash -c "stdbuf -oL -eL $exe $cfg 2>&1 | perl -MTime::HiRes=time -ne 'BEGIN{\$|=1} printf \"%.3f %s\", time, \$_' > $OUT/logs/$a.log" < /dev/null > /dev/null 2>&1 &)
      printf '%s\t%s\t%s\t%s\n' $a $exe "$(git -C $src rev-parse --short HEAD 2>/dev/null)" "$(readlink -f $exe)" >> $OUT/agents.txt
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
  *) echo "usage: [LTOM=1] [CONCEPT_WT=<worktree>] [CONCEPT_AGENTS=\"…\"] $0 start [out_dir] | stop | status"; exit 2;;
esac
