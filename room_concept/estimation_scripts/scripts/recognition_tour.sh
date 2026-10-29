#!/bin/bash
# ROOM-RECOGNITION gold tour (spec 2026-10-22). Real route of gold_tour.sh, 6 legs; doors are picked by RELATION, never by name
# (door-identity-design.md part 3: a door_N name is a graph slot, not an identity). The controller resolves the argument of `cross`:
#   cross new    a live door of the current room with no recall_* child and no passage match (nothing learnt about it yet)
#   cross <room> the live door that has a recall_<room> child (ltsm posts it under the door that leads to that remembered room)
#   1 room_1 --cross new--> room_2 (new)         2 cross room_1       3 room_1 --cross new--> room_3 (new; room_2's door is excluded)
#   4 room_3 --cross new--> the REMEMBERED room_2 (recognition + Swap; room_1's door has a recall child, so it is excluded)
#   5 cross room_3 (from the recognised room_2)   6 cross room_1
# The controller HOLDS a step until its door resolves; the harness scans (look viewpoint + explore) only to make a `new` door appear.
# usage: recognition_tour.sh <run_name> [explore_s=60]      env WT / WT_AGENTS / SCAN_CAP_S (default 600)
# Writes crossing/<run>/legs.tsv (leg windows + per-log line ranges), copies ltsm_memory.json, runs recognition_report.py.
# Stop agents with SIGTERM only (run_crossing.sh collect does); never edit this script while a run uses it.
LOCK=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ai_fleet_tour.lock; echo $$ > $LOCK; trap 'rm -f $LOCK' EXIT
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # helpers come from THIS script's own directory (the worktree), data stays under $E
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E; L=crossing/$1/logs
EXPLORE_S=${2:-60}; SCAN_CAP_S=${SCAN_CAP_S:-600}
export WT=${WT:-/home/pbustos/robocomp/components/active_inference.worktrees/rt-build} WT_AGENTS=${WT_AGENTS:-"controller door_concept table_concept room_concept ltsm_agent"}
export FLEET="retina room_concept residual_concept controller door_concept ltsm_agent table_concept"
echo stop > $B/controller/remote_cmd.txt
# fresh state: no last pose, no name high-water mark, no ltsm memory, no place maps
rm -f ../etc/last_robot_pose.txt ../etc/room_name_hwm.txt $B/ltsm_agent/etc/ltsm_memory.json $B/ltsm_agent/etc/ltsm_memory.json.prev
rm -rf $B/retina/etc/place_maps/; rm -f $WT/ltsm_agent/etc/recognition_trace.csv $B/ltsm_agent/etc/recognition_trace.csv
# close every door BEFORE the fleet starts (the cross mission opens the one it needs)
for d in DOOR_1 DOOR_0 DOOR_2; do python3 $S/door.py close $d; done
sleep 3
NORUN=1 bash $S/run_crossing.sh crossing/$1 -5.33 3.66 0.58 >/dev/null 2>&1
sleep 25
for i in $(seq 1 90); do grep -a -q 'room polygon LOADED' $L/door_concept.log 2>/dev/null && break; sleep 2; done; sleep 5
room_now() { grep -a "current room changed" $L/controller.log | tail -1 | awk '{print $NF}'; }
nl() { wc -l < $L/$1.log 2>/dev/null || echo 0; }
now_ms() { date +%s%3N; }
TSV=crossing/$1/legs.tsv; printf 'leg\tstatus\tt0_ms\tt1_ms\tltsm0\tltsm1\tdoor0\tdoor1\troom0\troom1\n' > $TSV
run_leg() {   # label cross_arg here   (sets r)
  echo run > $B/controller/remote_cmd.txt; sleep 0.7   # the controller reads remote_cmd.txt every 300 ms
  printf 'mission "leg %s"\n  explore %s until=explored(%s)>=0.9 budget_s=%ss\n  cross %s retry=2\n' $1 $3 $3 $EXPLORE_S $2 > $B/controller/remote_cmd.txt
  t0=$(date +%s); r="FAILED (no end in 600 s)"
  for i in $(seq 1 1200); do
    grep -a -q "mission 'leg $1': completed" $L/controller.log && { r="OK in $(( $(date +%s)-t0 )) s"; break; }
    grep -a -q -E "mission 'leg $1': (failed|aborted)" $L/controller.log && { r="FAILED: $(grep -a -o -E "mission 'leg $1': (failed|aborted)[^\"]*" $L/controller.log | tail -1)"; break; }
    sleep 0.5
  done; }
# No door resolves yet ("door_unresolved" after the controller's own hold): make one appear. First the epistemic LOOK viewpoint
# door_concept offers for a remembered door ("[look] aff_door_look_<d> OFFERED at (x,y) yaw=Y", after the last room switch), then
# one explore; the caller re-issues the mission. Bounded by SCAN_CAP_S in the caller.
scan_once() {   # tag
  local tag=$1 lk; lk=$(awk '/\[room\] LET GO/{l=""} /\[look\] aff_door_look_.* OFFERED at/{l=$0} END{print l}' $L/door_concept.log \
                 | grep -o -E "OFFERED at \(-?[0-9.]+,-?[0-9.]+\) yaw=[-+]?[0-9.]+" | tr -d '()' | tr ',=' '  ' | awk '{print $3, $4, $6}')
  echo run > $B/controller/remote_cmd.txt; sleep 0.7
  if [ -n "$lk" ]; then set -- $lk
    printf 'mission "lookpose%s"\n  go_to pose:%s(%s, %s, %s)\n' "$tag" "$here" $1 $2 $3 > $B/controller/remote_cmd.txt
    for i in $(seq 1 90); do grep -a -q -E "mission 'lookpose$tag': (completed|failed|aborted)" $L/controller.log && break; sleep 1; done
  fi
  echo run > $B/controller/remote_cmd.txt; sleep 0.7
  printf 'mission "look%s"\n  explore %s budget_s=90s\n' "$tag" "$here" > $B/controller/remote_cmd.txt
  for i in $(seq 1 60); do grep -a -q -E "mission 'look$tag': (completed|failed|aborted)" $L/controller.log && break; sleep 3; done; }
ok=0; SW0=0; scan_s=0
declare -A ARG=([1]=new [2]=room_1 [3]=new [4]=new [5]=room_3 [6]=room_1)
for leg in 1 2 3 4 5 6; do
  here=$(room_now); [ -z "$here" ] && here=room_1
  T0=$(now_ms); A0=$(nl ltsm_agent); D0=$(nl door_concept); R0=$(nl room_concept)
  a=${ARG[$leg]}; [ $leg = 4 ] && SW0=$A0
  ts=$(date +%s)
  run_leg $leg $a $here
  case "$r" in *wrong_room*) sleep 10; here=$(room_now); [ -z "$here" ] && here=room_1; run_leg "${leg}r" $a $here;; esac
  n=0
  while case "$r" in *door_unresolved*) true;; *) false;; esac && [ $(( $(date +%s)-ts )) -lt $SCAN_CAP_S ]; do
    n=$((n+1))
    if [ $leg = 5 ] && [ $n = 1 ]; then   # fallback: ltsm logs at swap time which face of the recognised room is the crossed door's
      sl=$(tail -n +$((SW0+1)) $L/ltsm_agent.log | grep -a '\[swap\] crossed door' | tail -1)
      d=$(python3 -I $S/recognition_report.py --leg5-door $L/ltsm_agent.log $SW0 $L/door_concept.log $D0)
      echo "leg 5: no recall child; ltsm line '${sl:-absent}' -> door '$d'"
      if [ -n "$d" ]; then a=$d; run_leg 5f $a $here; continue; fi
    fi
    scan_once "${leg}_$n"; run_leg "${leg}s$n" $a $here
  done
  scan_s=$(( $(date +%s)-ts ))
  if [ $leg = 4 ]; then   # the window closes on the Swap line (ltsm acts after the cross) or a 20 s settle timeout
    for i in $(seq 1 20); do tail -n +$((A0+1)) $L/ltsm_agent.log | grep -a -q '\[recog\] SWAP requested' && break; sleep 1; done; sleep 2
  fi
  now=$(room_now)
  echo "leg $leg ($here --cross $a--> now $now): $r"
  case "$r" in OK*) st=OK; ok=$((ok+1));; *) st=FAILED;; esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $leg $st $T0 $(now_ms) $A0 $(nl ltsm_agent) $D0 $(nl door_concept) $R0 $(nl room_concept) >> $TSV
  [ $st = OK ] || break
done
echo "=== $ok/6 legs"
echo stop > $B/controller/remote_cmd.txt
# restore the world: open the doors the tour crossed (DOOR_1, DOOR_0, DOOR_2)
for d in DOOR_1 DOOR_0 DOOR_2; do python3 $S/door.py open $d; done
# collect + stop the fleet (SIGTERM; ltsm persists its memory on exit), then grade
bash $S/run_crossing.sh crossing/$1 collect >/dev/null 2>&1
cp $B/ltsm_agent/etc/ltsm_memory.json crossing/$1/ltsm_memory.json 2>/dev/null
cp $WT/ltsm_agent/etc/recognition_trace.csv crossing/$1/recognition_trace.csv 2>/dev/null || cp $B/ltsm_agent/etc/recognition_trace.csv crossing/$1/recognition_trace.csv 2>/dev/null
python3 $S/recognition_report.py crossing/$1 | tee crossing/$1/recognition_report.txt
exit ${PIPESTATUS[0]}
