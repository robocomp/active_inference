#!/bin/bash
# ROOM-RECOGNITION gold tour (spec 2026-10-22). Real route of gold_tour.sh, 6 legs:
#   1 room_1 --door_1 (Webots DOOR_1)--> room_2 (new)      2 back to room_1
#   3 room_1 --door_2 (Webots DOOR_0)--> room_3 (new)
#   4 room_3 --<door born in room_3 that is not door_2, Webots DOOR_2>--> the REMEMBERED room_2 (recognition + Swap)
#   5 back through DOOR_2's carried passage (ordinary recall)      6 room_3 --door_2--> room_1 (ordinary recall)
# DOOR_2's name is discovered at run time. Leg 4 scans (repeated explore) until that door is born, bounded by SCAN_CAP_S.
# usage: recognition_tour.sh <run_name> [explore_s=60]      env WT / WT_AGENTS / SCAN_CAP_S (default 600)
# Writes crossing/<run>/legs.tsv (leg windows + per-log line ranges), copies ltsm_memory.json, runs recognition_report.py.
# Stop agents with SIGTERM only (run_crossing.sh collect does); never edit this script while a run uses it.
LOCK=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ai_fleet_tour.lock; echo $$ > $LOCK; trap 'rm -f $LOCK' EXIT
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E; L=crossing/$1/logs
EXPLORE_S=${2:-60}; SCAN_CAP_S=${SCAN_CAP_S:-600}
export WT=${WT:-/home/pbustos/robocomp/components/active_inference.worktrees/rt-build} WT_AGENTS=${WT_AGENTS:-"controller door_concept table_concept room_concept ltsm_agent"}
export FLEET="retina room_concept residual_concept controller door_concept ltsm_agent table_concept"
echo stop > $B/controller/remote_cmd.txt
# fresh state: no last pose, no name high-water mark, no ltsm memory, no place maps
rm -f ../etc/last_robot_pose.txt ../etc/room_name_hwm.txt $B/ltsm_agent/etc/ltsm_memory.json $B/ltsm_agent/etc/ltsm_memory.json.prev
rm -rf $B/retina/etc/place_maps/; rm -f $WT/ltsm_agent/etc/recognition_trace.csv $B/ltsm_agent/etc/recognition_trace.csv
# close every door BEFORE the fleet starts (the cross mission opens the one it needs)
for d in DOOR_1 DOOR_0 DOOR_2; do python3 scripts/door.py close $d; done
sleep 3
NORUN=1 bash scripts/run_crossing.sh crossing/$1 -5.33 3.66 0.58 >/dev/null 2>&1
sleep 25
for i in $(seq 1 90); do grep -a -q 'room polygon LOADED' $L/door_concept.log 2>/dev/null && break; sleep 2; done; sleep 5
room_now() { grep -a "current room changed" $L/controller.log | tail -1 | awk '{print $NF}'; }
new_door() { awk '/\[room\] LET GO/{buf=""} /created instance for node/{buf=buf"\n"$0} END{print buf}' $L/door_concept.log \
             | grep -o "node 'door_[0-9]*'" | tr -d "'" | awk '{print $2}' | grep -v -x "$1" | tail -1; }
# LEG 6 ONLY (leg 5 comes from ltsm's '[swap] crossed door' line). Door to cross NEXT in room $1, other than $2; $3 = preferred;
# $4 = door_concept.log line count to window from. Candidates: doors RESTORED in that
# remembered room by door_concept (after a recognition swap the face of DOOR_2 carries the room's OWN restored name) plus doors
# born in it. Logs how it was found.
discover_door() {
  local room=$1 ex=$2 pref=$3 cands pick src
  cands=$( { tail -n +$((${4:-0}+1)) $L/door_concept.log | grep -a -o -E "'door_[0-9]+' RESTORED in remembered '$room'" | grep -o "door_[0-9]*"; new_door "$ex" | grep -v '^$'; } | grep -v -x "$ex" | awk '!s[$0]++')
  if [ -n "$pref" ] && echo "$cands" | grep -q -x "$pref"; then pick=$pref; src="preferred name present among restored/born doors of $room"
  else pick=$(echo "$cands" | tail -1); src="last restored/born door of $room other than $ex (candidates: $(echo $cands))"; fi
  echo "discover_door($room): '$pick' - $src" >&2; echo "$pick"; }
nl() { wc -l < $L/$1.log 2>/dev/null || echo 0; }
now_ms() { date +%s%3N; }
TSV=crossing/$1/legs.tsv; printf 'leg\tstatus\tt0_ms\tt1_ms\tltsm0\tltsm1\tdoor0\tdoor1\troom0\troom1\n' > $TSV
run_leg() {   # leg door here   (sets r)
  echo run > $B/controller/remote_cmd.txt; sleep 2
  printf 'mission "leg %s"\n  explore %s until=explored(%s)>=0.9 budget_s=%ss\n  cross %s retry=2\n' $1 $3 $3 $EXPLORE_S $2 > $B/controller/remote_cmd.txt
  t0=$(date +%s); r="FAILED (no end in 600 s)"
  for i in $(seq 1 200); do
    grep -a -q "mission 'leg $1': completed" $L/controller.log && { r="OK in $(( $(date +%s)-t0 )) s"; break; }
    grep -a -q -E "mission 'leg $1': (failed|aborted)" $L/controller.log && { r="FAILED: $(grep -a -o -E "mission 'leg $1': (failed|aborted)[^\"]*" $L/controller.log | tail -1)"; break; }
    sleep 3
  done; }
ok=0; d2=""; d5=""; D5=0; SW0=0; SD0=0; scan_s=0
for leg in 1 2 3 4 5 6; do
  here=$(room_now); [ -z "$here" ] && here=room_1
  T0=$(now_ms); A0=$(nl ltsm_agent); D0=$(nl door_concept); R0=$(nl room_concept)
  case $leg in 1|2) d=door_1;; 3) d=door_2;;
    4) SW0=$A0; SD0=$D0; d2=$(new_door door_2); ts=$(date +%s)
       # scan until DOOR_2 is born (the gold_tour driver gave up after ONE 90 s explore although it was born later)
       while [ -z "$d2" ] && [ $(( $(date +%s)-ts )) -lt $SCAN_CAP_S ]; do
         n=$(( $(date +%s)-ts )); echo run > $B/controller/remote_cmd.txt; sleep 2
         printf 'mission "scan%s"\n  explore %s budget_s=90s\n' $n "$here" > $B/controller/remote_cmd.txt
         for i in $(seq 1 40); do grep -a -q -E "mission 'scan$n': (completed|failed|aborted)" $L/controller.log && break; d2=$(new_door door_2); [ -n "$d2" ] && break; sleep 3; done
         d2=$(new_door door_2)
       done
       scan_s=$(( $(date +%s)-ts )); echo "leg 4: DOOR_2 scan took ${scan_s} s -> '${d2}'"
       [ -z "$d2" ] && { echo "leg 4: DOOR_2 never born in $here within ${SCAN_CAP_S} s"; break; }; d=$d2;;
    5) # the door back: ltsm logs, at swap time, which face of the recognised room is the crossed door's ("[swap] crossed door
       # rA_door_M <-> K face rN_door_K" or "... face new"); door_concept restores K's doors under their remembered base names
       sl=$(tail -n +$((SW0+1)) $L/ltsm_agent.log | grep -a '\[swap\] crossed door' | tail -1)
       d=$(python3 -I scripts/recognition_report.py --leg5-door $L/ltsm_agent.log $SW0 $L/door_concept.log $SD0)
       if [ -z "$d" ]; then echo "leg 5: door discovery failed (ltsm line: '${sl:-absent}')"
         printf '5\tDISCOVERY_FAILED\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $T0 $(now_ms) $A0 $(nl ltsm_agent) $D0 $(nl door_concept) $R0 $(nl room_concept) >> $TSV; break; fi
       echo "leg 5: door '$d' from ltsm: $sl"; d5=$d; D5=$D0;;
    6) # window from leg 5's ROOM SWITCH (last '[room] LET GO' in door_concept.log after leg 5 began), not leg 5's start
       sw=$(awk -v s=$D5 'NR>s && /\[room\] LET GO/{n=NR} END{print n+0}' $L/door_concept.log); [ "$sw" -gt 0 ] || sw=$D5
       d=$(discover_door "$(room_now)" "$d5" door_2 $sw); [ -z "$d" ] && d=door_2;; esac
  run_leg $leg $d $here
  case "$r" in *wrong_room*) sleep 10; here=$(room_now); [ -z "$here" ] && here=room_1; run_leg "${leg}r" $d $here;; esac
  if [ $leg = 4 ]; then   # the window closes on the Swap line (ltsm acts after the cross) or a 20 s settle timeout
    for i in $(seq 1 20); do tail -n +$((A0+1)) $L/ltsm_agent.log | grep -a -q '\[recog\] SWAP requested' && break; sleep 1; done; sleep 2
  fi
  now=$(room_now)
  echo "leg $leg ($here --$d--> now $now): $r"
  case "$r" in OK*) st=OK; ok=$((ok+1));; *) st=FAILED;; esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $leg $st $T0 $(now_ms) $A0 $(nl ltsm_agent) $D0 $(nl door_concept) $R0 $(nl room_concept) >> $TSV
  [ $st = OK ] || break
done
echo "=== $ok/6 legs; DOOR_2 named '$d2'; scan ${scan_s} s"
echo stop > $B/controller/remote_cmd.txt
# restore the world: open the doors the tour crossed (DOOR_1, DOOR_0, DOOR_2)
for d in DOOR_1 DOOR_0 DOOR_2; do python3 scripts/door.py open $d; done
# collect + stop the fleet (SIGTERM; ltsm persists its memory on exit), then grade
bash scripts/run_crossing.sh crossing/$1 collect >/dev/null 2>&1
cp $B/ltsm_agent/etc/ltsm_memory.json crossing/$1/ltsm_memory.json 2>/dev/null
cp $WT/ltsm_agent/etc/recognition_trace.csv crossing/$1/recognition_trace.csv 2>/dev/null || cp $B/ltsm_agent/etc/recognition_trace.csv crossing/$1/recognition_trace.csv 2>/dev/null
python3 scripts/recognition_report.py crossing/$1 | tee crossing/$1/recognition_report.txt
exit ${PIPESTATUS[0]}
