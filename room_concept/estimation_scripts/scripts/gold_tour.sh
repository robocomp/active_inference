#!/bin/bash
# GOLD tour (2026-10-22 layout): room_1 -DOOR_1-> room_3 -> room_1 -DOOR_0-> room_2 -DOOR_2-> room_3 (never-crossed door into a KNOWN room).
# -DOOR_2-> room_2 (the LOOP: no passage memory for DOOR_2) -> room_3 -DOOR_1-> room_1. DOOR_2's name is discovered
# at run time (a door born in room_3 that is not the crossed door_1). Reports which room the loop entry became.
# usage: loop_tour.sh <run_name> [explore_s=60]   (env WT / WT_AGENTS as round_trips_door1.sh)
LOCK=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ai_fleet_tour.lock; echo $$ > $LOCK; trap 'rm -f $LOCK' EXIT
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E; L=crossing/$1/logs
EXPLORE_S=${2:-60}
export WT=${WT:-/home/pbustos/robocomp/components/active_inference.worktrees/interaction-anchor} WT_AGENTS=${WT_AGENTS:-"controller door_concept table_concept room_concept"}
export FLEET="retina room_concept residual_concept controller door_concept ltsm_agent table_concept"
echo stop > $B/controller/remote_cmd.txt
rm -f ../etc/last_robot_pose.txt ../etc/room_name_hwm.txt $B/ltsm_agent/etc/ltsm_memory.json $B/ltsm_agent/etc/ltsm_memory.json.prev
# close every door BEFORE the fleet starts, so room_1's first estimate cannot see through a door left open (user 10-23)
python3 scripts/door.py close; python3 scripts/door.py close DOOR_0; python3 scripts/door.py close DOOR_2
sleep 3
NORUN=1 bash scripts/run_crossing.sh crossing/$1 -5.33 3.66 0.58 >/dev/null 2>&1
sleep 25
# wait until a room exists (door_concept loads its polygon) — the controller rejects 'explore room_1' with wrong_room otherwise (gold5/gold7)
for i in $(seq 1 90); do grep -a -q 'room polygon LOADED' $L/door_concept.log 2>/dev/null && break; sleep 2; done; sleep 5
room_now() { grep -a "current room changed" $L/controller.log | tail -1 | awk '{print $NF}'; }
# newest door born in the current room whose name is not $1 (after the last room switch in door_concept.log)
new_door() { awk '/\[room\] LET GO/{buf=""} /created instance for node/{buf=buf"\n"$0} END{print buf}' $L/door_concept.log \
             | grep -o "node 'door_[0-9]*'" | tr -d "'" | awk '{print $2}' | grep -v -x "$1" | tail -1; }
run_leg() {   # leg explore_s door here
  echo run > $B/controller/remote_cmd.txt; sleep 2
  printf 'mission "leg %s"\n  explore %s until=explored(%s)>=0.9 budget_s=%ss\n  cross %s retry=2\n' $1 $4 $4 $2 $3 > $B/controller/remote_cmd.txt
  t0=$(date +%s); r="FAILED (no end in 600 s)"
  for i in $(seq 1 200); do
    grep -a -q "mission 'leg $1': completed" $L/controller.log && { r="OK in $(( $(date +%s)-t0 )) s"; break; }
    grep -a -q -E "mission 'leg $1': (failed|aborted)" $L/controller.log && { r="FAILED: $(grep -a -o -E "mission 'leg $1': (failed|aborted)[^\"]*" $L/controller.log | tail -1)"; break; }
    sleep 3
  done; }
ok=0; d2=""
for leg in 1 2 3 4 5 6; do
  here=$(room_now); [ -z "$here" ] && here=room_1
  case $leg in 1|2) d=door_1;; 3|6) d=door_2;;
    4) # find DOOR_2 from room_3: explore first if it is not born yet
       d2=$(new_door door_2); if [ -z "$d2" ]; then run_explore=1; echo run > $B/controller/remote_cmd.txt; sleep 2
         printf 'mission "scan"\n  explore %s budget_s=90s\n' "$here" > $B/controller/remote_cmd.txt
         for i in $(seq 1 40); do grep -a -q "mission 'scan': \(completed\|failed\|aborted\)" $L/controller.log && break; sleep 3; done
         d2=$(new_door door_2); fi
       [ -z "$d2" ] && { echo "leg 4: DOOR_2 never born in $here"; break; }; d=$d2;;
    5) d=$d2;; esac
  run_leg $leg $EXPLORE_S $d $here
  case "$r" in *wrong_room*) sleep 10; here=$(room_now); [ -z "$here" ] && here=room_1; run_leg "${leg}r" $EXPLORE_S $d $here;; esac
  now=$(room_now)
  echo "leg $leg ($here --$d--> now $now): $r"
  case "$r" in OK*) ok=$((ok+1));; *) break;; esac
done
echo "=== $ok/6 legs; DOOR_2 named '$d2'"
grep -a -E "PROMOTING|entered FRESH|recall_room_[0-9]+ posted|DONE #" crossing/$1/logs/ltsm_agent.log | cut -c1-170 > crossing/$1/room_identity.txt
grep -a "entered FRESH\|ltsm announces" $L/door_concept.log | cut -c1-150 >> crossing/$1/room_identity.txt
echo "room identity trace: crossing/$1/room_identity.txt ($(wc -l < crossing/$1/room_identity.txt) lines)"
echo stop > $B/controller/remote_cmd.txt
