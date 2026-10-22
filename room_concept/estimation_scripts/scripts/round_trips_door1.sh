#!/bin/bash
# Round trips through DOOR_1 (door_1: room_1 <-> room_3, the large rectangular room) in threerooms-piso.wbt: 3 round
# trips = 6 crossings, each leg explores the current room then crosses. Fresh fleet, fresh ltsm memory and room-name
# counter. Same scaffolding as three_rooms_tour.sh.
# usage: three_rooms_tour.sh <run_name> [explore_s=60]   (env WT defaults to the interaction-anchor worktree, WT_AGENTS to
# "controller door_concept table_concept room_concept"; run_crossing.sh is always the MAIN checkout's ($E), so it works from either)
LOCK=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ai_fleet_tour.lock; echo $$ > $LOCK; trap 'rm -f $LOCK' EXIT
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E; L=crossing/$1/logs
EXPLORE_S=${2:-60}
export WT=${WT:-/home/pbustos/robocomp/components/active_inference.worktrees/interaction-anchor} WT_AGENTS=${WT_AGENTS:-"controller door_concept table_concept room_concept"}   # worktree binaries; data + run_crossing.sh stay in the main checkout (B)
export FLEET="retina room_concept residual_concept controller door_concept ltsm_agent table_concept"
echo stop > $B/controller/remote_cmd.txt
rm -f ../etc/last_robot_pose.txt ../etc/room_name_hwm.txt $B/ltsm_agent/etc/ltsm_memory.json
NORUN=1 bash scripts/run_crossing.sh crossing/$1 -5.33 3.66 0.58 >/dev/null 2>&1
python3 scripts/door.py close; python3 scripts/door.py close DOOR_0
for i in $(seq 1 150); do grep -a -q "created instance for node 'door_1'" $L/door_concept.log 2>/dev/null && break; sleep 2; done
grep -a -q "created instance for node 'door_1'" $L/door_concept.log || echo "WARN: 'door_1' not born yet in 300 s (leg 1 explores first)"
sleep 5
room_now() { grep -a "current room changed" $L/controller.log | tail -1 | awk '{print $NF}'; }
DOORS=(door_1 door_1 door_1 door_1 door_1 door_1)
WANT=(room_3 room_1 room_3 room_1 room_3 room_1)
ok=0
for leg in 1 2 3 4 5 6; do
  here=$(room_now); [ -z "$here" ] && here=room_1
  d=${DOORS[$((leg-1))]}
  echo run > $B/controller/remote_cmd.txt; sleep 2
  printf 'mission "leg %s"\n  explore %s budget_s=%ss\n  cross %s retry=2\n' $leg $here $EXPLORE_S $d > $B/controller/remote_cmd.txt
  t0=$(date +%s); r="FAILED (no end in 600 s)"
  for i in $(seq 1 200); do
    if grep -a -q "mission 'leg $leg': completed" $L/controller.log; then r="OK in $(( $(date +%s)-t0 )) s"; break; fi
    if grep -a -q -E "mission 'leg $leg': (failed|aborted)" $L/controller.log; then
      r="FAILED: $(grep -a -o -E "mission 'leg $leg': (failed|aborted)[^\"]*" $L/controller.log | tail -1)"; break; fi
    sleep 3
  done
  now=$(room_now)
  echo "leg $leg ($here --$d--> want ${WANT[$((leg-1))]}): $r  (now in $now)"
  case "$r" in OK*) ok=$((ok+1));; *) break;; esac
done
echo "=== $ok/6 legs"
echo stop > $B/controller/remote_cmd.txt
