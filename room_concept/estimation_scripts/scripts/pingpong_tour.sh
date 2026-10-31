#!/bin/bash
# PING-PONG tour (user 2026-10-30: "We have to go from room1 to room 2 and back again six times before even thinking of going into
# room 3"). Webots threerooms-piso; only DOOR_1 (room_1 <-> room_2) is used: DOOR_0 and DOOR_2 are closed for the whole run.
# Doors are picked by RELATION, never by name (the controller resolves the argument of `cross`, mission_door_relation.h):
#   leg 1        cross new      a live door of room_1 with no recall_* child and no passage match  -> room_2 (new)
#   legs 2,4,..  cross room_1   the door with a recall_room_1 child (ltsm posts it under the door that leads there)
#   legs 3,5,..  cross room_2   the door with a recall_room_2 child
# Every leg is GRADED ON GROUND TRUTH, not on the controller's word: when the mission completes, the robot's true pose must be on
# the expected side of DOOR_1's line (recog_proto15 "completed" a leg on a false room switch with the robot in the doorway).
# usage: pingpong_tour.sh <run_name> [round_trips=6]      env WT / WT_AGENTS / LEG_TIMEOUT_S (default 600) / CROSS_NAME
# CROSS_NAME=door_1: every leg is `cross door_1` BY NAME instead of by relation — for a controller older than the relation
#   picker (the 10-19 known-good fleet, feat/interaction-anchor), so the same graded tour can bisect a regression.
# Writes crossing/<run>/legs.tsv (leg windows, controller verdict, true side), gt_pose.tsv (1 Hz true pose), graph_trace.csv
# (graph_recorder), and runs trace_check.py. Stop agents with SIGTERM only (run_crossing.sh collect does).
LOCK=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ai_fleet_tour.lock; echo $$ > $LOCK; trap 'rm -f $LOCK; [ -n "$REC_PID" ] && kill -TERM $REC_PID 2>/dev/null' EXIT
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # helpers come from THIS script's own directory (the worktree), data stays under $E
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E; L=crossing/$1/logs
TRIPS=${2:-6}; LEG_TIMEOUT_S=${LEG_TIMEOUT_S:-600}
export WT=${WT:-/home/pbustos/robocomp/components/active_inference.worktrees/rt-build} WT_AGENTS=${WT_AGENTS:-"controller door_concept table_concept room_concept residual_concept ltsm_agent"}
export FLEET="retina room_concept residual_concept controller door_concept ltsm_agent table_concept"
DOOR1_Y=9.3796   # DOOR_1's line in the world (threerooms-piso.wbt, DEF DOOR_1 translation): room_1 is y < it, room_2 is y > it
echo stop > $B/controller/remote_cmd.txt
# ── fresh state: no last pose, no name high-water mark, no ltsm memory, no door memory, no place maps ──────────────────────
# memories are ARCHIVED, never just deleted (user 10-29: keep the LTSM graph on disk)
[ -f $B/ltsm_agent/etc/ltsm_memory.json ] && mkdir -p crossing/ltsm_archive && cp $B/ltsm_agent/etc/ltsm_memory.json crossing/ltsm_archive/ltsm_memory_before_$1_$(date +%Y%m%d_%H%M%S).json
[ -f $B/door_concept/etc/door_identities.csv ] && mkdir -p crossing/door_identities_archive && cp $B/door_concept/etc/door_identities.csv crossing/door_identities_archive/door_identities_before_$1_$(date +%Y%m%d_%H%M%S).csv
rm -f ../etc/last_robot_pose.txt ../etc/room_name_hwm.txt $B/ltsm_agent/etc/ltsm_memory.json $B/ltsm_agent/etc/ltsm_memory.json.prev
rm -f $B/door_concept/etc/door_identities.csv
rm -rf $B/retina/etc/place_maps/; rm -f $WT/ltsm_agent/etc/recognition_trace.csv $B/ltsm_agent/etc/recognition_trace.csv
# DOOR_1 starts OPEN (2026-10-31: `cross new` takes the closest open door; with all three shut it took the bathroom's);
# DOOR_0 and DOOR_2 stay shut for the whole run
for d in DOOR_0 DOOR_2; do python3 $S/door.py close $d; done; python3 $S/door.py open DOOR_1
sleep 3
# graph recorder (passive DSR participant, agent id 28): every structural write with its writer — for trace_check.py
REC=$(cd $S/../../.. && pwd)/graph_recorder/bin/graph_recorder; mkdir -p crossing/$1
[ -x $REC ] && { setsid $REC crossing/$1/graph_trace.csv > crossing/$1/graph_recorder.log 2>&1 < /dev/null & REC_PID=$!; }
NORUN=1 bash $S/run_crossing.sh crossing/$1 -5.33 3.66 0.58 >/dev/null 2>&1
sleep 25
for i in $(seq 1 90); do grep -a -q 'room polygon LOADED' $L/door_concept.log 2>/dev/null && break; sleep 2; done; sleep 5
room_now() { grep -a "current room changed" $L/controller.log | tail -1 | awk '{print $NF}'; }
nl() { wc -l < $L/$1.log 2>/dev/null || echo 0; }
now_ms() { date +%s%3N; }
true_y() { python3 $S/tp.py get 2>/dev/null | tail -1 | awk '{print $2}'; }
# GROUND TRUTH at 1 Hz: wall ms, Webots pose, and every agent log's line count (maps any log line to the true pose then)
GT=crossing/$1/gt_pose.tsv; printf 't_ms\tx\ty\tyaw\troom_concept\tltsm_agent\tdoor_concept\tcontroller\n' > $GT
( while [ -f $LOCK ]; do
    p=$(python3 $S/tp.py get 2>/dev/null | tail -1 | awk '{print $1"\t"$2"\t"$5}')
    [ -n "$p" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\n' $(now_ms) "$p" $(nl room_concept) $(nl ltsm_agent) $(nl door_concept) $(nl controller) >> $GT
    sleep 1
  done ) &
TSV=crossing/$1/legs.tsv; printf 'leg\targ\tverdict\ttrue_side\ttrue_y\tstatus\tt0_ms\tt1_ms\tgraph_room\n' > $TSV
run_leg() {   # label cross_arg   (sets r)
  echo run > $B/controller/remote_cmd.txt; sleep 0.7   # the controller reads remote_cmd.txt every 300 ms
  printf 'mission "leg %s"\n  cross %s retry=2\n' $1 $2 > $B/controller/remote_cmd.txt
  local t0; t0=$(date +%s); r="FAILED (no end in ${LEG_TIMEOUT_S} s)"
  for i in $(seq 1 $((LEG_TIMEOUT_S * 2))); do
    grep -a -q "mission 'leg $1': completed" $L/controller.log && { r="OK in $(( $(date +%s)-t0 )) s"; break; }
    grep -a -q -E "mission 'leg $1': (failed|aborted)" $L/controller.log && { r="FAILED: $(grep -a -o -E "mission 'leg $1': (failed|aborted)[^\"]*" $L/controller.log | tail -1)"; break; }
    sleep 0.5
  done; }
# `cross new` only needs a door to APPEAR: when the controller reports door_unresolved, one explore, then the leg again (bounded)
explore_once() {
  echo run > $B/controller/remote_cmd.txt; sleep 0.7
  printf 'mission "look%s"\n  explore %s budget_s=90s\n' "$1" "$(room_now)" > $B/controller/remote_cmd.txt
  for i in $(seq 1 60); do grep -a -q -E "mission 'look$1': (completed|failed|aborted)" $L/controller.log && break; sleep 3; done; }
ok=0; n_legs=$((2 * TRIPS))
for leg in $(seq 1 $n_legs); do
  if [ -n "$CROSS_NAME" ]; then a=$CROSS_NAME; elif [ $leg = 1 ]; then a=new; elif [ $((leg % 2)) = 0 ]; then a=room_1; else a=room_2; fi
  want=$([ $((leg % 2)) = 1 ] && echo room_2 || echo room_1)   # the room the leg must END in (true side of DOOR_1)
  T0=$(now_ms); ts=$(date +%s)
  run_leg $leg $a; n=0
  while case "$r" in *door_unresolved*) true;; *) false;; esac && [ $(( $(date +%s)-ts )) -lt $LEG_TIMEOUT_S ]; do
    n=$((n+1)); explore_once "${leg}_$n"; run_leg "${leg}s$n" $a
  done
  sleep 2   # the pose settles past the door
  y=$(true_y); side=$(awk -v y="$y" -v d=$DOOR1_Y 'BEGIN{ if (y=="") print "unknown"; else print (y > d ? "room_2" : "room_1") }')
  case "$r" in OK*) [ "$side" = "$want" ] && st=OK || st=FALSE_OK;; *) st=FAILED;; esac
  [ $st = OK ] && ok=$((ok+1))
  echo "leg $leg (cross $a, want $want): controller '$r', true side $side (y=$y, door line $DOOR1_Y), graph room $(room_now) -> $st"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $leg $a "$r" $side "$y" $st $T0 $(now_ms) "$(room_now)" >> $TSV
  [ $st = OK ] || break
done
echo "=== $ok/$n_legs legs ($TRIPS round trips)"
echo stop > $B/controller/remote_cmd.txt
for d in DOOR_1 DOOR_0 DOOR_2; do python3 $S/door.py open $d; done
# collect + stop the fleet (SIGTERM; ltsm persists its memory on exit), then check the recorded trace
bash $S/run_crossing.sh crossing/$1 collect >/dev/null 2>&1
[ -n "$REC_PID" ] && { sleep 3; kill -TERM $REC_PID 2>/dev/null; }   # after the fleet's own cleanup writes
cp $B/ltsm_agent/etc/ltsm_memory.json crossing/$1/ltsm_memory.json 2>/dev/null
cp $B/door_concept/etc/door_identities.csv crossing/$1/door_identities.csv 2>/dev/null
python3 -I $S/trace_check.py crossing/$1 | tee crossing/$1/trace_check.txt
[ $ok = $n_legs ]
