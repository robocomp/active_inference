#!/bin/bash
# Door-crossing reliability test WITH affordance-driven exploration between crossings: fresh fleet, 6 legs through
# the main door, each leg one mission script run by the controller's interpreter (mission_text.h):
#     mission "leg N"
#       explore <current room> budget_s=<S>s     ← the affordance selector has the base for S seconds (from leg 2)
#       cross door_1                              ← then the door passage, on the measured gap
# usage: round_trips_mission.sh <run_name> [explore_s=60]   (same env as round_trips.sh: WT / WT_AGENTS)
# TOUR LOCK (shared with chat_agent's fleet tool): held while this tour runs, i.e. while this PID is alive; chat_agent
# refuses to start/stop/restart fleet agents meanwhile. Removed on any exit; a crashed script's PID is dead, so it never sticks.
LOCK=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ai_fleet_tour.lock; echo $$ > $LOCK; trap 'rm -f $LOCK' EXIT
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E; L=crossing/$1/logs
EXPLORE_S=${2:-60}
echo stop > $B/controller/remote_cmd.txt
rm -f ../etc/last_robot_pose.txt $B/ltsm_agent/etc/ltsm_memory.json
NORUN=1 bash scripts/run_crossing.sh crossing/$1 -5.0 6.8 0 >/dev/null 2>&1
python3 scripts/door.py close; python3 scripts/door.py close DOOR_0
for i in $(seq 1 150); do grep -a -q "created instance for node 'door_1'" $L/door_concept.log 2>/dev/null && break; sleep 2; done
if ! grep -a -q "created instance for node 'door_1'" $L/door_concept.log; then
  echo "ABORT: 'door_1' never born in 300 s"; bash scripts/run_crossing.sh crossing/$1 collect >/dev/null 2>&1; exit 1
fi; sleep 5
room_now() { grep -a "current room changed" $L/controller.log | tail -1 | awk '{print $NF}'; }
ok=0
for leg in 1 2 3 4 5 6; do
  here=$(room_now); [ -z "$here" ] && here=room_1
  echo run > $B/controller/remote_cmd.txt; sleep 2   # arm the base output, as round_trips.sh does before each leg
  {
    echo "mission \"leg $leg\""
    [ $leg -gt 1 ] && echo "  explore $here budget_s=${EXPLORE_S}s"
    echo "  cross door_1 retry=2"
  } > $B/controller/remote_cmd.txt
  t0=$(date +%s); r="FAILED (no end in 600 s)"
  for i in $(seq 1 200); do
    if grep -a -q "mission 'leg $leg': completed" $L/controller.log; then r="OK in $(( $(date +%s)-t0 )) s"; break; fi
    if grep -a -q -E "mission 'leg $leg': (failed|aborted)" $L/controller.log; then
      r="FAILED: $(grep -a -o -E "mission 'leg $leg': (failed|aborted)[^\"]*" $L/controller.log | tail -1)"; break; fi
    sleep 3
  done
  echo "leg $leg (explore ${EXPLORE_S}s in $here, then cross door_1): $r  (now in $(room_now))"
  case "$r" in OK*) ok=$((ok+1));; *) break;; esac
done
echo "=== $ok/6 legs (explore + cross)"
echo stop > $B/controller/remote_cmd.txt
bash scripts/run_crossing.sh crossing/$1 collect >/dev/null 2>&1; echo "fleet stopped"
