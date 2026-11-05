#!/bin/bash
# SPIKE (exploration-tour design, risks 1-3): Estimate mode from an EMPTY memory.
#   leg 1: explore room_1 (budget 300 s)   leg 2: cross new   leg 3: explore the new room (budget 300 s)
# Watches: start-room promotion, crossing-born room estimation + promotion, doors found. Config restored on exit.
# usage: spike_explore.sh <run_name>
RUN=${1:?run name}
LOCK=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ai_fleet_tour.lock
if [ -s $LOCK ] && ps -p $(cat $LOCK) >/dev/null 2>&1; then echo "tour lock held by $(cat $LOCK)"; exit 3; fi
echo $$ > $LOCK
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E; L=crossing/$RUN/logs
export WT=/home/pbustos/robocomp/components/active_inference.worktrees/rt-build
export WT_AGENTS="controller door_concept room_concept residual_concept ltsm_agent"
export FLEET="retina room_concept residual_concept controller door_concept ltsm_agent"
RC_CFG=$WT/room_concept/etc/config.toml
cp $RC_CFG $RC_CFG.spike_bak
trap 'mv -f $RC_CFG.spike_bak $RC_CFG; echo stop > $B/controller/remote_cmd.txt; rm -f $LOCK' EXIT
sed -i 's/^MapMode = "[a-z]*"/MapMode = "estimate"/' $RC_CFG
grep -q '^MapMode = "estimate"' $RC_CFG || { echo "could not set MapMode=estimate"; exit 1; }
echo stop > $B/controller/remote_cmd.txt
# EMPTY memory: archive (never just delete) what is there, then clear
STAMP=$(date +%Y%m%d_%H%M%S); mkdir -p crossing/ltsm_archive crossing/door_identities_archive
[ -f $B/ltsm_agent/etc/ltsm_memory.json ] && cp $B/ltsm_agent/etc/ltsm_memory.json crossing/ltsm_archive/ltsm_memory_before_${RUN}_$STAMP.json
[ -f $B/door_concept/etc/door_identities.csv ] && cp $B/door_concept/etc/door_identities.csv crossing/door_identities_archive/door_identities_before_${RUN}_$STAMP.csv
for h in $B/ltsm_agent/etc/ltsm_history $WT/ltsm_agent/etc/ltsm_history; do
  [ -d "$h" ] && [ -n "$(ls -A "$h" 2>/dev/null)" ] && mv "$h" crossing/ltsm_archive/ltsm_history_before_${RUN}_${STAMP}_$(basename $(dirname $(dirname $(dirname "$h"))))
done
mkdir -p $B/ltsm_agent/etc/ltsm_history
rm -f ../etc/last_robot_pose.txt ../etc/room_name_hwm.txt $B/ltsm_agent/etc/ltsm_memory.json $B/ltsm_agent/etc/ltsm_memory.json.prev $B/door_concept/etc/door_identities.csv
rm -rf $B/retina/etc/place_maps/
for d in DOOR_0 DOOR_1 DOOR_2; do python3 scripts/door.py close $d; done
sleep 3
NORUN=1 bash scripts/run_crossing.sh crossing/$RUN -5.33 3.66 0.58 >/dev/null 2>&1
sleep 25
room_now() { grep -a "current room changed" $L/controller.log | tail -1 | awk '{print $NF}'; }
leg() {   # leg <n> <mission body> <timeout_s>
  echo run > $B/controller/remote_cmd.txt; sleep 2
  printf 'mission "leg %s"\n%s\n' "$1" "$2" > $B/controller/remote_cmd.txt
  local t0=$(date +%s) r="TIMEOUT ($3 s)"
  while (( $(date +%s) - t0 < $3 )); do
    if grep -a -q "mission 'leg $1': completed" $L/controller.log; then r="OK in $(( $(date +%s)-t0 )) s"; break; fi
    if grep -a -q -E "mission 'leg $1': (failed|aborted)" $L/controller.log; then
      r="FAILED: $(grep -a -o -E "mission 'leg $1': (failed|aborted)[^\"]*" $L/controller.log | tail -1)"; break; fi
    sleep 3
  done
  echo "leg $1: $r  room now: $(room_now)" | tee -a crossing/$RUN/legs.txt
}
leg 1 "  explore room_1 budget_s=300s" 360
leg 2 "  cross new retry=2" 600
here=$(room_now); [ -z "$here" ] && here=room_2
leg 3 "  explore $here budget_s=300s" 360
python3 scripts/tp.py get > crossing/$RUN/end_pose.txt 2>&1
bash scripts/run_crossing.sh crossing/$RUN collect >/dev/null 2>&1
echo "done: crossing/$RUN"
