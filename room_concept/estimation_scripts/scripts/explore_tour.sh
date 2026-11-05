#!/bin/bash
# EXPLORATION TOUR from scratch (spec docs/superpowers/specs/2026-11-05-exploration-tour-design.md): Estimate mode, EMPTY
# memory, no scripted legs. The controller's exploration drive (`explore_tour on`, controller/src/exploration_drive.h) explores
# each room until it is promoted (or ExploreTour.MaxRoomS), crosses the nearest found, uncrossed door, else returns through
# the room's first entry door; it ends with "[explore-tour] DONE". Config restored on exit; memory archived, never deleted.
# usage: explore_tour.sh <run_name>      env RUN_LIMIT_S (3600), WT (rt-build)
RUN=${1:?run name}
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCK=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ai_fleet_tour.lock
if [ -s $LOCK ] && ps -p $(cat $LOCK) >/dev/null 2>&1; then echo "tour lock held by $(cat $LOCK)"; exit 3; fi
echo $$ > $LOCK
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E; L=crossing/$RUN/logs
export WT=/home/pbustos/robocomp/components/active_inference.worktrees/rt-build
export WT_AGENTS="controller door_concept room_concept residual_concept ltsm_agent"
export FLEET="retina room_concept residual_concept controller door_concept ltsm_agent"
RC_CFG=$WT/room_concept/etc/config.toml
LT_CFG=$WT/ltsm_agent/etc/config.toml
cp $RC_CFG $RC_CFG.spike_bak; cp $LT_CFG $LT_CFG.spike_bak
trap 'mv -f $RC_CFG.spike_bak $RC_CFG; mv -f $LT_CFG.spike_bak $LT_CFG; echo stop > $B/controller/remote_cmd.txt; rm -f $LOCK' EXIT
sed -i 's/^MapMode = "[a-z]*"/MapMode = "estimate"/' $RC_CFG
# RECOGNITION ON (as warm_start_tour.sh; explore_tour2: re-entering the start room through its second doorway, recognition
# off, gave a NEW room_4 whose doors all looked uncrossed — the tour could not end)
sed -i '/^\[Recognition\]/,/^\[/ s/^enabled = false/enabled = true/' $LT_CFG
grep -q '^enabled = true   # RoomSwap' $LT_CFG || { echo "could not set Recognition.enabled"; exit 1; }
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
now_ms() { date +%s%3N; }
nl() { wc -l < $L/$1.log 2>/dev/null || echo 0; }
# GROUND TRUTH at 1 Hz (as warm_start_tour.sh): wall ms, Webots pose, every agent log's line count
GT=crossing/$RUN/gt_pose.tsv; printf 't_ms\tx\ty\tyaw\troom_concept\tltsm_agent\tdoor_concept\tcontroller\n' > $GT
( while [ -f $LOCK ]; do
    p=$(python3 scripts/tp.py get 2>/dev/null | tail -1 | awk '{print $1"\t"$2"\t"$5}')
    [ -n "$p" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\n' $(now_ms) "$p" $(nl room_concept) $(nl ltsm_agent) $(nl door_concept) $(nl controller) >> $GT
    sleep 1
  done ) &
RUN_LIMIT_S=${RUN_LIMIT_S:-3600}
echo run > $B/controller/remote_cmd.txt; sleep 2
echo "explore_tour on" > $B/controller/remote_cmd.txt
t0=$(date +%s); result="RUN LIMIT ($RUN_LIMIT_S s)"
while (( $(date +%s) - t0 < RUN_LIMIT_S )); do
  if grep -a -q "\[explore-tour\] DONE" $L/controller.log; then result="DONE in $(( $(date +%s)-t0 )) s"; break; fi
  sleep 5
done
echo "explore_tour: $result" | tee crossing/$RUN/result.txt
echo "explore_tour off" > $B/controller/remote_cmd.txt; sleep 2
# the drive's CSV from THIS run only (final review M2: the main checkout keeps an older one)
CSV=$(ls -t $B/controller/etc/exploration_drive.csv $WT/controller/etc/exploration_drive.csv 2>/dev/null | head -1)
if [ -n "$CSV" ] && [ "$CSV" -nt crossing/$RUN/legs.txt -o "$(stat -c %Y $CSV)" -ge "$t0" ]; then cp $CSV crossing/$RUN/
else echo "WARN: no exploration_drive.csv written by this run (newest: ${CSV:-none})"; fi
python3 scripts/tp.py get > crossing/$RUN/end_pose.txt 2>&1
bash scripts/run_crossing.sh crossing/$RUN collect >/dev/null 2>&1
python3 $S/grade_explore_tour.py crossing/$RUN
echo "done: crossing/$RUN"
