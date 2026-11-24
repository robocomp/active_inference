#!/bin/bash
# PARKED TEST + A/B/A VISIT, live (spec 2026-11-23 shared-engine, Tests item 3; plan rev 2 Task P1).
#   parked_test.sh <run_name> <object_DEF>     e.g.  parked_test.sh p1_nightstand NIGHTSTAND_R   (BED_1, TOILET_1)
# env: DWELL_S (60) · CONCEPT_WT (default: THIS script's checkout — the countable binary built there) · WT (the fleet's
#      checkout, default main, as the tour scripts) · WBT_PATH (threerooms-piso.wbt)
# Webots must be running threerooms-piso.wbt (the warm tour's world) — the controller books it (CLAIM -> ACK -> FREE).
#
# The run, in order:
#  1. Model.ReplayDump = true in the bed / nightstand / toilet configs of CONCEPT_WT for this run only (restored on exit), so
#     every live run also leaves replay dumps (Task R1) — as the objects_replay1 run did from the r1-dump worktree.
#  2. MAIN's warm_start_tour.sh <run_name> crossing/fixtures/three_rooms_door0 (main's copy: its graph recorder and START
#     handling; the objects_replay1 flow): the fleet from WT, warm-started in room_1 (kitchen);
#     ON_FLEET_UP starts the three countable agents from CONCEPT_WT (concept_agents.sh); LEGS goes to the object's room by
#     relation (bedroom = room_2: LEGS="room_2"; bathroom = room_3: LEGS="room_2 room_3", the fixture's names), and AFTER_LEG
#     runs aba_visit.sh right after the leg that enters it. The tour then ends as usual (fleet stopped, logs collected).
#     concept_agents.sh / aba_visit.sh / the reports are THIS checkout's (its harvest copies offers.csv + delivered.csv).
#  3. The countables are stopped (SIGTERM), harvested (ai2_log / offers / delivered), and this run's replay dumps moved into
#     crossing/<run_name>/replay/.
#  4. parked_report.py and aba_report.py on the run (their output in crossing/<run_name>/{parked,aba}_report.txt).
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts
RUN=${1:?usage: parked_test.sh <run_name> <object_DEF>}; DEF=${2:?usage: parked_test.sh <run_name> <object_DEF>}
OUT=$E/crossing/$RUN
export CONCEPT_WT=${CONCEPT_WT:-$(cd $S/../../.. && pwd)}
export CONCEPT_AGENTS="nightstand_concept bed_concept toilet_concept"
export DWELL_S=${DWELL_S:-60}
WBT=${WBT_PATH:-/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt}
[ -e $OUT ] && { echo "$OUT exists: pick a new run name"; exit 2; }
[ -x $CONCEPT_WT/countable_concept/bin/countable_concept ] || { echo "no countable binary in $CONCEPT_WT: build it first"; exit 2; }

# the object's room, from its Webots pose, with the warm tour's own door lines (warm_start_tour.sh DOOR1_Y / DOOR2_X)
read -r OX OY <<< "$(python3 $S/webots_object.py get $DEF | awk '{print $1, $2}')"
[ -n "$OY" ] || { echo "Webots: no node $DEF (is the world loaded?)"; exit 2; }
ROOM=$(awk -v x="$OX" -v y="$OY" 'BEGIN{ if (x > -1.705) print "room_3"; else if (y > 9.3796) print "room_2"; else print "room_1" }')
case $ROOM in
  room_2) LEGS="room_2"; N=1;;
  room_3) LEGS="room_2 room_3"; N=2;;
  *) echo "$DEF is in room_1 (the start room): this script visits room_2 / room_3 objects only"; exit 2;;
esac
echo "$DEF at ($OX, $OY) -> $ROOM: LEGS=\"$LEGS\", visit after leg $N; countables from $CONCEPT_WT ($(git -C $CONCEPT_WT rev-parse --short HEAD))"

# 1. ReplayDump on for this run only
CFGS=""
for a in $CONCEPT_AGENTS; do
  c=$CONCEPT_WT/$a/etc/config.toml; cp $c $c.p1_bak; CFGS="$CFGS $c"
  sed -i 's/^ReplayDump = false/ReplayDump = true /' $c
  grep -q '^ReplayDump = true' $c || { echo "could not set Model.ReplayDump in $c"; for f in $CFGS; do mv -f $f.p1_bak $f; done; exit 1; }
done
trap 'for f in $CFGS; do mv -f $f.p1_bak $f; done' EXIT
# a gain-query request file left by an earlier run would be answered again by the fresh agents: start with none
for a in $CONCEPT_AGENTS; do rm -f $CONCEPT_WT/$a/etc/gain_query.txt; done
STAMP=$OUT.start; mkdir -p $E/crossing; touch $STAMP   # the dumps this run writes are newer than this

# 2. the tour, with the visit after the leg that enters the object's room
ON_FLEET_UP="bash $S/concept_agents.sh start $OUT" LEGS="$LEGS" AFTER_LEG="$N:bash $S/aba_visit.sh $OUT $DEF" \
  bash $E/scripts/warm_start_tour.sh $RUN $E/crossing/fixtures/three_rooms_door0
echo "tour exit $?"

# 3. stop the countables gracefully (they own graph nodes: SIGTERM only), harvest, move this run's dumps
bash $S/concept_agents.sh stop
bash $S/concept_agents.sh harvest $OUT
mkdir -p $OUT/replay
for a in $CONCEPT_AGENTS; do
  find $CONCEPT_WT/$a/etc/replay -maxdepth 1 -name '*.bin' -newer $STAMP -exec mv {} $OUT/replay/ \; 2>/dev/null
done
rm -f $STAMP
echo "replay dumps: $(ls $OUT/replay | wc -l) in $OUT/replay"

# 4. the reports
python3 -I $S/parked_report.py $OUT | tee $OUT/parked_report.txt
python3 -I $S/aba_report.py $OUT | tee $OUT/aba_report.txt
