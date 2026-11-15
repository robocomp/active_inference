#!/bin/bash
# DETECTION CHECK for new concept agents (bedroom + bathroom of threerooms-piso.wbt): does retina detect each class on the
# Webots model at all, from where, how confidently? Run BEFORE cloning an agent for that class.
# Starts only retina + room_concept (retina's required peer; the controller is not started, so nothing drives), teleports the
# robot through fixed viewpoints, turning through yaws that put each object in view, and records every masks_zed slice with
# det_census.py. Ends with a per-class / per-viewpoint summary (det_census_report.py).
# usage: det_check.sh <run_name> [dwell_s=4]     Webots is SHARED: claim it first, release it after.
RUN=${1:?usage: det_check.sh <run_name> [dwell_s]}; DWELL=${2:-4}
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E
OUT=crossing/$RUN; mkdir -p $OUT/logs
export WT=/home/pbustos/robocomp/components/active_inference.worktrees/rt-build WT_AGENTS="room_concept"
export FLEET="retina room_concept"
echo stop > $B/controller/remote_cmd.txt
# start in the bedroom (room_2, the L room behind DOOR_1), facing the bed
NORUN=1 bash $S/run_crossing.sh $OUT -5.2 10.4 1.5708 >/dev/null 2>&1
sleep 20
python3 -I $S/det_census.py $OUT/det_census.csv > $OUT/logs/det_census.log 2>&1 &
CEN=$!
sleep 5
# viewpoint name, x, y, then the WORLD FACING directions (deg, 0 = +x, 90 = +y) to dwell at. ★Shadow's yaw 0 faces +y, so the
# teleport yaw is facing − 90° (detcheck2 used facings as yaws and looked 90° off). Objects (world): bed (-5.18,12.39),
# nightstands (-6.36,13.22) (-4.00,13.22); toilet (-0.87,12.87), bathtub (1.21,12.51), vanity/sink (1.44,9.71),
# mirror (1.68,9.71, z 1.45) on the east wall above the vanity.
VIEWS=(
  "bed_south      -5.20 10.40   75 90 105 112 67"
  "bed_east       -3.00 11.30   150 165 180"
  "bath_centre     0.30 11.20   125 55 -38 -53 -68"
  "vanity_front   -0.60 10.00   -20 -5 10"
)
for v in "${VIEWS[@]}"; do
  set -- $v; name=$1; x=$2; y=$3; shift 3
  for deg in "$@"; do
    th=$(python3 -c "import math; print(f'{math.radians($deg - 90):.4f}')")   # facing → Shadow yaw
    python3 $S/tp.py set $x $y $th > /dev/null 2>&1
    echo "$name $x $y $deg" > $OUT/pose.txt
    sleep $DWELL
  done
done
kill -TERM $CEN 2>/dev/null; sleep 1
bash $S/run_crossing.sh $OUT collect > /dev/null 2>&1   # stops retina + room_concept (SIGTERM)
python3 -I $S/det_census_report.py $OUT/det_census.csv | tee $OUT/report.txt
