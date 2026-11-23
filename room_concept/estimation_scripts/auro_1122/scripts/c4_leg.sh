#!/bin/bash
# AuRo C4 control (2026-11-23): one gyro-bias leg in piso.wbt. Bridge GyroBias = +0.002 rad/s (set by hand, restart by the owner).
#   c4_leg.sh prep <on|off> <rep>   stop fleet, set MotionCalibApply, wipe motion state (cold), start fleet, explore until room_1, stop
#   c4_leg.sh save <on|off> <rep>   (after the route ends) collect, stop fleet, file heading log + motion state into the run folder
B=/home/pbustos/robocomp/components/active_inference; R=$B/room_concept; E=$R/estimation_scripts; S=$E/scripts
cond=$2; rep=$3; r=c4_${cond}_r$rep; OUT=$E/crossing/$r
case "$1" in
  prep)
    bash $S/run_crossing.sh $E/crossing/c4_prev collect > /dev/null 2>&1   # stops whatever fleet is up (collect = stop + copy, into a scratch dir)
    v=$([ "$cond" = on ] && echo true || echo false)
    sed -i -E "s/^MotionCalibApply    = (true|false) /MotionCalibApply    = $v /" $R/etc/config.toml
    grep -q "^MotionCalibApply    = $v " $R/etc/config.toml || { echo "MotionCalibApply NOT SET"; exit 1; }
    grep -q "^MotionCalibApplyMask = -1" $R/etc/config.toml || { echo "mask is not -1"; exit 1; }
    rm -f $R/etc/motion_calib_state.csv
    mkdir -p $OUT; echo "cond=$cond MotionCalibApply=$v mask=-1 bridge_GyroBias=$(grep -a 'SensorNoise\] ON' ~/.local/logs/bridge.out | tail -1 | grep -o 'bias=[^ ]*') $(date +%T)" > $OUT/c4_conditions.txt
    (FLEET="room_concept residual_concept controller" NORUN=1 bash $S/run_crossing.sh $OUT here > /dev/null 2>&1 &)
    for i in $(seq 1 60); do [ -s $OUT/logs/controller.log ] && grep -a -q "operating\|Operating" $OUT/logs/controller.log && break; sleep 2; done
    sleep 15; echo run > $B/controller/remote_cmd.txt
    for i in $(seq 1 150); do grep -a -q "after 'room_1'" $OUT/logs/controller.log 2>/dev/null && break; sleep 2; done
    # never stop a route the operator already started (11-23 c4_on_r1: this stop disarmed the route at wp 38)
    if grep -a -q "\[mission\] START 'calib turns'" $OUT/logs/controller.log; then echo "$r route ALREADY RUNNING — no stop sent"; exit 0; fi
    echo stop > $B/controller/remote_cmd.txt; sleep 2
    grep -a -q "after 'room_1'" $OUT/logs/controller.log && echo "$r READY: room_1 exists, controller stopped" || echo "$r NOT READY: no room_1 after 5 min";;
  save)
    bash $S/run_crossing.sh $OUT collect > /dev/null 2>&1
    h=$(ls -t $R/tmp/heading/heading_*.csv | head -1); cp -p $h $OUT/; cp -p $R/etc/motion_calib_state.csv $OUT/ 2>/dev/null
    grep -a "\[mission\] STOP 'calib turns'" $OUT/logs/controller.log | tail -1
    echo "$r saved: $(basename $h)";;
esac
