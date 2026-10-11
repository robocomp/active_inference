#!/bin/bash
# Door-crossing test (plan 2026-10-11, Phases 0-1): estimate room 1 from a start pose, then hand over for a joystick
# crossing; the fleet keeps running until `run_crossing.sh <dir> stop`.
# usage: run_crossing.sh <out_dir> [x y theta]        (world: tworooms-piso-door1open.wbt loaded in Webots)
#        NORUN=1 run_crossing.sh <out_dir> x y theta  (teleport, controller left STOPPED)
#        run_crossing.sh <out_dir> here                (no teleport, controller left STOPPED: the robot stays put)
#        run_crossing.sh <out_dir> run|stop|collect   (controller run / controller stop / copy logs + stop the fleet)
S=$(dirname "$(readlink -f "$0")"); B=/home/pbustos/robocomp/components/active_inference; R=$B/room_concept
OUT=$(readlink -f "${1:?usage: run_crossing.sh <out_dir> [x y theta | run | stop | collect]}"); mkdir -p $OUT/logs
FLEET="retina room_concept residual_concept controller door_concept"
log(){ echo "$(date +%T) $*" | tee -a $OUT/crossing.log; }
stop_fleet(){ pkill -TERM -x door_concept; pkill -TERM -x controller; pkill -TERM -x residual_concep; pkill -TERM -x room_concept; pkill -TERM -x retina
  for i in $(seq 1 30); do sleep 1
    pgrep -x room_concept >/dev/null || pgrep -x controller >/dev/null || pgrep -x residual_concep >/dev/null \
      || pgrep -x door_concept >/dev/null || pgrep -x retina >/dev/null || return 0; done
  log "WARN fleet did not stop in 30 s"; }
collect(){
  cp $R/tmp/proto_room.csv $R/tmp/door_filter.csv $R/tmp/promotion_select.csv $R/tmp/map_layout.csv $R/tmp/layout_refine.csv \
     $R/tmp/slam_shadow.csv $R/tmp/refine_input.bin $OUT/ 2>/dev/null
  cp $R/tmp/sdf_localizer/gt_error.csv $OUT/ 2>/dev/null
  [[ -d $R/tmp/snapshots ]] && cp -r $R/tmp/snapshots $OUT/
  cp $(ls -t $R/tmp/agent_events_*.jsonl | head -1) $OUT/events.jsonl 2>/dev/null
  cp $B/door_concept/etc/door_events.csv $B/door_concept/etc/door_existence_log.csv $OUT/ 2>/dev/null
  for f in $(ls -t $B/layouts/learnt/*.svg | head -3); do cp $f $OUT/; done
}
case "$2" in
  run)     echo run  > $B/controller/remote_cmd.txt; log "controller RUN"; exit 0;;
  stop)    echo stop > $B/controller/remote_cmd.txt; log "controller STOP"; exit 0;;
  collect) echo stop > $B/controller/remote_cmd.txt; collect; stop_fleet; log "collected + fleet stopped"; exit 0;;
esac
stop_fleet
if [[ "$2" == "here" ]]; then
  python3 $S/tp.py get >> $OUT/teleport.txt 2>&1; log "start HERE (no teleport): $(tail -1 $OUT/teleport.txt)"
else
  x=${2:--5.187}; y=${3:-8.441}; th=${4:-2.496}   # apartment start 1: the one nearest DOOR_1
  python3 $S/tp.py set $x $y $th >> $OUT/teleport.txt 2>&1; sleep 4
  python3 $S/tp.py get >> $OUT/teleport.txt 2>&1; log "start ($x,$y,$th) landed: $(tail -1 $OUT/teleport.txt)"
fi
rm -rf $R/tmp/snapshots; rm -f $R/tmp/refine_input.bin $R/tmp/promotion_select.csv $R/tmp/proto_room.csv $R/tmp/map_layout.csv \
      $R/tmp/layout_refine.csv $R/tmp/door_filter.csv
for a in $FLEET; do
  (cd $B/$a && DISPLAY=:0 setsid nohup bin/$a etc/config.toml > $OUT/logs/$a.log 2>&1 < /dev/null &) ; sleep 4
done
if [[ "$2" == "here" || -n "$NORUN" ]]; then log "fleet up: $FLEET; controller left STOPPED"
else sleep 6; echo run > $B/controller/remote_cmd.txt; log "fleet up: $FLEET; controller RUN"; fi
