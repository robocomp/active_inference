#!/bin/bash
# Door-crossing test (plan 2026-10-11, Phases 0-1): estimate room 1 from a start pose, then hand over for a joystick
# crossing; the fleet keeps running until `run_crossing.sh <dir> stop`.
# usage: run_crossing.sh <out_dir> [x y theta]        (world: tworooms-piso-door1open.wbt loaded in Webots)
#        NORUN=1 run_crossing.sh <out_dir> x y theta  (teleport, controller left STOPPED)
#        run_crossing.sh <out_dir> here                (no teleport, controller left STOPPED: the robot stays put)
#        run_crossing.sh <out_dir> run|stop|collect   (controller run / controller stop / copy logs + stop the fleet)
#        CHAT=1 run_crossing.sh ...                    (also start/stop chat_agent, the robot chat window; pass CHAT=1
#                                                        to `collect` too so it is stopped — SIGTERM only, never -9)
S=$(dirname "$(readlink -f "$0")"); B=/home/pbustos/robocomp/components/active_inference; R=$B/room_concept
OUT=$(readlink -f "${1:?usage: run_crossing.sh <out_dir> [x y theta | run | stop | collect]}"); mkdir -p $OUT/logs
FLEET=${FLEET:-"retina room_concept residual_concept controller door_concept ltsm_agent chair_concept table_concept"}   # override: FLEET="..." run_crossing.sh ...
[[ -n "$CHAT" ]] && FLEET="$FLEET chat_agent"
log(){ echo "$(date +%T) $*" | tee -a $OUT/crossing.log; }
stop_fleet(){ pkill -TERM -x door_concept; pkill -TERM -x controller; pkill -TERM -x residual_concep; pkill -TERM -x room_concept; pkill -TERM -x retina; pkill -TERM -x ltsm_agent
  pkill -TERM -x chair_concept; pkill -TERM -x table_concept
  [[ -n "$CHAT" ]] && pkill -TERM -x chat_agent
  for i in $(seq 1 30); do sleep 1
    pgrep -x room_concept >/dev/null || pgrep -x controller >/dev/null || pgrep -x residual_concep >/dev/null \
      || pgrep -x door_concept >/dev/null || pgrep -x retina >/dev/null \
      || pgrep -x chair_concept >/dev/null || pgrep -x table_concept >/dev/null \
      || { [[ -n "$CHAT" ]] && pgrep -x chat_agent >/dev/null; } || return 0; done
  log "WARN fleet did not stop in 30 s"; }
collect(){
  cp $R/tmp/proto_room.csv $R/tmp/door_filter.csv $R/tmp/promotion_select.csv $R/tmp/map_layout.csv $R/tmp/layout_refine.csv \
     $R/tmp/slam_shadow.csv $R/tmp/refine_input.bin $OUT/ 2>/dev/null
  cp $R/tmp/sdf_localizer/gt_error.csv $OUT/ 2>/dev/null
  [[ -d $R/tmp/snapshots ]] && cp -r $R/tmp/snapshots $OUT/
  cp $(ls -t $R/tmp/agent_events_*.jsonl | head -1) $OUT/events.jsonl 2>/dev/null
  cp $B/door_concept/etc/door_events.csv $B/door_concept/etc/door_existence_log.csv $B/door_concept/etc/ai2_log.csv $B/door_concept/etc/ai2_log_phi.csv $OUT/ 2>/dev/null
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
  # Re-teleport while the landed yaw is off the requested one (try72: a robot still wedged from the previous run
  # landed at yaw -0.64 instead of 0, faced away from the door, and the door was never born).
  for k in 1 2 3; do
    python3 $S/tp.py set $x $y $th >> $OUT/teleport.txt 2>&1; sleep 4
    python3 $S/tp.py get >> $OUT/teleport.txt 2>&1
    yaw=$(tail -1 $OUT/teleport.txt | awk '{for(i=1;i<=NF;i++) if($i=="yaw") print $(i+1)}')
    python3 -c "import math,sys; d=abs(math.remainder(float('$yaw')-float('$th'),2*math.pi)); sys.exit(0 if d<0.1 else 1)" 2>/dev/null && break
  done
  log "start ($x,$y,$th) landed: $(tail -1 $OUT/teleport.txt)"
fi
rm -rf $R/tmp/snapshots; rm -f $R/tmp/refine_input.bin $R/tmp/promotion_select.csv $R/tmp/proto_room.csv $R/tmp/map_layout.csv \
      $R/tmp/layout_refine.csv $R/tmp/door_filter.csv
# WT=<worktree> WT_AGENTS="controller door_concept ...": run those agents' binaries (and config) from a worktree,
# still from the main agent directory so data files and controller/remote_cmd.txt stay where the scripts expect them.
for a in $FLEET; do
  exe=bin/$a; cfg=etc/config.toml
  if [[ -n "$WT" && " $WT_AGENTS " == *" $a "* ]]; then exe=$WT/$a/bin/$a; cfg=$WT/$a/etc/config.toml; fi
  (cd $B/$a && DISPLAY=:0 setsid nohup $exe $cfg > $OUT/logs/$a.log 2>&1 < /dev/null &) ; sleep 4
done
if [[ "$2" == "here" || -n "$NORUN" ]]; then log "fleet up: $FLEET; controller left STOPPED"
else sleep 6; echo run > $B/controller/remote_cmd.txt; log "fleet up: $FLEET; controller RUN"; fi
