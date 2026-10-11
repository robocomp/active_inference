#!/bin/bash
# Multi-start estimation campaign. usage: run_campaign.sh [first_id] [last_id]
S=$(dirname "$(readlink -f "$0")"); B=/home/pbustos/robocomp/components/active_inference; R=$B/room_concept
OUT=${OUT:-$HOME/robocomp/components/active_inference/room_concept/synthetic_rooms}; mkdir -p $OUT
first=${1:-0}; last=${2:-9}; TIMEOUT_S=${TIMEOUT_S:-600}; REFINE_S=${REFINE_S:-60}
log(){ echo "$(date +%T) $*" | tee -a $OUT/campaign.log; }
stop_fleet(){ pkill -TERM -x controller; pkill -TERM -x residual_concep; pkill -TERM -x room_concept
  for i in $(seq 1 30); do sleep 1; pgrep -x room_concept >/dev/null || pgrep -x controller >/dev/null || pgrep -x residual_concep >/dev/null || return 0; done
  log "WARN fleet did not stop in 30 s"; }
for id in $(seq $first $last); do
  read x y th < <(awk -F, -v k=$id 'NR>1 && $1==k{print $2,$3,$4}' $S/starts.csv)
  D=$OUT/run_$id; rm -rf $D; mkdir -p $D/logs
  log "run $id: start ($x,$y,$th)"
  stop_fleet
  python3 $S/tp.py set $x $y $th >> $D/teleport.txt 2>&1; sleep 4
  python3 $S/tp.py get >> $D/teleport.txt 2>&1; log "  landed: $(tail -1 $D/teleport.txt)"
  rm -f $R/tmp/refine_input.bin $R/tmp/promotion_select.csv $R/tmp/proto_room.csv $R/tmp/map_layout.csv $R/tmp/layout_refine.csv
  t0=$(date +%s)
  for a in room_concept residual_concept controller; do
    (cd $B/$a && DISPLAY=:0 setsid nohup bin/$a etc/config.toml > $D/logs/$a.log 2>&1 < /dev/null &) ; sleep 4
  done
  sleep 6; echo run > $B/controller/remote_cmd.txt
  promoted_at=""; status=timeout
  while (( $(date +%s) - t0 < TIMEOUT_S )); do
    sleep 5
    pgrep -x room_concept >/dev/null || { status=room_died; break; }
    p=$(tail -1 $R/tmp/proto_room.csv 2>/dev/null | awk -F, '{print $40}')   # column 'promoted'
    if [[ "$p" == "1" && -z "$promoted_at" ]]; then promoted_at=$(( $(date +%s) - t0 )); log "  promoted at ${promoted_at}s (wall clock since launch)"; fi
    if [[ -n "$promoted_at" ]] && (( $(date +%s) - t0 - promoted_at >= REFINE_S )); then status=ok; break; fi
  done
  echo stop > $B/controller/remote_cmd.txt; sleep 1
  log "  status $status"
  echo "status=$status promoted_wall_s=$promoted_at" > $D/status.txt
  cp $R/tmp/promotion_select.csv $R/tmp/map_layout.csv $R/tmp/layout_refine.csv $R/tmp/refine_input.bin $R/tmp/proto_room.csv $R/tmp/slam_shadow.csv $D/ 2>/dev/null
  cp $R/tmp/sdf_localizer/gt_error.csv $D/ 2>/dev/null
  cp $(ls -t $R/tmp/agent_events_*.jsonl | head -1) $D/events.jsonl
  cp $B/controller/affordance_select.jsonl $D/ 2>/dev/null
  f=$(ls -t $B/layouts/learnt/*.svg | head -1); [[ $(stat -c %Y $f) -ge $t0 ]] && cp $f $D/learnt.svg
done
stop_fleet; log "campaign $first..$last done"
