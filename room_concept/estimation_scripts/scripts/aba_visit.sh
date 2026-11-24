#!/bin/bash
# A/B/A VISIT of one object (shared-engine plan rev 2, Task P1; AuRo test b amended; spec Tests item 3): A -> B -> A, parked
# DWELL_S at each, on a fleet that is ALREADY UP with the robot in the object's room (parked_test.sh calls it from the warm
# tour's AFTER_LEG hook). Standpoints: aba_geometry.py plan (A = 1.5 m in front of the object, B = 45 deg round it at the same
# range; the feasibility rule is in that file and in docs/superpowers/results/2026-11-25-p1-predictions.md).
#   aba_visit.sh <run_dir> <object_DEF>        env DWELL_S (60), MOVE_TIMEOUT_S (180), CONCEPT_WT (the countables' checkout),
#                                              WBT_PATH (the world)
# Per segment it logs to <run_dir>:
#   aba_segments.tsv   claim (= the go_to is sent) / arrival / dwell start / dwell end (wall ms, the clock of gt_pose.tsv and of
#                      the agents' t_ms), the scripted standpoint (world and the published room frame it was sent in), the
#                      robot's Webots pose at the dwell start, the go_to and face outcomes
#   aba_promises.tsv   AT THE CLAIM: every node's last offers.csv row of the object's class (t_ms <= claim) — the engine's
#                      promise (predicted_gain at the offer's standpoint), read live from the agent's own etc/offers.csv
#   aba_plan_<DEF>.json  the plan and the ground truth
# Missions go through controller/remote_cmd.txt like the tour scripts: go_to pose:<room>(x, y, yaw) (the controller does not
# drive the yaw), then `move turn_deg=` to face the object (computed from the Webots pose), then `wait s=`.
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); B=/home/pbustos/robocomp/components/active_inference
RUN=$(readlink -f "${1:?usage: aba_visit.sh <run_dir> <object_DEF>}"); DEF=${2:?usage: aba_visit.sh <run_dir> <object_DEF>}
L=$RUN/logs; WBT=${WBT_PATH:-/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt}
CWT=${CONCEPT_WT:-$(cd $S/../../.. && pwd)}
DWELL_S=${DWELL_S:-60}; MOVE_TIMEOUT_S=${MOVE_TIMEOUT_S:-180}
now_ms() { date +%s%3N; }
room_now() { grep -a "current room changed" $L/controller.log | tail -1 | awk '{print $NF}'; }
log() { echo "[aba $DEF] $(date +%T) $*"; }

PLAN=$RUN/aba_plan_$DEF.json
python3 $S/aba_geometry.py plan $WBT $DEF --live > $PLAN || { log "plan FAILED"; exit 1; }
read -r FEAS CLS CX CY AX AY AYAW BX BY BYAW <<< "$(python3 -c "
import json,sys; p=json.load(open('$PLAN')); t=p['truth']
if not p['feasible']: print('0', t['cls']); sys.exit()
print(1, t['cls'], t['cx'], t['cy'], *p['A'], *p['B'])")"
[ "$FEAS" = 1 ] || { log "NO FEASIBLE STANDPOINTS (see $PLAN)"; exit 1; }
log "$CLS at ($CX, $CY): A ($AX, $AY)  B ($BX, $BY)  dwell ${DWELL_S}s"
OFFERS=$CWT/${CLS}_concept/etc/offers.csv
SEG=$RUN/aba_segments.tsv; PROM=$RUN/aba_promises.tsv
[ -f $SEG ] || printf 'def\tseg\tstand_x\tstand_y\tstand_yaw\troom\tpub_x\tpub_y\tpub_yaw\tfit\tt_claim_ms\tt_arrive_ms\tt_dwell0_ms\tt_dwell1_ms\tgoto\tturn_deg\tface\tgt_x\tgt_y\tgt_yaw\tdwell\n' > $SEG
[ -f $PROM ] || printf 'def\tseg\tt_claim_ms\tnode\tt_ms\tstandpoint_x\tstandpoint_y\tstandpoint_yaw\tpromised_nats\n' > $PROM

# run_mission <name> <body> <timeout_s>: sets M (completed | failed ... | aborted ... | TIMEOUT)
run_mission() {
  echo run > $B/controller/remote_cmd.txt; sleep 0.7   # the controller reads remote_cmd.txt every 300 ms
  printf 'mission "%s"\n%s\n' "$1" "$2" > $B/controller/remote_cmd.txt
  M=TIMEOUT
  for i in $(seq 1 $(($3 * 2))); do
    m=$(grep -a -o -E "mission '$1': (completed|failed|aborted)[^\"]{0,80}" $L/controller.log | tail -1)
    [ -n "$m" ] && { M=${m#*: }; break; }
    sleep 0.5
  done
  [ "$M" = TIMEOUT ] && echo stop > $B/controller/remote_cmd.txt
  true; }

record_promises() {   # seg t_claim: every node's last offer of the class at the claim
  python3 - "$OFFERS" "$DEF" "$1" "$2" >> $PROM <<'EOF'
import csv, sys
path, d, seg, tc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
last = {}
try:
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                if int(r['t_ms']) <= tc: last[r['node']] = r
            except (KeyError, ValueError): pass
except FileNotFoundError: pass
for n, r in sorted(last.items()):
    print('\t'.join([d, seg, str(tc), n, r['t_ms'], r['standpoint_x'], r['standpoint_y'], r['standpoint_yaw'], r['promised_nats']]))
EOF
}

visit() {   # seg world_x world_y world_yaw
  local seg=$1 tc ta t0 t1 room pub fit go turn face gt dw
  tc=$(now_ms); record_promises $seg $tc
  room=$(room_now)
  read -r px py pyaw fit <<< "$(python3 $S/aba_geometry.py to_room $2 $3 $4)"
  log "$seg: claim at $tc, go_to pose:$room($px, $py, $pyaw) [$fit]"
  run_mission "aba_${DEF}_${seg}_go" "  go_to pose:${room}(${px}, ${py}, ${pyaw})" $MOVE_TIMEOUT_S; go=$M
  ta=$(now_ms)
  turn=$(python3 $S/aba_geometry.py face $CX $CY)
  # a zero turn is refused by the controller ("nothing to do"): the degenerate case, not a tolerance
  if [ "$turn" = "0.0" ] || [ "$turn" = "-0.0" ]; then face=none
  else run_mission "aba_${DEF}_${seg}_face" "  move turn_deg=${turn}deg" 60; face=$M; fi
  t0=$(now_ms); gt=$(python3 $S/tp.py get 2>/dev/null | tail -1 | awk '{print $1"\t"$2"\t"$5}')
  [ -n "$gt" ] || gt=$(printf 'nan\tnan\tnan')   # keep the row's columns when Webots does not answer
  log "$seg: go_to $go, face $turn deg $face; dwell ${DWELL_S}s from $t0 at $(echo $gt | tr '\t' ' ')"
  run_mission "aba_${DEF}_${seg}_dwell" "  wait s=${DWELL_S}s" $((DWELL_S + 30)); dw=$M
  t1=$(now_ms)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $DEF $seg $2 $3 $4 "$room" $px $py $pyaw \
    "$fit" $tc $ta $t0 $t1 "$go" "$turn" "$face" "$gt" "$dw" | tr ' ' '_' >> $SEG   # spaces inside a field -> _
}

visit A1 $AX $AY $AYAW
visit B  $BX $BY $BYAW
visit A2 $AX $AY $AYAW
log "done: $SEG"
