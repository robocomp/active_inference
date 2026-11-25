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
#   THE PROMISE, AT THE CLAIM (P1 follow-up): before the robot moves, a gain query for the SCRIPTED standpoint is appended to
#                      the agent's etc/gain_query.txt (countable_concept/src/countable_gain_query.h: "<id> * x y yaw", the
#                      published room frame, yaw = the engine's math heading facing the object); every instance of the class
#                      answers in etc/gain_query.csv with EpistemicPlanner::gain_at — the very computation of its offers — and the
#                      update serial. The script waits (<= 5 s) for each answer. Two queries per segment: at the CLAIM (query_id,
#                      context) and on ARRIVAL, before the dwell, for the pose the robot reached (arr_query_id: the tested promise).
#   aba_promises.tsv   context only: every node's last offers.csv row at the claim (the planner's OWN standpoint's promise)
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
GQ_REQ=$CWT/${CLS}_concept/etc/gain_query.txt; GQ_ANS=$CWT/${CLS}_concept/etc/gain_query.csv
SEG=$RUN/aba_segments.tsv; PROM=$RUN/aba_promises.tsv
[ -f $SEG ] || printf 'def\tseg\tstand_x\tstand_y\tstand_yaw\troom\tpub_x\tpub_y\tpub_yaw\tfit\tt_claim_ms\tt_arrive_ms\tt_dwell0_ms\tt_dwell1_ms\tgoto\tturn_deg\tface\tgt_x\tgt_y\tgt_yaw\tdwell\tquery_id\tquery_answered\tarr_query_id\tarr_query_answered\n' > $SEG
[ -f $PROM ] || printf 'def\tseg\tt_claim_ms\tnode\tt_ms\tstandpoint_x\tstandpoint_y\tstandpoint_yaw\tpromised_nats\n' > $PROM

# run_mission <name> <body> <timeout_s>: sets M (completed | failed ... | aborted ... | TIMEOUT)
run_mission() {
  echo run > $B/controller/remote_cmd.txt; sleep 0.7   # the controller reads remote_cmd.txt every 300 ms
  printf 'mission "%s"\n%s\n' "$1" "$2" > $B/controller/remote_cmd.txt
  M=TIMEOUT
  for i in $(seq 1 $(($3 * 2))); do
    # co(mpleted)? as warm_start_tour.sh (3640149d): the line can be cut after "co"
    m=$(grep -a -o -E "mission '$1': (co(mpleted)?|failed|aborted)[^\"]{0,80}" $L/controller.log | tail -1)
    [ -n "$m" ] && { M=${m#*: }; case "$M" in co|co[^m]*) M=completed;; esac; break; }
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

gain_query() {   # id world_x world_y: a gain query for that standpoint, facing the object; sets QA = yes | no | frame_failed
  local pose
  if ! pose=$(python3 $S/aba_geometry.py query_pose "$2" "$3" "$CX" "$CY") || [ -z "$pose" ]; then
    QA=frame_failed; log "gain query $1: frame conversion FAILED (not asked)"; return; fi
  echo "$1 * $pose" >> "$GQ_REQ"
  QA=no
  for i in $(seq 1 25); do grep -q "^[0-9]*,$1," "$GQ_ANS" 2>/dev/null && { QA=yes; break; }; sleep 0.2; done
  log "gain query $1 at ($2, $3): $QA $(grep "^[0-9]*,$1," "$GQ_ANS" 2>/dev/null | cut -d, -f3,8 | tr '\n' ' ')"; }

# face_step seg: turn to face the object, re-computing the turn from the CURRENT Webots pose after every move, until the residual
# heading is within the controller's own heading goal tolerance (align_yaw_tol_rad) — a long pivot can stop short (live run
# p1_nightstand, B: 165.5 deg asked, the move skill's own budget ran out after 33 deg near the bed's foot corner).
# At most FACE_ATTEMPTS moves (3). Sets FACE = completed|failed followed by every attempt in [...], TURN0 = the first turn asked.
FACE_ATTEMPTS=${FACE_ATTEMPTS:-3}
HEADING_TOL_DEG=$(python3 $S/aba_geometry.py heading_tol) || { log "controller heading tolerance not found"; exit 1; }
face_step() {
  local k turn trace="" ok=0
  TURN0=nan
  for k in $(seq 1 $((FACE_ATTEMPTS + 1))); do
    turn=$(python3 $S/aba_geometry.py face "$CX" "$CY") || { trace+="a$k:pose_failed;"; break; }
    [ "$TURN0" = nan ] && TURN0=$turn
    if awk -v t="$turn" -v tol="$HEADING_TOL_DEG" 'BEGIN{ exit !((t < 0 ? -t : t) <= tol) }'; then
      trace+="a$k:${turn}deg:within;"; ok=1; break; fi
    [ $k -gt $FACE_ATTEMPTS ] && { trace+="a$k:${turn}deg:out_of_attempts;"; break; }   # the last check after the last move
    run_mission "aba_${DEF}_${1}_face$k" "  move turn_deg=${turn}deg" 60
    trace+="a$k:${turn}deg:${M// /_};"
  done
  if [ $ok = 1 ]; then FACE="completed[${trace%;}]"; else FACE="failed[${trace%;}]"; fi
  log "$1: face $FACE (tolerance ${HEADING_TOL_DEG} deg)"; }

visit() {   # seg world_x world_y world_yaw
  local seg=$1 tc ta=0 t0=0 t1=0 room px=nan py=nan pyaw=nan fit=none go turn=nan face=none gx=nan gy=nan gyaw=nan dw=none
  local qid qa aqid=0 aqa=no out gt
  tc=$(now_ms); record_promises "$seg" "$tc"
  # context: the promise of the scripted standpoint at the claim, before the robot moves (ids are epoch ms: increasing)
  qid=$tc; gain_query "$qid" "$2" "$3"; qa=$QA
  room=$(room_now)
  # a frame that cannot be converted for the query cannot be trusted for the drive either
  if [ "$qa" != frame_failed ] && out=$(python3 $S/aba_geometry.py to_room "$2" "$3" "$4") && [ -n "$out" ]; then
    read -r px py pyaw fit <<< "$out"
    log "$seg: claim at $tc, go_to pose:$room($px, $py, $pyaw) [$fit]"
    run_mission "aba_${DEF}_${seg}_go" "  go_to pose:${room}(${px}, ${py}, ${pyaw})" "$MOVE_TIMEOUT_S"; go=$M
    ta=$(now_ms)
    face_step "$seg"; face=$FACE; turn=$TURN0
    gt=$(python3 $S/tp.py get 2>/dev/null | tail -1 | awk '{print $1, $2, $5}')
    [ -n "$gt" ] && read -r gx gy gyaw <<< "$gt"
    # THE TESTED PROMISE: asked on arrival, from the arrival Sigma, for where the robot now stands, before the dwell
    # (parked at rho = 1 the dwell is one look: its net change is what this one hypothetical update predicts)
    if [ "$gx" != nan ]; then aqid=$(now_ms); gain_query "$aqid" "$gx" "$gy"; aqa=$QA; fi
    t0=$(now_ms)
    log "$seg: go_to $go, face $turn deg $face; dwell ${DWELL_S}s from $t0 at ($gx, $gy, $gyaw)"
    run_mission "aba_${DEF}_${seg}_dwell" "  wait s=${DWELL_S}s" "$((DWELL_S + 30))"; dw=$M
    t1=$(now_ms)
  else
    go=frame_failed
    log "$seg: frame conversion FAILED — not driving, no dwell (this segment gets NO VERDICT)"
  fi
  # one quoted argument per column; spaces inside a field -> _
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$DEF" "$seg" "$2" "$3" "$4" "${room:-none}" "$px" "$py" "$pyaw" "${fit// /_}" "$tc" "$ta" "$t0" "$t1" "${go// /_}" "$turn" \
    "${face// /_}" "$gx" "$gy" "$gyaw" "${dw// /_}" "$qid" "$qa" "$aqid" "$aqa" >> "$SEG"
}

# The class agent must hold an instance before the A1 claim query, or nobody answers it (live run p1_toilet: the leg ended
# ~20 s before toilet_2's first update; the claim query got no answer and the whole visit had NO VERDICT). An instance exists
# once the agent writes an offer row stamped after this script started. Wait for it (<= NODE_WAIT_S, 120); if none comes the
# claim query is still asked and the report gives NO VERDICT, as before.
T_START=$(now_ms); NODE_WAIT_S=${NODE_WAIT_S:-120}
# an engine without offers (main's, for the live A/B) shows its instance in ai2_log.csv instead (pkt_ts, column 4, epoch ms)
AI2=$CWT/${CLS}_concept/etc/ai2_log.csv
has_instance() { awk -F, -v t="$T_START" 'NR > 1 && $1 + 0 >= t { f = 1; exit } END { exit !f }' "$OFFERS" 2>/dev/null ||
                 awk -F, -v t="$T_START" 'NR > 1 && $4 + 0 >= t { f = 1; exit } END { exit !f }' "$AI2" 2>/dev/null; }
for i in $(seq 1 $((NODE_WAIT_S * 2))); do has_instance && break; sleep 0.5; done
if has_instance; then log "instance up after $(( ($(now_ms) - T_START) / 1000 )) s"
else log "NO $CLS instance after ${NODE_WAIT_S}s: claiming anyway (A1 will have no verdict)"; fi

visit A1 $AX $AY $AYAW
visit B  $BX $BY $BYAW
visit A2 $AX $AY $AYAW
log "done: $SEG"
