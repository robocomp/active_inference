#!/bin/bash
# WARM-START tour (docs/superpowers/specs/2026-11-01-ltsm-warm-start-design.md): the fleet starts in room_1 ALREADY knowing
# room_1 and room_2 (a fixture made by make_warm_fixture.sh), and goes on to the third room:
#   leg 1  cross room_2   (the door with a recall_room_2 child: DOOR_1)          want room_2
#   leg 2  cross new      (room_2's door nobody remembers the far side of: DOOR_2) want room_3
#          (a THREE-room fixture: leg 2 is `cross room_3` — the door with a recall_room_3 child)
#   leg 3  cross room_2   (back through DOOR_2)                                    want room_2
#   leg 4  cross room_1   (back through DOOR_1)                                    want room_1
# Graded on GROUND TRUTH (threerooms-piso.wbt): DOOR_1's line y = 9.3796 (room_1 below), DOOR_2's line x = -1.705 (room_3 east).
# The fixture is COPIED in (never written); its checksums are verified after the run.
# usage: warm_start_tour.sh <run_name> <fixture_dir>      env WT / WT_AGENTS / LEG_TIMEOUT_S (600) / N_LEGS
LOCK=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ai_fleet_tour.lock; echo $$ > $LOCK
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E; L=crossing/$1/logs
FIX=$(cd "$2" 2>/dev/null && pwd) || { echo "usage: $0 <run_name> <fixture_dir>"; rm -f $LOCK; exit 2; }
(cd $FIX && sha256sum -c --quiet SHA256SUMS) || { echo "fixture $FIX: checksums FAIL before the run"; rm -f $LOCK; exit 1; }
LEG_TIMEOUT_S=${LEG_TIMEOUT_S:-600}
export WT=${WT:-/home/pbustos/robocomp/components/active_inference.worktrees/rt-build} WT_AGENTS=${WT_AGENTS:-"controller door_concept table_concept room_concept residual_concept ltsm_agent"}
export FLEET="retina room_concept residual_concept controller door_concept ltsm_agent table_concept"
DOOR1_Y=9.3796; DOOR2_X=-1.705
# ── the two switches, for THIS run only (restored on exit) ──
RC_CFG=$WT/room_concept/etc/config.toml; LT_CFG=$WT/ltsm_agent/etc/config.toml
cp $RC_CFG $RC_CFG.warm_bak; cp $LT_CFG $LT_CFG.warm_bak
trap 'mv -f $RC_CFG.warm_bak $RC_CFG; mv -f $LT_CFG.warm_bak $LT_CFG; rm -f $LOCK; [ -n "$REC_PID" ] && kill -TERM $REC_PID 2>/dev/null' EXIT
sed -i 's/^MapMode = "[a-z]*"/MapMode = "ltsm"/' $RC_CFG
sed -i 's/^# Room = "room_1"/Room = "room_1"/' $LT_CFG
# RECOGNITION ON (user 2026-11-01, warm11: R1 entered through DOOR_0 — a door never crossed in the memory — was promoted as a
# NEW room_4 because recognition was off; ltsm then placed it on top of the remembered room_1)
sed -i '/^\[Recognition\]/,/^\[/ s/^enabled = false/enabled = true/' $LT_CFG
grep -q '^MapMode = "ltsm"' $RC_CFG && grep -q '^Room = "room_1"' $LT_CFG && grep -q '^enabled = true   # RoomSwap' $LT_CFG \
    || { echo "could not set MapMode=ltsm / WarmStart.Room / Recognition.enabled"; exit 1; }
echo stop > $B/controller/remote_cmd.txt
# ── archive what is there (never just delete), then install the fixture (COPIES) ──
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p crossing/ltsm_archive crossing/door_identities_archive
[ -f $B/ltsm_agent/etc/ltsm_memory.json ] && cp $B/ltsm_agent/etc/ltsm_memory.json crossing/ltsm_archive/ltsm_memory_before_$1_$STAMP.json
[ -f $B/door_concept/etc/door_identities.csv ] && cp $B/door_concept/etc/door_identities.csv crossing/door_identities_archive/door_identities_before_$1_$STAMP.csv
for h in $B/ltsm_agent/etc/ltsm_history $WT/ltsm_agent/etc/ltsm_history; do
  [ -d "$h" ] && [ -n "$(ls -A "$h" 2>/dev/null)" ] && mv "$h" crossing/ltsm_archive/ltsm_history_before_$1_${STAMP}_$(basename $(dirname $(dirname $(dirname "$h"))))
done
rm -f ../etc/last_robot_pose.txt $B/ltsm_agent/etc/ltsm_memory.json $B/ltsm_agent/etc/ltsm_memory.json.prev
rm -rf $B/retina/etc/place_maps/; rm -f $WT/ltsm_agent/etc/recognition_trace.csv $B/ltsm_agent/etc/recognition_trace.csv
mkdir -p $B/ltsm_agent/etc/ltsm_history
cp $FIX/ltsm_history/* $B/ltsm_agent/etc/ltsm_history/ && chmod u+w $B/ltsm_agent/etc/ltsm_history/*
cp $FIX/door_identities.csv $B/door_concept/etc/door_identities.csv && chmod u+w $B/door_concept/etc/door_identities.csv
cp $FIX/room_name_hwm.txt ../etc/room_name_hwm.txt && chmod u+w ../etc/room_name_hwm.txt
echo "fixture installed: $FIX (hwm $(cat ../etc/room_name_hwm.txt))"
# DOOR_1 OPEN (120 deg, user 2026-11-01) at start: the fixture remembers it open (pingpong41 ended with the leaf at ~105 deg) and door_concept restores that
# leaf belief at full confidence (PENDING model fix: a restored leaf should lose precision with the time since it was seen).
# DOOR_0 and DOOR_2 shut (DOOR_2 is the way to room_3: the cross mission opens it).
python3 $S/door.py open DOOR_1 120; for d in DOOR_0 DOOR_2; do python3 $S/door.py close $d; done
sleep 3
# graph recorder (passive DSR participant, agent id 28): every structural write with its writer — for trace_check.py
REC=$(cd $S/../../.. && pwd)/graph_recorder/bin/graph_recorder; mkdir -p crossing/$1
[ -x $REC ] && { setsid $REC crossing/$1/graph_trace.csv > crossing/$1/graph_recorder.log 2>&1 < /dev/null & REC_PID=$!; }
NORUN=1 bash $S/run_crossing.sh crossing/$1 -5.33 3.66 0.58 >/dev/null 2>&1
room_now() { grep -a "current room changed" $L/controller.log | tail -1 | awk '{print $NF}'; }
nl() { wc -l < $L/$1.log 2>/dev/null || echo 0; }
now_ms() { date +%s%3N; }
true_y() { python3 $S/tp.py get 2>/dev/null | tail -1 | awk '{print $2}'; }
# GROUND TRUTH at 1 Hz: wall ms, Webots pose, and every agent log's line count (maps any log line to the true pose then)
GT=crossing/$1/gt_pose.tsv; printf 't_ms\tx\ty\tyaw\troom_concept\tltsm_agent\tdoor_concept\tcontroller\n' > $GT
( while [ -f $LOCK ]; do
    p=$(python3 $S/tp.py get 2>/dev/null | tail -1 | awk '{print $1"\t"$2"\t"$5}')
    [ -n "$p" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\n' $(now_ms) "$p" $(nl room_concept) $(nl ltsm_agent) $(nl door_concept) $(nl controller) >> $GT
    sleep 1
  done ) &
# ── READY: room_1 adopted from ltsm's start-up recall and localised, its doors restored, recall_room_2 posted ──
ready=0
for i in $(seq 1 120); do
  grep -a -q "MapMode=ltsm: layout of 'room_1'" $L/room_concept.log 2>/dev/null \
    && grep -a -q "\[restore\] 'room_1' recalled" $L/door_concept.log 2>/dev/null \
    && grep -a -q "\[recall\] recall_room_2 posted" $L/ltsm_agent.log 2>/dev/null && { ready=1; break; }
  sleep 1
done
echo "--- start-up ---"
grep -a -m1 "MapMode=ltsm: still waiting" $L/room_concept.log
grep -a -m1 "\[warm\] start-up recall" $L/ltsm_agent.log
grep -a -m1 "MapMode=ltsm: layout of" $L/room_concept.log
grep -a -m3 "\[reloc\]" $L/room_concept.log
grep -a -m1 "\[restore\] 'room_1' recalled" $L/door_concept.log
grep -a "\[restore\].*room_1" $L/door_concept.log | grep -a -v "recalled" | head -3
grep -a -m1 "\[recall\] recall_room_2 posted" $L/ltsm_agent.log
[ $ready = 1 ] || { echo "ABORT: not ready in 120 s"; bash $S/run_crossing.sh crossing/$1 collect >/dev/null 2>&1; exit 1; }
sleep 5
# ── LOCALISATION GOOD ENOUGH TO ACT ON (user 2026-11-01, warm12: the pose drifted 0.2-0.4 m in room_1, the door was misplaced
# and everything after failed: "check confidence is good enough to proceed"). loc_ok.py: median scan residual over 5 s.
# Logged every 2 s for the whole run (crossing/<run>/loc.txt) so a drift DURING a leg is visible afterwards.
( while [ -f $LOCK ]; do echo "$(now_ms) $(python3 -I $S/loc_ok.py $B/room_concept)" >> crossing/$1/loc.txt; sleep 2; done ) &
loc_gate() {   # label max_wait_s
  local i; for i in $(seq 1 $2); do python3 -I $S/loc_ok.py $B/room_concept > /dev/null && break; sleep 1; done
  local v; v=$(python3 -I $S/loc_ok.py $B/room_concept); echo "[$1] $v"
  case "$v" in "loc OK"*) return 0;; *) return 1;; esac; }
loc_gate start 60 || { echo "ABORT: localisation not good enough to start"; echo stop > $B/controller/remote_cmd.txt
                      bash $S/run_crossing.sh crossing/$1 collect >/dev/null 2>&1; exit 1; }
run_leg() {   # label cross_arg [close_arg]   (sets r: the crossing; c: the close)
  # THE LEG'S PLAN (user 2026-11-01): `cross X`, then — a healthy habit — `close <room just left>`: the plan claims
  # door_concept's close affordance of the door behind (mission op `close`). The leg is the CROSSING: r is OK once this leg's
  # passage logs DONE; the leg ends when the mission ends (the close succeeded or failed), and c reports the close.
  local l0; l0=$(nl controller)
  echo run > $B/controller/remote_cmd.txt; sleep 0.7   # the controller reads remote_cmd.txt every 300 ms
  # OPEN IS A STEP OF THE PLAN (user 2026-11-01, warm13: DOOR_2 shut but remembered open — the cross never asked for it):
  # `open X` approaches, looks, and opens when the door is SEEN shut (satisfied when seen open); then `cross X`.
  if [ -n "$3" ]; then printf 'mission "leg %s"\n  open %s\n  cross %s retry=2\n  close %s\n' $1 $2 $2 $3 > $B/controller/remote_cmd.txt
  else printf 'mission "leg %s"\n  open %s\n  cross %s retry=2\n' $1 $2 $2 > $B/controller/remote_cmd.txt; fi
  local t0; t0=$(date +%s); r="FAILED (no end in ${LEG_TIMEOUT_S} s)"; c="-"; local crossed=""
  for i in $(seq 1 $((LEG_TIMEOUT_S * 2))); do
    if [ -z "$crossed" ] && tail -n +$((l0+1)) $L/controller.log | grep -a -v "asked to stop after open" | grep -a -q -E "passage DONE|mission 'leg $1': co"; then
      crossed=1; r="OK in $(( $(date +%s)-t0 )) s"; fi
    if grep -a -q -E "mission 'leg $1': co(mpleted)?" $L/controller.log; then [ -n "$3" ] && c="closed"; break; fi
    if grep -a -q -E "mission 'leg $1': (failed|aborted)" $L/controller.log; then
      f=$(grep -a -o -E "mission 'leg $1': (failed|aborted)[^\"]*" $L/controller.log | tail -1)
      if [ -n "$crossed" ]; then c="NOT closed ($f)"; else r="FAILED: $f"; fi; break; fi
    sleep 0.5
  done; }
# `cross new` only needs a door to APPEAR: when the controller reports door_unresolved, one explore, then the leg again (bounded)
explore_once() {
  echo run > $B/controller/remote_cmd.txt; sleep 0.7
  printf 'mission "look%s"\n  explore %s budget_s=90s\n' "$1" "$(room_now)" > $B/controller/remote_cmd.txt
  for i in $(seq 1 60); do grep -a -q -E "mission 'look$1': (completed|failed|aborted)" $L/controller.log && break; sleep 3; done; }
true_xy() { python3 $S/tp.py get 2>/dev/null | tail -1 | awk '{print $1, $2}'; }
side_of() { awk -v x="$1" -v y="$2" -v dy=$DOOR1_Y -v dx=$DOOR2_X 'BEGIN{ if (y=="") print "unknown"; else if (x > dx) print "room_3"; else if (y > dy) print "room_2"; else print "room_1" }'; }
TSV=crossing/$1/legs.tsv; printf 'leg\targ\tverdict\ttrue_side\ttrue_xy\tstatus\tt0_ms\tt1_ms\tgraph_room\n' > $TSV
# THE PLAN FOLLOWS WHAT THE FIXTURE KNOWS (its room_name_hwm): two rooms remembered -> leg 2 crosses into a NEW room (room_3);
# three remembered -> every leg goes by relation, room_3 included (user 2026-11-01: "start the next tour with the three rooms").
# R1_LOOP=1 (user 2026-11-01): room_1 -> room_2 -> room_3 -> back into room_1 through DOOR_0 (the bathroom's second door, never
# crossed in the memory: `cross new` from room_3). Tests RECOGNITION: the room entered must become room_1, not a new room_4.
if [ -n "$R1_LOOP" ]; then
  # leg 4 (R1_LOOP_BACK=1): back to room_3 through DOOR_0 — leaving the recognised room_1 is what commits the swap to ltsm's
  # memory (it is held until the robot leaves): a run that ends in room_1 never saves the room_3 <-> room_1 passage.
  ARGS=(room_2 room_3 new room_3); WANT=(room_2 room_3 room_1 room_3); BEHIND=(room_1 room_2 room_3 room_1)
  N_LEGS=${N_LEGS:-$([ -n "$R1_LOOP_BACK" ] && echo 4 || echo 3)}
elif [ "$(cat $FIX/room_name_hwm.txt)" -ge 3 ]; then
  ARGS=(room_2 room_3 room_2 room_1); WANT=(room_2 room_3 room_2 room_1); BEHIND=(room_1 room_2 room_3 room_2)
else
  ARGS=(room_2 new room_2 room_1); WANT=(room_2 room_3 room_2 room_1); BEHIND=(room_1 room_2 room_3 room_2)   # close behind: the room left
fi
echo "leg plan: ${ARGS[*]} (fixture remembers $(cat $FIX/room_name_hwm.txt) rooms)"
ok=0; n_legs=${N_LEGS:-4}
for leg in $(seq 1 $n_legs); do
  a=${ARGS[$((leg-1))]}; want=${WANT[$((leg-1))]}
  loc_gate "before leg $leg" 30 || { echo "leg $leg NOT STARTED: localisation not good enough"; break; }
  T0=$(now_ms); ts=$(date +%s)
  run_leg $leg $a ${BEHIND[$((leg-1))]}; n=0
  # the next door AS SOON AS IT EXISTS: a door named by relation (room_X) that is not resolvable yet (its recall not posted)
  # is asked for again every 2 s; only `cross new` explores, since that door may still have to be found
  while case "$r" in *door_unresolved*) true;; *) false;; esac && [ $(( $(date +%s)-ts )) -lt $LEG_TIMEOUT_S ]; do
    n=$((n+1))
    if [ "$a" = new ]; then explore_once "${leg}_$n"; else sleep 2; fi
    run_leg "${leg}s$n" $a ${BEHIND[$((leg-1))]}
  done
  # the ground-truth track is sampled at 1 Hz: wait (<= 5 s) for it to show the crossing instead of a fixed pause
  for i in $(seq 1 10); do side=$(python3 -I $S/true_room.py crossing/$1/gt_pose.tsv); [ "$side" = "$want" ] && break; sleep 0.5; done
  xy=$(true_xy)
  case "$r" in OK*) [ "$side" = "$want" ] && st=OK || st=FALSE_OK;; *) st=FAILED;; esac
  [ $st = OK ] && ok=$((ok+1))
  echo "leg $leg (cross $a, want $want): controller '$r', true side $side (xy $xy), graph room $(room_now) -> $st; door behind: $c"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $leg $a "$r" $side "$xy" $st $T0 $(now_ms) "$(room_now)" >> $TSV
  [ $st = OK ] || break
done
echo "=== $ok/$n_legs legs"
grep -a "\[room\].*room_3\|new room.*room_" $L/room_concept.log | head -3
echo stop > $B/controller/remote_cmd.txt
for d in DOOR_1 DOOR_0 DOOR_2; do python3 $S/door.py open $d; done
bash $S/run_crossing.sh crossing/$1 collect >/dev/null 2>&1
[ -n "$REC_PID" ] && { sleep 3; kill -TERM $REC_PID 2>/dev/null; }
cp $B/door_concept/etc/door_identities.csv crossing/$1/door_identities.csv 2>/dev/null
python3 -I $S/trace_check.py crossing/$1 | tee crossing/$1/trace_check.txt
(cd $FIX && sha256sum -c --quiet SHA256SUMS) && echo "fixture UNCHANGED" || echo "fixture CHANGED"
[ $ok = $n_legs ]
