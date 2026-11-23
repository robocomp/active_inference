#!/bin/bash
# AuRo item 14 (2026-11-22): removal test. Warm three-room tour (fixture three_rooms_door0, legs R2 R3 R2 R1) with chair_concept,
# table_concept, nightstand_concept + ltom_agent, LTOM starting EMPTY (written, not deleted). After leg 2 (robot in room_3) three
# learned objects are moved out of the house (Webots): TABLE_ROUND + CHAIR_2 (room_1, next to each other) and NIGHTSTAND_R
# (room_2, the bedroom). The robot returns to room_2 (leg 3: nightstand) and room_1 (leg 4: chair + table).
# World: threerooms-piso.wbt as reloaded 2026-11-22 19:26 (closet, chairs 4/5, WoodenTable(1) out).
# usage: REPS="1 2 3" bash c14_removal_1122.sh
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; S=$E/scripts
SUM=$E/crossing/c14_series_1122.txt
OBJS="TABLE_ROUND CHAIR_2 NIGHTSTAND_R"
CA="chair_concept table_concept nightstand_concept"
export LOOK_S=90
LT=$B/ltom_agent/etc; BK=$E/crossing/ltom_repertory_backup_1122_c14
mkdir -p $BK; [ -f $BK/ltom_repertory.json ] || cp -p $LT/ltom_repertory.json $LT/ltom_repertory.csv $LT/ltom_repertory.json.prev $BK/ 2>/dev/null
declare -A HOME0
for o in $OBJS; do HOME0[$o]=$(python3 $S/webots_object.py get $o | cut -d' ' -f1,2); done
restore() { for o in $OBJS; do python3 $S/webots_object.py set $o ${HOME0[$o]} > /dev/null; done; }
trap 'restore; cp -p $BK/ltom_repertory.json $LT/; [ -f $BK/ltom_repertory.csv ] && cp -p $BK/ltom_repertory.csv $LT/' EXIT
echo "series start $(date +%T) main $(git -C $B rev-parse --short HEAD) homes: $(for o in $OBJS; do echo -n "$o=${HOME0[$o]// /,} "; done)" >> $SUM
for rep in ${REPS:-1 2 3}; do
  r=c14_r$rep
  python3 -I -c "import json,sys; b=json.load(open(sys.argv[1])); json.dump({'entries': [], 'humans': b.get('humans', []), 'version': b.get('version', 1)}, open(sys.argv[2], 'w'))" $BK/ltom_repertory.json $LT/ltom_repertory.json
  rm -f $LT/ltom_repertory.csv
  restore; sleep 1
  mkdir -p $E/crossing/$r
  export ON_FLEET_UP="LTOM=1 CONCEPT_WT=$B CONCEPT_AGENTS='$CA' bash $S/concept_agents.sh start $E/crossing/$r"
  i=0; MV=""
  for o in $OBJS; do MV="$MV python3 $S/webots_object.py set $o $((30 + i)) 30 >> $E/crossing/$r/removed.txt;"; i=$((i + 1)); done
  export AFTER_LEG="2:echo \"removed t_ms \$(date +%s%3N)\" >> $E/crossing/$r/removed.txt; $MV echo \"homes ${HOME0[*]}\" >> $E/crossing/$r/removed.txt"
  (cd $S && bash warm_start_tour.sh $r $E/crossing/fixtures/three_rooms_door0 > $E/crossing/$r.console 2>&1)
  LTOM=1 CONCEPT_AGENTS="$CA" bash $S/concept_agents.sh stop > /dev/null 2>&1
  bash $S/concept_agents.sh harvest $E/crossing/$r > /dev/null
  cp $LT/ltom_repertory.json $E/crossing/$r/ltom_repertory_end.json 2>/dev/null
  restore
  legs=$(grep -a "^=== " $E/crossing/$r.console | tail -1)
  echo "$r $(date +%T) | ${legs:-NO RESULT}" >> $SUM
  [ -f $E/crossing/STOP_SERIES ] && { echo "STOP_SERIES seen, stopping" >> $SUM; break; }
done
echo "series done $(date +%T)" >> $SUM
