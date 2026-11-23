#!/bin/bash
# AuRo C3(a) (2026-11-21): one furnished warm three-room tour (three_rooms_door0, legs R2 R3 R2 R1) with every object concept
# agent + ltom_agent, from an EMPTY object memory; graded by scripts/c3a_objects.py against the Webots ground truth.
# usage: RUN=c3a_1 [CONCEPT_AGENTS="..."] bash c3a_run.sh
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; S=$E/scripts
RUN=${RUN:-c3a_1}; LT=$B/ltom_agent/etc; BK=$E/crossing/ltom_repertory_backup_$RUN
export LOOK_S=${LOOK_S:-90}
mkdir -p $BK; [ -f $BK/ltom_repertory.json ] || cp -p $LT/ltom_repertory.json $LT/ltom_repertory.csv $LT/ltom_repertory.json.prev $BK/ 2>/dev/null
# EMPTY of objects, humans kept. Written, never deleted: a missing file makes ltom_agent load ltom_repertory.json.prev.
python3 -I -c "import json,sys; b=json.load(open(sys.argv[1])); json.dump({'entries': [], 'humans': b.get('humans', []), 'version': b.get('version', 1)}, open(sys.argv[2], 'w'))" $BK/ltom_repertory.json $LT/ltom_repertory.json
rm -f $LT/ltom_repertory.csv
CA=${CONCEPT_AGENTS:-"chair_concept table_concept bottle_concept hood_concept refrigerator_concept cabinet_concept nightstand_concept bed_concept toilet_concept"}
echo "c3a start $(date +%T) run $RUN build $(git -C $B rev-parse --short HEAD) libdsr_api $(stat -c %y /usr/local/lib/libdsr_api.so | cut -d. -f1) agents: $CA"
export ON_FLEET_UP="LTOM=1 CONCEPT_WT=${CONCEPT_WT:-$B} CONCEPT_AGENTS='$CA' bash $S/concept_agents.sh start $E/crossing/$RUN"
(cd $S && bash warm_start_tour.sh $RUN $E/crossing/fixtures/three_rooms_door0 > $E/crossing/$RUN.console 2>&1)
LTOM=1 CONCEPT_AGENTS="$CA" bash $S/concept_agents.sh stop
cp $LT/ltom_repertory.json $E/crossing/$RUN/ltom_repertory_end.json
grep -a "repertory .*ltom_repertory.json:" $E/crossing/$RUN/logs/ltom_agent.log | head -1
bash $S/concept_agents.sh harvest $E/crossing/$RUN   # from the checkout each agent RAN from (11-22: main-only copy left c3a_3 with c3a_2's countable logs)
cp -p $BK/ltom_repertory.json $BK/ltom_repertory.csv $LT/   # the owner's memory back as it was
cmp $BK/ltom_repertory.json $LT/ltom_repertory.json && echo "ltom repertory restored"
grep -a "^=== " $E/crossing/$RUN.console | tail -1
cat $E/crossing/$RUN/trace_check.txt 2>/dev/null
python3 $S/c3a_objects.py $E/crossing/$RUN
echo "c3a done $(date +%T)"
