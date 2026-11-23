#!/bin/bash
# AuRo item 13 (2026-11-22): the out-of-view moved-chair test, repeated. Protocol = c3b_*_hidden1 (11-21): warm three-room tour
# (fixture three_rooms_door0), chair_concept + ltom_agent only, LTOM starts EMPTY (written, not deleted: no .prev load), LTSM from
# the fixture (warm_start_tour.sh archives the old history and deletes ltsm_memory.json[.prev]). CHAIR_3 starts at (X0,Y0); after
# leg 2 (robot in room_3) it is moved 1.0 m to (-6.35, 8.60), hidden from the DOOR_2 entry; the robot returns (leg 3) and acts.
# Arms alternate per repetition: on = chair epistemic offer on, off = ChairConcept.EpistemicOfferEnabled=false. That flag turns off
# EVERY chair's offer; only chair(3) is in room_2, so in room_2 it removes exactly the moved chair's offer.
# usage: REPS="1 2 3 4 5" bash c3b_series_1122.sh
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; S=$E/scripts
CFG=$B/chair_concept/etc/config.toml
SUM=$E/crossing/c3b5_series_1122.txt
X0=-7.4323; Y0=8.5130; X1=-6.35; Y1=8.60
export LOOK_S=90
LT=$B/ltom_agent/etc; BK=$E/crossing/ltom_repertory_backup_1122_c3b5
mkdir -p $BK; [ -f $BK/ltom_repertory.json ] || cp -p $LT/ltom_repertory.json $LT/ltom_repertory.csv $LT/ltom_repertory.json.prev $BK/ 2>/dev/null
# whatever happens (error, Ctrl-C), the chair config and the owner's object memory go back as they were
trap 'sed -i "s/^EpistemicOfferEnabled = false/EpistemicOfferEnabled = true/" $CFG; cp -p $BK/ltom_repertory.json $LT/; [ -f $BK/ltom_repertory.csv ] && cp -p $BK/ltom_repertory.csv $LT/' EXIT
echo "series start $(date +%T) main $(git -C $B rev-parse --short HEAD) chair bin $(stat -c %y $B/chair_concept/bin/chair_concept | cut -d. -f1) libdsr_api $(stat -c %y /usr/local/lib/libdsr_api.so | cut -d. -f1)" >> $SUM
for rep in ${REPS:-1 2 3 4 5}; do
  arms="on off"; [ $((rep % 2)) = 0 ] && arms="off on"      # alternate which arm goes first
  for arm in $arms; do
    r=c3b5_${arm}_r$rep
    python3 -I -c "import json,sys; b=json.load(open(sys.argv[1])); json.dump({'entries': [], 'humans': b.get('humans', []), 'version': b.get('version', 1)}, open(sys.argv[2], 'w'))" $BK/ltom_repertory.json $LT/ltom_repertory.json
    rm -f $LT/ltom_repertory.csv
    [ $arm = off ] && sed -i 's/^EpistemicOfferEnabled = true/EpistemicOfferEnabled = false/' $CFG
    python3 $S/webots_object.py set CHAIR_3 $X0 $Y0 > /dev/null; sleep 1
    export ON_FLEET_UP="LTOM=1 CONCEPT_WT=$B CONCEPT_AGENTS='chair_concept' bash $S/concept_agents.sh start $E/crossing/$r"
    export AFTER_LEG="2:echo \"chair moved t_ms \$(date +%s%3N)\" >> $E/crossing/$r/chair_move.txt; python3 $S/webots_object.py set CHAIR_3 $X1 $Y1 >> $E/crossing/$r/chair_move.txt; sleep 1; python3 $S/webots_object.py get CHAIR_3 >> $E/crossing/$r/chair_move.txt"
    (cd $S && bash warm_start_tour.sh $r $E/crossing/fixtures/three_rooms_door0 > $E/crossing/$r.console 2>&1)
    LTOM=1 CONCEPT_AGENTS='chair_concept' bash $S/concept_agents.sh stop > /dev/null 2>&1
    bash $S/concept_agents.sh harvest $E/crossing/$r > /dev/null
    cp $LT/ltom_repertory.json $E/crossing/$r/ltom_repertory_end.json 2>/dev/null
    grep -a "EpistemicOfferEnabled" $CFG >> $E/crossing/$r/chair_move.txt
    [ $arm = off ] && sed -i 's/^EpistemicOfferEnabled = false/EpistemicOfferEnabled = true/' $CFG
    python3 $S/webots_object.py set CHAIR_3 $X0 $Y0 > /dev/null
    legs=$(grep -a "^=== " $E/crossing/$r.console | tail -1)
    echo "$r $(date +%T) | ${legs:-NO RESULT} | chair picks: $(grep -a -c "aff-select\] -> 'aff_chair" $E/crossing/$r/logs/controller.log 2>/dev/null)" >> $SUM
    [ -f $E/crossing/STOP_SERIES ] && { echo "STOP_SERIES seen, stopping" >> $SUM; break 2; }
  done
done
cp -p $BK/ltom_repertory.json $LT/; [ -f $BK/ltom_repertory.csv ] && cp -p $BK/ltom_repertory.csv $LT/
grep -n "^EpistemicOfferEnabled" $CFG >> $SUM
echo "series done $(date +%T)" >> $SUM
