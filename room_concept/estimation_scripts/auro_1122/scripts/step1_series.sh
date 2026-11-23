#!/bin/bash
# Step 1 (2026-11-23): object offers priced by existence information instead of the 5-nat floor (worktree feat/existence-info).
# 3 furnished C3a tours at the default room scale 0.35, then 2 with AffordanceRoomGainScale = 1.0. Cabinet out, LTOM empty,
# per-class ai2 logs harvested. The controller config is restored on any exit.
B=/home/pbustos/robocomp/components/active_inference; S=/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/1c86b8d6-47c4-4a7b-8f4b-04a583e967d4/scratchpad
WT=$B/../active_inference.worktrees/existence-info
CFG=$B/controller/etc/config.toml; cp -p $CFG $S/controller_config.before_step1
trap 'cp -p $S/controller_config.before_step1 $CFG; echo "controller config restored"' EXIT
CA="chair_concept table_concept bottle_concept hood_concept refrigerator_concept nightstand_concept bed_concept toilet_concept"
for spec in "s1_rs035_1 0.35" "s1_rs035_2 0.35" "s1_rs035_3 0.35" "s1_rs100_1 1.0" "s1_rs100_2 1.0"; do
  set -- $spec; r=$1; rs=$2
  [ -f $B/room_concept/estimation_scripts/crossing/STOP_SERIES ] && break
  sed -i -E "s/^AffordanceRoomGainScale *= *[0-9.]+/AffordanceRoomGainScale = $rs/" $CFG
  grep -q "^AffordanceRoomGainScale = $rs" $CFG || { echo "room scale $rs NOT SET"; exit 1; }
  RUN=$r CONCEPT_WT=$WT CONCEPT_AGENTS="$CA" bash $S/c3a_run.sh > $S/$r.out 2>&1
  echo "$r $(date +%T) $(grep -a '^=== ' $S/$r.out | tail -1) | $(grep -a -m1 'AffordanceRoomGainScale' $B/room_concept/estimation_scripts/crossing/$r/logs/controller.log) | harvest: $(grep -c ' -> ' $S/$r.out) copied, $(grep -c REFUSED $S/$r.out) refused"
done
