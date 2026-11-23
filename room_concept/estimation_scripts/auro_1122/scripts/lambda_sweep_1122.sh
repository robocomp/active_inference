#!/bin/bash
# AuRo item 4 (2026-11-22): travel-cost sweep. One furnished C3a tour per λ (controller AffordanceLambdaCost, nats/m), cabinet
# out, LTOM empty. Order 0.1 -> 0.4 -> 0.2 (default last). The config is restored to its original line on any exit.
B=/home/pbustos/robocomp/components/active_inference; S=/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/1c86b8d6-47c4-4a7b-8f4b-04a583e967d4/scratchpad
CFG=$B/controller/etc/config.toml; cp -p $CFG $S/controller_config.before_lambda
trap 'cp -p $S/controller_config.before_lambda $CFG; echo "controller config restored"' EXIT
CA="chair_concept table_concept bottle_concept hood_concept refrigerator_concept nightstand_concept bed_concept toilet_concept"
for L in ${LAMBDAS:-0.1 0.4 0.2}; do
  [ -f $B/room_concept/estimation_scripts/crossing/STOP_SERIES ] && break
  sed -i -E "s/^AffordanceLambdaCost *= *[0-9.]+/AffordanceLambdaCost = $L/" $CFG
  grep -q "^AffordanceLambdaCost = $L " $CFG || { echo "λ=$L NOT SET"; exit 1; }
  r=lam_${L/./p}
  RUN=$r CONCEPT_WT=$B/../active_inference.worktrees/cabinet-exit CONCEPT_AGENTS="$CA" bash $S/c3a_run.sh > $S/$r.out 2>&1
  grep -a "^AffordanceLambdaCost\|AffordanceLambdaCost" $B/room_concept/estimation_scripts/crossing/$r/logs/controller.log | head -1 > $B/room_concept/estimation_scripts/crossing/$r/lambda.txt
  echo "$r $(date +%T) $(grep -a '^=== ' $S/$r.out | tail -1) | $(cat $B/room_concept/estimation_scripts/crossing/$r/lambda.txt | cut -c1-80)"
done
