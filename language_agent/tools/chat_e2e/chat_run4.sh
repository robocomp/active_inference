#!/bin/bash
# End-to-end chat test: fleet parked (HOLD) in room_1 + ltom_agent, chair_concept and language_agent from the object-colour worktree.
# The user types in the language_agent window; ends when crossing/<run>/stop exists.
A=/home/pbustos/robocomp/components/active_inference; E=$A/room_concept/estimation_scripts; RUN=$1; L=$E/crossing/$RUN/logs
S=/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-language-agent/680f5d6c-7b33-47dc-a863-26056842a0a4/scratchpad
mkdir -p $E/crossing/$RUN/ltom_before
for f in ltom_repertory.json ltom_repertory.json.prev ltom_repertory.csv; do [ -f $A/ltom_agent/etc/$f ] && cp -p $A/ltom_agent/etc/$f $E/crossing/$RUN/ltom_before/; done
( for i in $(seq 1 200); do grep -a -q "MapMode=ltsm: layout of" $L/room_concept.log 2>/dev/null && break; sleep 1; done
  mkdir -p $L
  cd $A/ltom_agent && DISPLAY=:0 setsid nohup stdbuf -oL -eL /home/pbustos/robocomp/components/active_inference.worktrees/object-colour/ltom_agent/bin/ltom_agent etc/config.toml > $L/ltom_agent.log 2>&1 < /dev/null &
  sleep 2
  WT=/home/pbustos/robocomp/components/active_inference.worktrees/object-colour OBJECTS=chair_concept bash $E/scripts/run_objects.sh start $E/crossing/$RUN
  sleep 3
  cd ${LA_DIR:-/home/pbustos/robocomp/components/active_inference.worktrees/object-colour/language_agent} && DISPLAY=:0 QT_QPA_PLATFORM=xcb setsid nohup stdbuf -oL -eL ${LA_BIN:-bin/language_agent} etc/config.toml > $L/language_agent.log 2>&1 < /dev/null &
  echo OBJECTS_AND_CHAT_STARTED ) &
cd $E
HOLD=1 WT=$S/wt_shim3 WT_AGENTS="retina controller door_concept room_concept residual_concept ltsm_agent" N_LEGS=0 bash $S/goto_tour4.sh $RUN crossing/fixtures/three_rooms_door0
echo TOUR_EXIT=$?
pkill -TERM -x language_agent
WT=/nonexistent OBJECTS=chair_concept bash $E/scripts/run_objects.sh stop
pkill -TERM -x ltom_agent; sleep 6
pgrep -x ltom_agent >/dev/null && echo LTOM_STILL_RUNNING; pgrep -x language_agent >/dev/null && echo LANGUAGE_STILL_RUNNING
for d in DOOR_0 DOOR_1 DOOR_2; do python3 /home/pbustos/robocomp/components/active_inference.worktrees/rt-build/room_concept/estimation_scripts/scripts/door.py open $d; done
echo RUN_FINISHED
