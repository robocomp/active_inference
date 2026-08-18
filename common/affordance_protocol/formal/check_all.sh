#!/bin/bash
# Exhaustively model-check the affordance protocol over every combination of the three
# design choices. Each combination is a real pair of code paths in the tree.
JAR="$1"; cd "$(dirname "$0")"
printf "%-7s %-7s %-7s | %-9s %-9s %-9s %s\n" LEVEL REFUSE REARM deadlock Progress NewsArr "verdict"
printf -- "------- ------- ------- + --------- --------- --------- -------\n"
for L in TRUE FALSE; do for R in TRUE FALSE; do for A in TRUE FALSE; do
  cat > cfg.cfg <<CFG
SPECIFICATION Spec
CONSTANTS LevelTriggered = $L ConsumerRefuses = $R RearmUnchanged = $A
PROPERTY Progress
INVARIANT TypeOK
CFG
  out=$(java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -config cfg.cfg -deadlock AffordanceProtocol.tla 2>&1)
  dl=$(echo "$out" | grep -c "Deadlock reached")
  # rerun WITHOUT -deadlock so TLC actually reports deadlocks, then check the property
  out2=$(java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -config cfg.cfg AffordanceProtocol.tla 2>&1)
  dead=$(echo "$out2" | grep -qi "deadlock" && echo YES || echo no)
  prog=$(echo "$out2" | grep -qi "temporal properties were violated" && echo VIOLATED || echo ok)
  verdict="SOUND"; [ "$dead" = YES ] && verdict="DEADLOCK"; [ "$prog" = VIOLATED ] && verdict="NO-PROGRESS"
  printf "%-7s %-7s %-7s | %-9s %-9s %-9s %s\n" "$L" "$R" "$A" "$dead" "$prog" "-" "$verdict"
done; done; done
