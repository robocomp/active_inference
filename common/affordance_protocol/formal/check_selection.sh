#!/bin/bash
# Sweep the three design choices that were live on 2026-08-19. Each is a real code path.
JAR="$1"; cd "$(dirname "$0")"
printf "%-6s %-6s %-8s %-7s | %-9s %-9s %-11s %s\n" REARM HOLD TIMEOUT ARGMAX deadlock Observes WireProgress verdict
printf -- "------ ------ -------- ------- + --------- --------- ----------- -------\n"
for A in TRUE FALSE; do for H in TRUE FALSE; do for O in TRUE FALSE; do for M in TRUE FALSE; do
  cat > sel.cfg <<CFG
SPECIFICATION Spec
CONSTANTS Cells = {c1,c2} Reachable = {c2} Favoured = c1
          RearmUnchanged = $A ConsumerHoldsRefused = $H OfferTimeout = $O RefusalMovesArgmax = $M
INVARIANT TypeOK ObservedAreReachable
PROPERTY Observes
CFG
  out=$(java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -config sel.cfg AffordanceSelection.tla 2>&1)
  # ★TRUST NOTHING THAT DID NOT RUN — but tell a VERDICT from a TOOL FAILURE. TLC prefixes both with
  # "Error:", so a naive grep for that calls a real deadlock a broken run (it did). Check for the
  # verdicts FIRST, and only then for the failures that mean the model never executed.
  if ! echo "$out" | grep -qE "[0-9]+ states generated"; then
      printf "%-6s %-6s %-8s %-7s | %s\n" "$A" "$H" "$O" "$M" "✗ NO STATES EXPLORED — not a verdict"
      echo "$out" | grep -iE "TLC attempted|was not assigned|Unknown operator" | head -1 | sed 's/^/        /'
      continue
  fi
  dead=$(echo "$out" | grep -qi "deadlock" && echo YES || echo no)
  obs=$(echo "$out"  | grep -qi "(was|were) violated" && echo VIOLATED || echo ok)
  cat > sel2.cfg <<CFG
SPECIFICATION Spec
CONSTANTS Cells = {c1,c2} Reachable = {c2} Favoured = c1
          RearmUnchanged = $A ConsumerHoldsRefused = $H OfferTimeout = $O RefusalMovesArgmax = $M
PROPERTY WireProgress
CFG
  out2=$(java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -config sel2.cfg -deadlock AffordanceSelection.tla 2>&1)
  wp=$(echo "$out2" | grep -qi "(was|were) violated" && echo VIOLATED || echo ok)
  v="SOUND"; [ "$obs" = VIOLATED ] && v="NEVER-OBSERVES"; [ "$dead" = YES ] && v="DEADLOCK"
  printf "%-6s %-6s %-8s %-7s | %-9s %-9s %-11s %s\n" "$A" "$H" "$O" "$M" "$dead" "$obs" "$wp" "$v"
done; done; done; done
rm -f sel.cfg sel2.cfg
