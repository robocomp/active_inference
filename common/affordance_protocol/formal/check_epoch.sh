#!/bin/bash
# ── CAN A REPORT BE APPLIED TO THE WRONG OFFER? ────────────────────────────────────────────────
# One question, two configurations. EpochCarried=FALSE is the code as it stands and MUST fail:
# a model in which the known hazard does not reproduce is not modelling it. TRUE must hold.
JAR="$1"; cd "$(dirname "$0")"
printf "%-14s %-28s %s\n" EPOCH "BeliefIsHonest" verdict
printf -- "-------------- ---------------------------- -------\n"
for E in FALSE TRUE; do
  cat > epoch.cfg <<CFG
SPECIFICATION Spec
CONSTANTS Cells = {c1,c2} Reachable = {c2} EpochCarried = $E
INVARIANT TypeOK ObservedAreReachable BeliefIsHonest
CFG
  out=$(java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -config epoch.cfg AffordanceEpoch.tla 2>&1)
  # ★VERDICTS FIRST, TOOL FAILURES SECOND — a naive grep for "Error:" calls a real violation a
  # broken run, which is how sixteen rows once printed SOUND without executing a single state.
  if ! echo "$out" | grep -qE "[0-9]+ states generated"; then
      printf "%-14s %-28s %s\n" "$E" "-" "✗ NO STATES EXPLORED — not a verdict"
      echo "$out" | grep -iE "TLC attempted|was not assigned|Unknown operator|Attempted to" | head -2 | sed 's/^/        /'
      continue
  fi
  if echo "$out" | grep -q "Invariant BeliefIsHonest is violated"; then
      st=$(echo "$out" | grep -oE "[0-9]+ states generated" | head -1)
      printf "%-14s %-28s %s\n" "$E" "VIOLATED" "ABA reproduces ($st)"
  else
      st=$(echo "$out" | grep -oE "[0-9]+ states generated" | head -1)
      printf "%-14s %-28s %s\n" "$E" "holds" "no stale report survives ($st)"
  fi
done
