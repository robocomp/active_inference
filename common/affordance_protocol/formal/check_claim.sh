#!/bin/bash
# ── CAN THE TWO AGENTS DISAGREE ABOUT WHAT IS BEING EXECUTED? ──────────────────────────────────
# One question, two configurations. ClaimCarried=FALSE is the protocol as it stood on 2026-08-23 and
# MUST violate both AgreementObservable and NoFalseAbandon: a model in which the known hazard does
# not reproduce is not modelling it (check_epoch.sh's rule, kept). TRUE must hold both, AND must
# still terminate — a fix that buys agreement with a deadlock is not a fix.
JAR="$1"; cd "$(dirname "$0")"
[ -z "$JAR" ] && { echo "usage: $0 <tla2tools.jar>"; exit 2; }

printf "%-8s %-22s %-18s %-12s %s\n" CLAIM AgreementObservable NoFalseAbandon Terminates states
printf -- "-------- ---------------------- ------------------ ------------ ----------\n"

run() {   # $1=constant  $2=property-list  $3=extra
  cat > claim.cfg <<CFG
SPECIFICATION Spec
CONSTANTS Cells = {c1,c2} ClaimCarried = $1 MaxSteps = 2 Timeout = 2
INVARIANT TypeOK $2
$3
CFG
  java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -config claim.cfg AffordanceExecutionClaim.tla 2>&1
}

# ★★"states generated" IS NOT ENOUGH, AND I LEARNED THAT THE SAME WAY THIS DIRECTORY DID.
# The first version of this harness copied check_epoch.sh's guard and printed a full table —
# FALSE VIOLATED / TRUE holds, exactly the expected shape — from a run where TLC had ABORTED after
# four states with "unable to fingerprint" (a record compared against a string). A crashed run still
# prints "N states generated", so that line proves the model STARTED, never that it FINISHED.
# The only positive evidence of a completed exploration is TLC's own closing line, so that is what is
# required here; anything else is reported as a broken run and never as a verdict.
completed() { echo "$1" | grep -qE "Model checking completed|is violated"; }
crashed()   { echo "$1" | grep -qE "unable to fingerprint|Attempted to|TLC threw|was not assigned|Unknown operator|is not a legal"; }

verdict() {  # $1=output $2=name -> "VIOLATED" | "holds" | a loud failure
  if crashed "$1";      then echo "TLC-CRASHED"; return; fi
  if ! completed "$1";  then echo "NOT-FINISHED"; return; fi
  if echo "$1" | grep -q "Invariant $2 is violated"; then echo "VIOLATED"; else echo "holds"; fi
}

for C in FALSE TRUE; do
  # ★VERDICTS FIRST, TOOL FAILURES SECOND. A naive grep for "Error:" calls a real violation a broken
  # run — the failure mode check_epoch.sh records as sixteen rows printing SOUND without executing
  # a single state. Every column below is read only after "states generated" is confirmed present.
  a=$(run "$C" "AgreementObservable" "")
  va=$(verdict "$a" AgreementObservable)
  n=$(run "$C" "NoFalseAbandon" "")
  vn=$(verdict "$n" NoFalseAbandon)
  t=$(run "$C" "" "PROPERTY Terminates")
  if crashed "$t"; then vt="TLC-CRASHED"
  elif echo "$t" | grep -qE "Temporal properties were violated"; then vt="VIOLATED"
  elif completed "$t"; then vt="holds"
  else vt="NOT-FINISHED"; fi
  st=$(echo "$a" | grep -oE "[0-9]+ states generated" | head -1)
  printf "%-8s %-22s %-18s %-12s %s\n" "$C" "$va" "$vn" "$vt" "${st:-—}"
done

echo
echo "Expected: FALSE violates both invariants (the live failure reproduces); TRUE holds all three."
echo "A run where FALSE 'holds' is a model that has abstracted the defect away — fix the model, not"
echo "the table. TLC-CRASHED or NOT-FINISHED in ANY cell invalidates the WHOLE table, not just that"
echo "cell: the two columns are separate TLC runs over the same spec, and a spec that cannot be"
echo "explored in one configuration was not explored in the other either."
