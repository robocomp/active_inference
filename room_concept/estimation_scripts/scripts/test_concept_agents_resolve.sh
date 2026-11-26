#!/bin/bash
# Print-only test of concept_agents.sh's binary choice (shared-engine final fix wave F1): which binary each class resolves to
# under the defaults, COUNTABLE_AGENTS and LEGACY_AGENTS. Builds a fake checkout (no agent runs): legacy bin/<a> for
# bottle/chair/table/hood/refrigerator, a countable link <a>/countable/<a> for chair/table/hood/refrigerator, and bin/<a> as a
# countable link for nightstand/bed/toilet.  Usage: bash test_concept_agents_resolve.sh   (exit 0 = all pass)
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T=$(mktemp -d); trap 'rm -rf $T' EXIT
mkdir -p $T/countable_concept/bin; printf '#!/bin/sh\n' > $T/countable_concept/bin/countable_concept; chmod +x $T/countable_concept/bin/countable_concept
for a in bottle_concept chair_concept table_concept hood_concept refrigerator_concept; do
  mkdir -p $T/$a/bin $T/$a/etc; printf '#!/bin/sh\n' > $T/$a/bin/$a; chmod +x $T/$a/bin/$a
done
for a in chair_concept table_concept hood_concept refrigerator_concept; do
  mkdir -p $T/$a/countable; ln -s ../../countable_concept/bin/countable_concept $T/$a/countable/$a
done
for a in nightstand_concept bed_concept toilet_concept; do
  mkdir -p $T/$a/bin $T/$a/etc; ln -s ../../countable_concept/bin/countable_concept $T/$a/bin/$a
done
ALLA="bottle_concept chair_concept table_concept hood_concept refrigerator_concept nightstand_concept bed_concept toilet_concept"
fail=0
check() {  # check <label> "<expected: agent=mode ...>" [env assignments...]
  local label=$1 want=$2; shift 2
  local got; got=$(env "$@" CONCEPT_WT=$T CONCEPT_AGENTS="$ALLA" bash $S/concept_agents.sh resolve 2>$T/err)
  echo "--- $label"; echo "$got" | awk -F'\t' '{printf "  %-22s %-9s %s\n", $1, $2, $4}'
  [ -s $T/err ] && sed 's/^/  stderr: /' $T/err
  for kv in $want; do
    local a=${kv%=*} m=${kv#*=} line
    line=$(echo "$got" | awk -F'\t' -v a=$a '$1==a')
    local gm cfg; gm=$(echo "$line" | cut -f2); cfg=$(echo "$line" | cut -f4)
    local wc=config.toml; [ $m = countable ] && [ -L $T/$a/countable/$a ] && wc=countable.toml
    if [ "$gm" != "$m" ] || [ "$(basename "$cfg")" != $wc ]; then echo "  FAIL $a: want $m ($wc), got $gm ($(basename "$cfg"))"; fail=1; fi
  done
}
DEF="bottle_concept=legacy nightstand_concept=countable bed_concept=countable toilet_concept=countable"
check "defaults" "$DEF chair_concept=legacy table_concept=legacy hood_concept=legacy refrigerator_concept=legacy"
check "COUNTABLE_AGENTS=chair hood fridge" "$DEF chair_concept=countable table_concept=legacy hood_concept=countable refrigerator_concept=countable" \
  COUNTABLE_AGENTS="chair_concept hood_concept refrigerator_concept"
check "COUNTABLE_AGENTS=table (cut: refused, legacy)" "$DEF table_concept=legacy chair_concept=legacy" COUNTABLE_AGENTS="table_concept"
grep -q "CUT from part 1" $T/err || { echo "  FAIL: no loud refusal for the countable table"; fail=1; }
check "COUNTABLE_AGENTS=chair hood fridge + LEGACY_AGENTS=hood (override)" \
  "$DEF chair_concept=countable hood_concept=legacy refrigerator_concept=countable table_concept=legacy" \
  COUNTABLE_AGENTS="chair_concept hood_concept refrigerator_concept" LEGACY_AGENTS="hood_concept"
[ $fail = 0 ] && echo "ALL PASS" || echo "FAILED"
exit $fail
