#!/bin/bash
# Warm-start fixture (docs/superpowers/specs/2026-11-01-ltsm-warm-start-design.md): what one run left behind, frozen, so a
# later run can START with that memory (warm_start_tour.sh installs a COPY; the fixture itself is never written).
# usage: make_warm_fixture.sh <fixture_name> <ltsm_generation.json> <door_identities.csv> <room_name_hwm>
#   writes crossing/fixtures/<fixture_name>/{ltsm_history/<gen>.json + <gen>_layout.json, door_identities.csv,
#   room_name_hwm.txt, SHA256SUMS, README}
set -e
[ $# -eq 4 ] || { echo "usage: $0 <fixture_name> <ltsm_generation.json> <door_identities.csv> <room_name_hwm>"; exit 2; }
E=/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts
F=$E/crossing/fixtures/$1; GEN=$2; DOORS=$3; HWM=$4
[ -e "$F" ] && { echo "fixture $F exists: refusing to overwrite"; exit 1; }
LAYOUT=${GEN%.json}_layout.json
for f in "$GEN" "$LAYOUT" "$DOORS"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
mkdir -p $F/ltsm_history
cp "$GEN" "$LAYOUT" $F/ltsm_history/
cp "$DOORS" $F/door_identities.csv
printf '%s\n' "$HWM" > $F/room_name_hwm.txt
printf 'warm-start fixture %s, made %s\n  ltsm generation: %s\n  door table: %s\n  room_name_hwm: %s\n' \
  "$1" "$(date -Iseconds)" "$GEN" "$DOORS" "$HWM" > $F/README
(cd $F && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
chmod -R a-w $F
echo "fixture: $F"; cat $F/SHA256SUMS
