#!/bin/bash
# Smoke: the fit recovers the generated zed envelope (min .12, max .55, soft .05, hit .9) within 10%, and the false alarm (.10, from the
# label-0 views: masks of non-doors through the same front end) within .02. Run from door_concept/.
tmp=$(mktemp --suffix=.csv) || exit 1; trap 'rm -f "$tmp"' EXIT
python3 -I tests/gen_detector_fit_fixture.py "$tmp" 3000 || exit 1
out=$(python3 -I tools/fit_detector_envelopes.py "$tmp" --dry-run) || exit 1
echo "$out"
python3 -I - "$out" <<'P'
import re,sys
m=re.search(r"zed:.*min_fill=([\d.]+) max_fill=([\d.]+) soft=([\d.]+) hit_rate=([\d.]+)",sys.argv[1])
got=list(map(float,m.groups())); want=[0.12,0.55,0.05,0.9]
bad=[(g,w) for g,w in zip(got,want) if abs(g-w)>0.1*w]
fa=re.search(r"zed: false_alarm=([\d.]+)",sys.argv[1])
if fa is None or abs(float(fa.group(1))-0.10)>0.02: bad.append(("false_alarm",fa and fa.group(1)))
print("FAIL" if bad else "PASS", bad); sys.exit(1 if bad else 0)
P
