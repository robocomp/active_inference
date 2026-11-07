#!/bin/bash
# Smoke: the fit recovers the fixture's generating zed envelope (min .12, max .55, soft .05, hit .9) within 10%. Run from door_concept/.
out=$(python3 -I tools/fit_detector_envelopes.py tests/data/detector_fit_fixture.csv --dry-run) || exit 1
echo "$out"
python3 -I - "$out" <<'P'
import re,sys
m=re.search(r"zed:.*min_fill=([\d.]+) max_fill=([\d.]+) soft=([\d.]+) hit_rate=([\d.]+)",sys.argv[1])
got=list(map(float,m.groups())); want=[0.12,0.55,0.05,0.9]
bad=[(g,w) for g,w in zip(got,want) if abs(g-w)>0.1*w]
print("FAIL" if bad else "PASS", bad); sys.exit(1 if bad else 0)
P
