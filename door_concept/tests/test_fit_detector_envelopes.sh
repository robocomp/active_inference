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

# Beta-prior posterior mean: 1 false report in 21 label-0 views, default false_alarm 0.05 (ricoh), prior worth 20 views. With a front end of
# exactly 1 the posterior mean is (1 + 20*0.05)/(21 + 20) = 2/41 = 0.04878; the fixture's front end is 0.9987 (env at fill 0.5), so expect 0.0488 +- 0.0015.
python3 -I - "$tmp" <<'P'
import sys
h="cycle,door,camera,view_id,q,fill_h,fill_v,aspect,vis,kappa,hit,label\n"
rows=[]
for i in range(25): rows.append(f"{i},d{i},ricoh,{i},0.5,0.9,0.9,1,1,1,1,1")
for i in range(21): rows.append(f"{100+i},n{i},ricoh,{100+i},0.05,0.9,0.9,1,1,1,{1 if i==0 else 0},0")
open(sys.argv[1],"w").write(h+"\n".join(rows)+"\n")
P
out=$(python3 -I tools/fit_detector_envelopes.py "$tmp" --fix-envelope --dry-run) || exit 1
echo "$out"
python3 -I - "$out" <<'P'
import re,sys
m=re.search(r"ricoh: false_alarm=([\d.]+) posterior mean from (\d+) label-0 views",sys.argv[1])
ok=m is not None and abs(float(m.group(1))-2/41)<0.0015 and int(m.group(2))==21
print("PASS" if ok else "FAIL", m and m.groups()); sys.exit(0 if ok else 1)
P
