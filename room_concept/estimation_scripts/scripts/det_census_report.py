#!/usr/bin/env python3
"""det_census_report.py <det_census.csv> — per class and per viewpoint: how many frames detected it, confidence, range.
The classes the new bedroom/bathroom agents need are listed first, with an explicit 0 when never detected."""
import csv, statistics, sys
from collections import defaultdict

NEED = ["bed", "nightstand", "toilet", "bathtub", "sink", "mirror"]
rows = list(csv.DictReader(open(sys.argv[1])))
frames_at = defaultdict(set)              # viewpoint -> frames recorded there
hits = defaultdict(lambda: defaultdict(list))   # label -> viewpoint -> [(conf, range)]
for r in rows:
    vp = " ".join(r["pose"].split()[:1] + r["pose"].split()[3:4]) or "?"   # "<name> <deg>"
    frames_at[vp].add(r["frame"])
    try:
        hits[r["label"]][vp].append((float(r["conf"]), float(r["range_m"])))
    except ValueError:
        pass
print(f"{len(rows)} mask slices over {len({r['frame'] for r in rows})} frames")
print(f"{'class':14}{'frames':>8}{'conf med':>10}{'conf max':>10}{'range med':>11}  best viewpoints")
for lab in NEED + sorted(l for l in hits if l not in NEED):
    allv = [c for vp in hits[lab].values() for c in vp]
    if not allv:
        print(f"{lab:14}{0:>8}{'-':>10}{'-':>10}{'-':>11}  NEVER DETECTED"); continue
    confs = [c for c, _ in allv]; rngs = [d for _, d in allv if d == d]
    best = sorted(hits[lab].items(), key=lambda kv: -len(kv[1]))[:3]
    bs = ", ".join(f"{vp} ({len(v)}/{len(frames_at[vp])})" for vp, v in best)
    print(f"{lab:14}{len(allv):>8}{statistics.median(confs):>10.2f}{max(confs):>10.2f}"
          f"{(statistics.median(rngs) if rngs else float('nan')):>11.2f}  {bs}")
