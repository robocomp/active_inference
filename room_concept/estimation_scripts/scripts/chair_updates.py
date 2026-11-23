#!/usr/bin/env python3
"""Chair belief updates per run (2026-11-23, chair fixation-precision fix).   chair_updates.py <run_dir>...

Per chair node (chair_ai2_log.csv, one row per matched mask frame, chair_fitter.cpp:608):
  rows, UPDATES (gated == 0: the Laplace update ran), of which OFF-CENTRE (centroid_r > 0.60, the old gate's line),
  std at first row / min / last, and the planar error against Webots truth (world frame, run_frames.make_to_world;
  nearest chair of the world within 1 m of the belief) at the first and the last row.
DRIFT: for every update with centroid_r > 0.60, the change in planar error across it (next row minus previous row);
  reported as median and the share of off-centre updates that increased the error by more than 5 cm.
BIRTHS: "BIRTH 'chair_N'" lines in logs/chair_concept.log per node name (a re-birth restarts sigma at the prior).
"""
import csv, math, os, re, statistics, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_frames import make_to_world
import c3a_objects as C

CENTRE = 0.60


def one(run):
    run = run.rstrip('/')
    tw = make_to_world(run)
    gt = [g for g in C.gt_objects(C.W.WBT) if g[1] == 'chair']
    rows = defaultdict(list)
    for r in csv.DictReader(open(f'{run}/chair_ai2_log.csv')):
        try:
            rows[r['node']].append(dict(t=int(r['t_ms']), cx=float(r['cx']), cy=float(r['cy']), s=max(float(r['std_cx']), float(r['std_cy'])),
                                        g=int(r['gated']), c=float(r['centroid_r']) if r['centroid_r'] not in ('', 'nan') else float('nan')))
        except (KeyError, ValueError):
            pass
    births = Counter(m.group(1) for l in open(f'{run}/logs/chair_concept.log', errors='replace')
                     for m in [re.search(r"BIRTH '(chair_\d+)'", l)] if m)
    out = []
    drift = []
    for n, R in rows.items():
        R.sort(key=lambda x: x['t'])
        def err(x):
            w = tw(x['cx'], x['cy'], x['t'])
            if not w: return float('nan')
            d = min((math.dist(w, (g[2], g[3])) for g in gt), default=float('nan'))
            return d if d <= 1.0 else float('nan')
        upd = [i for i, x in enumerate(R) if x['g'] == 0]
        off = [i for i in upd if x_c(R[i]) > CENTRE]
        for i in off:
            if 0 < i < len(R) - 1:
                a, b = err(R[i - 1]), err(R[i + 1])
                if a == a and b == b: drift.append(b - a)
        out.append(dict(node=n, rows=len(R), updates=len(upd), off_centre_updates=len(off), births=births.get(n, 0),
                        s_first=R[0]['s'], s_min=min(x['s'] for x in R), s_last=R[-1]['s'], err_first=err(R[0]), err_last=err(R[-1])))
    return out, drift, births


def x_c(x):
    return x['c'] if x['c'] == x['c'] else 0.0


if __name__ == '__main__':
    for run in sys.argv[1:]:
        out, drift, births = one(run)
        name = os.path.basename(run.rstrip('/'))
        stuck = [o for o in out if o['s_min'] >= 0.299]
        print(f'== {name}: {len(out)} chair nodes, {sum(o["updates"] for o in out)} updates ({sum(o["off_centre_updates"] for o in out)} off-centre), '
              f'{len(stuck)} never below the 0.30 prior; births {sum(births.values())} for {len(births)} names, re-born names {sum(1 for v in births.values() if v > 1)}')
        for o in sorted(out, key=lambda o: o['node']):
            print(f"   {o['node']:10s} rows {o['rows']:5d} upd {o['updates']:4d} (off {o['off_centre_updates']:4d}) births {o['births']} "
                  f"σ {o['s_first']:.3f}->{o['s_min']:.3f}->{o['s_last']:.3f}  err {o['err_first']:.2f}->{o['err_last']:.2f} m")
        if drift:
            print(f'   off-centre update error change: median {statistics.median(drift):+.3f} m, n {len(drift)}, share > +5 cm {sum(d > 0.05 for d in drift) / len(drift):.2f}')
