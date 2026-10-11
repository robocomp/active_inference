#!/usr/bin/env python3
"""Grade every run_<id> of the estimation campaign against the apartment ground truth -> results.csv + panel figure."""
import sys, re, glob, os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from shapely.geometry import Polygon
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/robocomp/components/active_inference/room_concept/estimation_scripts/synthetic/campaign_v2_LT')
import json
ROOMS={r['id']:np.array(r['polygon']) for r in json.load(open(os.path.join(OUT,'rooms.json')))}
def score(V, gtp):
    Pl = Polygon(V).buffer(0); I = Pl.intersection(gtp).area
    return I / Pl.union(gtp).area, I / Pl.area, I / gtp.area
rows, panels = [], []
for D in sorted(glob.glob(f'{OUT}/run_*'), key=lambda d: int(d.split('_')[-1])):
    rid = int(D.split('_')[-1]); st = open(f'{D}/status.txt').read().strip() if os.path.exists(f'{D}/status.txt') else 'missing'
    r = dict(run=rid, status=st.split()[0].split('=')[1] if '=' in st else st)
    try:
        g = pd.read_csv(f'{D}/gt_error.csv', on_bad_lines='skip'); t = (g.ts_ms - g.ts_ms.iloc[0]) / 1000
        A = np.c_[g.gt_x_world, g.gt_y_world]; E = np.c_[g.est_x, g.est_y]
        am, em = A.mean(0), E.mean(0); U, _, Vt = np.linalg.svd((A - am).T @ (E - em)); R = Vt.T @ U.T
        if np.linalg.det(R) < 0: Vt[-1] *= -1; R = Vt.T @ U.T
        err = np.linalg.norm((A - am) @ R.T + em - E, axis=1)
        GTW = ROOMS[rid]; gtp = Polygon((GTW - am) @ R.T + em)
        pr = pd.read_csv(f'{D}/proto_room.csv', on_bad_lines='skip')
        prom = pr[pr.promoted == 1]
        tp = (prom.ts_ms.iloc[0] - pr.ts_ms.iloc[0]) / 1000 if len(prom) else np.nan
        mpath = t <= (tp if np.isfinite(tp) else t.iloc[-1])
        r.update(t_promote_s=tp, path_to_promote_m=np.hypot(*np.diff(A[mpath.values], axis=0).T).sum(),
                 path_total_m=np.hypot(*np.diff(A, axis=0).T).sum(), duration_s=t.iloc[-1],
                 pose_err_med_m=np.median(err), pose_err_p95_m=np.percentile(err, 95), pose_err_max_m=err.max())
        ml = [l.strip().split(';') for l in open(f'{D}/map_layout.csv') if not l.startswith('#')]
        P = np.array([[float(c) for c in v.split(',')] for v in ml[-1][2].split()])
        if os.path.exists(f'{D}/promotion_select.csv'):   # the outline promotion ADOPTED (chosen by description length)
            ch = [l.strip().split(';') for l in open(f'{D}/promotion_select.csv') if not l.startswith('#') and l.split(';')[2] == '1']
            if ch: P = np.array([[float(c) for c in v.split(',')] for v in ch[0][3].split()])
        r['promoted_verts'] = len(P); r['promoted_iou'], r['promoted_prec'], r['promoted_rec'] = score(P, gtp)
        if os.path.exists(f'{D}/learnt.svg'):
            Q = np.array([[float(c) for c in v.split(',')] for v in re.search(r'points="([^"]+)"', open(f'{D}/learnt.svg').read()).group(1).split()])
            Q[:, 0] *= -1
            r['refined_verts'] = len(Q); r['refined_iou'], r['refined_prec'], r['refined_rec'] = score(Q, gtp)
        else: Q = None
        panels.append((rid, gtp, P, Q, E, r))
    except Exception as ex:
        r['error'] = repr(ex)[:120]
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/results.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250); print(df.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
ok = df[df.status == 'ok'] if 'status' in df else df
for c in ['promoted_iou', 'refined_iou', 'pose_err_med_m', 't_promote_s']:
    if c in ok: print(f'{c}: median {ok[c].median():.3f}  min {ok[c].min():.3f}  max {ok[c].max():.3f}')
if panels:
    n = len(panels); cols = 5; fig, ax = plt.subplots((n + cols - 1) // cols, cols, figsize=(4 * cols, 4 * ((n + cols - 1) // cols)), squeeze=False)
    for a, (rid, gtp, P, Q, E, r) in zip(ax.flat, panels):
        a.plot(*gtp.exterior.xy, color='#4a9', lw=3, alpha=.45); a.plot(*np.r_[P, P[:1]].T, 'k-', lw=.8)
        if Q is not None: a.plot(*np.r_[Q, Q[:1]].T, 'r-', lw=.8)
        a.plot(*E.T, color='#e80', lw=.6); a.plot(0, 0, 'bo', ms=4)
        a.set_title(f"run {rid}: IoU {r.get('promoted_iou', np.nan):.3f} -> {r.get('refined_iou', np.nan):.3f}", fontsize=9); a.set_aspect('equal'); a.axis('off')
    for a in list(ax.flat)[n:]: a.axis('off')
    plt.tight_layout(); plt.savefig(f'{OUT}/layouts_panel.png', dpi=80)
