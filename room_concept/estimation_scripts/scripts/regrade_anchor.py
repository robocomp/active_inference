#!/usr/bin/env python3
"""Re-score refined/promoted layouts in the map's TRUE frame (anchored at the start pose: the estimate starts at (0,0,0)
and gt_theta is offset by +90 deg from the map heading) and by best rigid fit (shape only). The campaign grader aligns by
an SVD fit of the whole trajectory, which inherits the pose error (up to 0.9 deg here = 11 cm across a 7 m room)."""
import sys, json, re, os, numpy as np, pandas as pd
from shapely.geometry import Polygon
from scipy.optimize import minimize
FURN = []
if os.environ.get('FURNITURE'):   # furniture.json (estimation_scripts/scripts/furniture.py, e.g. estimation_scripts/apartment/furniture.json), world XY, drawn in each panel
    from shapely.geometry import shape
    FURN = [shape(o['geometry']) for o in json.load(open(os.environ['FURNITURE']))]
D = sys.argv[1]; rooms = json.load(open(f'{D}/rooms.json')); rows = []; panels = []
def iou(P, G): P = P.buffer(0); I = P.intersection(G).area; return I / P.union(G).area
def shapely_tf(g, p0, th):
    from shapely import affinity
    return affinity.rotate(affinity.translate(g, -p0[0], -p0[1]), th, origin=(0, 0), use_radians=True)
def tf(V, th, t): c, s = np.cos(th), np.sin(th); return V @ np.array([[c, -s], [s, c]]).T + t
for rid in range(len(rooms)):
    R = f'{D}/run_{rid}'
    if not os.path.exists(f'{R}/learnt.svg'): continue
    g = pd.read_csv(f'{R}/gt_error.csv', on_bad_lines='skip')
    p0 = np.array([g.gt_x_world.iloc[0], g.gt_y_world.iloc[0]]); th = np.pi / 2 - g.gt_theta.iloc[0]
    GT = np.array(rooms[rid]['polygon']); G = Polygon(tf(GT - p0, th, 0))          # GT in the start frame
    Q = np.array([[float(c) for c in v.split(',')] for v in re.search(r'points="([^"]+)"', open(f'{R}/learnt.svg').read()).group(1).split()]); Q[:, 0] *= -1
    ml = [l.strip().split(';') for l in open(f'{R}/map_layout.csv') if not l.startswith('#')]
    P = np.array([[float(c) for c in v.split(',')] for v in ml[-1][2].split()])
    if os.path.exists(f'{R}/promotion_select.csv'):
        ch = [l.strip().split(';') for l in open(f'{R}/promotion_select.csv') if not l.startswith('#') and l.split(';')[2] == '1']
        if ch: P = np.array([[float(c) for c in v.split(',')] for v in ch[0][3].split()])
    def best(V):
        f = lambda x: -iou(Polygon(tf(V, x[0], x[1:])), G)
        r = minimize(f, [0, 0, 0], method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-6, initial_simplex=[[0,0,0],[.01,0,0],[0,.03,0],[0,0,.03]]))
        return -r.fun, np.degrees(r.x[0]), np.hypot(*r.x[1:]), r.x
    sr, dth, dt, xq = best(Q); spr = best(P)[0]
    E = np.c_[g.est_x, g.est_y]; F_start = [h for h in (shapely_tf(f, p0, th) for f in FURN) if G.contains(h.centroid)]   # props stored outside the apartment are not drawn
    panels.append((rid, G, F_start, tf(P, *(lambda x: (x[0], x[1:]))(best(P)[3])), tf(Q, xq[0], xq[1:]), E, spr, sr))
    rows.append(dict(run=rid, prom_shape=spr, prom_anchor=iou(Polygon(P), G), ref_anchor=iou(Polygon(Q), G), ref_shape=sr, frame_err_deg=dth, frame_err_m=dt, verts=len(Q)))
df = pd.DataFrame(rows); df.to_csv(f'{D}/results_anchor.csv', index=False, float_format='%.4f')
print(df.round(4).to_string(index=False)); print('median ref_anchor %.4f min %.4f | ref_shape median %.4f min %.4f' % (df.ref_anchor.median(), df.ref_anchor.min(), df.ref_shape.median(), df.ref_shape.min()))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
n = len(panels); cols = 5; fig, ax = plt.subplots((n + cols - 1) // cols, cols, figsize=(4 * cols, 4 * ((n + cols - 1) // cols)), squeeze=False)
for a, (rid, G, FS, P, Q, E, sp, sr) in zip(ax.flat, panels):
    for f in FS:
        for part in getattr(f, 'geoms', [f]): a.fill(*part.exterior.xy, color='0.55', alpha=.55, lw=0)
    a.plot(*G.exterior.xy, color='#4a9', lw=3, alpha=.45); a.plot(*np.r_[P, P[:1]].T, 'k-', lw=.8); a.plot(*np.r_[Q, Q[:1]].T, 'r-', lw=.8)
    a.plot(*E.T, color='#e80', lw=.6); a.plot(0, 0, 'bo', ms=4)
    a.set_title(f"{sys.argv[2] if len(sys.argv) > 2 else 'run'} {rid}: IoU {sp:.3f} -> {sr:.3f}", fontsize=9); a.set_aspect('equal'); a.axis('off')
for a in list(ax.flat)[n:]: a.axis('off')
plt.tight_layout(); plt.savefig(f'{D}/layouts_panel' + ('_furniture' if FURN else '') + '.png', dpi=80)
