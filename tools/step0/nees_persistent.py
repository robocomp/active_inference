#!/usr/bin/env python3
"""Step 0, NEES of the countables on the persistent path, for runs that nees_agents.py cannot score.
   nees_persistent.py [--gt mesh|boundingobject] <run_dir> [<run_dir> ...]   (WBT_PATH selects the world; --gt as in nees_agents.py, default mesh)

Why a second script: nees_agents.py needs c3a_instances.csv (ltom_repertory_end.json) and a dwell of >= 10 samples after
the 3 s trim in each room window (legs.tsv). The objects_live* runs have neither: the legs run straight into the next
leg, so every window is empty. This script keeps the SAME chain as nees_agents.py (world -> internal by a rigid fit
per room visit -> published by the logged gauge at the row's time) and the SAME NEES = e^T P^-1 e with the agent's
diagonal sigma, but changes two things:
  * room membership comes from the ground-truth position inside the Webots room polygon (wbt_rooms), not from legs.tsv;
    the fit uses every sample inside that room, after the first 3 s in it.
  * node -> Webots object is the nearest object of the node's class within the c3a radius, taken at each row.
Reuses nees_agents.py / c3a_objects.py for the fit, the gauge and the ground-truth list.
"""
import csv, glob, math, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, '..', '..', 'room_concept', 'estimation_scripts', 'scripts')
sys.path.insert(0, SC)
import c3a_objects as C
import nees_agents as N
import wbt_rooms as W

CLASSES = ['nightstand', 'bed', 'toilet']
TRIM_MS = 3000
# Truth point = the Webots origin by default. With NEES_CENTRE=1 it is the centre of the graded footprint instead
# (mesh bounding-box centre in the object frame, rotated by the object's yaw). Toilet: Toilet.proto IndexedFaceSets x 0.2
# (Transform scale 0.2) span x [-0.435, 0.303], y [-0.303, 0.286], so the centre is (-0.066, -0.009) from the origin.
USE_CENTRE = os.environ.get('NEES_CENTRE') == '1'


def room_windows(run, polys):
    est = [(int(r['ts_ms']), float(r['est_x']), float(r['est_y']), float(r['gt_x_world']), float(r['gt_y_world']))
           for r in csv.DictReader(open(f'{run}/gt_error.csv'))]
    P = np.array([[e[3], e[4]] for e in est])
    where = np.full(len(est), '', dtype=object)
    for name, poly in polys.items():
        where[W.inside(poly, P)] = name
    fits = {}
    first = {}
    for i, e in enumerate(est):
        r = where[i]
        if r:
            first.setdefault(r, e[0])
    for r, t0 in first.items():
        S = [e for i, e in enumerate(est) if where[i] == r and e[0] >= t0 + TRIM_MS]
        if len(S) >= 10:
            th, t, rms = C.rigid_fit([(e[1], e[2]) for e in S], [(e[3], e[4]) for e in S])
            fits[r] = (th, t, rms, S[0][0], S[-1][0], len(S))
    return fits


def to_map(fit, wx, wy):
    th, t = fit[0], fit[1]
    dx, dy = wx - t[0], wy - t[1]
    return math.cos(th) * dx + math.sin(th) * dy, -math.sin(th) * dx + math.cos(th) * dy


def to_world(fit, mx, my):
    th, t = fit[0], fit[1]
    return math.cos(th) * mx - math.sin(th) * my + t[0], math.sin(th) * mx + math.cos(th) * my + t[1]


def run_rows(run, polys, gt):
    fits = room_windows(run, polys)
    G = N.gauges(run); Gts = [g[0] for g in G]
    out = []
    if USE_CENTRE and N.GT == 'boundingobject':   # (--gt mesh already carries the footprint centre)
        sys.path.insert(0, os.path.join(HERE, '..', 'gt_mesh'))
        from mesh_extents import known_extents
        KE = known_extents()
        import mesh_extents as ME
        yaw = {n: v['yaw'] for n, v in ME.world_layers(W.WBT).items() if v['cls'] == 'toilet'}
        gt = [(o[0], o[1], *ME.to_world(o[2], o[3], yaw[o[0]], KE['toilet'][4], KE['toilet'][5]))
              if o[1] == 'toilet' and o[0] in yaw else o for o in gt]
    for cls in CLASSES:
        p = f'{run}/{cls}_ai2_log.csv'
        if not os.path.exists(p): continue
        cands = [o for o in gt if o[1] == cls]
        for r in csv.DictReader(open(p)):
            t = int(N.f(r, 'pkt_ts'))
            cx, cy, sx, sy = N.f(r, 'cx'), N.f(r, 'cy'), N.f(r, 'std_cx'), N.f(r, 'std_cy')
            if not (sx > 0 and sy > 0 and math.isfinite(cx)): continue
            _, g0, g1, gth = N.gauge_at(G, Gts, t)
            # published -> internal (inverse gauge) for the nearest-object match, then per room fit
            dx, dy = cx - g0, cy - g1
            mx, my = math.cos(gth) * dx + math.sin(gth) * dy, -math.sin(gth) * dx + math.cos(gth) * dy
            best = None
            for room, fit in fits.items():
                if not fit[3] <= t <= fit[4]: continue
                wx, wy = to_world(fit, mx, my)
                for o in cands:
                    d = math.hypot(o[2] - wx, o[3] - wy)
                    if d <= C.R[cls] and (best is None or d < best[0]): best = (d, room, fit, o)
            if best is None: continue
            _, room, fit, o = best
            if not W.inside(polys[room], np.array([[o[2], o[3]]]))[0]: continue   # the object must lie in the room being visited
            ax, ay = to_map(fit, o[2], o[3])
            tx = math.cos(gth) * ax - math.sin(gth) * ay + g0
            ty = math.sin(gth) * ax + math.cos(gth) * ay + g1
            n = ((cx - tx) / sx) ** 2 + ((cy - ty) / sy) ** 2
            out.append((cls, r['node'], o[0], room, t, n / 2, math.hypot(cx - tx, cy - ty), math.hypot(sx, sy)))
    return out


if __name__ == '__main__':
    rooms, _, _ = W.rooms()
    polys = {k: v['poly'] for k, v in rooms.items()}
    args, it = [], iter(sys.argv[1:])
    for a in it:
        if a == '--gt': N.GT = next(it)
        elif a.startswith('--gt='): N.GT = a[5:]
        else: args.append(a)
    if N.GT not in ('mesh', 'boundingobject'): sys.exit('--gt must be mesh (default) or boundingobject')
    gt = N.gt_objects(W.WBT)
    print('world:', W.WBT, '| ground truth:', N.GT)
    allr = defaultdict(list)
    for run in args:
        run = os.path.abspath(run.rstrip('/'))
        rows = run_rows(run, polys, gt)
        with open(f'{run}/nees_persistent_rows.csv', 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['class', 'node', 'gt_object', 'room', 't_ms', 'nees_per_dof', 'err_m', 'sigma_m'])
            w.writerows(rows)
        fits = room_windows(run, polys)
        print(os.path.basename(run), {k: f'rms {v[2]:.3f} m, n {v[5]}' for k, v in fits.items()})
        for r in rows: allr[(os.path.basename(run), r[0], r[1], r[2], r[3])].append(r)
    print('run | class | node | gt object | room | n rows | median NEES/DOF | frac<=1 | median err m | median sigma m | last-row NEES/DOF')
    for k, v in sorted(allr.items()):
        n = np.array([x[5] for x in v])
        print(' | '.join(map(str, k)), f'| {len(v)} | {np.median(n):.2f} | {np.mean(n <= 1):.2f} | '
              f'{np.median([x[6] for x in v]):.3f} | {np.median([x[7] for x in v]):.3f} | {n[-1]:.2f}')
