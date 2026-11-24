#!/usr/bin/env python3
"""Per-agent NEES calibration from existing run logs (AuRo, 2026-11-22). No new runs.
   nees_agents.py [--gt mesh|boundingobject] <run_dir> [<run_dir> ...]      prints a table; writes <run>/nees_rows.csv
   --gt mesh (default): object truth = centre of the graded visual-mesh layer; boundingobject: the earlier numbers (see GT below)

NEES = e^T P^-1 e with the agent's own published sigma; NEES/DOF has median ~0.69 (chi2 median / DOF: 0.455 for 1 DOF,
0.693 for 2, 0.789 for 3) and P(NEES/DOF <= 1) = 0.68 / 0.63 / 0.61 when the sigma is calibrated (Gaussian, DOF 1/2/3).
States, ground truth and frames:
  room   robot pose (x, y, theta) in the map frame, proto_room.csv x,y,theta,sig_x,sig_y,sig_theta (room_concept's
         published pose posterior; diagonal only, no cross terms logged). Truth = Webots pose (gt_error.csv gt_*_world,
         gt_theta) taken to the map frame by ONE rigid 2-D fit per room visit (legs.tsv windows, trimmed 3 s), the same
         fit c3a_objects.py grades with; the yaw offset is one constant per run (circular median). The fit absorbs a
         constant offset per visit, so this is the pose error AROUND each visit's mean: a lower bound on the true error.
  door   width w and height h (door_concept ai2_log.csv w,h,std_w,std_h); frame-free. Truth = every Webots Door here is
         size 1.0 x 2.1 m (threerooms-piso.wbt DOOR_0..2). Rows assigned to a Webots door are not needed: all three equal.
  objects  centre (cx, cy) in the object's room frame (<cls>_ai2_log.csv cx,cy,std_cx,std_cy; diagonal only).
         Node -> Webots object = the grader's match (c3a_instances.csv verdict=match). Truth = the Webots centre taken
         world -> map (the visit fit nearest the row's time) -> room frame (the LTSM layout's room pose), i.e. the
         inverse of the grader's to_world. Unmatched / phantom nodes are skipped. Split by visit: on the RETURN to a room
         the map frame has moved (c3a: room_1 rotated ~0.1 rad, shifted ~0.5 m between first visit and return) while a
         remembered object keeps its old room-frame centre, and its sigma carries no room-frame term.
Every logged update row counts as one sample; consecutive rows are strongly autocorrelated, so n overstates the
independent evidence (the number of instances / visits is printed too).
"""
import csv, glob, json, math, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c3a_objects as C

DOOR_W, DOOR_H = 1.0, 2.1
OBJ = ['chair', 'table', 'refrigerator', 'hood', 'nightstand', 'bed', 'toilet']
COUNTABLE = {'nightstand', 'bed', 'toilet'}
FIXTURE_FRAMES = os.environ.get('NEES_FIXTURE_FRAMES') == '1'
# Ground truth of an object centre: 'mesh' = the centre of the class's graded layer read from the VISUAL mesh (tools/gt_mesh/mesh_extents.py:
# bed mattress, nightstand body, toilet footprint, chair seat, table top, fridge body, hood canopy); 'boundingobject' = the numbers of
# the earlier runs (Webots origin, plus the 11-22 fridge offset). A class with no mesh layer (Desk, WoodenTable, RoundTable) keeps the old number.
GT = 'mesh'
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'tools', 'gt_mesh'))


def gt_objects(wbt, mode=None):
    """[(name, class, x, y)] as C.gt_objects, with the centre of the graded mesh layer when mode (default GT) is 'mesh'."""
    base = C.gt_objects(wbt)
    if (mode or GT) != 'mesh': return base
    import mesh_extents as ME
    lay = ME.world_layers(wbt)
    out = []
    for nm, c, x, y in base:
        L = lay.get(nm)
        if L is None or L['cls'] != c: out.append((nm, c, x, y)); continue
        ox, oy = C.POSE_OVERRIDE.get(nm, (L['x'], L['y']))      # the run's live origin, when a script moved the object
        out.append((nm, c, *ME.to_world(ox, oy, L['yaw'], L['box'][4], L['box'][5])))
    return out


def visit_fits(run):
    legs = {int(r['leg']): (int(r['t0_ms']), int(r['t1_ms']), r['arg'])
            for r in csv.DictReader(open(f'{run}/legs.tsv'), delimiter='\t')}
    gt = [int(r['t_ms']) for r in csv.DictReader(open(f'{run}/gt_pose.tsv'), delimiter='\t')]
    wins, prev_t, prev_room = [], gt[0], 'room_1'
    for k in sorted(legs):
        t0, t1, arg = legs[k]
        wins.append((prev_room, prev_t, t0)); prev_t, prev_room = t1, arg
    wins.append((prev_room, prev_t, gt[-1]))
    est = [(int(r['ts_ms']), float(r['est_x']), float(r['est_y']), float(r['gt_x_world']), float(r['gt_y_world']),
            float(r['est_theta']), float(r['gt_theta'])) for r in csv.DictReader(open(f'{run}/gt_error.csv'))]
    fits = []
    for rm, a, b in wins:
        S = [e for e in est if a + 3000 <= e[0] <= b - 3000]
        if len(S) >= 10:
            th, t, rms = C.rigid_fit([(e[1], e[2]) for e in S], [(e[3], e[4]) for e in S])
            fits.append((rm, a, b, th, t, rms))
    return fits, est


def nearest_fit(fits, t_ms):
    return min(fits, key=lambda f: 0 if f[1] <= t_ms <= f[2] else min(abs(t_ms - f[1]), abs(t_ms - f[2])))


def in_visit(fits, t_ms):
    return next((f for f in fits if f[1] + 3000 <= t_ms <= f[2] - 3000), None)


def world_to_map(f, wx, wy):
    th, t = f[3], f[4]
    dx, dy = wx - t[0], wy - t[1]
    return math.cos(th) * dx + math.sin(th) * dy, -math.sin(th) * dx + math.cos(th) * dy


def room_poses(run):
    lay = sorted(glob.glob(f'{run}/ltsm_history/*_layout.json')) or \
        sorted(glob.glob(f'{os.path.dirname(run)}/fixtures/{C.FIXTURE}/ltsm_history/*_layout.json'))
    return {'room_' + r['name'].rsplit('_', 1)[-1]: (r['x'], r['y'], r['yaw']) for r in json.load(open(lay[-1]))['rooms']}


def gauges(run):
    """[(ts_ms, gx, gy, gtheta)] = room_concept's T_pub_int (internal -> published), proto_room.csv gauge_* columns
    (room_scene_graph.cpp:1363). est_x/est_y and proto x,y are in the INTERNAL frame; DSR nodes (objects) in the PUBLISHED one."""
    out = []
    for r in csv.DictReader(open(f'{run}/proto_room.csv')):
        try: out.append((int(r['ts_ms']), float(r['gauge_x']), float(r['gauge_y']), float(r['gauge_theta'])))
        except (KeyError, ValueError): pass
    return out


def gauge_at(G, ts, t):
    import bisect
    i = bisect.bisect_right(ts, t) - 1
    return G[max(i, 0)]


def f(r, k):
    try: return float(r[k])
    except (KeyError, ValueError): return float('nan')


def run_nees(run):
    rows = []   # (agent, state, dof, nees, unit_id)
    fits, est = visit_fits(run)
    if not fits: return rows
    # yaw offset Webots -> map, one constant per run
    off = [math.remainder(e[5] - e[6], 2 * math.pi) for e in est if in_visit(fits, e[0])]
    yaw_off = math.atan2(np.median(np.sin(off)), np.median(np.cos(off))) if off else 0.0
    # --- room: robot pose. gt_error rows carry the truth; proto_room rows carry sigma; join by time (<= 100 ms) ---
    t_run = [int(l.split('\t')[0]) for l in open(f'{run}/gt_pose.tsv') if l[:1].isdigit()]
    t_lo, t_hi = t_run[0] - 60000, t_run[-1] + 5000   # a log row outside the run's own window is a stale copy (11-22:
    # c3a_3's bed/nightstand/toilet logs are byte-identical to c3a_2's) and is dropped
    pp = f'{run}/proto_room.csv'
    pr = [] if not os.path.exists(pp) else [(int(r['ts_ms']), f(r, 'x'), f(r, 'y'), f(r, 'theta'), f(r, 'sig_x'), f(r, 'sig_y'), f(r, 'sig_theta'))
          for r in csv.DictReader(open(pp)) if r['ts_ms'].isdigit()]
    pr = [p for p in pr if all(math.isfinite(v) for v in p[1:]) and min(p[4:]) > 0]
    pts = np.array([p[0] for p in pr])
    S = [e for e in est if in_visit(fits, e[0])]
    th_all, t_all, _ = C.rigid_fit([(e[1], e[2]) for e in S], [(e[3], e[4]) for e in S]) if len(S) >= 10 else (0, (0, 0), 0)
    f_all = ('run', 0, 0, th_all, t_all, 0)
    for e in est:
        fv = in_visit(fits, e[0])
        if fv is None or not len(pts): continue
        i = int(np.argmin(np.abs(pts - e[0])))
        if abs(pts[i] - e[0]) > 100: continue
        _, x, y, th, sx, sy, st = pr[i]
        gx, gy = world_to_map(fv, e[3], e[4])
        gth = e[6] + yaw_off
        ex, ey, et = x - gx, y - gy, math.remainder(th - gth, 2 * math.pi)
        rows.append(('room', 'robot pose x,y,theta', 3, (ex / sx) ** 2 + (ey / sy) ** 2 + (et / st) ** 2, f'{fv[0]}@{fv[1]}'))
        rows.append(('room', 'robot position x,y', 2, (ex / sx) ** 2 + (ey / sy) ** 2, f'{fv[0]}@{fv[1]}',
                     math.hypot(ex, ey), math.hypot(sx, sy)))
        rows.append(('room', 'robot heading theta', 1, (et / st) ** 2, f'{fv[0]}@{fv[1]}'))
        ax, ay = world_to_map(f_all, e[3], e[4])   # one fit for the whole run: keeps each visit's offset (upper bound)
        ex2, ey2 = x - ax, y - ay
        rows.append(('room', 'robot position x,y, one fit per run', 2, (ex2 / sx) ** 2 + (ey2 / sy) ** 2, f'{fv[0]}@{fv[1]}',
                     math.hypot(ex2, ey2), math.hypot(sx, sy)))
    # --- door: width, height ---
    p = f'{run}/ai2_log.csv'
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            w, h, sw, sh = f(r, 'w'), f(r, 'h'), f(r, 'std_w'), f(r, 'std_h')
            if not (sw > 0 and sh > 0 and math.isfinite(w) and math.isfinite(h)): continue
            rows.append(('door', 'width w, height h', 2, ((w - DOOR_W) / sw) ** 2 + ((h - DOOR_H) / sh) ** 2, r['node']))
            rows.append(('door', 'width w', 1, ((w - DOOR_W) / sw) ** 2, r['node']))
            rows.append(('door', 'height h', 1, ((h - DOOR_H) / sh) ** 2, r['node']))
    # --- objects: centre in the room frame ---
    ip = f'{run}/c3a_instances.csv'
    if not os.path.exists(ip): return rows
    rp = room_poses(run)
    G = gauges(run); Gts = [g[0] for g in G]
    gtobj = {o[0]: o for o in gt_objects(C.W.WBT)}
    match = {r['name']: (r['room'], gtobj[r['gt_object']]) for r in csv.DictReader(open(ip))
             if r['verdict'] == 'match' and r['gt_object'] in gtobj}
    for cls in OBJ:
        p = f'{run}/{cls}_ai2_log.csv'
        if not os.path.exists(p): continue
        for r in csv.DictReader(open(p)):
            if r['node'] not in match: continue
            room, (_, _, ox, oy) = match[r['node']]
            t = int(f(r, 't_ms') if 't_ms' in r else f(r, 'pkt_ts'))
            if not t_lo <= t <= t_hi: continue
            cx, cy, sx, sy = f(r, 'cx'), f(r, 'cy'), f(r, 'std_cx'), f(r, 'std_cy')
            if not (sx > 0 and sy > 0 and math.isfinite(cx) and (room in rp or not FIXTURE_FRAMES)): continue
            fv = nearest_fit(fits, t)
            nth = sum(1 for g in fits if g[0] == fv[0] and g[1] < fv[1])   # earlier visits to the same room
            mx, my = world_to_map(fv, ox, oy)               # world -> room_concept's internal frame (this visit's fit)
            if FIXTURE_FRAMES:                                # the 11-22 first pass: fixture layout room pose (WRONG, kept for comparison)
                rx, ry, ryaw = rp[room]
                dx, dy = mx - rx, my - ry
                gx, gy = math.cos(ryaw) * dx + math.sin(ryaw) * dy, -math.sin(ryaw) * dx + math.cos(ryaw) * dy
            else:                                             # internal -> published by the logged gauge at the row's time
                if fv[0] != room: continue                    # object row outside a visit to its own room
                _, g0, g1, gth = gauge_at(G, Gts, t)
                gx = math.cos(gth) * mx - math.sin(gth) * my + g0
                gy = math.sin(gth) * mx + math.cos(gth) * my + g1
            rows.append((cls, 'centre cx,cy (room frame), ' + ('first visit' if nth == 0 else 'return visit'), 2, ((cx - gx) / sx) ** 2 + ((cy - gy) / sy) ** 2, r['node'],
                         math.hypot(cx - gx, cy - gy), math.hypot(sx, sy)))
    return rows


if __name__ == '__main__':
    args = []
    it = iter(sys.argv[1:])
    for a in it:
        if a == '--gt': GT = next(it)
        elif a.startswith('--gt='): GT = a[5:]
        else: args.append(a)
    if GT not in ('mesh', 'boundingobject'): sys.exit('--gt must be mesh or boundingobject')
    print(f'ground truth: {GT}')
    allrows = defaultdict(list); units = defaultdict(set); runs_of = defaultdict(set); extra = defaultdict(list)
    for run in args:
        run = os.path.abspath(run.rstrip('/'))
        rows = run_nees(run)
        with open(f'{run}/nees_rows.csv', 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['agent', 'state', 'dof', 'nees', 'unit', 'err_m', 'sigma_m'])
            for r in rows: w.writerow(list(r) + [''] * (7 - len(r)))
        for r in rows:
            k = (r[0], r[1], r[2]); allrows[k].append(r[3] / r[2]); units[k].add((run, r[4])); runs_of[k].add(os.path.basename(run))
            if len(r) > 5: extra[k].append((r[5], r[6]))
    CAL = {1: (0.455, 0.68), 2: (0.693, 0.63), 3: (0.789, 0.61)}
    print('agent | state | DOF | n rows | instances/visits | runs | median NEES/DOF (calibrated) | frac NEES/DOF<=1 (calibrated) | median |e| m | median sigma m')
    for k in sorted(allrows, key=lambda k: (OBJ.index(k[0]) + 2 if k[0] in OBJ else ['room', 'door'].index(k[0]), k[1])):
        v = np.array(allrows[k]); cm, cf = CAL[k[2]]
        e = extra.get(k)
        em = f'{np.median([a for a, b in e]):.3f} | {np.median([b for a, b in e]):.3f}' if e else '- | -'
        print(f'{k[0]} | {k[1]} | {k[2]} | {len(v)} | {len(units[k])} | {len(runs_of[k])} | {np.median(v):.2f} ({cm}) | '
              f'{np.mean(v <= 1):.2f} ({cf}) | {em}')
