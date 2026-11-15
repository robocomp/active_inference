#!/usr/bin/env python3
"""Offline check: cast simulated Helios beams from the logged robot poses against the TRUE Webots shapes,
then score the returns against the FITTED box with carve_box's own rule (surface / interior / through).
Answers: which part of the real object do the 'against' beams hit, and why.
Usage: carve_replay.py RUN_DIR"""
import csv, sys, math, collections
import numpy as np

RUN = sys.argv[1]
SIG_SENSOR = 0.03; PD = 0.85; PC = 0.05
H_ORIGIN = 1.075 + 0.0325
ELEV = np.deg2rad(np.linspace(-15.5, 54.5, 32))
AZ = np.deg2rad(np.arange(0, 360, 0.4))

# ---- ground truth scene, WORLD frame, axis-aligned boxes (xmin,xmax,ymin,ymax,zmin,zmax,label) ----
def box(cx, cy, sx, sy, z0, z1, lab): return (cx - sx/2, cx + sx/2, cy - sy/2, cy + sy/2, z0, z1, lab)
S = []
# bed (-5.18246,12.3911) rot -pi/2: local x -> world -y, local y -> world +x
bx, by = -5.18246, 12.3911
S.append(box(bx, by, 1.75, 2.1, 0.25, 0.65, 'bed_mattress'))
S.append(box(bx, by - 1.01, 1.75, 0.1, 0.0, 0.83, 'bed_footboard'))
S.append(box(bx, by + 1.01, 1.75, 0.1, 0.0, 1.25, 'bed_headboard'))
for nx in (-6.3615, -4.0035):
    ny = 13.216
    S.append(box(nx, ny, 0.504, 0.508, 0.41, 0.62, 'ns_body'))
    for dx in (-0.22, 0.22):
        for dy in (-0.22, 0.22):
            S.append(box(nx + dx, ny + dy, 0.04, 0.04, 0.0, 0.41, 'ns_leg'))
# toilet (-0.874478,12.8652) rot -pi/2: local (x,y) -> world (y, -x) offset
tx, ty = -0.874478, 12.8652
def tbox(lx, ly, sx, sy, zc, sz, lab): return box(tx + ly, ty - lx, sy, sx, zc - sz/2, zc + sz/2, lab)
S.append(tbox(0.048, 0, 0.51, 0.48, 0.35, 0.22, 'toilet_bowl'))
S.append(tbox(-0.04, 0, 0.5, 0.36, 0.12, 0.24, 'toilet_pedestal'))
S.append(tbox(-0.32, -0.01, 0.22, 0.58, 0.628, 0.78, 'toilet_tank'))
# walls (from threerooms-piso.wbt; rotation pi/2 swaps the size axes)
S.append(box(0.11232, 13.5799, 3.5, 0.2, 0, 3, 'wall'))
S.append(box(-5.02, 13.58, 7.0, 0.2, 0, 3, 'wall'))
S.append(box(-8.49, 10.32, 0.2, 6.5, 0, 3, 'wall'))
S.append(box(-1.715, 12.36, 0.2, 2.2, 0, 3, 'wall'))
S.append(box(-1.715, 9.76, 0.2, 1.02, 0, 3, 'wall'))
S.append(box(-3.005, 9.25, 2.47, 0.2, 0, 3, 'wall'))
S.append(box(1.79, 10.34, 0.2, 6.5, 0, 3, 'wall'))
B = np.array([s[:6] for s in S]); LAB = [s[6] for s in S]

def cast(o, d):
    """first hit of rays o + t d (d unit, N x 3) against boxes, floor z=0, ceiling z=3. returns t, label idx"""
    N = d.shape[0]; tbest = np.full(N, 30.0); lbest = np.full(N, -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / d
        for k in range(B.shape[0]):
            t1 = (B[k, [0, 2, 4]] - o) * inv; t2 = (B[k, [1, 3, 5]] - o) * inv
            tn = np.nanmax(np.minimum(t1, t2), axis=1); tf = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tn <= tf) & (tf > 0.09) & (tn < tbest)
            tn = np.where(tn < 0.09, tf, tn)
            tbest = np.where(hit, tn, tbest); lbest = np.where(hit, k, lbest)
        for zp, lab in ((0.0, -2), (3.0, -3)):
            t = (zp - o[2]) / d[:, 2]
            hit = (t > 0.09) & (t < tbest)
            tbest = np.where(hit, t, tbest); lbest = np.where(hit, lab, lbest)
    return tbest, lbest

def Phi(x):
    from math import erfc
    return 0.5 * np.vectorize(lambda v: erfc(-v * 0.70710678))(x)

def carve(O, P, cx, cy, yaw, w, d, H, ssurf):
    """carve_box, verbatim logic, vectorised; returns per-beam (occ, interior, through)"""
    c, s = math.cos(yaw), math.sin(yaw)
    def loc(q): dx, dy = q[..., 0] - cx, q[..., 1] - cy; return np.stack([c*dx + s*dy, -s*dx + c*dy, q[..., 2]], -1)
    Ol = loc(O[None, :])[0]; Pl = loc(P); D = Pl - Ol; L = np.linalg.norm(D, axis=1)
    hw, hd = w/2, d/2
    tn = np.zeros(len(P)); tf = np.full(len(P), 1e9); miss = np.zeros(len(P), bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        for a, h in ((0, hw), (1, hd)):
            dd = D[:, a]; small = np.abs(dd) < 1e-6
            miss |= small & ((Ol[a] < -h) | (Ol[a] > h))
            t1 = (-h - Ol[a]) / dd; t2 = (h - Ol[a]) / dd
            lo = np.minimum(t1, t2); hi = np.maximum(t1, t2)
            tn = np.where(small, tn, np.maximum(tn, lo)); tf = np.where(small, tf, np.minimum(tf, hi))
    ok = (~miss) & (tn <= tf) & (tf >= 0)
    tm = np.clip(0.5*(tn + tf), 0, 1); zm = Ol[2] + tm * D[:, 2]
    sz = SIG_SENSOR
    from scipy.special import ndtr
    zw = np.clip(ndtr((zm - 0)/sz) - ndtr((zm - H)/sz), 0, 1)
    ss = math.sqrt(SIG_SENSOR**2 + ssurf**2); st = ss / np.maximum(L, 1e-6)
    pr = ndtr((1 - tn)/st); pt = ndtr((1 - tf)/st)
    ok &= (zw >= 1e-3) & (pr*zw >= 1e-3)
    inside = (pr - pt) * zw; depth = np.maximum(0, (1 - tn) * L)
    ws = np.exp(-0.5 * depth**2 / ss**2)
    z = np.zeros(len(P))
    return np.where(ok, inside*ws, z), np.where(ok, inside*(1-ws), z), np.where(ok, pt*zw, z), zm


def carve3d(O, P, cx, cy, yaw, parts, ssurf):
    """3-D slab test against each part (dx,dy,sx,sy,z0,z1) in the box frame; same soft surface as carve_box"""
    from scipy.special import ndtr
    c, s = math.cos(yaw), math.sin(yaw)
    def loc(q): dx, dy = q[..., 0] - cx, q[..., 1] - cy; return np.stack([c*dx + s*dy, -s*dx + c*dy, q[..., 2]], -1)
    Ol = loc(O[None, :])[0]; Pl = loc(P); D = Pl - Ol; L = np.linalg.norm(D, axis=1)
    ss = math.sqrt(SIG_SENSOR**2 + ssurf**2); st = ss / np.maximum(L, 1e-6)
    occ = np.zeros(len(P)); inn = np.zeros(len(P)); thr = np.zeros(len(P))
    for (pdx, pdy, sx, sy, z0, z1) in parts:
        lo = np.array([pdx - sx/2, pdy - sy/2, z0]); hi = np.array([pdx + sx/2, pdy + sy/2, z1])
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (lo - Ol) / D; t2 = (hi - Ol) / D
        tn = np.nanmax(np.minimum(t1, t2), axis=1); tf = np.nanmin(np.maximum(t1, t2), axis=1)
        ok = (tn <= tf) & (tf >= 0); tn = np.maximum(tn, 0)
        pr = ndtr((1 - tn)/st); pt = ndtr((1 - tf)/st)
        ok &= pr >= 1e-3
        inside = pr - pt; depth = np.maximum(0, (1 - tn) * L); ws = np.exp(-0.5 * depth**2 / ss**2)
        occ += np.where(ok, inside*ws, 0); inn += np.where(ok, inside*(1-ws), 0); thr += np.where(ok, pt*(1-ws), 0)
    return occ, inn, thr

# ---- room <-> world: bedroom from the three objects (room axes = world axes); robot from gt_pose ----
gt = np.array([(int(r['t_ms']), float(r['x']), float(r['y']), float(r['yaw'])) for r in csv.DictReader(open(f'{RUN}/gt_pose.tsv'), delimiter='\t')])
GTC = {'bed_10': (-5.18246, 12.3911), 'nightstand_6': (-4.0035, 13.216), 'nightstand_7': (-6.3615, 13.216), 'toilet_6': (tx, ty)}
FIT = {'bed_10': (-0.36, 3.04), 'nightstand_6': (0.87, 3.89), 'nightstand_7': (-1.60, 3.91), 'toilet_6': (-2.32, 1.10)}
tb = np.mean([np.subtract(GTC[k], FIT[k]) for k in ('bed_10', 'nightstand_6', 'nightstand_7')], axis=0)
for k in ('bed_10', 'nightstand_6', 'nightstand_7'):
    print(k, 'fitted-box error vs GT (cm): %.1f' % (100*np.linalg.norm(np.add(FIT[k], tb) - GTC[k])))
XF = {'bedroom': (np.eye(2), tb, 0.0)}
c90 = -math.pi/2; R3 = np.array([[math.cos(c90), -math.sin(c90)], [math.sin(c90), math.cos(c90)]])
XF['bathroom'] = (R3, np.array(GTC['toilet_6']) - R3 @ np.array(FIT['toilet_6']), c90)   # toilet centred on GT
CENTRE_ON_GT = len(sys.argv) > 2
def robot_world(ts, rg):
    if ts < gt[0, 0] or ts > gt[-1, 0]: return None
    k = int(np.searchsorted(gt[:, 0], ts)); k = max(1, min(k, len(gt) - 1))
    a, b = gt[k-1], gt[k]
    if b[0] - a[0] > 2500: return None
    f = (ts - a[0]) / (b[0] - a[0]); dyaw = ((b[3] - a[3] + math.pi) % (2*math.pi)) - math.pi
    return (a[1] + f*(b[1]-a[1]), a[2] + f*(b[2]-a[2])), a[3] + f*dyaw


def world_parts_rel(labels, centre):
    out = []
    for (x0, x1, y0, y1, z0, z1), lab in zip(B[:, :6], LAB):
        if lab in labels:
            cxw, cyw = 0.5*(x0+x1), 0.5*(y0+y1)
            if abs(cxw - centre[0]) < 1.3 and abs(cyw - centre[1]) < 1.3:
                out.append((cxw - centre[0], cyw - centre[1], x1 - x0, y1 - y0, z0, z1))
    return out
COMPOSED = {'bed_10': world_parts_rel({'bed_mattress', 'bed_footboard', 'bed_headboard'}, GTC['bed_10']),
            'nightstand_6': world_parts_rel({'ns_body', 'ns_leg'}, GTC['nightstand_6']),
            'nightstand_7': world_parts_rel({'ns_body', 'ns_leg'}, GTC['nightstand_7']),
            'toilet_6': world_parts_rel({'toilet_bowl', 'toilet_pedestal', 'toilet_tank'}, GTC['toilet_6'])}

# ---- per class ----
ROOM_OF = {'bed': 'bedroom', 'nightstand': 'bedroom', 'toilet': 'bathroom'}
dirs = np.array([[math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)] for e in ELEV for a in AZ])
for cls in ('bed', 'nightstand', 'toilet'):
    rg = ROOM_OF[cls]
    if rg not in XF: print(cls, 'no transform'); continue
    R, t, dth = XF[rg]
    rows = list(csv.DictReader(open(f'{RUN}/{cls}_ai2_log.csv')))
    by = collections.defaultdict(list)
    for r in rows: by[r['node']].append(r)
    for node, rs in by.items():
        if len(rs) < 200: continue
        agg = collections.defaultdict(float); n = 0; prev = None; logged = [0, 0]; zhist = collections.Counter()
        for i, r in enumerate(rs):
            try:
                L = float(r['ex_L']); ts = int(float(r['pkt_ts']))
                cx, cy, yaw, w, h, H = (float(r[k]) for k in ('cx', 'cy', 'yaw', 'w', 'h', 'H'))
                ssurf = math.sqrt(max(0, 0.5*(float(r['std_cx'])**2 + float(r['std_cy'])**2)))
                lfe = float(r['ex_lfree_eff'])
            except (ValueError, KeyError):
                prev = None; continue
            drop = prev is not None and L < prev - 1e-4 and lfe > 0; prev = L
            if not drop or i % 2: continue      # drop cycles, every other one (cost)
            rp = robot_world(ts, rg)
            if rp is None: continue
            (px, py), th = rp
            # helios 0.155 m behind the base along the robot's facing (+y at yaw 0)
            fx, fy = -math.sin(th), math.cos(th)
            O = np.array([px - 0.155*fx, py - 0.155*fy, H_ORIGIN])
            tt, lb = cast(O, dirs); P = O + dirs * tt[:, None]
            cw = R @ np.array([cx, cy]) + t
            if CENTRE_ON_GT: cw = cw - (R @ np.array(FIT[node]) + t) + np.array(GTC[node])
            yw = yaw + dth
            for tag, (o_, i_, t_) in (('2d_cuboid', carve(O, P, cw[0], cw[1], yw, w, h, H, ssurf)[:3]),
                                      ('3d_cuboid', carve3d(O, P, cw[0], cw[1], yw, [(0, 0, w, h, 0, H)], ssurf)),
                                      ('3d_composed', carve3d(O, P, cw[0], cw[1], yw - (yw if True else 0) * 0, COMPOSED[node], ssurf) if False else
                                                      carve3d(O, P, cw[0], cw[1], 0.0, COMPOSED[node], ssurf))):
                agg[(tag, 'occ')] += o_.sum(); agg[(tag, 'interior')] += i_.sum(); agg[(tag, 'through')] += t_.sum()
            logged[0] += float(r['ex_locc']); logged[1] += float(r['ex_lfree']); n += 1
        if n == 0: continue
        print(f'\n== {cls} {node}: {n} drop cycles replayed; logged sums occ {logged[0]:.0f}, through {logged[1]:.0f}')
        pdv, pcv = 0.85, 0.05; lo_, lf_ = math.log(pdv/pcv), math.log((1-pdv)/(1-pcv))
        for tag in ('2d_cuboid', '3d_cuboid', '3d_composed'):
            o_, i_, t_ = agg[(tag, 'occ')], agg[(tag, 'interior')], agg[(tag, 'through')]
            print(f'   {tag:12s} occ {o_:6.0f}  interior {i_:6.0f}  through {t_:6.0f}   for/against {o_/max(1e-9, i_ + t_):6.2f}   raw dL/cycle {(o_*lo_ + (i_+t_)*lf_)/n:+7.1f}')
