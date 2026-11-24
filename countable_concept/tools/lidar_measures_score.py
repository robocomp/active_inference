#!/usr/bin/env python3
"""Score a countable_concept live run against the Webots ground truth: spec 2026-11-15-lidar-measures-design.md §8,
plan Task 10 Step 2.

usage: lidar_measures_score.py RUN_DIR [OFF_DIR] [--ref LIVE6_DIR] [--shift]

Per run dir: <cls>_ai2_log.csv for cls in nightstand, bed, toilet (one row per instance per fit cycle) and gt_pose.tsv.
Missing files / missing columns (flag-off era runs have no lidar_only ... Itr_L) are tolerated: those lines print n/a.

Conventions (printed again in the header of every report):
  * look      = maximal run of FRESH rows (gated == 0 and lidar_only != 1) of one instance with no pkt_ts gap > 2 s.
  * settled   = rows from the first fresh row with range <= 2.5 m onward (all rows, fresh or LiDAR-only).
  * Sigma     = PASS iff max std <= 0.15 m over the settled rows AND at least 95 % of the settled rows have
                rms/3 <= std <= 3*rms (rms = rms of the true error of that DOF over the settled rows).
  * NEES      = (ex/std_cx)^2 + (ey/std_cy)^2 + (eH/std_H)^2 per settled row (diagonal Sigma; expected median ~ 2.37 for 3 DOF).
  * drift     = per instance, the last fitted centre of each look; the step between consecutive looks projected on the
                direction to the nearest wall and to the nearest other GT object; PASS iff the largest such step <= 0.10 m.
  * rate      = rows / (last - first pkt_ts) per instance; PASS iff >= 8 Hz.
  * parked    = the gt_pose interval containing the row has speed < 0.02 m/s and yaw rate < 0.02 rad/s.
  * updated   = rows whose e / spose_* / c_* / gn_* belong to THIS cycle: Itr_cam > 0 or Itr_L > 0. Those columns hold
                their last value on every other row (final review Task 13 M8); Itr_* is reset to 0 every cycle and set only
                after an ACCEPTED flag-on update, which is also the only cycle that staged spose_* / c_* AND ran the GN. Rows
                that staged but whose update was rejected (step bound) or had no usable rays are therefore not counted.
  * live10 (spec 7c.7 Pass "Live", plan "Task 10 again"; n/a on logs before e_* / spose_* / gn_*):
      |e| <= 3 sigma   = per component |e_x|, |e_y|, |e_psi| <= 3 spose_* on every updated row (PASS iff all);
                         the same against the posterior's own se_* is printed [info].
      Sigma_pose       = parked updated rows vs proto_room.csv's pose sigma at the nearest ts_ms (<= 100 ms):
                         sqrt(spose_x^2 + spose_y^2) vs sqrt(sig_x^2 + sig_y^2) (rotation-invariant: proto_room is in the
                         room's internal frame, the edge in the published one; their gauge rotation mixes x and y) and
                         spose_psi vs sig_theta; PASS iff >= 95 % of rows are within 1 % on both.
      gn               = over updated rows, the gn_it distribution and the share of gn_conv = 0, split camera-only
                         (Itr_L == 0) vs with rays (Itr_L > 0) [info; with rays gn_conv = 0 is usually theta on a ray edge,
                         ruling T14-7; on camera-only rows it should be ~0, final review M6].
  * beam (spec 7c.8 Pass "Live11", per instance; n/a on logs before beam_staged ... gap_rays) [info]:
      staged           = beam_staged over rows that staged rays (median / max): rays staged by ray geometry;
      nolabel          = beam_nolabel summed and its median over those rows: rays left out for lack of camera information;
      nomem            = rows with nomem_cycle == 1 (a sweep but no membership: rays skipped) and the longest run of
                         consecutive such rows; nosweep = rows with nomem_cycle == 2 (no fresh Helios sweep), split gated /
                         admissible (settles Task 15's question: do GATED cycles lack membership or lack a sweep);
      gap rays         = gap_rays (pi > 0.99, e < -5c after the accepted update, stated limit 1): summed, max, rows > 0;
      qsum             = beam_qsum (Sum q over the staged rays; rev 3.3.11: measured <=> staged and qsum > 0): median over
                         staged rows and the count of staged rows with qsum 0; n/a on logs before Task 22 (live10/11).
                         beam_kappa is the old sweep weight, an instrument since rev 3.3.11 (it multiplies nothing).
Numbers are read with float() on the CSV text (written with '.'); output uses explicit format specs; no locale module.
"""
import argparse, bisect, csv, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, 'scene_threerooms.csv')
CLASSES = ('bed', 'nightstand', 'toilet')
BEAM_COLS = ('beam_staged', 'beam_nolabel', 'beam_kappa', 'nomem_cycle', 'gap_rays')
# Appended by Task 22 (spec 7c.8 rev 3.3.11): Sum q over the staged rays. Absent from live10/live11 logs (printed n/a there).
BEAM_QSUM = 'beam_qsum'
NEW_COLS = ('lidar_only', 'lidar_msum', 'rho_cam', 'rho_L', 'rho_chain', 'dphi_cam', 'dz_cam',
            'bcam_H', 'bL_H', 'sbcam_H', 'sbL_H', 'ray_n', 'Itr_cam', 'Itr_L')
# The localisation latent on (x, y): e_x, e_y (spec 7c.7, the robot's pose error) or, in logs before it, bchain_x, bchain_y.
CHAIN_COLS = (('e_x', 'e_y'), ('bchain_x', 'bchain_y'))
GAP_MS = 2000
NEAR_M = 2.5
STD_MAX = 0.15
FRAC_OK = 0.95
DRIFT_MAX = 0.10
RATE_MIN = 8.0
E_NSIG = 3.0
SPOSE_REL = 0.01
PROTO_MATCH_MS = 100
# Room frame -> world: the room frame's origin is the door the robot entered through (DOOR_1 for the bedroom, DOOR_2 for
# the bathroom, threerooms-piso.wbt, see room_concept/estimation_scripts/scripts/true_room.py), rotated by the room's yaw.
# Cross-check on objects_live6: bed / nightstands land 6 / 11 / 7 cm from GT with a rotation of 0 (carve-replay result).
# ROOMS is only the NOMINAL frame (it identifies which room a fitted frame belongs to). By default each run's room frames are
# FITTED (fit_room_frames, below): the published frame is not the door frame (wall-offset check 2026-11-17: bedroom is
# (-4.844, 9.321) rot +0.0126, not DOOR_1 -- the nominal origin put every object 7.5 / 5.8 cm and 0.7 deg off).
# Override with --room NAME=x,y,rot (NAME in bedroom, bathroom); --pin fits the translation to the class's own GT
# (then the centre error of a single-object room is 0 by construction: reported as such).
ROOMS = {'bedroom': (-4.769, 9.3796, 0.0), 'bathroom': (-1.705, 10.77, -math.pi / 2)}
ROOM_OF = {'bed': 'bedroom', 'nightstand': 'bedroom', 'toilet': 'bathroom'}
# True top height per class as the box fit sees it: bed = mattress top, nightstand = body top, toilet = highest part (tank).
TOP = {'bed': 0.65, 'nightstand': 0.62, 'toilet': 0.90}   # toilet 0.90 as in the live6 results doc (tank top is 1.018)
# Webots node centre of the toilet (the live6 comparison used it; the bbox centre of its parts is (-0.884, 12.929)).
NODE_CENTRE = {'toilet': (-0.874478, 12.8652)}
SHIFT_TOL = {'w': 0.03, 'h': 0.03, 'H': 0.05}   # nightstand prior shifted +0.06 / +0.10: closer to the truth than to the shifted mean


FIT_MIN_SAMPLES = 20
FIT_MAX_RMS = 0.05
FIT_EDGE_MS = 2000      # samples this close to the start/end of a gauge segment (door crossings) are dropped
FIT_MATCH_MS = 100


def fit_room_frames(run):
    """Per-run room -> world rigid transform, by least squares of the robot's GT world position (gt_error.csv
    gt_x_world/gt_y_world) against its estimated pose mapped into the PUBLISHED room frame (proto_room.csv x,y,theta through
    the proto gauge T_pub_int = gauge_x/y/theta). Samples = rows where the gauge is valid (the graph says the robot is in
    that room's frame: fs_free_cells > 0), minus FIT_EDGE_MS at both ends of each gauge segment (door crossings).
    A fitted frame belongs to the room whose nominal origin (ROOMS) is nearest. Returns {room: dict(origin, rot, rms, n)
    or dict(error=...)} for every room in ROOMS."""
    out = {k: dict(error='no usable samples (no gauge segment near the nominal origin)') for k in ROOMS}
    try:
        gt = [(fnum(r['ts_ms']), fnum(r['gt_x_world']), fnum(r['gt_y_world'])) for r in csv.DictReader(open(os.path.join(run, 'gt_error.csv')))]
        pr = list(csv.DictReader(open(os.path.join(run, 'proto_room.csv'))))
    except (OSError, KeyError) as e:
        return {k: dict(error='cannot read gt_error.csv / proto_room.csv (%s)' % e) for k in ROOMS}
    gt = [g for g in gt if g[1] == g[1] and g[2] == g[2]]
    gts = [g[0] for g in gt]
    segs = []     # consecutive rows sharing one gauge triple
    for r in pr:
        G = tuple(fnum(r.get(k)) for k in ('gauge_x', 'gauge_y', 'gauge_theta'))
        t, x, y = fnum(r.get('ts_ms')), fnum(r.get('x')), fnum(r.get('y'))
        if any(v != v for v in G + (t, x, y)) or not fnum(r.get('fs_free_cells')) > 0: continue
        i = bisect.bisect_left(gts, t)
        cand = [j for j in (i - 1, i) if 0 <= j < len(gt)]
        if not cand: continue
        j = min(cand, key=lambda j: abs(gts[j] - t))
        if abs(gts[j] - t) > FIT_MATCH_MS: continue
        c, s = math.cos(G[2]), math.sin(G[2])
        row = (c * x - s * y + G[0], s * x + c * y + G[1], gt[j][1], gt[j][2], t)
        if segs and segs[-1][0] == G: segs[-1][1].append(row)
        else: segs.append((G, [row]))
    best = {}
    for G, v in segs:
        v = [a for a in v if v[0][4] + FIT_EDGE_MS <= a[4] <= v[-1][4] - FIT_EDGE_MS]
        n = len(v)
        if n < 3: continue
        mx, my = sum(a[0] for a in v) / n, sum(a[1] for a in v) / n
        gx, gy = sum(a[2] for a in v) / n, sum(a[3] for a in v) / n
        sxx = sum((a[0] - mx) * (a[2] - gx) + (a[1] - my) * (a[3] - gy) for a in v)
        sxy = sum((a[0] - mx) * (a[3] - gy) - (a[1] - my) * (a[2] - gx) for a in v)
        th = math.atan2(sxy, sxx)
        c, s = math.cos(th), math.sin(th)
        ox, oy = gx - (c * mx - s * my), gy - (s * mx + c * my)
        rr = rms([math.hypot(c * a[0] - s * a[1] + ox - a[2], s * a[0] + c * a[1] + oy - a[3]) for a in v])
        room = min(ROOMS, key=lambda k: math.hypot(ox - ROOMS[k][0], oy - ROOMS[k][1]))
        if math.hypot(ox - ROOMS[room][0], oy - ROOMS[room][1]) > 1.0: continue
        if room not in best or n > best[room]['n']:
            best[room] = dict(origin=(ox, oy), rot=th, rms=rr, n=n)
    for room, f in best.items():
        if f['n'] < FIT_MIN_SAMPLES: f['error'] = 'only %d samples (< %d)' % (f['n'], FIT_MIN_SAMPLES)
        elif not f['rms'] <= FIT_MAX_RMS: f['error'] = 'rms %.4f m > %.2f m' % (f['rms'], FIT_MAX_RMS)
        out[room] = f
    return out


def fnum(s):
    try: return float(s)
    except (TypeError, ValueError): return float('nan')


def rms(v): return math.sqrt(sum(x * x for x in v) / len(v)) if v else float('nan')
def med(v):
    v = [x for x in v if x == x]
    return statistics.median(v) if v else float('nan')
def fmt(x, spec='.3f'): return 'n/a' if x != x else format(x, spec)
def verdict(ok): return 'PASS' if ok else 'FAIL'


# ---- ground truth ----------------------------------------------------------------------------------------------------

def load_scene(path):
    rows = list(csv.DictReader(open(path)))
    box = lambda r: tuple(fnum(r[k]) for k in ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'))
    parts = [(r['label'], box(r)) for r in rows]
    walls = [b for l, b in parts if l == 'wall']
    objs = []

    def union(bs): return (min(b[0] for b in bs), max(b[1] for b in bs), min(b[2] for b in bs), max(b[3] for b in bs),
                           min(b[4] for b in bs), max(b[5] for b in bs))
    bed = [b for l, b in parts if l.startswith('bed_')]
    if bed: objs.append(('bed', 'bed', union(bed)))
    nbodies = [b for l, b in parts if l == 'ns_body']
    legs = [b for l, b in parts if l == 'ns_leg']
    for i, nb in enumerate(sorted(nbodies, key=lambda b: b[0])):
        mine = [nb] + [g for g in legs if nb[0] - 0.05 <= g[0] and g[1] <= nb[1] + 0.05 and nb[2] - 0.05 <= g[2] <= nb[3] + 0.05]
        objs.append(('nightstand', 'nightstand_%d' % i, union(mine)))
    toi = [b for l, b in parts if l.startswith('toilet_')]
    if toi: objs.append(('toilet', 'toilet', union(toi)))
    gt = []
    for cls, name, u in objs:
        cx, cy = NODE_CENTRE.get(cls, (0.5 * (u[0] + u[1]), 0.5 * (u[2] + u[3])))
        gt.append(dict(cls=cls, name=name, cx=cx, cy=cy, dx=u[1] - u[0], dy=u[3] - u[2],
                       top=TOP[cls]))
    return gt, walls


def mesh_gt(gtobjs, wbt):
    """The scene's bed / nightstands / toilet re-read from the VISUAL meshes of the world (tools/gt_mesh/mesh_extents.py): the bed's
    mattress, the nightstand body, the footprint of the toilet's bowl + tank. cx, cy = the graded box's centre in the world, dx, dy =
    its axis-aligned world extents. `top` keeps its value: height is not a graded quantity. Objects the world does not have are kept."""
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'gt_mesh'))
    import mesh_extents as ME
    lay = ME.world_layers(wbt)
    by = {'bed': [], 'nightstand': [], 'toilet': []}
    for L in lay.values():
        if L['cls'] in by: by[L['cls']].append(L)
    for c in by: by[c].sort(key=lambda L: L['cx'])           # nightstand_%d of load_scene is numbered by x
    idx = {'bed': 0, 'nightstand': 0, 'toilet': 0}
    out = []
    for g in gtobjs:
        k = idx[g['cls']]; idx[g['cls']] += 1
        if k >= len(by[g['cls']]): out.append(g); continue
        L = by[g['cls']][k]; c, sn = abs(math.cos(L['yaw'])), abs(math.sin(L['yaw'])); lx, ly = L['box'][0], L['box'][1]
        out.append(dict(g, cx=L['cx'], cy=L['cy'], dx=c * lx + sn * ly, dy=sn * lx + c * ly))
    return out


def room_tf(room, rooms):
    ox, oy, rot = rooms[room]
    c, s = math.cos(rot), math.sin(rot)
    to_world = lambda x, y: (ox + c * x - s * y, oy + s * x + c * y)
    to_room = lambda X, Y: (c * (X - ox) + s * (Y - oy), -s * (X - ox) + c * (Y - oy))
    return to_world, to_room, rot


def true_dims(g, yaw_world):
    """true (w, h) of the GT footprint along a box whose yaw (world) is yaw_world: w along the box's x axis."""
    swap = abs(math.sin(yaw_world)) > math.sqrt(0.5)
    return (g['dy'], g['dx']) if swap else (g['dx'], g['dy'])


# ---- run loading -----------------------------------------------------------------------------------------------------

def load_gt_pose(path):
    out = []
    try:
        for r in csv.DictReader(open(path), delimiter='\t'):
            out.append((fnum(r['t_ms']), fnum(r['x']), fnum(r['y']), fnum(r['yaw'])))
    except (OSError, KeyError):
        return []
    return out


def parked_at(gt, t):
    for k in range(1, len(gt)):
        if gt[k - 1][0] <= t <= gt[k][0]:
            dt = (gt[k][0] - gt[k - 1][0]) / 1000.0
            if dt <= 0 or dt > 3.0: return None
            v = math.hypot(gt[k][1] - gt[k - 1][1], gt[k][2] - gt[k - 1][2]) / dt
            dy = ((gt[k][3] - gt[k - 1][3] + math.pi) % (2 * math.pi)) - math.pi
            return v < 0.02 and abs(dy) / dt < 0.02
    return None


def load_instances(run, cls):
    """-> (dict node -> list of row dicts with numeric fields, header columns) or ({}, [])"""
    path = os.path.join(run, cls + '_ai2_log.csv')
    if not os.path.exists(path): return {}, []
    inst, cols = {}, []
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        for r in rd:
            if r.get('node') in (None, 'node'): continue
            d = {k: fnum(v) for k, v in r.items() if k != 'node'}
            d['node'] = r['node']
            inst.setdefault(r['node'], []).append(d)
    return inst, cols


def load_proto_sigma(run):
    """proto_room.csv -> sorted list of (ts_ms, sig_x, sig_y, sig_theta), finite rows only ([] if absent)."""
    out = []
    try:
        for r in csv.DictReader(open(os.path.join(run, 'proto_room.csv'))):
            v = (fnum(r.get('ts_ms')), fnum(r.get('sig_x')), fnum(r.get('sig_y')), fnum(r.get('sig_theta')))
            if all(x == x for x in v): out.append(v)
    except OSError:
        return []
    out.sort()
    return out


def nearest(seq, keys, t, tol):
    i = bisect.bisect_left(keys, t)
    cand = [j for j in (i - 1, i) if 0 <= j < len(seq)]
    if not cand: return None
    j = min(cand, key=lambda j: abs(keys[j] - t))
    return seq[j] if abs(keys[j] - t) <= tol else None


def pctl(v, q):
    v = sorted(x for x in v if x == x)
    if not v: return float('nan')
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def is_updated(r): return r.get('Itr_cam', 0) > 0 or r.get('Itr_L', 0) > 0


def live10_checks(rows, gtpose, proto, proto_ts):
    """The live10 additions on one instance's rows (see the module docstring). {} when the log predates them."""
    m = {}
    if not rows or not all(k in rows[0] for k in ('e_x', 'spose_x', 'gn_it', 'Itr_cam')): return m
    U = [r for r in rows if is_updated(r)]
    m['n_updated'] = len(U)
    # |e| <= 3 sigma of Sigma_pose (PASS) and of e's own posterior sigma (info)
    worst_p, worst_e, bad = 0.0, 0.0, 0
    for r in U:
        rp = [abs(r['e_' + k]) / r['spose_' + k] if r['spose_' + k] > 0 else (0.0 if r['e_' + k] == 0 else float('inf'))
              for k in ('x', 'y', 'psi')]
        re_ = [abs(r['e_' + k]) / r['se_' + k] for k in ('x', 'y', 'psi') if r['se_' + k] > 0]
        worst_p = max(worst_p, max(rp))
        if re_: worst_e = max(worst_e, max(re_))
        bad += any(x > E_NSIG for x in rp)
    m['e_worst_pose'] = worst_p if U else float('nan')
    m['e_worst_post'] = worst_e if U else float('nan')
    m['e_bad'] = bad
    m['e_ok'] = bool(U) and bad == 0
    # parked Sigma_pose vs proto_room's pose sigma at the same stamp
    rel_xy, rel_psi = [], []
    if proto:
        for r in U:
            if not (gtpose and parked_at(gtpose, r['pkt_ts']) is True): continue
            q = nearest(proto, proto_ts, r['pkt_ts'], PROTO_MATCH_MS)
            if q is None: continue
            a, b = math.hypot(r['spose_x'], r['spose_y']), math.hypot(q[1], q[2])
            if b > 0: rel_xy.append(abs(a - b) / b)
            if q[3] > 0: rel_psi.append(abs(r['spose_psi'] - q[3]) / q[3])
    m['sp_n'] = min(len(rel_xy), len(rel_psi))
    m['sp_xy_med'], m['sp_xy_p95'] = med(rel_xy), pctl(rel_xy, 0.95)
    m['sp_psi_med'], m['sp_psi_p95'] = med(rel_psi), pctl(rel_psi, 0.95)
    both = [x <= SPOSE_REL and y <= SPOSE_REL for x, y in zip(rel_xy, rel_psi)]
    m['sp_frac'] = sum(both) / len(both) if both else float('nan')
    m['sp_ok'] = bool(both) and m['sp_frac'] >= FRAC_OK
    # GN: iterations and the share of unconverged updates, camera-only vs with rays
    for tag, sel in (('cam', lambda r: r.get('Itr_L', 0) == 0), ('ray', lambda r: r.get('Itr_L', 0) > 0)):
        V = [r for r in U if sel(r)]
        it = [r['gn_it'] for r in V]
        m['gn_%s_n' % tag] = len(V)
        m['gn_%s_q' % tag] = tuple(pctl(it, q) for q in (0.0, 0.25, 0.5, 0.75, 1.0))
        m['gn_%s_unconv' % tag] = sum(1 for r in V if r['gn_conv'] == 0) / len(V) if V else float('nan')
    return m


def beam_checks(rows):
    """The beam-staging columns of one instance's rows (see the module docstring). {} when the log predates them."""
    if not rows or not all(k in rows[0] for k in BEAM_COLS): return {}
    m = {}
    S = [r for r in rows if r['beam_staged'] > 0]
    m['n_staged_rows'] = len(S)
    m['staged_med'] = med([r['beam_staged'] for r in S])
    m['staged_max'] = max((r['beam_staged'] for r in S), default=float('nan'))
    m['nolabel_sum'] = sum(r['beam_nolabel'] for r in rows)
    m['nolabel_med'] = med([r['beam_nolabel'] for r in S])
    m['kappa_med'] = med([r['beam_kappa'] for r in S])
    m['qsum_med'] = med([r[BEAM_QSUM] for r in S]) if rows and BEAM_QSUM in rows[0] else None
    m['qsum_zero'] = sum(1 for r in S if r[BEAM_QSUM] == 0) if m['qsum_med'] is not None else None
    run, best = 0, 0
    for r in rows:
        run = run + 1 if r['nomem_cycle'] == 1 else 0
        best = max(best, run)
    m['nomem'] = sum(1 for r in rows if r['nomem_cycle'] == 1)
    m['nomem_run'] = best
    m['nomem_gated'] = sum(1 for r in rows if r['nomem_cycle'] == 1 and r.get('gated', 0) == 1)
    m['nosweep'] = sum(1 for r in rows if r['nomem_cycle'] == 2)
    m['nosweep_gated'] = sum(1 for r in rows if r['nomem_cycle'] == 2 and r.get('gated', 0) == 1)
    m['gap_sum'] = sum(r['gap_rays'] for r in rows)
    m['gap_max'] = max((r['gap_rays'] for r in rows), default=0)
    m['gap_rows'] = sum(1 for r in rows if r['gap_rays'] > 0)
    return m


def is_fresh(r): return r.get('gated', 0) == 0 and not (r.get('lidar_only', 0) == 1)


def looks_of(rows):
    """index ranges (a, b) of maximal fresh runs with no pkt_ts gap > 2 s"""
    out, a, prev = [], None, None
    for i, r in enumerate(rows):
        if is_fresh(r):
            if a is not None and r['pkt_ts'] - prev > GAP_MS:
                out.append((a, i - 1)); a = None
            if a is None: a = i
            prev = r['pkt_ts']
        elif a is not None and r['pkt_ts'] - prev > GAP_MS:
            out.append((a, i - 1)); a = None
        # a non-fresh row inside the gap budget does not break the look (it carries no mask, not a loss of view)
    if a is not None:
        last = max(i for i, r in enumerate(rows) if is_fresh(r))
        out.append((a, last))
    return out


# ---- scoring ---------------------------------------------------------------------------------------------------------

def score_run(run, gtobjs, walls, overrides, pin, shift):
    gtpose = load_gt_pose(os.path.join(run, 'gt_pose.tsv'))
    proto = load_proto_sigma(run)
    proto_ts = [q[0] for q in proto]
    res, notes = {}, []
    fits = fit_room_frames(run)
    rooms, failed = {}, {}
    for k in ROOMS:
        if k in overrides:
            rooms[k] = overrides[k]; notes.append('room %s: manual override origin (%.3f, %.3f) rot %.4f' % ((k,) + overrides[k]))
        elif 'error' in fits[k] and 'origin' not in fits[k]:
            failed[k] = fits[k]['error']
        elif 'error' in fits[k]:
            failed[k] = fits[k]['error']
            notes.append('room %s: fitted origin (%.3f, %.3f) rot %.4f rms %.4f m n %d' % (k, fits[k]['origin'][0], fits[k]['origin'][1], fits[k]['rot'], fits[k]['rms'], fits[k]['n']))
        else:
            f = fits[k]
            rooms[k] = (f['origin'][0], f['origin'][1], f['rot'])
            notes.append('room %s: fitted origin (%.3f, %.3f) rot %+.4f rad rms %.4f m n %d  (nominal (%.3f, %.3f) rot %.4f)' % (
                k, f['origin'][0], f['origin'][1], f['rot'], f['rms'], f['n'], *ROOMS[k]))
    for k, why in failed.items():
        notes.append('FAIL room %s: frame fit rejected: %s; its classes are NOT scored (use --room %s=x,y,rot to override)' % (k, why, k))
    data = {c: load_instances(run, c) for c in CLASSES}
    if pin:   # translation from the class's own GT: the mean over the room's final fits
        for room in rooms:
            diffs = []
            for c in CLASSES:
                if ROOM_OF[c] != room: continue
                for node, rows in data[c][0].items():
                    if not rows: continue
                    ox, oy, rot = rooms[room]
                    cs, sn = math.cos(rot), math.sin(rot)
                    x, y = rows[-1]['cx'], rows[-1]['cy']
                    wx, wy = cs * x - sn * y, sn * x + cs * y
                    cands = [g for g in gtobjs if g['cls'] == c]
                    if not cands: continue
                    g = min(cands, key=lambda g: math.hypot(g['cx'] - (ox + wx), g['cy'] - (oy + wy)))
                    diffs.append((g['cx'] - wx, g['cy'] - wy))
            if diffs:
                rooms[room] = (statistics.mean(d[0] for d in diffs), statistics.mean(d[1] for d in diffs), rooms[room][2])
                notes.append('pinned %s origin to (%.3f, %.3f)' % (room, rooms[room][0], rooms[room][1]))
    for c in CLASSES:
        inst, cols = data[c]
        if ROOM_OF[c] in failed: continue
        if not inst:
            notes.append('%s: no %s_ai2_log.csv' % (c, c)); continue
        missing = [k for k in NEW_COLS if k not in cols]
        if not any(all(k in cols for k in pair) for pair in CHAIN_COLS): missing += list(CHAIN_COLS[0])
        if missing: notes.append('%s: columns absent (flag-off era): %s' % (c, ','.join(missing[:3]) + ('...' if len(missing) > 3 else '')))
        to_world, to_room, rot = room_tf(ROOM_OF[c], rooms)
        cands = [g for g in gtobjs if g['cls'] == c]
        for node, rows in sorted(inst.items()):
            if not rows: continue
            last = rows[-1]
            wx, wy = to_world(last['cx'], last['cy'])
            g = min(cands, key=lambda g: math.hypot(g['cx'] - wx, g['cy'] - wy))
            d_final = math.hypot(g['cx'] - wx, g['cy'] - wy)
            m = dict(node=node, cls=c, gt=g['name'], gtd=g, final_err=d_final, rows=len(rows))
            m['phantom'] = d_final > 0.75
            gx, gy = to_room(g['cx'], g['cy'])                 # GT centre in the room frame (axes of the fit)
            m['H_err_final'] = abs(last['H'] - g['top'])
            m['H_final'] = last['H']
            # looks and settled window
            looks = looks_of(rows)
            m['n_looks'] = len(looks)
            start = next((i for i, r in enumerate(rows) if is_fresh(r) and r.get('range', 99) <= NEAR_M), None)
            m['settled'] = 0
            # Over-confidence over EVERY row with a posterior (not only the settled window, so an instance never seen within
            # NEAR_M is still checked): the share of rows whose error exceeds 3 posterior std, per DOF. A calibrated Gaussian
            # gives 0.27 %; rows are autocorrelated, so this is reported, not bounded.
            twa = lambda r: true_dims(g, r.get('yaw', 0.0) + rot)
            allr = [r for r in rows if r.get('std_cx', 0) > 0]
            m['z3'] = {}
            for k, ef, sk in (('cx', lambda r: r['cx'] - gx, 'std_cx'), ('cy', lambda r: r['cy'] - gy, 'std_cy'),
                              ('H', lambda r: r['H'] - g['top'], 'std_H'), ('w', lambda r: r['w'] - twa(r)[0], 'std_w'),
                              ('h', lambda r: r['h'] - twa(r)[1], 'std_h')):
                rs = [r for r in allr if r.get(sk, 0) > 0]
                m['z3'][k] = (sum(1 for r in rs if abs(ef(r)) > 3 * r[sk]) / len(rs)) if rs else float('nan')
            if start is not None:
                S = rows[start:]
                m['settled'] = len(S)
                tw = lambda r: true_dims(g, r.get('yaw', 0.0) + rot)
                errs = {'cx': [r['cx'] - gx for r in S], 'cy': [r['cy'] - gy for r in S], 'H': [r['H'] - g['top'] for r in S],
                        'w': [r['w'] - tw(r)[0] for r in S], 'h': [r['h'] - tw(r)[1] for r in S]}
                stds = {'cx': [r['std_cx'] for r in S], 'cy': [r['std_cy'] for r in S], 'H': [r['std_H'] for r in S],
                        'w': [r['std_w'] for r in S], 'h': [r['std_h'] for r in S]}
                dof = {}
                for k in errs:
                    e, s = errs[k], stds[k]
                    rr = rms(e)
                    inside = sum(1 for x in s if rr / 3 <= x <= 3 * rr) / len(s)
                    below = sum(1 for x in s if x < rr / 3) / len(s)
                    ok = max(s) <= STD_MAX and inside >= FRAC_OK
                    dof[k] = dict(rms=rr, smin=min(s), smax=max(s), inside=inside, below=below, ok=ok)
                m['dof'] = dof
                nees = []
                for i, r in enumerate(S):
                    if r['std_cx'] > 0 and r['std_cy'] > 0 and r['std_H'] > 0:
                        nees.append((errs['cx'][i] / r['std_cx']) ** 2 + (errs['cy'][i] / r['std_cy']) ** 2 +
                                    (errs['H'][i] / r['std_H']) ** 2)
                m['nees'] = med(nees)
            # drift between consecutive looks (world frame, last fitted centre of each look)
            cen = [to_world(rows[b]['cx'], rows[b]['cy']) for a, b in looks]
            worst, wstep, worst_what = 0.0, 0.0, '-'
            others = [(o['cx'], o['cy'], o['name']) for o in gtobjs if o['name'] != g['name']]
            for k in range(1, len(cen)):
                sx, sy = cen[k][0] - cen[k - 1][0], cen[k][1] - cen[k - 1][1]
                wstep = max(wstep, math.hypot(sx, sy))
                px, py = cen[k - 1]
                tg = []
                for b in walls:
                    qx, qy = min(max(px, b[0]), b[1]), min(max(py, b[2]), b[3])
                    tg.append((math.hypot(qx - px, qy - py), qx, qy, 'wall'))
                nw = min(tg) if tg else None
                no = min(((math.hypot(ox - px, oy - py), ox, oy, nm) for ox, oy, nm in others), default=None)
                for t in (nw, no):
                    if t is None or t[0] < 1e-6: continue
                    toward = (sx * (t[1] - px) + sy * (t[2] - py)) / t[0]
                    if toward > worst: worst, worst_what = toward, t[3]
            m['drift'] = worst; m['drift_what'] = worst_what; m['max_step'] = wstep
            # rate
            dur = (rows[-1]['pkt_ts'] - rows[0]['pkt_ts']) / 1000.0
            m['rate'] = len(rows) / dur if dur > 0 else float('nan')
            dts = [(rows[i]['pkt_ts'] - rows[i - 1]['pkt_ts']) / 1000.0 for i in range(1, len(rows))]
            dts = [x for x in dts if 0 < x < 2.0]
            m['rate_med'] = 1.0 / statistics.median(dts) if dts else float('nan')
            # LiDAR measures
            if 'lidar_only' in rows[0]:
                m['lidar_only_share'] = sum(1 for r in rows if r['lidar_only'] == 1) / len(rows)
                for k in ('rho_cam', 'rho_L', 'rho_chain'):
                    m[k] = med([r[k] for r in rows if r.get('lidar_only', 0) == 0 or k == 'rho_L'])
                pk = [r for r in rows if gtpose and parked_at(gtpose, r['pkt_ts']) is True]
                m['n_parked'] = len(pk)
                m['dphi_parked'] = med([abs(r['dphi_cam']) for r in pk]) if pk and 'dphi_cam' in pk[0] else float('nan')
                m['dz_parked'] = med([abs(r['dz_cam']) for r in pk]) if pk and 'dz_cam' in pk[0] else float('nan')
            m['live10'] = live10_checks(rows, gtpose, proto, proto_ts)
            m['beam'] = beam_checks(rows)
            if m['live10'] and not proto: notes.append('%s: no proto_room.csv sigma: Sigma_pose check n/a' % node)
            # shift: final size vs the truth
            tw_, th_ = true_dims(g, last.get('yaw', 0.0) + rot)
            m['w_err'] = last['w'] - tw_; m['h_err'] = last['h'] - th_; m['H_err'] = last['H'] - g['top']
            res[node] = m
    return res, notes, failed


# ---- report ----------------------------------------------------------------------------------------------------------

def report(label, res, notes, shift):
    P = print
    P('=' * 100)
    P('RUN %s' % label)
    for n in notes: P('  note: ' + n)
    for node, m in sorted(res.items()):
        g = m['gtd']
        P('-' * 100)
        P('%s -> GT %s (%.3f, %.3f, top %.2f)  rows %d  looks %d  settled rows %d%s' % (
            node, g['name'], g['cx'], g['cy'], g['top'], m['rows'], m['n_looks'], m['settled'],
            '   PHANTOM (>0.75 m from every GT of its class)' if m['phantom'] else ''))
        P('  final centre error      %s m  (live6 ref: bed 0.055, nightstands 0.02-0.06, toilet 0.23)  [info]' % fmt(m['final_err']))
        P('  final |H - true top|    %s m  (H %s, top %.2f)  [info; compare with the live6 column]' % (fmt(m['H_err_final']), fmt(m['H_final']), g['top']))
        if 'dof' in m:
            for k in ('cx', 'cy', 'H', 'w', 'h'):
                d = m['dof'][k]
                P('  %-2s rms %s  std min %s max %s  inside[rms/3,3rms] %5.1f%% (below %5.1f%%)   %s' % (
                    k, fmt(d['rms'], '.4f'), fmt(d['smin'], '.4f'), fmt(d['smax'], '.4f'), 100 * d['inside'], 100 * d['below'], verdict(d['ok'])))
            P('  median NEES (3 dof: cx,cy,H; expected ~2.4)  %s  [info]' % fmt(m['nees'], '.2f'))
        else:
            P('  no fresh-mask row from within %.1f m: DOF checks not scored   FAIL' % NEAR_M)
        P('  rows with |err| > 3 std (all %d rows; calibrated Gaussian 0.27%%)  %s  [info]' % (
            m['rows'], '  '.join('%s %5.1f%%' % (k, 100 * v) for k, v in m['z3'].items())))
        P('  largest centre step toward %s between looks %s m (largest step %s m)   %s' % (
            m['drift_what'], fmt(m['drift'], '.3f'), fmt(m['max_step'], '.3f'), verdict(m['drift'] <= DRIFT_MAX)))
        P('  fit rate %s Hz (median-dt %s Hz)   %s' % (fmt(m['rate'], '.1f'), fmt(m['rate_med'], '.1f'), verdict(m['rate'] >= RATE_MIN)))
        if 'lidar_only_share' in m:
            P('  lidar_only share %.1f %%  medians rho_cam %s rho_L %s rho_chain %s;  parked rows %d: |dphi_cam| %s |dz_cam| %s  [info]' % (
                100 * m['lidar_only_share'], fmt(m['rho_cam'], '.3f'), fmt(m['rho_L'], '.3f'), fmt(m['rho_chain'], '.3f'),
                m['n_parked'], fmt(m['dphi_parked'], '.4f'), fmt(m['dz_parked'], '.4f')))
        else:
            P('  lidar_only / rho / dphi / dz: n/a (columns absent)  [info]')
        L = m.get('live10') or {}
        if L:
            P('  live10: updated rows (Itr_cam > 0 or Itr_L > 0) %d of %d' % (L['n_updated'], m['rows']))
            P('  |e| <= %.0f sigma of Sigma_pose: worst |e|/spose %s, rows over %d   %s ;  worst |e|/se (posterior) %s [info]' % (
                E_NSIG, fmt(L['e_worst_pose'], '.2f'), L['e_bad'], verdict(L['e_ok']) if L['n_updated'] else 'n/a',
                fmt(L['e_worst_post'], '.2f')))
            P('  parked Sigma_pose vs proto_room sigma (%d rows): xy rel med %s p95 %s ; psi rel med %s p95 %s ; within %.0f %% %s   %s' % (
                L['sp_n'], fmt(L['sp_xy_med'], '.4f'), fmt(L['sp_xy_p95'], '.4f'), fmt(L['sp_psi_med'], '.4f'),
                fmt(L['sp_psi_p95'], '.4f'), 100 * SPOSE_REL, fmt(100 * L['sp_frac'], '.1f') + '%' if L['sp_n'] else 'n/a',
                verdict(L['sp_ok']) if L['sp_n'] else 'n/a'))
            for tag, nm in (('cam', 'camera-only'), ('ray', 'with rays')):
                q = L['gn_%s_q' % tag]
                P('  gn %-11s n %d: gn_it min/p25/med/p75/max %s ; gn_conv = 0 share %s  [info]' % (
                    nm, L['gn_%s_n' % tag], '/'.join(fmt(x, '.0f') for x in q), fmt(L['gn_%s_unconv' % tag], '.3f')))
        else:
            P('  live10 (|e| <= 3 sigma, Sigma_pose vs proto_room, gn_it / gn_conv): n/a (columns absent)  [info]')
        B = m.get('beam') or {}
        if B:
            qs = ('n/a (column absent)' if B['qsum_med'] is None
                  else '%s (staged rows with qsum 0: %d)' % (fmt(B['qsum_med'], '.2f'), B['qsum_zero']))
            P('  beam: staged rows %d, staged med %s max %s, kappa med %s (instrument) ; qsum med %s ; nolabel sum %d med %s ; '
              'nomem rows %d (gated %d) longest run %d ; no-sweep rows %d (gated %d) ; gap rays sum %d max %d rows %d  [info]' % (
                B['n_staged_rows'], fmt(B['staged_med'], '.0f'), fmt(B['staged_max'], '.0f'), fmt(B['kappa_med'], '.3f'), qs,
                B['nolabel_sum'], fmt(B['nolabel_med'], '.0f'), B['nomem'], B['nomem_gated'], B['nomem_run'], B['nosweep'],
                B['nosweep_gated'], B['gap_sum'], B['gap_max'], B['gap_rows']))
        else:
            P('  beam (staged / nolabel / nomem / gap rays): n/a (columns absent)  [info]')
        if shift:
            for k, nm in (('w_err', 'w'), ('h_err', 'h'), ('H_err', 'H')):
                P('  shift: final %s err vs TRUE size %+.3f m (tol %.2f; the shifted prior would read +%.2f)   %s' % (
                    nm, m[k], SHIFT_TOL[nm], 0.10 if nm == 'H' else 0.06, verdict(abs(m[k]) <= SHIFT_TOL[nm])))


def side_by_side(named):
    P = print
    P('=' * 100)
    P('SIDE BY SIDE (per GT object; instances that matched the same GT are listed separately)')
    names = [n for n, _ in named]
    P('%-26s' % 'GT / node / metric' + ''.join('%14s' % n for n in names))
    gts = sorted({(m['gt'], node) for _, res in named for node, m in res.items()})
    rows = [('final_err', 'centre err m', '.3f'), ('H_err_final', '|H-top| m', '.3f'), ('drift', 'drift m', '.3f'),
            ('rate', 'rate Hz', '.1f'), ('nees', 'NEES med', '.2f'), ('lidar_only_share', 'lidar_only', '.3f')]
    for gname in sorted({g for g, _ in gts}):
        for key, lab, spec in rows:
            line = '%-26s' % ('%s %s' % (gname, lab))
            for _, res in named:
                ms = [m for m in res.values() if m['gt'] == gname and not m['phantom']]
                v = min((m.get(key, float('nan')) for m in ms), default=float('nan')) if ms else float('nan')
                line += '%14s' % fmt(v, spec)
            P(line)
        for k in ('cx', 'cy', 'H', 'w', 'h'):
            line = '%-26s' % ('%s %s std-check' % (gname, k))
            for _, res in named:
                ms = [m for m in res.values() if m['gt'] == gname and not m['phantom'] and 'dof' in m]
                line += '%14s' % (verdict(ms[0]['dof'][k]['ok']) if ms else 'n/a')
            P(line)
    for _, res in named:
        pass
    P('phantoms (instances > 0.75 m from every GT): ' + '; '.join(
        '%s: %s' % (n, ','.join(k for k, m in res.items() if m['phantom']) or 'none') for n, res in named))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run'); ap.add_argument('off', nargs='?')
    ap.add_argument('--ref', help='objects_live6 dir (flag-off era) for the side-by-side')
    ap.add_argument('--shift', action='store_true', help='score final w, h, H against the TRUE size (shifted-class run)')
    ap.add_argument('--pin', action='store_true', help="fit each room's translation to the class's own GT (placement NOT independent)")
    ap.add_argument('--room', action='append', default=[], metavar='NAME=x,y,rot', help='override a room origin')
    ap.add_argument('--scene', default=SCENE)
    ap.add_argument('--gt', choices=('mesh', 'boundingobject'), default='mesh',
                    help='ground truth of the objects: mesh = the graded layer of the visual mesh read from --wbt (bed mattress, nightstand body, toilet '
                         'footprint); boundingobject = the earlier numbers (--scene csv, built from the collision boxes)')
    ap.add_argument('--wbt', default=os.environ.get('WBT_PATH', '/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt'),
                    help='world the run was made in (default: the live world; WBT_PATH overrides)')
    a = ap.parse_args()
    rooms = {}
    for s in a.room:
        k, v = s.split('=')
        rooms[k] = tuple(float(x) for x in v.split(','))
    gtobjs, walls = load_scene(a.scene)
    if a.gt == 'mesh': gtobjs = mesh_gt(gtobjs, a.wbt)
    print('ground truth: ' + ('mesh layers of ' + a.wbt if a.gt == 'mesh' else 'boundingobject numbers of ' + a.scene))
    print('lidar_measures_score  (spec 2026-11-15 §8)')
    print('look = fresh-mask run, gap <= %d ms; settled = from the first fresh row with range <= %.1f m' % (GAP_MS, NEAR_M))
    print('Sigma PASS = max std <= %.2f m AND >= %.0f %% of settled rows with rms/3 <= std <= 3*rms (per DOF, rms of the true error)' % (STD_MAX, 100 * FRAC_OK))
    print('drift PASS = largest step toward nearest wall / neighbour between looks <= %.2f m;  rate PASS >= %.0f Hz' % (DRIFT_MAX, RATE_MIN))
    print('room->world: FITTED PER RUN from gt_error.csv GT world track vs proto_room.csv est pose x gauge (see the room lines '
          'under each RUN); a room with < %d samples or rms > %.2f m FAILS and is not scored; --room overrides.' % (FIT_MIN_SAMPLES, FIT_MAX_RMS) +
          ('  [--pin: translation fitted to GT]' if a.pin else ''))
    print('GT: ' + '; '.join('%s (%.3f, %.3f) %.2fx%.2f top %.2f' % (g['name'], g['cx'], g['cy'], g['dx'], g['dy'], g['top']) for g in gtobjs))
    named, main_failed = [], {}
    for label, d, sh in (('RUN ' + a.run, a.run, a.shift), ('OFF ' + str(a.off), a.off, False), ('REF ' + str(a.ref), a.ref, False)):
        if not d: continue
        res, notes, failed = score_run(d, gtobjs, walls, rooms, a.pin, sh)
        report(label, res, notes, sh)
        named.append((label.split()[0], res))
        if failed and len(named) == 1: main_failed = failed
    if len(named) > 1: side_by_side(named)
    fails = sum(1 for _, res in named[:1] for m in res.values() for k in m.get('dof', {}) if not m['dof'][k]['ok'])
    print('=' * 100)
    print('SUMMARY (%s): %d object instance(s), %d DOF std-check FAIL line(s), %d drift FAIL, %d rate FAIL' % (
        named[0][0] if named else '-', len(named[0][1]) if named else 0, fails,
        sum(1 for m in named[0][1].values() if m['drift'] > DRIFT_MAX) if named else 0,
        sum(1 for m in named[0][1].values() if not m['rate'] >= RATE_MIN) if named else 0))
    L = [m['live10'] for m in named[0][1].values() if m.get('live10')] if named else []
    if L:
        print('SUMMARY live10: %d |e| > 3 sigma FAIL, %d Sigma_pose-vs-proto_room FAIL (of %d with parked rows)' % (
            sum(1 for x in L if x['n_updated'] and not x['e_ok']), sum(1 for x in L if x['sp_n'] and not x['sp_ok']),
            sum(1 for x in L if x['sp_n'])))
    else:
        print('SUMMARY live10: n/a (log predates e_* / spose_* / gn_*)')
    if main_failed:
        print('ROOM FRAME FAIL (main run): ' + '; '.join('%s: %s' % kv for kv in main_failed.items()))
        sys.exit(2)


if __name__ == '__main__':
    main()
