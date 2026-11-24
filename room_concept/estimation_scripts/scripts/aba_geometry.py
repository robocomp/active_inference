#!/usr/bin/env python3
"""Ground truth, standpoints and frames for the parked test and the A/B/A visit (shared-engine plan rev 2, Task P1).

    aba_geometry.py plan  <wbt> <DEF> [--live]            standpoints A, B (world) as JSON on stdout
    aba_geometry.py truth <wbt> <DEF> [--live]            the object's graded ground truth as JSON
    aba_geometry.py to_room <x> <y> <yaw>                 a WORLD pose -> the published frame of the room the robot is in now
                                                          (live: room_concept/tmp/sdf_localizer/gt_error.csv + tmp/proto_room.csv)
    aba_geometry.py query_pose <sx> <sy> <cx> <cy>       a WORLD standpoint and object centre -> the standpoint in the published
                                                          room frame and the heading facing the centre (math convention, the
                                                          engine's Standpoint yaw) — the gain query's pose
    aba_geometry.py heading_tol                          the controller's heading goal tolerance (deg, align_yaw_tol_rad)
    aba_geometry.py face <x> <y>                          the turn (deg, + = left) that points the robot (Webots pose now) at the
                                                          world point (x, y)
--live reads the object's pose from Webots (webots_object.py get) instead of the .wbt file.

GROUND TRUTH (spec 2026-11-23 shared-engine, "Graded quantities" + AuRo A3 amendment): the extent of the mesh surfaces the
likelihood fits. Nightstand = body box, toilet = footprint of bowl + tank (tools/gt_mesh/mesh_extents.py); bed = w 1.766 (frame
union blanket), h (depth) 2.09-2.12 (frame incl. headboard: an interval), H 0.72 (blanket top), centred on the mattress centre
(the frame envelope is +-1.06 about it: docs/superpowers/results/2026-11-25-bed-depth-diagnosis.md section 0).
theta = (cx, cy, H, w, h, yaw): w along the wall, h = depth away from it, R(yaw)·y = the direction front -> back (spec "State
naming"). The BACK direction is the outward normal of the room-polygon edge nearest the object's centre (the wall it stands
against), so no per-class orientation table is needed.

STANDPOINTS (the rule, stated in docs/superpowers/results/2026-11-25-p1-predictions.md):
  * nominal: A is 1.5 m in front of the object's FRONT face, on the front normal through the centre, so its range to the centre
    is z0 = h/2 + 1.5; B is 45 deg round the centre at the SAME range (the spec's "same range": then rho's |dz| term is zero
    along an arc and the prediction rests on the bearing change alone).
  * feasible(p): the robot's disc of radius r = circumscribed body radius 0.3252 m (ROBOT_GEOMETRY.md: it must turn in place to
    face the object) + the controller's RouteSafetyMargin (controller/etc/config.toml) lies inside the room polygon and clear
    of every footprint in the room, the object's own included. A geometric feasibility test, not a tuning threshold.
  * chosen range: z = the feasible range nearest z0 (1 cm grid, ties to the smaller) at which A AND one B side are both
    feasible; B's side is the feasible one, and when both are, the one with more clearance. No view-change threshold: rho is
    whatever the logged path gives; the planned values are reported for the registered prediction.
FRAMES: the agents' theta and the controller's go_to poses are in the PUBLISHED room frame. room_concept's est_x/est_y
(gt_error.csv) are its INTERNAL frame; published = R(g_theta)·internal + g with the gauge (g_x, g_y, g_theta) it logs in
proto_room.csv (as c3a_objects.py, 11-22 frame fix). internal <- world is fitted per visit: rotation = circular mean of
est_theta − (gt_theta − pi/2) (gt_error's gt_theta is the Webots yaw + pi/2), translation = mean residual of the positions.
"""
import bisect, csv, json, math, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'tools', 'gt_mesh'))
B = '/home/pbustos/robocomp/components/active_inference'
LIVE_GT_ERROR = f'{B}/room_concept/tmp/sdf_localizer/gt_error.csv'
LIVE_PROTO_ROOM = f'{B}/room_concept/tmp/proto_room.csv'
ROBOT_CIRCUMSCRIBED_M = 0.3252          # ROBOT_GEOMETRY.md: circumscribed body radius of the shipping footprint
FRONT_CLEARANCE_M = 1.5                 # the brief: A is 1.5 m in front of the object
B_ANGLE_RAD = math.pi / 4               # the brief: B is 45 deg round the object at the same range
GRID_M = 0.01

# AuRo A3 (spec "Graded quantities", 11-25): the bed's graded box is NOT the mattress mesh.
BED_A3 = dict(w=1.766, h=(2.09, 2.12), H=0.72)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def route_safety_margin(cfg=f'{B}/controller/etc/config.toml'):
    for line in open(cfg):
        m = re.match(r'\s*RouteSafetyMargin\s*=\s*([-+0-9.eE]+)', line)
        if m: return float(m.group(1))
    raise KeyError(f'RouteSafetyMargin not in {cfg}')


def webots_pose(defn):
    """(x, y, yaw) of a Webots node now (webots_object.py get: x y z qx qy qz qw)."""
    out = subprocess.run([sys.executable, os.path.join(HERE, 'webots_object.py'), 'get', defn], capture_output=True,
                         text=True, check=True).stdout.split()
    x, y, qx, qy, qz, qw = float(out[0]), float(out[1]), *map(float, out[3:7])
    return x, y, math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def robot_pose():
    """(x, y, yaw) of the robot in Webots now (tp.py get)."""
    out = subprocess.run([sys.executable, os.path.join(HERE, 'tp.py'), 'get'], capture_output=True, text=True,
                         check=True).stdout.split()
    return float(out[0]), float(out[1]), float(out[out.index('yaw') + 1])


# ------------------------------------------------------------------------------------------------ the room and its furniture
def room_polygons():
    import wbt_rooms as W
    rooms, _, _ = W.rooms()
    return {k: np.asarray(v['poly'], float) for k, v in rooms.items()}


def _point_in(poly, p):
    import wbt_rooms as W
    return bool(W.inside(poly, np.asarray([p], float))[0])


def footprints(wbt):
    """[(name, shapely polygon)] of everything standing on the floor that the furniture parser can read, plus [(name, why)]
    of the nodes it could not. Graded classes use their graded layer (the bed its A3 frame box); PROTO nodes without a
    boundingObject in the world use the one in their cached PROTO file."""
    import furniture as F
    import mesh_extents as M
    from shapely.geometry import MultiPoint, box
    text = open(wbt).read()
    protos = {os.path.basename(u)[:-len('.proto')]: u for u in re.findall(r'EXTERNPROTO\s+"([^"]+)"', text)}
    t = F.tokens(text); i = 0; top = []
    while i < len(t):
        if t[i] == 'EXTERNPROTO': i += 2; continue
        if t[i] == 'DEF': i += 2; continue
        if i + 1 < len(t) and t[i + 1] == '{': n, i = F.parse_node(t, i); top.append(n)
        else: i += 1
    skip_types = {'Wall', 'Door', 'Floor', 'Ceiling', 'PointLight', 'TexturedBackground', 'TexturedBackgroundLight',
                  'WorldInfo', 'Viewpoint', 'Shadow', 'BeerBottle', 'PortraitPainting', 'PhotoFrame'}
    out, failed = [], []

    def name(n):
        v = n.get('name')
        while isinstance(v, list) and v: v = v[0]
        return v.strip('"') if isinstance(v, str) else n['_type']

    def visit(n):
        if n['_type'] == 'Group':
            for c in n.get('children', []):
                if isinstance(c, dict): visit(c)
            return
        nm = name(n)
        if n['_type'] in skip_types or nm in F.SKIP or 'translation' not in n: return
        x, y, _, yaw = F.T(n)
        try:
            cls = M.classify(n)
            if cls == 'bed':
                g = box(-BED_A3['h'][1] / 2, -BED_A3['w'] / 2, BED_A3['h'][1] / 2, BED_A3['w'] / 2)   # local x = length
                out.append((nm, F.place(g, x, y, yaw))); return
            if cls is not None:
                b = M.layer_box(cls, n, wbt)
                if b is not None:
                    g = box(b[4] - b[0] / 2, b[5] - b[1] / 2, b[4] + b[0] / 2, b[5] + b[1] / 2)
                    out.append((nm, F.place(g, x, y, yaw))); return
            bo = n.get('boundingObject')
            if bo is None and n['_type'] in protos:
                pt = F.tokens(open(M.asset(protos[n['_type']].split('/projects/objects/')[1])).read())
                k = pt.index('boundingObject')
                bo, _ = F.parse_value(pt, k + 1)
            if isinstance(bo, dict) and bo['_type'] == 'Mesh':
                u = bo['url'][0]; u = (u[0] if isinstance(u, list) else u).strip('"')
                p = os.path.join(os.path.dirname(wbt), u)
                if u.endswith('.dae'): g = F.dae_footprint(p)
                else:
                    v = M.file_vertices(p); v = v[(v[:, 2] >= F.Z_MIN) & (v[:, 2] <= F.Z_MAX)]
                    g = MultiPoint([tuple(q) for q in v[:, :2]]).convex_hull
                if g is None: raise ValueError('empty mesh footprint')
                out.append((nm, F.place(g, x, y, yaw))); return
            gs = F.prims(bo)
            if not gs: raise ValueError(f'no boundingObject primitive ({n["_type"]})')
            from shapely.ops import unary_union
            out.append((nm, F.place(unary_union(gs), x, y, yaw)))
        except Exception as e:   # reported, never silent: a missing footprint makes a standpoint look freer than it is
            failed.append((nm, f'{type(e).__name__}: {e}', [round(x, 3), round(y, 3)]))

    for n in top: visit(n)
    return out, failed


# ------------------------------------------------------------------------------------------------ ground truth
def object_truth(wbt, defn, live=False):
    import furniture as F
    import mesh_extents as M
    nm, cls, x, y, _, yaw, node = M.graded(wbt, defn)
    pose_src = 'wbt'
    if live:
        x, y, yaw = webots_pose(defn); pose_src = 'webots'
    b = M.layer_box(cls, node, wbt)
    cx, cy = M.to_world(x, y, yaw, b[4], b[5])
    polys = room_polygons()
    room = next((k for k, p in polys.items() if _point_in(p, (cx, cy))), None)
    if room is None: raise ValueError(f'{defn}: centre ({cx:.3f}, {cy:.3f}) is in no room polygon')
    poly = polys[room]
    # back = outward normal of the nearest polygon edge (the wall it stands against)
    best = None
    for a, c in zip(poly, np.roll(poly, -1, axis=0)):
        d = c - a; L2 = float(d @ d)
        if L2 == 0.0: continue                                   # degenerate edge (repeated vertex)
        s = min(max(float((np.array([cx, cy]) - a) @ d) / L2, 0.0), 1.0)
        q = a + s * d; dist = math.hypot(q[0] - cx, q[1] - cy)
        if best is None or dist < best[0]: best = (dist, q)
    back = np.array([best[1][0] - cx, best[1][1] - cy]); back /= np.linalg.norm(back)
    lx_dir = np.array([math.cos(yaw), math.sin(yaw)])
    along_x = abs(float(lx_dir @ back)) >= abs(float(np.array([-math.sin(yaw), math.cos(yaw)]) @ back))
    h_mesh, w_mesh = (b[0], b[1]) if along_x else (b[1], b[0])
    if cls == 'bed':
        w, h, H, src = BED_A3['w'], BED_A3['h'], BED_A3['H'], 'AuRo A3 (spec Graded quantities): frame u blanket w, frame+headboard h, blanket top H'
    else:
        w, h, H, src = w_mesh, (h_mesh, h_mesh), b[3], f'mesh layer ({M.__name__}.layer_box)'
    return dict(defn=defn, name=nm, cls=cls, room_poly=room, pose_src=pose_src, origin=[x, y, yaw], cx=cx, cy=cy,
                back=[float(back[0]), float(back[1])], yaw_theta_world=math.atan2(-back[0], back[1]),
                w=w, h=list(h), H=H, wall_gap=best[0], source=src)


# ------------------------------------------------------------------------------------------------ standpoints
def plan(wbt, defn, live=False):
    from shapely.geometry import Point, Polygon
    T = object_truth(wbt, defn, live)
    r = ROBOT_CIRCUMSCRIBED_M + route_safety_margin()
    room = Polygon(room_polygons()[T['room_poly']])
    inner = room.buffer(-r)
    fps, failed = footprints(wbt)
    obst = [(n, g) for n, g in fps if g.intersects(room.buffer(0.5))]
    c = np.array([T['cx'], T['cy']]); f = -np.array(T['back'])
    hmax = max(T['h'])
    z0 = hmax / 2 + FRONT_CLEARANCE_M

    def rot(v, a): return np.array([math.cos(a) * v[0] - math.sin(a) * v[1], math.sin(a) * v[0] + math.cos(a) * v[1]])

    def clearance(p):
        P = Point(*p)
        if not inner.contains(P): return -1.0
        return min([room.exterior.distance(P) - r] + [g.distance(P) - r for _, g in obst])

    cands = []
    for k in range(int(round((max(z0 * 2.5, 4.0)) / GRID_M)) + 1):
        z = k * GRID_M
        if z <= hmax / 2: continue
        A = c + z * f
        if clearance(A) < 0: continue
        sides = [(s, c + z * rot(f, s * B_ANGLE_RAD)) for s in (+1, -1)]
        sides = [(s, p, clearance(p)) for s, p in sides if clearance(p) >= 0]
        if not sides: continue
        s, Bp, cl = max(sides, key=lambda q: q[2])
        cands.append((abs(z - z0), z, A, s, Bp))
    if not cands:
        return dict(truth=T, feasible=False, why='no range with both A and a B side feasible', r_clear=r, z0=z0,
                    footprints_failed=failed)
    _, z, A, s, Bp = min(cands, key=lambda q: (q[0], q[1]))
    face = lambda p: math.atan2(-(c - p)[0], (c - p)[1])     # Webots/robot yaw: 0 faces +y, CCW positive
    import aba_model as V
    plan_rho = V.planned_rho(A, Bp, c, T['w'], max(T['h']), np.array(T['back']))
    # a node whose footprint could not be read is listed with its origin's distance to the nearer standpoint (an
    # unreadable node far from both changes nothing; a near one makes the plan suspect)
    unread = [dict(name=n, why=w, origin=o, dist_to_standpoint_m=round(min(math.hypot(o[0] - p[0], o[1] - p[1]) for p in (A, Bp)), 2))
              for n, w, o in failed]
    return dict(truth=T, feasible=True, r_clear=r, z0=z0, z=z, moved_m=z - z0, side=s,
                A=[float(A[0]), float(A[1]), face(A)], B=[float(Bp[0]), float(Bp[1]), face(Bp)],
                planned_rho=plan_rho, obstacles=[n for n, _ in obst], footprints_failed=unread)


# ------------------------------------------------------------------------------------------------ frames
def read_gt_error(path):
    rows = []
    if not os.path.exists(path): return rows
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((int(r['ts_ms']), float(r['est_x']), float(r['est_y']), float(r['est_theta']),
                             float(r['gt_x_world']), float(r['gt_y_world']), float(r['gt_theta'])))
            except (KeyError, ValueError):
                continue
    return rows


def read_gauge(path):
    G = []
    if not os.path.exists(path): return G
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try: G.append((int(r['ts_ms']), float(r['gauge_x']), float(r['gauge_y']), float(r['gauge_theta'])))
            except (KeyError, ValueError): pass
    return G


def fit_world_to_internal(rows):
    """(phi, t, rms): internal = R(phi)·world + t, from (ts, ex, ey, eth, gx, gy, gth) rows of ONE visit."""
    if not rows: raise ValueError('no gt_error rows in the window')
    d = [wrap(eth - (gth - math.pi / 2)) for _, _, _, eth, _, _, gth in rows]
    phi = math.atan2(sum(math.sin(a) for a in d), sum(math.cos(a) for a in d))
    R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    W = np.array([[g[4], g[5]] for g in rows]); E = np.array([[g[1], g[2]] for g in rows])
    t = (E - W @ R.T).mean(0)
    rms = float(np.sqrt(np.mean(np.sum((W @ R.T + t - E) ** 2, 1))))
    return phi, t, rms


class Frames:
    """world <-> published room frame for one visit: a fit on gt_error rows of the window, the gauge at a time."""
    def __init__(self, gt_rows, gauge_rows, t0_ms, t1_ms):
        win = [r for r in gt_rows if t0_ms <= r[0] <= t1_ms]
        self.phi, self.t, self.rms = fit_world_to_internal(win)
        self.n = len(win)
        self.G = gauge_rows; self.Gts = [g[0] for g in gauge_rows]

    def gauge(self, t_ms):
        if not self.G: return 0.0, 0.0, 0.0
        _, gx, gy, gth = self.G[max(bisect.bisect_right(self.Gts, t_ms) - 1, 0)]
        return gx, gy, gth

    def world_to_pub(self, x, y, yaw, t_ms):
        c, s = math.cos(self.phi), math.sin(self.phi)
        ix, iy = c * x - s * y + self.t[0], s * x + c * y + self.t[1]
        gx, gy, gth = self.gauge(t_ms); c, s = math.cos(gth), math.sin(gth)
        return c * ix - s * iy + gx, s * ix + c * iy + gy, wrap(yaw + self.phi + gth)

    def pub_to_world(self, x, y, yaw, t_ms):
        gx, gy, gth = self.gauge(t_ms); c, s = math.cos(gth), math.sin(gth)
        dx, dy = x - gx, y - gy
        ix, iy = c * dx + s * dy, -s * dx + c * dy
        c, s = math.cos(self.phi), math.sin(self.phi)
        dx, dy = ix - self.t[0], iy - self.t[1]
        return c * dx + s * dy, -s * dx + c * dy, wrap(yaw - self.phi - gth)


def main(argv):
    cmd = argv[0]
    if cmd in ('plan', 'truth'):
        live = '--live' in argv
        wbt, defn = [a for a in argv[1:] if not a.startswith('--')][:2]
        print(json.dumps(plan(wbt, defn, live) if cmd == 'plan' else object_truth(wbt, defn, live), indent=1))
        return 0
    if cmd == 'to_room':
        x, y, yaw = map(float, argv[1:4])
        rows = read_gt_error(LIVE_GT_ERROR)
        t1 = rows[-1][0]
        window_s = float(os.environ.get('ABA_FIT_WINDOW_S', '8'))
        fr = Frames(rows, read_gauge(LIVE_PROTO_ROOM), t1 - int(window_s * 1000), t1)
        px, py, pyaw = fr.world_to_pub(x, y, yaw, t1)
        print(f'{px:.4f} {py:.4f} {pyaw:.4f} fit_n={fr.n} fit_rms={fr.rms:.4f} phi={fr.phi:.4f}')
        return 0
    if cmd == 'query_pose':
        # a standpoint (world) and the object centre (world) -> the standpoint in the published room frame and the heading
        # that faces the centre THERE, in the engine's Standpoint convention (math heading atan2(dy, dx), 0 = +x; the
        # convention of rc::nbv::face_viewpoint, which the offer uses)
        sx, sy, cx, cy = map(float, argv[1:5])
        rows = read_gt_error(LIVE_GT_ERROR); t1 = rows[-1][0]
        window_s = float(os.environ.get('ABA_FIT_WINDOW_S', '8'))
        fr = Frames(rows, read_gauge(LIVE_PROTO_ROOM), t1 - int(window_s * 1000), t1)
        px, py, _ = fr.world_to_pub(sx, sy, 0.0, t1); qx, qy, _ = fr.world_to_pub(cx, cy, 0.0, t1)
        print(f'{px:.4f} {py:.4f} {math.atan2(qy - py, qx - px):.6f}')
        return 0
    if cmd == 'heading_tol':
        # the controller's own heading goal tolerance: |yaw error| under which its goal alignment is done
        # (controller/src/controller_types.h align_yaw_tol_rad; the config sets no key for it), in degrees
        hdr = f'{B}/controller/src/controller_types.h'
        for line in open(hdr):
            m = re.match(r'\s*float\s+align_yaw_tol_rad\s*=\s*([-+0-9.eE]+)f?\s*;', line)
            if m: print(f'{math.degrees(float(m.group(1))):.3f}'); return 0
        print(f'align_yaw_tol_rad not found in {hdr}', file=sys.stderr); return 1
    if cmd == 'face':
        tx, ty = float(argv[1]), float(argv[2])
        x, y, yaw = robot_pose()
        want = math.atan2(-(tx - x), ty - y)
        print(f'{math.degrees(wrap(want - yaw)):.1f}')
        return 0
    print(__doc__); return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
