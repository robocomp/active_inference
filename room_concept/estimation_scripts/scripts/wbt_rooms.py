#!/usr/bin/env python3
"""Ground-truth room polygons for threerooms-piso.wbt (world frame, metres).

Walls come from the .wbt: every `Wall {}` node (translation, rotation about z, size =
thickness x length x height) is rasterised as an oriented rectangle. The room_1 partition
walls are a `Solid` whose boundingObject is meshes/wall.dae (no transform, world coords);
its triangles are rasterised too. Door openings are closed by a 0.2 m thick strip on the wall
line (door width = size.y). Free space = 4-connected component of a seed point on a 1 cm grid;
its outer boundary is traced and Douglas-Peucker simplified.

Usage:
  python3 wbt_rooms.py                       # print room_2 / room_3 polygons
  python3 wbt_rooms.py --score room_3 est.csv  # align est (x,y CSV, any frame) and score
"""
import csv, math, re, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WBT = __import__("os").environ.get("WBT_PATH", "/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt")   # WBT_PATH: grade an old run against the world it ran in
MESH_DIR = os.path.join(os.path.dirname(WBT), "meshes")
RES = 0.01
BOX = (-9.0, 4.5, -1.0, 14.2)  # x0, x1, y0, y1
# door -> (axis of the wall the door sits in, wall-normal coordinate of the wall line)
# DOOR_2 translation in the .wbt is x=-1.615; the wall it sits in is the x=-1.715 partition.
DOOR_WALL = {"DOOR_0": ("x", 7.226), "DOOR_1": ("x", 9.3796), "DOOR_2": ("y", -1.715)}
SEEDS = {"room_3": (-6.5, 11.5), "room_2": (-3.0, 8.2), "room_1": (-5.33, 3.66)}  # room_2 = box behind DOOR_0 (x -4.2..-1.8) + east part


def _block(text, start):
    """Return the text of the {...} block whose opening brace follows position start."""
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    raise ValueError("unbalanced block")


def parse_wbt(path=WBT):
    text = open(path, encoding="utf8").read()
    walls, doors, meshes = [], {}, []
    for m in re.finditer(r"(?m)^\s*Wall \{", text):
        b = _block(text, m.start())
        t = re.search(r"\btranslation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)", b)
        r = re.search(r"\brotation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)", b)
        s = re.search(r"\bsize ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)", b)
        n = re.search(r'\bname "([^"]*)"', b)
        tx, ty = float(t.group(1)), float(t.group(2))
        ang = float(r.group(4)) if r else 0.0
        sx, sy = float(s.group(1)), float(s.group(2))
        walls.append(dict(name=n.group(1) if n else "?", c=(tx, ty), ang=ang, size=(sx, sy)))
    for m in re.finditer(r"DEF (DOOR_\d) Door \{", text):
        b = _block(text, m.start())
        t = re.search(r"(?m)^\s*translation ([-\d.e]+) ([-\d.e]+)", b)
        s = re.search(r"\bsize ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)", b)
        doors[m.group(1)] = dict(c=(float(t.group(1)), float(t.group(2))), width=float(s.group(2)))
    for m in re.finditer(r"boundingObject Mesh \{\s*url \[\s*\"([^\"]*wall\.dae)\"", text):
        meshes.append(os.path.join(os.path.dirname(path), m.group(1)))
    return walls, doors, meshes


def mesh_triangles(path):
    s = open(path, encoding="utf8").read()
    pos = re.search(r'<float_array[^>]*id="Room-positions-array"[^>]*>([^<]*)<', s)
    v = np.array([float(x) for x in pos.group(1).split()]).reshape(-1, 3)[:, :2]
    tri = re.search(r"<triangles[^>]*>.*?<p>([^<]*)</p>", s, re.S)
    p = np.array([int(x) for x in tri.group(1).split()]).reshape(-1, 3, 3)[:, :, 0]  # VERTEX offset 0
    tris = v[p]  # (T,3,2)
    return tris


def grid():
    x0, x1, y0, y1 = BOX
    xs = np.arange(x0, x1, RES) + RES / 2
    ys = np.arange(y0, y1, RES) + RES / 2
    X, Y = np.meshgrid(xs, ys)  # rows = y, cols = x
    return xs, ys, X, Y


def rasterise(walls, doors, meshes):
    xs, ys, X, Y = grid()
    occ = np.zeros(X.shape, bool)
    for w in walls:
        a = w["ang"]; c, s = math.cos(-a), math.sin(-a)
        dx, dy = X - w["c"][0], Y - w["c"][1]
        lx, ly = c * dx - s * dy, s * dx + c * dy
        occ |= (np.abs(lx) <= w["size"][0] / 2) & (np.abs(ly) <= w["size"][1] / 2)
    for f in meshes:
        for (a, b, cc) in mesh_triangles(f):
            lo = np.minimum(np.minimum(a, b), cc); hi = np.maximum(np.maximum(a, b), cc)
            i0, i1 = np.searchsorted(xs, lo[0]), np.searchsorted(xs, hi[0])
            j0, j1 = np.searchsorted(ys, lo[1]), np.searchsorted(ys, hi[1])
            if i1 <= i0 or j1 <= j0:
                continue
            px, py = X[j0:j1, i0:i1], Y[j0:j1, i0:i1]
            d = (b[1] - cc[1]) * (a[0] - cc[0]) + (cc[0] - b[0]) * (a[1] - cc[1])
            if abs(d) < 1e-12:
                continue
            l1 = ((b[1] - cc[1]) * (px - cc[0]) + (cc[0] - b[0]) * (py - cc[1])) / d
            l2 = ((cc[1] - a[1]) * (px - cc[0]) + (a[0] - cc[0]) * (py - cc[1])) / d
            inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (1 - l1 - l2 >= -1e-9)
            occ[j0:j1, i0:i1] |= inside
    for name, (axis, line) in DOOR_WALL.items():  # close door openings on the wall line
        d = doors[name]; half = d["width"] / 2; t = 0.1
        if axis == "x":
            occ |= (np.abs(X - d["c"][0]) <= half) & (np.abs(Y - line) <= t)
        else:
            occ |= (np.abs(Y - d["c"][1]) <= half) & (np.abs(X - line) <= t)
    return occ


def dilate(mask, r):
    out = np.zeros_like(mask)
    pad = np.pad(mask, r)
    H, W = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= pad[r + dy:r + dy + H, r + dx:r + dx + W]
    return out


SEAL_CELLS = 2  # 2 cm: wall corners meeting across a 2 cm slit (H top 13.46 vs N 13.48) are sealed


def component(occ, seed):
    """Free-space component of seed. Walls are sealed by SEAL_CELLS for connectivity only;
    the true free cells within SEAL_CELLS of that sealed component are then restored."""
    comp_s = _flood(~dilate(occ, SEAL_CELLS), seed)
    return _flood(~occ, seed) & dilate(comp_s, SEAL_CELLS)


def _flood(free, seed):
    H, W = free.shape
    j, i = np.searchsorted(grid()[1], seed[1]), np.searchsorted(grid()[0], seed[0])
    comp = np.zeros_like(free)
    stack = [(j, i)]; comp[j, i] = True
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= yy < H and 0 <= xx < W and free[yy, xx] and not comp[yy, xx]:
                comp[yy, xx] = True; stack.append((yy, xx))
    if comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any():
        raise RuntimeError("free component leaks to the grid border: room not closed")
    return comp


def trace_outline(comp):
    """Directed boundary edges with interior on the left, chained into the outer loop."""
    H, W = comp.shape
    ox, oy = BOX[0], BOX[2]
    nxt = {}
    def add(p, q):
        nxt.setdefault(p, []).append(q)
    ys_idx, xs_idx = np.nonzero(comp)
    for j, i in zip(ys_idx, xs_idx):
        x0, x1 = ox + i * RES, ox + (i + 1) * RES
        y0, y1 = oy + j * RES, oy + (j + 1) * RES
        if j == 0 or not comp[j - 1, i]:
            add((x0, y0), (x1, y0))
        if i == W - 1 or not comp[j, i + 1]:
            add((x1, y0), (x1, y1))
        if j == H - 1 or not comp[j + 1, i]:
            add((x1, y1), (x0, y1))
        if i == 0 or not comp[j, i - 1]:
            add((x0, y1), (x0, y0))
    loops, seen = [], set()
    for start in list(nxt):
        for q0 in nxt[start]:
            if (start, q0) in seen:
                continue
            loop = [start]; p, q = start, q0; seen.add((p, q))
            while q != start:
                loop.append(q)
                cand = [r for r in nxt[q] if (q, r) not in seen]
                r = cand[0]; seen.add((q, r)); p, q = q, r
            loops.append(loop)
    def area(l):
        a = np.array(l); return 0.5 * np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1])
    loops.sort(key=lambda l: -abs(area(l)))
    outer = loops[0]
    if area(outer) < 0:
        outer = outer[::-1]
    if len(loops) > 1:
        print(f"# note: {len(loops)-1} hole loop(s) ignored", file=sys.stderr)
    return simplify(np.array(outer), 0.02)


def dp(pts, tol):
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a; n = np.hypot(*ab)
    d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / (n if n > 0 else 1)
    k = int(np.argmax(d))
    if d[k] > tol:
        return np.vstack([dp(pts[:k + 1], tol)[:-1], dp(pts[k:], tol)])
    return np.vstack([a, b])


def simplify(loop, tol):
    # closed loop: start at the vertex farthest from loop[0], simplify both halves
    k = int(np.argmax(np.hypot(*(loop - loop[0]).T)))
    loop = np.vstack([loop[k:], loop[:k]])
    closed = np.vstack([loop, loop[:1]])
    m = int(np.argmax(np.hypot(*(closed - closed[0]).T)))
    A = dp(closed[:m + 1], tol); B = dp(closed[m:], tol)
    return np.vstack([A[:-1], B[:-1]])


def area_of(poly):
    p = np.asarray(poly); return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])


def rooms():
    walls, doors, meshes = parse_wbt()
    occ = rasterise(walls, doors, meshes)
    out = {}
    for name, seed in SEEDS.items():
        comp = component(occ, seed)
        poly = trace_outline(comp)
        out[name] = dict(poly=poly, area=abs(area_of(poly)), cells=int(comp.sum()))
    return out, doors, walls


def boundary_dist(p, q):
    """min distance from each point of p to polyline/polygon q boundary."""
    q = np.asarray(q); qn = np.roll(q, -1, axis=0)
    best = np.full(len(p), np.inf)
    for a, b in zip(q, qn):
        ab = b - a; L2 = max(ab @ ab, 1e-12)
        t = np.clip(((p - a) @ ab) / L2, 0, 1)
        d = np.hypot(*(p - (a + t[:, None] * ab)).T)
        best = np.minimum(best, d)
    return best


def sample_boundary(poly, step=0.02):
    p = np.asarray(poly); pn = np.roll(p, -1, axis=0); pts = []
    for a, b in zip(p, pn):
        L = np.hypot(*(b - a)); n = max(int(L / step), 1)
        t = np.arange(n) / n
        pts.append(a + t[:, None] * (b - a))
    return np.vstack(pts)


def inside(poly, P):
    poly = np.asarray(poly); res = np.zeros(len(P), bool)
    x, y = P[:, 0], P[:, 1]
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        cond = (a[1] > y) != (b[1] > y)
        xi = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1] + 1e-300)
        res ^= cond & (x < xi)
    return res


def iou(p, q, step=0.02):
    allp = np.vstack([p, q])
    x0, y0 = allp.min(0) - 0.05; x1, y1 = allp.max(0) + 0.05
    xs = np.arange(x0, x1, step); ys = np.arange(y0, y1, step)
    X, Y = np.meshgrid(xs, ys); P = np.column_stack([X.ravel(), Y.ravel()])
    A, B = inside(p, P), inside(q, P)
    u = np.count_nonzero(A | B)
    return np.count_nonzero(A & B) / max(u, 1)


def align_and_score(truth, est):
    T = np.asarray(truth); E = np.asarray(est, float)
    cT = T.mean(0); cE = E.mean(0)
    Tb = sample_boundary(T, 0.05)

    def place(th, t):
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        return (E - cE) @ R.T + cT + t

    best = (-1, 0.0, np.zeros(2))
    for th in np.radians(np.arange(0, 360, 1.0)):
        v = iou(T, place(th, np.zeros(2)), 0.05)
        if v > best[0]:
            best = (v, th, np.zeros(2))
    _, th, t = best
    step_t, step_r = 0.1, math.radians(2)
    cur = iou(T, place(th, t), 0.01)
    while step_t > 0.002:
        improved = False
        for dth, dt in [(step_r, 0), (-step_r, 0)] + [(0, d) for d in [np.array(v) * step_t for v in ((1,0),(-1,0),(0,1),(0,-1))]]:
            if isinstance(dt, np.ndarray):
                cand = (th, t + dt)
            else:
                cand = (th + dth, t)
            v = iou(T, place(*cand), 0.01)
            if v > cur:
                cur, th, t = v, cand[0], cand[1]; improved = True
        if not improved:
            step_t *= 0.5; step_r *= 0.5
    A = place(th, t)
    d1 = boundary_dist(sample_boundary(A, 0.02), T)
    d2 = boundary_dist(Tb, A)
    return dict(theta_deg=math.degrees(th), t=t, iou=cur, area_truth=abs(area_of(T)),
                area_est=abs(area_of(A)), hausdorff=max(d1.max(), d2.max()),
                aligned=A)


def main(argv):
    res, doors, walls = rooms()
    if len(argv) >= 3 and argv[0] == "--score":
        name = argv[1]
        rows = []
        with open(argv[2], newline="") as f:
            for r in csv.reader(f):
                try:
                    rows.append((float(r[0]), float(r[1])))
                except (ValueError, IndexError):
                    continue
        r = align_and_score(res[name]["poly"], rows)
        print(f"{name}: IoU={r['iou']:.3f}  area_truth={r['area_truth']:.2f} m2  "
              f"area_est={r['area_est']:.2f} m2  max_boundary_dist={r['hausdorff']:.3f} m  "
              f"(alignment: rot {r['theta_deg']:.1f} deg, t=({r['t'][0]:.3f},{r['t'][1]:.3f}))")
        return
    print("# wall rectangles:", len(walls), " doors:", {k: v["c"] for k, v in doors.items()})
    for name, doors_on in (("room_3", ("DOOR_1", "DOOR_2")), ("room_2", ("DOOR_0", "DOOR_2"))):
        p = res[name]["poly"]
        print(f"{name}: area={res[name]['area']:.3f} m2, {len(p)} vertices (world, CCW):")
        print("  " + " ".join(f"({x:.3f},{y:.3f})" for x, y in p))
        for d in doors_on:
            dist = boundary_dist(np.array([doors[d]["c"]]), p)[0]
            print(f"  {d} centre distance to boundary: {dist:.3f} m")


if __name__ == "__main__":
    main(sys.argv[1:])
