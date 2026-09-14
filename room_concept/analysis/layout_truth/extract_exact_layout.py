#!/usr/bin/env python3
"""Extract the EXACT interior wall polygon of the Webots `piso` apartment from the
collision mesh that the simulated LiDAR actually hits, and compare it against the two
other layouts of the same place that live in this repo.

Sources
-------
exact   webots-shadow/worlds/piso/meshes/wall.dae  -- the `boundingObject` of Solid "Wall"
        in piso.wbt.  It is a THICK solid (buffer(air, 0.11) - air), so the polygon we
        want is its INNER ring, not the outer hull.
traced  room_concept/datasets/webots_piso/truth.txt -- hand-traced from piso_layout.svg
        on a 10 cm grid at 60 px/m.
agent   active_inference/layouts/apartamento_layout.svg -- what the LIVE agent loads in
        `given` mode (config [Scenario.apartamento] RoomLayoutSvg), via SvgRoomLoader
        with flip_y=false, mirror_x=true.

Everything is reported in the WEBOTS WORLD frame (x,y on the floor plane, z up).
"""
import xml.etree.ElementTree as ET
import re, sys, os
import numpy as np
from shapely.geometry import Polygon, LineString, Point

NS = "{http://www.collada.org/2005/11/COLLADASchema}"
WB   = "/home/pbustos/robocomp/components/webots-shadow/worlds/piso"
RC   = "/home/pbustos/robocomp/components/active_inference/room_concept"
LAY  = "/home/pbustos/robocomp/components/active_inference/layouts"

# ---------------------------------------------------------------- 1. the exact mesh

def dae_positions(path):
    mesh = list(ET.parse(path).getroot().iter(NS + "geometry"))[0].find(NS + "mesh")
    for s in mesh.findall(NS + "source"):
        if s.attrib["id"].endswith("-positions"):
            return np.fromstring(s.find(NS + "float_array").text, sep=" ").reshape(-1, 3)
    raise SystemExit("no positions in " + path)


def exact_interior_ring():
    """The inner ring of the thick wall solid.

    The mesh has no ring structure of its own (it is a triangle soup), so we recover the
    inner ring by RECONSTRUCTING the solid the way thicken_walls.py built it and checking
    that the reconstruction reproduces the mesh vertex set exactly.  The candidate loop is
    the one thicken_walls.py records as the source; we VERIFY it rather than trust it.
    """
    LOOP = [(-0.000, 6.892), (-0.215, 0.000), (-3.236, 0.078), (-3.232, 0.792),
            (-4.222, 0.820), (-4.184, 2.506), (-4.329, 2.497), (-4.331, 0.814),
            (-5.207, 0.822), (-5.184, 0.068), (-8.262, 0.178), (-8.508, 0.437),
            (-8.392, 7.090), (-5.932, 7.102), (-5.920, 7.163), (-6.336, 7.169),
            (-6.330, 8.026), (-5.884, 8.007), (-5.871, 9.256), (-4.196, 9.231),
            (-4.296, 5.022), (-4.212, 5.012), (-4.172, 7.166), (-0.222, 7.166)]
    P = dae_positions(os.path.join(WB, "meshes", "wall.dae"))
    xy = np.unique(np.round(P[:, :2], 6), axis=0)
    L = np.array(LOOP)
    # (a) every loop vertex is an exact mesh vertex
    dmax = max(np.linalg.norm(xy - p, axis=1).min() for p in L)
    # (b) every remaining mesh vertex lies OUTSIDE the loop (it belongs to the outer ring
    #     or to a door reveal) -- i.e. the loop really is the innermost ring.
    poly = Polygon(L)
    rest = [q for q in xy if np.linalg.norm(L - q, axis=1).min() > 1e-6]
    # a door jamb sits exactly ON the wall face, so "inside" must mean inside AND off the
    # boundary by more than float noise.
    inside = [q for q in rest if poly.contains(Point(*q))
              and poly.exterior.distance(Point(*q)) > 1e-5]
    # (c) every non-loop vertex sits at the wall thickness outside the loop (or further, at a
    #     mitred corner), or exactly on it (a door jamb).  Nothing sits at some other depth.
    dep = np.array([poly.exterior.distance(Point(*q)) for q in rest])
    print(f"[verify] wall.dae: {len(P)} verts, {len(xy)} distinct xy")
    print(f"[verify] all 24 loop vertices present in mesh, max mismatch {dmax:.3e} m")
    print(f"[verify] mesh vertices strictly inside the loop: {len(inside)}  (must be 0)")
    print(f"[verify] the other {len(rest)} vertices lie OUTSIDE at depth "
          f"{dep.min():.3f}..{dep.max():.3f} m (wall thickness 0.110, mitres deeper, "
          f"{int((dep<1e-5).sum())} door jambs at 0)")
    assert dmax < 1e-6 and len(inside) == 0
    return L


def ccw(pts):
    pts = np.asarray(pts, float)
    a = np.cross(pts - pts[0], np.roll(pts, -1, axis=0) - pts[0]).sum()
    return pts if a > 0 else pts[::-1]

# ---------------------------------------------------------------- 2. the other two

def read_ws_layout(path):
    line = open(path).read().strip().split("\n")[0]
    name, pts = line.split(";", 1)
    return name, np.array([[float(v) for v in tok.split(",")] for tok in pts.split()])


def read_agent_svg(path):
    txt = open(path).read()
    m = re.search(r'id="room_contour"[^>]*points="([^"]+)"', txt, re.S)
    if m is None:
        m = re.search(r'points="([^"]+)"[^>]*id="room_contour"', txt, re.S)
    pts = np.array([[float(v) for v in t.split(",")] for t in m.group(1).split()])
    pts[:, 0] *= -1.0          # SvgRoomLoader(mirror_x=True, flip_y=False)
    return pts[::-1]           # the loader reverses after the mirror

# ---------------------------------------------------------------- 3. comparison

def boundary_dist(poly, pts):
    b = poly.exterior
    return np.array([b.distance(Point(*p)) for p in pts])


def sample_edges(pts, step=0.02):
    """Dense samples along a closed polygon, with the index of the edge each came from."""
    out, eid = [], []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        L = np.linalg.norm(b - a)
        k = max(2, int(L / step))
        for t in np.linspace(0, 1, k, endpoint=False):
            out.append(a + (b - a) * t); eid.append(i)
    return np.array(out), np.array(eid)


def signed_offsets(ref, other):
    """For every edge of `ref`, the distribution of distance from points sampled on that
    edge to the boundary of `other`, signed + when `other` is OUTSIDE ref (ref too small)."""
    P = Polygon(other)
    s, eid = sample_edges(np.asarray(ref, float))
    d = boundary_dist(P, s)
    sign = np.array([1.0 if P.contains(Point(*p)) else -1.0 for p in s])
    return s, eid, d * sign


def dihedral_align(src, ref):
    """Best of the 8 axis transforms + free translation, then a free rigid (+reflection)
    refinement, minimising sum of squared distance from src vertices to ref's boundary."""
    from scipy.optimize import minimize
    R = Polygon(ref)
    def cost(p, S):
        c, s = np.cos(p[2]), np.sin(p[2])
        T = S @ np.array([[c, -s], [s, c]]).T + p[:2]
        return float((boundary_dist(R, T) ** 2).sum())
    best = None
    for mir in (False, True):
        S0 = src.copy()
        if mir: S0[:, 0] *= -1
        for k in range(4):
            th = k * np.pi / 2
            c, s = np.cos(th), np.sin(th)
            S1 = S0 @ np.array([[c, -s], [s, c]]).T
            t0 = ref.mean(0) - S1.mean(0)
            r = minimize(cost, [t0[0], t0[1], 0.0], args=(S0,),
                         method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-12, maxiter=20000, maxfev=20000))
            # re-seed with the quadrant rotation folded in
            r2 = minimize(cost, [t0[0], t0[1], th], args=(S0,), method="Nelder-Mead",
                          options=dict(xatol=1e-7, fatol=1e-12, maxiter=20000, maxfev=20000))
            for rr in (r, r2):
                if best is None or rr.fun < best[0]:
                    best = (rr.fun, mir, rr.x.copy())
    _, mir, p = best
    S = src.copy()
    if mir: S[:, 0] *= -1
    c, s = np.cos(p[2]), np.sin(p[2])
    return S @ np.array([[c, -s], [s, c]]).T + p[:2], mir, p


def report(tag, ref, other):
    print(f"\n=== {tag} ===")
    Pr, Po = Polygon(ref), Polygon(other)
    print(f"  area  exact {Pr.area:8.4f} m2   other {Po.area:8.4f} m2   "
          f"delta {Po.area-Pr.area:+.4f} m2 ({100*(Po.area-Pr.area)/Pr.area:+.2f}%)")
    print(f"  IoU   {Pr.intersection(Po).area/Pr.union(Po).area:.4f}")
    bbr, bbo = np.ptp(ref,0), np.ptp(other,0)
    print(f"  bbox  exact {bbr[0]:.3f} x {bbr[1]:.3f}   other {bbo[0]:.3f} x {bbo[1]:.3f}")
    _, eid, d = signed_offsets(ref, other)
    ad = np.abs(d)
    print(f"  per-SAMPLE |offset| over the exact boundary (2 cm sampling, n={len(d)}):")
    print(f"     mean {ad.mean():.4f}  median {np.median(ad):.4f}  p90 {np.percentile(ad,90):.4f}"
          f"  p95 {np.percentile(ad,95):.4f}  max {ad.max():.4f}   RMS {np.sqrt((d**2).mean()):.4f}")
    print(f"  per-WALL mean signed offset (+ = other lies OUTSIDE the exact wall):")
    print(f"     {'#':>3} {'from':>18} {'to':>18} {'len':>6} {'mean':>8} {'|max|':>7}")
    rows=[]
    for i in range(len(ref)):
        m = eid == i
        if not m.any(): continue
        a, b = ref[i], ref[(i+1) % len(ref)]
        L = np.linalg.norm(b-a)
        rows.append((i, a, b, L, d[m].mean(), np.abs(d[m]).max()))
    for i,a,b,L,mu,mx in rows:
        print(f"     {i:3d} ({a[0]:7.3f},{a[1]:6.3f}) ({b[0]:7.3f},{b[1]:6.3f}) {L:6.3f} "
              f"{mu:+8.4f} {mx:7.4f}")
    wl = np.array([r[3] for r in rows]); wm = np.array([abs(r[4]) for r in rows])
    print(f"  length-weighted mean |per-wall offset| = {(wl*wm).sum()/wl.sum():.4f} m")
    return d


def main():
    L = ccw(exact_interior_ring())
    out = os.path.join(RC, "datasets", "webots_piso", "truth_exact.txt")
    with open(out, "w") as f:
        f.write("piso_exact;" + " ".join(f"{x:.4f},{y:.4f}" for x, y in L) + "\n")
    print(f"\n[write] {out}  ({len(L)} vertices, CCW, Webots world frame, area "
          f"{Polygon(L).area:.4f} m2)")

    # agent layout -- already in the webots frame after the loader's mirror_x
    ag = read_agent_svg(os.path.join(LAY, "apartamento_layout.svg"))
    report("EXACT mesh  vs  AGENT layout (apartamento_layout.svg, as the agent reads it)", L, ag)

    # traced truth.txt -- different frame, needs alignment
    _, tr = read_ws_layout(os.path.join(RC, "datasets", "webots_piso", "truth.txt"))
    tra, mir, p = dihedral_align(tr, L)
    print(f"\n[align] truth.txt -> webots: mirror_x={mir}  theta={np.degrees(p[2]):+.4f} deg  "
          f"t=({p[0]:+.4f},{p[1]:+.4f})")
    report("EXACT mesh  vs  TRACED truth.txt (rigidly aligned, 10 cm grid)", L, tra)
    # also the agent layout vs the traced one, for completeness
    aga, _, _ = dihedral_align(tr, ag)
    report("AGENT layout  vs  TRACED truth.txt (rigidly aligned)", ag, aga)

    np.save("/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/c1e48398-243f-4bda-b915-064cb4be9944/scratchpad/exact.npy", L)
    np.save("/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/c1e48398-243f-4bda-b915-064cb4be9944/scratchpad/agent.npy", ag)
    np.save("/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/c1e48398-243f-4bda-b915-064cb4be9944/scratchpad/traced.npy", tra)

main()
