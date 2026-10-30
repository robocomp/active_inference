#!/usr/bin/env python3
"""
adoption_mock.py -- a seconds-fast Python MODEL of room_concept's room-adoption relocalisation.

WHY
  Live tours (Webots + 7 agents, ~10 min) are too slow to debug one decision.  In tour recog_proto9 the robot crossed
  DOOR_2 from room_3 into room_2 (a room remembered only through DOOR_1).  ltsm proposed "this is room_2, robot here";
  room_concept's adoption search instead jumped 4.96 m / 179 deg to a rival pose: the robot standing in the ONE doorway
  room_2's memory knows (DOOR_1).  Hypothesis: the scan through the unknown DOOR_2 hole misfits K's outline at the true
  pose, while the rival pose explains the through-door returns with the known DOOR_1 aperture, so the scan likelihood
  beats the prior's weak uniform floor (prior_wrong = 0.1).  Commit 7ba0ad18 (`Params::prior_only`) restricts the
  adoption search to the prior's own modes.  This tool lets us ask such questions in seconds.

!! CAVEAT -- READ THIS BEFORE QUOTING A NUMBER !!
  This is a MODEL of the agent's likelihood, not the agent (project rule: "the bench must run the agent's code").
  It is for finding hypotheses fast; any conclusion is confirmed in the C++ `gn_selftest late_recall` section before a
  fix ships.

WHAT IT DOES
  1. Parses /home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt (a small VRML parser, nothing
     hand-typed): `Wall` protos -> 2-D rectangles; the `Solid "Wall"` whose boundingObject is meshes/wall.dae (rooms 1 and
     3 are built from that mesh, not from Wall protos) -> the mesh sliced at the LiDAR height; `Door` protos -> two jamb
     rectangles each, the leaf is OMITTED (an open hole of size.y - 2*frameSize.y = 0.9 m).  Furniture is NOT an
     obstacle (the reloc band is a wall band).  Nested Group/Solid/Transform translation+rotation+scale are composed.
  2. Simulated 2-D LiDAR: 360 beams, 10 m, sigma 1 cm, exact ray-casting against the segments; beams through an open
     door continue into the next room.  Misses (> 10 m) are dropped, as the agent drops them.
  3. Remembered room K = room_2 as ltsm remembers it.  DERIVATION: rasterise every world segment (2 cm cells), seal BOTH
     faces of DOOR_1 and DOOR_2 across the jambs (so the polygon edge at a door is the flat wall face -- what a
     lidar-built layout holds), flood-fill the free cells from a seed inside room_2 (-5.0, 11.5), take the outline of
     that region (matplotlib/contourpy) and simplify it (Douglas-Peucker, 2.5 cm).  DOOR_2 is therefore solid wall in K
     and there is NO aperture for it; DOOR_1 is the only Aperture, seated on K's nearest edge exactly as
     DoorApertures::seat_in_room does.  Coordinates are WORLD coordinates; the frame does not matter for this question
     (every quantity is a pose relative to K's own outline).
  4. The relocaliser's scoring, ported from room_concept/src/reloc_search.{h,cpp} (+ wall_segmenter.cpp for the lines):
     PointTerm (per-return robust SDF likelihood), LikelihoodRaster, the doorway Cells / Aperture / BeyondLines mixture
     (cell_mixture, beyond_extent), structural yaws, lattice maxima, IRLS Gauss-Newton polish with EM sigma annealing
     and the per-wall common-mode marginalisation, merge by chi^2, Laplace evidence  l* + 1/2 log|S|,
     log_prior_modes with prior_wrong, and prior_only.  Parameters: etc/config.toml (RelocOutlierFrac .30,
     RelocWallOffsetSigma .03, RelocBodyClearance .30, SigmaSdf .15) and the Params defaults (max_points 150, max_yaws 8,
     max_seeds 24, max_modes 4, polish_iters 40, chi2_merge 11.34, list_wrong .1, opening_cell_m 1, beyond_extent 20).
  5. Scenarios (GT poses): in DOOR_2's aperture facing into room_2; 0.5 m and 1.0 m inside room_2 (measured from K's
     wall face at DOOR_2); a control in the middle of room_2 far from both doors.  Prior = ONE mode at GT +
     (0.1, -0.1, 3 deg), S = diag(.3^2, .3^2, .15^2).  Search (a) "as it was" (prior_only False, prior_wrong .1) and
     (b) prior_only True.  Seed pose handed to the search = the prior mean (seed_reloc_prior moved the tracker there).

SIMPLIFICATIONS / DEVIATIONS (the likelihood itself is a literal port)
  * float64 numpy instead of float32 C++; vectorised over points, so results differ at the 1e-6 level.
  * RNG: numpy default_rng instead of std::mt19937 -- the RANSAC lines (BeyondLines, structural yaws) can differ from
    the agent's by a hair on the same scan; the C++ seeds are fixed (20261021 / ep.ts) so they differ here too.
  * Polish: the SAME IRLS Gauss-Newton (analytic Jacobian J = (u, u x r), LM damping, EM sigma anneal, Woodbury
    common-mode) -- not a finite-difference one.
  * The scan is the 2-D lidar plane; the agent's "wall band" is a projection of ~10 rings (clutter/ceiling excluded here).
  * The door aperture's belief inputs are assumed: prior_closed = 0.5 / p_open unmeasured (-> p_shut = 1-(1-hazard)*0.5)
    / sigma_along 0.10 m (door_concept's real values for DOOR_1 in recog_proto9 are not known to this tool).
  * The mock stops at the search's modes (+ prior); the agent then runs mode_posterior / msprt_decide against the
    held pose and `explains_like_tracking` -- NOT modelled.
  * The yaw convention is "robot +x axis heading in the room frame" (q = R(theta) p + t, as reloc_search).  The agent's
    controller convention (theta = forward - 90 deg) is irrelevant to the likelihood.

USAGE
  python3 adoption_mock.py                      # the four scenarios, PNGs to the default scratch dir
  python3 adoption_mock.py --pose -2.2 10.77 3.14159   # one custom scenario (yaw in RADIANS)
  python3 adoption_mock.py --out DIR --seed 3
"""
import argparse
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import contourpy  # noqa: E402

WORLD = "/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt"
DEFAULT_OUT = ("/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/"
               "ea929cb0-5597-4d47-8f24-8055f99877f1/scratchpad/adoption_mock/")
TWO_PI = 2.0 * math.pi
INF = float("inf")


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  1. WORLD: a small VRML parser + geometry extraction
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass
class Node:
    type: str
    name: str = ""
    fields: dict = field(default_factory=dict)


_TOK = re.compile(r'"[^"]*"|#[^\n]*|[{}\[\]]|[^\s{}\[\],"#]+')
_NUM = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$')


def _tokens(text):
    return [t for t in _TOK.findall(text) if not t.startswith("#")]


class _Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self, k=0):
        return self.t[self.i + k] if self.i + k < len(self.t) else None

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def is_node_start(self):
        t = self.peek()
        if t in ("DEF", "USE"):
            return True
        return t is not None and t[0].isupper() and self.peek(1) == "{" and t not in ("TRUE", "FALSE")

    def node(self):
        name = ""
        if self.peek() == "DEF":
            self.next()
            name = self.next()
        if self.peek() == "USE":
            self.next()
            return Node("USE", self.next())
        typ = self.next()
        assert self.next() == "{", typ
        n = Node(typ, name)
        while self.peek() != "}":
            fname = self.next()
            if fname == "hidden":            # `hidden translation_4 ...` -- a Webots solver note, not a field we use
                fname = "hidden_" + self.next()
            n.fields[fname] = self.value()
        self.next()
        return n

    def value(self):
        t = self.peek()
        if t == "[":
            self.next()
            items = []
            while self.peek() != "]":
                if self.is_node_start():
                    items.append(self.node())
                else:
                    v = self.value()
                    items.extend(v if isinstance(v, list) else [v])
            self.next()
            return items
        if self.is_node_start():
            return self.node()
        if t is not None and t.startswith('"'):
            return self.next().strip('"')
        if t == "IS":
            self.next()
            self.next()
            return None
        if t in ("TRUE", "FALSE", "NULL"):
            return {"TRUE": True, "FALSE": False, "NULL": None}[self.next()]
        nums = []
        while self.peek() is not None and _NUM.match(self.peek()):
            nums.append(float(self.next()))
        if not nums:                          # an unknown bareword: consume it so the parse cannot loop
            return self.next()
        return nums if len(nums) > 1 else nums[0]

    def top(self):
        out = []
        while self.peek() is not None:
            if self.peek() in ("EXTERNPROTO", "IMPORTABLE"):
                self.next()
                self.next()
                continue
            if self.peek() == "PROTO":
                raise ValueError("inline PROTO not supported")
            out.append(self.node())
        return out


def _rot(axis_angle):
    x, y, z, a = axis_angle
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _local(node):
    M = np.eye(4)
    tr = node.fields.get("translation")
    if isinstance(tr, list) and len(tr) == 3:
        M[:3, 3] = tr
    ro = node.fields.get("rotation")
    if isinstance(ro, list) and len(ro) == 4:
        M[:3, :3] = _rot(ro)
    sc = node.fields.get("scale")
    if isinstance(sc, (int, float)):
        M[:3, :3] = M[:3, :3] * sc
    elif isinstance(sc, list) and len(sc) == 3:
        M[:3, :3] = M[:3, :3] @ np.diag(sc)
    return M


def _vec(v, n, default):
    return list(v) if isinstance(v, list) and len(v) == n else list(default)


@dataclass
class World:
    segments: list = field(default_factory=list)   # (p, q) 2-D world points
    walls: list = field(default_factory=list)      # (name, corners 4x2)
    doors: dict = field(default_factory=dict)      # DEF name -> dict(centre, tangent, normal, width, jambs, ...)
    meshes: list = field(default_factory=list)     # (name, n_segments)


def _slice_collada(path, M, zc):
    txt = open(path).read()
    txt = re.sub(r'\sxmlns="[^"]+"', "", txt, count=1)
    root = ET.fromstring(txt)
    mesh = root.find(".//mesh")
    vert = mesh.find("vertices")
    pos_id = [i.get("source")[1:] for i in vert.findall("input") if i.get("semantic") == "POSITION"][0]
    src = [s for s in mesh.findall("source") if s.get("id") == pos_id][0]
    P = np.array(src.find("float_array").text.split(), float).reshape(-1, 3)
    P = (M[:3, :3] @ P.T).T + M[:3, 3]
    segs = []
    for tri in list(mesh.findall("triangles")) + list(mesh.findall("polylist")):
        ins = tri.findall("input")
        stride = max(int(i.get("offset")) for i in ins) + 1
        voff = [int(i.get("offset")) for i in ins if i.get("semantic") == "VERTEX"][0]
        idx = np.array(tri.find("p").text.split(), int).reshape(-1, stride)[:, voff].reshape(-1, 3)
        for a, b, c in idx:
            pts = [P[a], P[b], P[c]]
            cut = []
            for u, v in ((0, 1), (1, 2), (2, 0)):
                zu, zv = pts[u][2] - zc, pts[v][2] - zc
                if (zu < 0) != (zv < 0):
                    t = zu / (zu - zv)
                    cut.append(pts[u][:2] + t * (pts[v][:2] - pts[u][:2]))
            if len(cut) == 2 and np.linalg.norm(cut[0] - cut[1]) > 1e-9:
                segs.append((cut[0], cut[1]))
    return segs


def parse_world(path=WORLD, lidar_z=0.5):
    nodes = _Parser(_tokens(open(path).read())).top()
    W = World()
    base = os.path.dirname(path)

    def add_rect(M, cx, cy, sx, sy, name):
        loc = np.array([[cx - sx / 2, cy - sy / 2, 0], [cx + sx / 2, cy - sy / 2, 0],
                        [cx + sx / 2, cy + sy / 2, 0], [cx - sx / 2, cy + sy / 2, 0]], float)
        w = (M[:3, :3] @ loc.T).T[:, :2] + M[:2, 3]
        for k in range(4):
            W.segments.append((w[k], w[(k + 1) % 4]))
        return w

    def walk(n, M):
        M = M @ _local(n)
        if n.type == "Wall":
            sz = _vec(n.fields.get("size"), 3, (1, 1, 1))
            z0 = M[2, 3]
            if z0 <= lidar_z <= z0 + sz[2]:       # Wall's origin is its base; the lintel at z=2.1 is above the band
                W.walls.append((n.fields.get("name", "wall"), add_rect(M, 0, 0, sz[0], sz[1], n.fields.get("name"))))
        elif n.type == "Door":
            sz = _vec(n.fields.get("size"), 3, (0.2, 1, 2.4))
            fs = _vec(n.fields.get("frameSize"), 3, (0.05, 0.05, 0.05))
            fh = n.fields.get("frameHeight", 2.0)
            jambs = []
            if fh > lidar_z:
                for sgn in (-1, 1):
                    jambs.append(add_rect(M, 0, sgn * (0.5 * sz[1] - 0.5 * fs[1]), 2 * fs[0] + sz[0], fs[1], "jamb"))
            W.doors[n.name or n.fields.get("name", "door")] = dict(
                centre=M[:2, 3].copy(), normal=M[:2, 0].copy(), tangent=M[:2, 1].copy(),
                width=sz[1] - 2 * fs[1], jamb_half=0.5 * (2 * fs[0] + sz[0]), jamb_in=0.5 * sz[1] - fs[1],
                name=n.fields.get("name", ""))
        elif n.type == "Solid":
            bo = n.fields.get("boundingObject")
            if isinstance(bo, Node) and bo.type == "Mesh":
                urls = bo.fields.get("url", [])
                urls = urls if isinstance(urls, list) else [urls]
                for u in urls:
                    if isinstance(u, str) and os.path.basename(u) == "wall.dae":
                        segs = _slice_collada(os.path.join(base, u), M, lidar_z)
                        W.segments.extend(segs)
                        W.meshes.append((u, len(segs)))
        ch = n.fields.get("children")
        for c in (ch if isinstance(ch, list) else []):
            if isinstance(c, Node):
                walk(c, M)

    for n in nodes:
        walk(n, np.eye(4))
    return W


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  2. SIMULATED LIDAR
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def raycast(segs, origin, angles, max_range):
    """Range of the first segment each ray hits (inf when none within max_range)."""
    P = np.array([s[0] for s in segs])
    S = np.array([s[1] for s in segs]) - P
    d = np.stack([np.cos(angles), np.sin(angles)], 1)                       # (B,2)
    rxs = d[:, None, 0] * S[None, :, 1] - d[:, None, 1] * S[None, :, 0]     # (B,N)
    qp = P - origin
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[None, :, 0] * S[None, :, 1] - qp[None, :, 1] * S[None, :, 0]) / rxs
        u = (qp[None, :, 0] * d[:, None, 1] - qp[None, :, 1] * d[:, None, 0]) / rxs
    ok = (np.abs(rxs) > 1e-12) & (t > 1e-6) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf)
    r = t.min(1)
    return np.where(r <= max_range, r, np.inf)


def lidar_scan(world, pose, rng, n_beams=360, max_range=10.0, sigma=0.01):
    """Robot-frame points (n,2) of one scan from the ground-truth `pose` (x, y, yaw)."""
    a = np.arange(n_beams) * TWO_PI / n_beams
    r = raycast(world.segments, np.array(pose[:2]), pose[2] + a, max_range)
    hit = np.isfinite(r)
    r = r[hit] + rng.normal(0, sigma, hit.sum())
    return np.stack([r * np.cos(a[hit]), r * np.sin(a[hit])], 1)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  3. REMEMBERED ROOM K
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _dp(pts, tol):
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / L if L > 1e-12 \
        else np.linalg.norm(pts - a, axis=1)
    k = int(np.argmax(d))
    if d[k] > tol:
        return np.vstack([_dp(pts[:k + 1], tol)[:-1], _dp(pts[k:], tol)])
    return np.vstack([a, b])


def build_memory_room(world, seed_xy=(-5.0, 11.5), res=0.02, dp_tol=0.025):
    """Room K: free-space region of the seed with every door sealed on both faces -> simplified CCW polygon."""
    segs = list(world.segments)
    for d in world.doors.values():
        c, n, t = d["centre"], d["normal"], d["tangent"]
        for s in (-d["jamb_half"], d["jamb_half"]):
            e = d["jamb_in"] + 0.02
            segs.append((c + s * n - e * t, c + s * n + e * t))
    P = np.array([s[0] for s in segs]); Q = np.array([s[1] for s in segs])
    lo = np.minimum(P, Q).min(0) - 0.5
    hi = np.maximum(P, Q).max(0) + 0.5
    nx, ny = int(np.ceil((hi[0] - lo[0]) / res)), int(np.ceil((hi[1] - lo[1]) / res))
    occ = np.zeros((ny, nx), bool)
    for p, q in zip(P, Q):
        n = max(2, int(np.linalg.norm(q - p) / (res / 4)) + 1)
        pts = p + (q - p) * np.linspace(0, 1, n)[:, None]
        ix = np.clip(((pts[:, 0] - lo[0]) / res).astype(int), 0, nx - 1)
        iy = np.clip(((pts[:, 1] - lo[1]) / res).astype(int), 0, ny - 1)
        occ[iy, ix] = True
    sx, sy = int((seed_xy[0] - lo[0]) / res), int((seed_xy[1] - lo[1]) / res)
    assert not occ[sy, sx], "seed is inside a wall"
    free = np.zeros_like(occ)
    stack = [(sy, sx)]
    free[sy, sx] = True
    while stack:                                   # 4-connected flood fill
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and not occ[yy, xx] and not free[yy, xx]:
                free[yy, xx] = True
                stack.append((yy, xx))
    pad = np.pad(free.astype(float), 1)
    gen = contourpy.contour_generator(z=pad)
    lines = gen.lines(0.5)
    ln = max(lines, key=len)
    xy = np.stack([lo[0] + (ln[:, 0] - 1 + 0.5) * res, lo[1] + (ln[:, 1] - 1 + 0.5) * res], 1)
    xy = xy[:-1] if np.allclose(xy[0], xy[-1]) else xy
    # Douglas-Peucker on the closed ring: split at the two mutually farthest vertices
    i0 = 0
    i1 = int(np.argmax(np.linalg.norm(xy - xy[i0], axis=1)))
    ring = np.vstack([xy[i0:i1 + 1], ]), np.vstack([xy[i1:], xy[:i0 + 1]])
    poly = np.vstack([_dp(ring[0], dp_tol)[:-1], _dp(ring[1], dp_tol)[:-1]])
    if polygon_area_signed(poly) < 0:
        poly = poly[::-1]
    return poly, free, lo, res


def polygon_area_signed(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def seat_in_room(a, b, poly):
    """DoorApertures::seat_in_room: project the mouth onto the nearest polygon edge; outward normal; recess."""
    c = 0.5 * (a + b)
    best, e0, e1 = INF, None, None
    for i in range(len(poly)):
        p0, p1 = poly[i], poly[(i + 1) % len(poly)]
        e = p1 - p0
        L2 = e @ e
        if L2 < 1e-12:
            continue
        t = np.clip((c - p0) @ e / L2, 0, 1)
        dist = np.linalg.norm(c - (p0 + t * e))
        if dist < best:
            best, e0, e1 = dist, p0, p1
    u = (e1 - e0) / np.linalg.norm(e1 - e0)
    n_out = (1.0 if polygon_area_signed(poly) > 0 else -1.0) * np.array([u[1], -u[0]])
    onto = lambda q: e0 + ((q - e0) @ u) * u  # noqa: E731
    return onto(a), onto(b), n_out, float(np.linalg.norm(c - onto(c)))


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  4. THE RELOCALISER (port of reloc_search.{h,cpp}, wall_segmenter.cpp)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass
class Aperture:
    a: np.ndarray
    b: np.ndarray
    n_out: np.ndarray
    open: float          # 1 - prior_closed
    p_shut: float
    sigma_along: float


@dataclass
class PriorMode:
    mu: np.ndarray
    S: np.ndarray
    w: float = 1.0


@dataclass
class Params:
    sigma_sdf: float = 0.15
    outlier_frac: float = 0.30
    raster_res: float = 0.05
    lattice_step: float = 0.10
    body_clearance: float = 0.30
    max_points: int = 150
    max_yaws: int = 8
    fallback_yaws: int = 36
    max_seeds: int = 24
    max_modes: int = 4
    polish_iters: int = 40
    wall_offset_sigma: float = 0.03
    chi2_merge: float = 11.34
    apertures: list = field(default_factory=list)
    opening_cell_m: float = 1.0
    list_wrong: float = 0.1
    beyond_extent: float = 20.0
    prior_modes: list = field(default_factory=list)
    prior_wrong: float = 0.1
    prior_only: bool = False


class PointTerm:
    def __init__(self, p, extent):
        self.log_in_norm = math.log((1 - p.outlier_frac) * 2 / (math.sqrt(TWO_PI) * p.sigma_sdf))
        self.log_out = math.log(p.outlier_frac / max(extent, 1e-3))
        self.inv_2s2 = 1.0 / (2 * p.sigma_sdf ** 2)

    def ll(self, d):
        li = self.log_in_norm - d * d * self.inv_2s2
        return np.logaddexp(li, self.log_out), li


def polygon_extent(poly):
    return float(np.linalg.norm(poly.max(0) - poly.min(0)))


def closest_on_polygon(poly, q):
    """(closest point (n,2), edge index (n,)) for points q (n,2)."""
    A = poly
    B = np.roll(poly, -1, 0)
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = np.clip(((q[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    C = A[None] + t[..., None] * AB[None]
    d2 = ((q[:, None, :] - C) ** 2).sum(2)
    e = d2.argmin(1)
    return C[np.arange(len(q)), e], e


def polygon_distance(poly, q):
    c, _ = closest_on_polygon(poly, q)
    return np.linalg.norm(q - c, axis=1)


def inside_polygon(poly, q):
    inn = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        a, b = poly[i], poly[j]
        if (a[1] > q[1]) != (b[1] > q[1]) and q[0] < (b[0] - a[0]) * (q[1] - a[1]) / (b[1] - a[1]) + a[0]:
            inn = not inn
        j = i
    return inn


def polygon_area(poly):
    return abs(polygon_area_signed(poly))


# ── doorway cells (build_cells) ─────────────────────────────────────────────────────────────────────────────────
@dataclass
class Cells:
    first: list
    count: list
    leaf: np.ndarray
    pi: list
    q: list


def _Phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def build_cells(poly, p):
    ne = len(poly)
    first, count, cell_len, cell_edge, cell_k = [], [], [], [], []
    cell = max(p.opening_cell_m, 1e-2)
    for e in range(ne):
        L = np.linalg.norm(poly[(e + 1) % ne] - poly[e])
        n = max(1, int(math.ceil(L / cell)))
        first.append(len(cell_len)); count.append(n)
        for k in range(n):
            cell_len.append(L / n); cell_edge.append(e); cell_k.append(k)
    N = len(cell_len)
    leaf = np.zeros(N); leaf_w = np.zeros(N)
    pis, qs = [], []
    w = min(max(p.list_wrong, 0), 1)
    for ap in p.apertures:
        m = 0.5 * (ap.a + ap.b)
        best, be = INF, 0
        for e in range(ne):
            e0 = poly[e]; ab = poly[(e + 1) % ne] - e0
            L2 = ab @ ab
            if L2 < 1e-12:
                continue
            t = min(max((m - e0) @ ab / L2, 0), 1)
            d = np.linalg.norm(m - e0 - t * ab)
            if d < best:
                best, be = d, e
        e0 = poly[be]; u = poly[(be + 1) % ne] - e0; u = u / np.linalg.norm(u)
        ta, tb = (ap.a - e0) @ u, (ap.b - e0) @ u
        t0, t1 = min(ta, tb), max(ta, tb)
        cover = np.zeros(N); csum = 0.0
        for ci in range(N):
            if cell_edge[ci] != be:
                continue
            cv = 0.0
            for j in range(8):
                sx = (cell_k[ci] + (j + 0.5) / 8) * cell_len[ci]
                if ap.sigma_along > 1e-4:
                    cv += _Phi((t1 - sx) / ap.sigma_along) - _Phi((t0 - sx) / ap.sigma_along)
                else:
                    cv += 1.0 if t0 <= sx <= t1 else 0.0
            cover[ci] = cv / 8; csum += cover[ci]
        pi = ((1 - w) * cover / csum if csum > 0 else 0.0) + (w if csum > 0 else 1.0) / N
        leaf += cover * min(max(1 - ap.open, 0), 1)
        leaf_w += cover
        pis.append(np.asarray(pi) * np.ones(N)); qs.append(min(max(ap.p_shut, 0), 1))
    leaf = np.where(leaf_w > 0, leaf / np.where(leaf_w > 0, leaf_w, 1), 0.0)
    return Cells(first, count, leaf, pis, qs)


def exit_cells(poly, cells, origin, origin_inside, q):
    """Per point: the boundary cell its ray reaches first (-1 none), and whether the return lies PAST it."""
    n = len(q)
    if not origin_inside:
        return np.full(n, -1), np.zeros(n, bool)
    ne = len(poly)
    A = poly; S = np.roll(poly, -1, 0) - poly
    r = q - origin                                                 # (n,2)
    rxs = r[:, None, 0] * S[None, :, 1] - r[:, None, 1] * S[None, :, 0]
    qp = A - origin
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[None, :, 0] * S[None, :, 1] - qp[None, :, 1] * S[None, :, 0]) / rxs
        u = (qp[None, :, 0] * r[:, None, 1] - qp[None, :, 1] * r[:, None, 0]) / rxs
    ok = (np.abs(rxs) >= 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
    tt = np.where(ok, t, np.inf)
    e = tt.argmin(1)
    best = tt[np.arange(n), e]
    cnt = np.array(cells.count); fst = np.array(cells.first)
    uu = u[np.arange(n), e]
    k = np.maximum(np.minimum(cnt[e] - 1, (uu * cnt[e]).astype(int)), 0)
    cell = np.where(np.isfinite(best), fst[e] + k, -1)
    return cell, best < 1.0


@dataclass
class BeyondLines:
    r: np.ndarray = None
    t: np.ndarray = None
    seg: np.ndarray = None


# ── wall segmenter (wallseg::segment, no corners) ───────────────────────────────────────────────────────────────
@dataclass
class WallSeg:
    normal: np.ndarray
    d: float
    phi: float
    resid_var: float
    npts: int
    info: np.ndarray
    inliers: np.ndarray


def _fit_pca(pts, min_points=3):
    if len(pts) < min_points:
        return None
    c = pts.mean(0)
    dp = pts - c
    ev, evec = np.linalg.eigh(dp.T @ dp)
    n = evec[:, 0]
    return n, float(n @ c), max(ev[0], 0) / len(pts)


def _info_phi_d(pts, n, sigma2):
    t = np.array([-n[1], n[0]])
    s = pts @ t
    L = np.array([[(s * s).sum(), -s.sum()], [-s.sum(), len(s)]])
    return L / sigma2


def wallseg_segment(pts, rng, sensor_sigma=0.02, chi2_inlier=3.841, chi2_merge=5.991, conf=0.99,
                    max_iters=200, min_points=3, max_segments=16, min_remaining=6):
    N = len(pts)
    segs = []
    if N < min_points:
        return segs
    s2 = sensor_sigma ** 2
    lo, hi = pts.min(0), pts.max(0)
    ext = hi - lo
    area = ext[0] * ext[1]
    ratio = area / max(1e-6, np.linalg.norm(ext) * math.sqrt(TWO_PI) * sensor_sigma)
    band = sensor_sigma * math.sqrt(2 * math.log(ratio)) if ratio > 1 else sensor_sigma * math.sqrt(chi2_inlier)
    band2 = band * band
    remaining = np.arange(N)

    def mk(n, d, ipts, inl):
        if d > 0:
            n, d = -n, -d
        phi = math.atan2(n[1], n[0])
        rv = s2 + _fit_pca(ipts)[2]
        return WallSeg(n, d, phi, rv, len(inl), _info_phi_d(ipts, n, rv), inl)

    while len(remaining) >= max(min_points, min_remaining) and len(segs) < max_segments:
        M = len(remaining)
        R = pts[remaining]
        best_n, best_d, best_sup, k_needed, it = None, 0.0, 0, max_iters, 0
        while it < k_needed and it < max_iters:
            it += 1
            i, j = rng.integers(0, M, 2)
            if i == j:
                continue
            e = R[j] - R[i]
            L = np.linalg.norm(e)
            if L < 1e-6:
                continue
            n = np.array([-e[1], e[0]]) / L
            d = n @ R[i]
            sup = int((((R @ n - d) ** 2) <= band2).sum())
            if sup > best_sup:
                best_n, best_d, best_sup = n, d, sup
                w = sup / M
                denom = math.log(max(1e-9, 1 - w * w))
                k_needed = min(max_iters, max(1, int(math.ceil(math.log(1 - conf) / denom))))
        if best_sup < min_points:
            break
        sel = ((R @ best_n - best_d) ** 2) <= band2
        line = _fit_pca(R[sel], min_points)
        if line is None:
            break
        n, d, _ = line
        sel = ((R @ n - d) ** 2) <= band2
        if sel.sum() < min_points:
            break
        n, d, _ = _fit_pca(R[sel], min_points)
        inl = remaining[sel]
        segs.append(mk(n, d, pts[inl], inl))
        remaining = remaining[~sel]
    merged = True
    while merged and chi2_merge > 0:
        merged = False
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                A, B = segs[a], segs[b]
                dl = np.array([wrap(A.phi - B.phi), A.d - B.d])
                try:
                    S = np.linalg.inv(A.info) + np.linalg.inv(B.info)
                    chi2 = float(dl @ np.linalg.inv(S) @ dl)
                except np.linalg.LinAlgError:
                    continue
                if not np.isfinite(chi2) or chi2 > chi2_merge:
                    continue
                inl = np.concatenate([A.inliers, B.inliers])
                line = _fit_pca(pts[inl], min_points)
                if line is None:
                    continue
                segs[a] = mk(line[0], line[1], pts[inl], inl)
                del segs[b]
                merged = True
                break
            if merged:
                break
    return segs


def beyond_lines(pts, p, rng):
    n = len(pts)
    if not p.apertures or n < 10:
        return BeyondLines()
    bl = BeyondLines(np.full(n, INF), np.zeros(n), np.full(n, -1))
    for j, sg in enumerate(wallseg_segment(pts, rng)):
        for i in sg.inliers:
            bl.r[i] = abs(sg.normal @ pts[i] - sg.d)
            bl.t[i] = np.array([-sg.normal[1], sg.normal[0]]) @ pts[i]
            bl.seg[i] = j
    return bl


# ── cell_mixture ────────────────────────────────────────────────────────────────────────────────────────────────
def cell_mixture(room_ll, cell_of, passed, cells, p, term, bl):
    n = len(room_ll)
    room_post = np.ones(n)
    log_clut = math.log(max(p.outlier_frac, 1e-6)) - math.log(max(p.beyond_extent, 1e-3))
    sig2 = 1.0 / (2 * term.inv_2s2)
    if bl.r is not None and len(bl.r) == n:
        with np.errstate(over="ignore"):
            beyond = np.logaddexp(term.log_in_norm - np.where(np.isfinite(bl.r), bl.r, 1e9) ** 2 * term.inv_2s2, log_clut)
        beyond = np.where(np.isfinite(bl.r), beyond, log_clut)
    else:
        beyond = np.full(n, log_clut)
    ll = float(room_ll.sum())
    groups = {int(c): np.where(cell_of == c)[0] for c in np.unique(cell_of[cell_of >= 0])}
    delta = {}
    for cid, idx in groups.items():
        cf = cells.leaf[cid]
        lc = math.log(cf) if cf > 0 else -INF
        l1c = math.log1p(-cf) if cf < 1 else -INF
        o = np.where(passed[idx], beyond[idx], term.log_out)
        with np.errstate(invalid="ignore"):
            so = float(np.logaddexp(l1c + o, lc + room_ll[idx]).sum())
        ss = float(room_ll[idx].sum())
        sel = idx[passed[idx] & (bl.seg[idx] >= 0)] if bl.seg is not None and len(bl.seg) == n else idx[:0]
        for line in np.unique(bl.seg[sel]) if len(sel) else []:
            m = sel[bl.seg[sel] == line]
            nn = len(m); st = bl.t[m].sum(); st2 = (bl.t[m] ** 2).sum()
            mean = st / nn
            var_t = max(st2 / nn - mean * mean, sig2 / 12.0)
            so -= (math.log(math.pi * p.beyond_extent) - math.log(TWO_PI)
                   + 0.5 * math.log(nn / sig2) + 0.5 * math.log(nn * var_t / sig2))
        delta[cid] = so - ss
    stay = {}
    for k in range(len(cells.pi)):
        pi = cells.pi[k]; q = cells.q[k]
        hi = max([0.0] + list(delta.values()))
        zs = (q + (1 - q)) * math.exp(-hi)
        for cid, d in delta.items():
            zs += (1 - q) * pi[cid] * (math.exp(d - hi) - math.exp(-hi))
        ll += hi + math.log(max(zs, 1e-300))
        for cid, d in delta.items():
            r = (1 - q) * pi[cid] * math.exp(d - hi) / max(zs, 1e-300)
            stay[cid] = stay.get(cid, 1.0) * (1 - min(max(r, 0), 1))
    for cid, idx in groups.items():
        rho = stay.get(cid, 1.0)
        if rho >= 1:
            continue
        cf = cells.leaf[cid]
        lc = math.log(cf) if cf > 0 else -INF
        l1c = math.log1p(-cf) if cf < 1 else -INF
        i = idx[passed[idx]]
        with np.errstate(invalid="ignore"):
            leafp = np.exp(lc + room_ll[i] - np.logaddexp(l1c + beyond[i], lc + room_ll[i]))
        room_post[i] = rho + (1 - rho) * leafp
    return ll, room_post


def weighted_median(d, w):
    o = np.argsort(d, kind="stable")
    d, w = d[o], w[o]
    if len(d) == 0:
        return float("nan")
    if np.all(w == w[0]):
        return float(d[len(d) // 2])
    tot = w.sum()
    acc = np.cumsum(w)
    k = int(np.searchsorted(acc, 0.5 * tot - 1e-15))
    return float(d[min(k, len(d) - 1)])


def _world_pts(pts, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    rp = np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], 1)
    return rp, rp + pose[:2]


def log_likelihood(poly, pts, pose, p, extent, bl=None, want_median=False):
    """reloc::log_likelihood.  Returns (l, door-weighted median |d|, plain median |d|)."""
    pose = np.asarray(pose, float)
    if bl is None:
        bl = beyond_lines(pts, p, np.random.default_rng(20261021)) if p.apertures else BeyondLines()
    term = PointTerm(p, extent)
    on = bool(p.apertures)
    cells = build_cells(poly, p) if on else Cells([], [], np.zeros(0), [], [])
    _, q = _world_pts(pts, pose)
    ds = polygon_distance(poly, q)
    room_ll, _ = term.ll(ds)
    o_in = on and inside_polygon(poly, pose[:2])
    ap_of, passed = exit_cells(poly, cells, pose[:2], o_in, q) if on else (np.full(len(pts), -1), np.zeros(len(pts), bool))
    ll, post = cell_mixture(room_ll, ap_of, passed, cells, p, term, bl)
    return ll, weighted_median(ds, post), float(np.median(ds))


# ── structural yaws ─────────────────────────────────────────────────────────────────────────────────────────────
def structural_yaws(poly, pts, p, rng):
    if len(poly) < 3 or len(pts) < 10:
        return [], 0
    segs = wallseg_segment(pts, rng)
    if not segs:
        return [], 0
    ccw = 1.0 if polygon_area_signed(poly) > 0 else -1.0
    edges = []
    for i in range(len(poly)):
        ab = poly[(i + 1) % len(poly)] - poly[i]
        L = np.linalg.norm(ab)
        if L < 1e-4:
            continue
        edges.append((math.atan2(ccw * ab[0], -ccw * ab[1]), L))
    K = 720
    binw = TWO_PI / K
    dens = np.zeros(K)
    for sg in segs:
        var_phi = 1e-2
        if abs(np.linalg.det(sg.info)) > 1e-12:
            var_phi = max(np.linalg.inv(sg.info)[0, 0], 0.0)
        sig = math.sqrt(max(var_phi, binw * binw))
        half = min(K // 2, int(math.ceil(3 * sig / binw)))
        ks = np.arange(-half, half + 1)
        kern = np.exp(-0.5 * (ks * binw) ** 2 / sig ** 2)
        for psi, L in edges:
            theta = wrap(psi - sg.phi)
            c = int(round((theta + math.pi) / binw)) % K
            np.add.at(dens, (c + ks) % K, sg.npts * L * kern)
    left, right = np.roll(dens, -1), np.roll(dens, 1)
    pk = np.where((dens > 0) & (dens >= left) & (dens > right))[0]
    pk = pk[np.argsort(-dens[pk], kind="stable")][:p.max_yaws]
    return [-math.pi + int(i) * binw for i in pk], len(segs)


# ── polish ──────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Mode:
    pose: np.ndarray
    cov: np.ndarray
    log_lik: float
    median_abs: float
    log_weight: float = 0.0
    weight: float = 0.0
    log_prior: float = 0.0
    half_logdet: float = 0.0


def polish(poly, pts, start, p, extent, bl):
    on = bool(p.apertures)
    cells = build_cells(poly, p) if on else Cells([], [], np.zeros(0), [], [])
    pl = Params(**{**p.__dict__})
    term = PointTerm(pl, extent)
    inv_s2 = 1.0 / pl.sigma_sdf ** 2
    prior = np.diag([1 / max(extent ** 2, 1e-3), 1 / max(extent ** 2, 1e-3), 1 / math.pi ** 2])
    n = len(pts)
    state = {}

    def evaluate(pose, want_H):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rp = np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], 1)
        q = rp + pose[:2]
        cl, edge = closest_on_polygon(poly, q)
        diff = q - cl
        d = np.linalg.norm(diff, axis=1)
        room, li = term.ll(d)
        o_in = on and inside_polygon(poly, pose[:2])
        ap, ps = exit_cells(poly, cells, pose[:2], o_in, q) if on else (np.full(n, -1), np.zeros(n, bool))
        ll, post = cell_mixture(room, ap, ps, cells, p, term, bl)
        if not want_H:
            return ll, None, None
        gamma = post * np.exp(li - room)
        state["wd2"], state["w"] = float((gamma * d * d).sum()), float(gamma.sum())
        ok = d >= 1e-6
        u = np.where(ok[:, None], diff / np.where(ok, d, 1)[:, None], 0.0)
        J = np.stack([u[:, 0], u[:, 1], -u[:, 0] * rp[:, 1] + u[:, 1] * rp[:, 0]], 1)
        w = np.where(ok, gamma * inv_s2, 0.0)
        H = prior + (J * w[:, None]).T @ J
        g = ((w * d)[:, None] * J).sum(0)
        es = []
        for e in range(len(poly)):
            m = edge == e
            es.append((float(w[m].sum()), (J[m] * w[m][:, None]).sum(0)))
        state["es"] = es
        return ll, H, g

    x = np.array(start, float)
    ll_cur, H, g = evaluate(x, True)
    lam = 1e-3
    for _ in range(p.polish_iters):
        A = H.copy()
        A[np.diag_indices(3)] *= (1 + lam)
        try:
            step = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        xn = x + step
        xn[2] = wrap(xn[2])
        ll_new, Hn, gn = evaluate(xn, True)
        if ll_new >= ll_cur:
            x, H, g, ll_cur = xn, Hn, gn, ll_new
            lam = max(lam * 0.3, 1e-6)
            moved = False
            if state["w"] > 3.0:
                sig_em = math.sqrt(state["wd2"] / state["w"])
                if np.isfinite(sig_em) and 0 < sig_em < 0.99 * pl.sigma_sdf:
                    pl.sigma_sdf = sig_em
                    term = PointTerm(pl, extent)
                    inv_s2 = 1.0 / pl.sigma_sdf ** 2
                    ll_cur, H, g = evaluate(x, True)
                    moved = True
            if moved:
                continue
            nw = np.linalg.solve(H, -g)
            if np.all(np.isfinite(nw)) and nw @ H @ nw < 1e-6:
                break
        else:
            lam *= 10
    H_marg = H
    if p.wall_offset_sigma > 0:
        _, H, _ = evaluate(x, True)
        H_marg = H.copy()
        inv_c2 = 1.0 / p.wall_offset_sigma ** 2
        for sw, swj in state["es"]:
            if sw > 0:
                H_marg = H_marg - np.outer(swj, swj) / (inv_c2 + sw)
    pose = x.copy(); pose[2] = wrap(pose[2])
    cov = np.linalg.inv(H_marg)
    ll, med, _ = log_likelihood(poly, pts, pose, p, extent, bl)      # at the ORIGINAL sigma_sdf, as the C++ does
    return Mode(pose, cov, ll, med)


def log_det3(m):
    try:
        L = np.linalg.cholesky(m)
    except np.linalg.LinAlgError:
        return 0.0
    return float(2 * np.log(np.diag(L)).sum())


def log_prior_modes(p, pose, cov, area):
    if not p.prior_modes:
        return 0.0
    wrong = min(max(p.prior_wrong, 1e-9), 1 - 1e-9)
    log_u = -math.log(max(area, 1e-6) * TWO_PI)
    wsum = sum(max(0, c.w) for c in p.prior_modes)
    lg = -INF
    for c in p.prior_modes:
        if not c.w > 0 or wsum <= 0:
            continue
        d = np.asarray(pose) - c.mu
        d[2] = wrap(d[2])
        Sg = c.S + cov
        maha = float(d @ np.linalg.solve(Sg, d))
        logdet = float(np.linalg.slogdet(Sg)[1])
        lc = math.log(c.w / wsum) - 0.5 * maha - 0.5 * (3 * math.log(TWO_PI) + logdet)
        lg = np.logaddexp(lg, lc)
    return float(np.logaddexp(math.log1p(-wrong) + lg, math.log(wrong) + log_u))


@dataclass
class Result:
    modes: list
    ess: float = 0.0
    n_yaws: int = 0
    n_segments: int = 0
    n_starts: int = 0


def build_raster(poly, p):
    pad = 4 * p.sigma_sdf
    lo, hi = poly.min(0), poly.max(0)
    res = p.raster_res
    x0, y0 = lo[0] - pad, lo[1] - pad
    nx = int(math.ceil((hi[0] - lo[0] + 2 * pad) / res)) + 1
    ny = int(math.ceil((hi[1] - lo[1] + 2 * pad) / res)) + 1
    extent = polygon_extent(poly)
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    dist = polygon_distance(poly, np.stack([X.ravel(), Y.ravel()], 1)).reshape(ny, nx)
    term = PointTerm(p, extent)
    return dict(x0=x0, y0=y0, res=res, nx=nx, ny=ny, extent=extent, dist=dist, ll=term.ll(dist)[0], ll_out=term.log_out)


def subsample(pts, m):
    if m <= 0 or len(pts) <= m:
        return pts
    stride = len(pts) / m
    return pts[[int(i * stride) for i in range(m)]]


def search(poly, pts_all, seeds, p, seed=12345):
    R = build_raster(poly, p)
    extent = R["extent"]
    pts = subsample(pts_all, p.max_points)
    rng = np.random.default_rng(seed)
    res = Result([])
    yaws, res.n_segments = structural_yaws(poly, subsample(pts_all, 4 * p.max_points), p, rng)
    if not yaws:
        yaws = [-math.pi + TWO_PI * k / p.fallback_yaws for k in range(p.fallback_yaws)]
    res.n_yaws = len(yaws)
    prior_only = p.prior_only and bool(p.prior_modes)
    maxima = []
    if not prior_only:
        m = max(1, int(round(p.lattice_step / R["res"])))
        step = m * R["res"]
        lx, ly = R["nx"] // m, R["ny"] // m
        I, J = np.meshgrid(np.arange(lx), np.arange(ly))
        T = np.stack([R["x0"] + I * step, R["y0"] + J * step], -1)
        dgrid = R["dist"][J * m, I * m]
        valid = (dgrid >= p.body_clearance) & np.array(
            [[inside_polygon(poly, T[j, i]) for i in range(lx)] for j in range(ly)])
        for yaw in yaws:
            c, s = math.cos(yaw), math.sin(yaw)
            ci = np.floor((c * pts[:, 0] - s * pts[:, 1]) / R["res"]).astype(int)
            cj = np.floor((s * pts[:, 0] + c * pts[:, 1]) / R["res"]).astype(int)
            a = ci[None, None, :] + (I * m)[..., None]
            b = cj[None, None, :] + (J * m)[..., None]
            inb = (a >= 0) & (b >= 0) & (a < R["nx"]) & (b < R["ny"])
            val = np.where(inb, R["ll"][np.clip(b, 0, R["ny"] - 1), np.clip(a, 0, R["nx"] - 1)], R["ll_out"])
            score = np.where(valid, val.sum(-1), -INF)
            pad = np.pad(score, 1, constant_values=-INF)
            ismax = valid.copy()
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    w = pad[1 + dj:1 + dj + ly, 1 + di:1 + di + lx]
                    tie_lose = (dj < 0) or (dj == 0 and di < 0)
                    ismax &= ~((w > score) | ((w == score) & tie_lose))
            for j, i in zip(*np.where(ismax)):
                maxima.append((score[j, i], np.array([R["x0"] + i * step, R["y0"] + j * step, yaw])))
        maxima.sort(key=lambda t: -t[0])
        maxima = maxima[:p.max_seeds]
    starts = []
    if not prior_only:
        starts += [m[1] for m in maxima] + [np.asarray(s, float) for s in seeds]
    starts += [pm.mu for pm in p.prior_modes]
    res.n_starts = len(starts)
    pts_polish = subsample(pts_all, 4 * p.max_points)
    bl = beyond_lines(pts_polish, p, np.random.default_rng(20261021))
    modes = []
    for st in starts:
        md = polish(poly, pts_polish, st, p, extent, bl)
        if not np.all(np.isfinite(md.pose)) or not np.isfinite(md.log_lik):
            continue
        merged = False
        for k, other in enumerate(modes):
            dl = md.pose - other.pose
            dl[2] = wrap(dl[2])
            chi2 = float(dl @ np.linalg.solve(md.cov + other.cov, dl))
            if np.isfinite(chi2) and chi2 < p.chi2_merge:
                if md.log_lik > other.log_lik:
                    modes[k] = md
                merged = True
                break
        if not merged:
            modes.append(md)
    area = polygon_area(poly) if p.prior_modes else 0.0
    for md in modes:
        md.half_logdet = 0.5 * log_det3(md.cov)
        md.log_prior = log_prior_modes(p, md.pose, md.cov, area) if p.prior_modes else 0.0
        md.log_weight = md.log_lik + md.half_logdet + md.log_prior
    modes.sort(key=lambda m: -m.log_weight)
    modes = modes[:p.max_modes]
    if modes:
        top = modes[0].log_weight
        z = sum(math.exp(m.log_weight - top) for m in modes)
        logz = top + math.log(z)
        w2 = 0.0
        for m in modes:
            m.log_weight -= logz
            m.weight = math.exp(m.log_weight)
            w2 += m.weight ** 2
        res.ess = 1.0 / w2 if w2 > 0 else 1.0
    res.modes = modes
    return res


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  5. SCENARIOS, REPORT, PLOTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def make_scenarios(world, K, door2_mouth, n_out2):
    """Named GT poses (x, y, yaw).  Facing INTO room_2 = against the outward normal of K's wall at DOOR_2."""
    d2 = world.doors["DOOR_2"]
    yaw = math.atan2(-n_out2[1], -n_out2[0])
    m = door2_mouth
    sc = [("in_door", np.array([*d2["centre"], yaw])),
          ("in_0.5m", np.array([*(m - 0.5 * n_out2), yaw])),
          ("in_1.0m", np.array([*(m - 1.0 * n_out2), yaw]))]
    # control: the interior point farthest from both door centres with >= 1 m clearance to K's walls
    lo, hi = K.min(0), K.max(0)
    best, bp = -1, None
    for x in np.arange(lo[0], hi[0], 0.1):
        for y in np.arange(lo[1], hi[1], 0.1):
            q = np.array([x, y])
            if not inside_polygon(K, q) or polygon_distance(K, q[None])[0] < 1.0:
                continue
            dd = min(np.linalg.norm(q - dv["centre"]) for dv in (world.doors["DOOR_1"], d2))
            if dd > best:
                best, bp = dd, q
    sc.append(("control_mid", np.array([*bp, yaw])))
    return sc


def fmt_mode(m, gt, label=""):
    dxy = np.linalg.norm(m.pose[:2] - gt[:2])
    dth = math.degrees(abs(wrap(m.pose[2] - gt[2])))
    return (f"{label:>3} ({m.pose[0]:7.3f},{m.pose[1]:7.3f},{math.degrees(m.pose[2]):7.1f}deg)  "
            f"d_xy={dxy:5.2f} m  d_th={dth:6.1f} deg  w={m.weight:.4f}  "
            f"log_lik={m.log_lik:9.2f}  1/2log|S|={m.half_logdet:7.2f}  log_prior={m.log_prior:8.2f}  "
            f"med|sdf|={m.median_abs:.3f}")


def draw_world(ax, world, K, apertures, zoom):
    for p_, q_ in world.segments:
        ax.plot([p_[0], q_[0]], [p_[1], q_[1]], color="0.8", lw=0.6, zorder=1)
    kc = np.vstack([K, K[:1]])
    ax.plot(kc[:, 0], kc[:, 1], color="tab:blue", lw=2.2, zorder=3, label="K outline (room_2 as remembered)")
    for ap in apertures:
        ax.plot([ap.a[0], ap.b[0]], [ap.a[1], ap.b[1]], color="tab:green", lw=5, alpha=.7, zorder=4,
                label="K aperture (DOOR_1)")
    ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3]); ax.set_aspect("equal")


def arrow(ax, pose, color, label, size=0.5, marker="o", ms=5):
    ax.plot(pose[0], pose[1], marker, color=color, ms=ms, zorder=6, label=label)
    ax.arrow(pose[0], pose[1], size * math.cos(pose[2]), size * math.sin(pose[2]), color=color,
             head_width=0.12, length_includes_head=True, zorder=6)


def plot_scenario(path, name, world, K, apertures, gt, prior_mu, scan, res_a, res_b, zoom):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    _, wp = _world_pts(scan, gt)
    for ax, res, title in ((axs[0], res_a, "(a) as it was: prior_only=False, prior_wrong=0.1"),
                           (axs[1], res_b, "(b) prior_only=True")):
        draw_world(ax, world, K, apertures, zoom)
        ax.scatter(wp[:, 0], wp[:, 1], s=5, color="tab:orange", zorder=5, label="scan at GT")
        arrow(ax, gt, "k", "GT", marker="*", ms=12)
        arrow(ax, prior_mu, "tab:purple", "prior mean", marker="s")
        for k, m in enumerate(res.modes):
            arrow(ax, m.pose, "tab:red" if k == 0 else "tab:brown", f"mode {k} w={m.weight:.3f}", marker="^")
        h, l = ax.get_legend_handles_labels()
        seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper left")
        ax.set_title(f"{name}  {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_floorplan(path, world, K, free, lo, res):
    fig, ax = plt.subplots(figsize=(11, 11))
    for p_, q_ in world.segments:
        ax.plot([p_[0], q_[0]], [p_[1], q_[1]], color="0.35", lw=0.7)
    ax.imshow(free, origin="lower", extent=[lo[0], lo[0] + free.shape[1] * res, lo[1], lo[1] + free.shape[0] * res],
              cmap="Blues", alpha=0.25, zorder=0)
    kc = np.vstack([K, K[:1]])
    ax.plot(kc[:, 0], kc[:, 1], color="tab:blue", lw=2.5, label="K polygon (room_2 memory)")
    ax.plot(K[:, 0], K[:, 1], "o", color="tab:blue", ms=4)
    for nm, d in world.doors.items():
        a, b = d["centre"] - 0.45 * d["tangent"], d["centre"] + 0.45 * d["tangent"]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:red" if nm == "DOOR_2" else "tab:green", lw=6, alpha=.8)
        ax.annotate(nm, d["centre"], color="k", fontsize=11, weight="bold", xytext=(6, 6), textcoords="offset points")
    ax.set_aspect("equal"); ax.legend(); ax.set_title("threerooms-piso: parsed 2-D walls at the lidar height, doors, room_2 memory K")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "YAW_RAD"), help="run ONE custom GT pose")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lidar-z", type=float, default=0.5, help="height of the lidar plane (m)")
    ap.add_argument("--prior-closed", type=float, default=0.5, help="DOOR_1 prior_closed (unmeasured = 0.5)")
    ap.add_argument("--sigma-along", type=float, default=0.10, help="DOOR_1 position sigma along the wall (m)")
    ap.add_argument("--prior-wrong", type=float, default=0.1)
    a = ap.parse_args()
    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)
    np.set_printoptions(precision=3, suppress=True)

    # ── world ───────────────────────────────────────────────────────────────────────────────────────────────────
    world = parse_world(WORLD, a.lidar_z)
    print(f"[world] {WORLD}")
    print(f"[world] {len(world.walls)} Wall protos at z={a.lidar_z} m: " + ", ".join(n for n, _ in world.walls))
    print(f"[world] mesh walls: {world.meshes}  -> {sum(n for _, n in world.meshes)} sliced segments")
    for nm, d in world.doors.items():
        print(f"[world] {nm:7s} '{d['name']}' centre=({d['centre'][0]:.3f},{d['centre'][1]:.3f}) "
              f"tangent=({d['tangent'][0]:.2f},{d['tangent'][1]:.2f}) normal=({d['normal'][0]:.2f},{d['normal'][1]:.2f}) "
              f"hole width={d['width']:.2f} m (leaf omitted)")
    print(f"[world] {len(world.segments)} segments total")

    # ── K ───────────────────────────────────────────────────────────────────────────────────────────────────────
    K, free, lo, res = build_memory_room(world)
    print(f"[K] room_2 memory polygon: {len(K)} vertices, area {polygon_area(K):.2f} m^2, bbox "
          f"x[{K[:,0].min():.2f},{K[:,0].max():.2f}] y[{K[:,1].min():.2f},{K[:,1].max():.2f}] (world frame)")
    for v in K:
        print(f"      ({v[0]:7.3f}, {v[1]:7.3f})")
    d1, d2 = world.doors["DOOR_1"], world.doors["DOOR_2"]
    a1, b1, n1, rec1 = seat_in_room(d1["centre"] - 0.5 * d1["width"] * d1["tangent"],
                                    d1["centre"] + 0.5 * d1["width"] * d1["tangent"], K)
    m2a, m2b, n2, rec2 = seat_in_room(d2["centre"] - 0.5 * d2["width"] * d2["tangent"],
                                      d2["centre"] + 0.5 * d2["width"] * d2["tangent"], K)
    m2 = 0.5 * (m2a + m2b)
    hazard = 1e-3
    ap1 = Aperture(a1, b1, n1, 1.0 - a.prior_closed, 1.0 - (1 - hazard) * 0.5, a.sigma_along)
    print(f"[K] DOOR_1 aperture (the ONLY one): mouth ({a1[0]:.2f},{a1[1]:.2f})-({b1[0]:.2f},{b1[1]:.2f}) "
          f"n_out=({n1[0]:.2f},{n1[1]:.2f}) recess {rec1:.3f} m; open={ap1.open:.2f} p_shut={ap1.p_shut:.3f}")
    print(f"[K] DOOR_2 is solid wall in K (K edge there: mouth-line centre ({m2[0]:.2f},{m2[1]:.2f}), n_out=({n2[0]:.2f},{n2[1]:.2f}), "
          f"door centre is {rec2:.3f} m behind it)")
    plot_floorplan(os.path.join(a.out, "floorplan.png"), world, K, free, lo, res)

    # ── scenarios ───────────────────────────────────────────────────────────────────────────────────────────────
    if a.pose:
        scen = [("custom", np.array(a.pose, float))]
    else:
        scen = make_scenarios(world, K, m2, n2)
    S = np.diag([0.3 ** 2, 0.3 ** 2, 0.15 ** 2])
    rng = np.random.default_rng(a.seed)
    rows = []
    zoom = (K[:, 0].min() - 3.2, K[:, 0].max() + 3.5, K[:, 1].min() - 2.2, K[:, 1].max() + 1.5)
    for name, gt in scen:
        scan = lidar_scan(world, gt, rng)
        mu = gt + np.array([0.1, -0.1, math.radians(3)])
        prior = [PriorMode(mu, S, 1.0)]
        base = dict(apertures=[ap1], prior_modes=prior, prior_wrong=a.prior_wrong)
        pa, pb = Params(**base, prior_only=False), Params(**base, prior_only=True)
        extent = polygon_extent(K)
        print("\n" + "=" * 150)
        print(f"SCENARIO {name}: GT=({gt[0]:.3f}, {gt[1]:.3f}, {math.degrees(gt[2]):.1f} deg)  "
              f"inside K: {inside_polygon(K, gt[:2])}  scan: {len(scan)} returns of 360 beams")
        print(f"  prior mean = GT + (0.1,-0.1,3deg) = ({mu[0]:.3f},{mu[1]:.3f},{math.degrees(mu[2]):.1f})  S=diag(.3^2,.3^2,.15^2)")
        for lab, pose in (("GT", gt), ("prior mean", mu)):
            ll, med, plain = log_likelihood(K, scan, pose, pa, extent)
            print(f"  scan at {lab:10s}: log_lik={ll:9.2f}  median|SDF| door-weighted={med:.3f}  plain={plain:.3f}")
        out = {}
        for tag, p in (("a", pa), ("b", pb)):
            t0 = time.time()
            r = search(K, scan, [mu], p)
            out[tag] = r
            print(f"  -- ({tag}) {'prior_only=False prior_wrong=%.2f (AS IT WAS)' % a.prior_wrong if tag == 'a' else 'prior_only=True'}: "
                  f"{len(r.modes)} modes, ESS={r.ess:.2f}, yaws={r.n_yaws}, wall-segs={r.n_segments}, starts={r.n_starts}, {time.time()-t0:.1f}s")
            for k, m in enumerate(r.modes):
                print("      " + fmt_mode(m, gt, f"#{k}"))
            top = r.modes[0]
            ll_t, med_t, plain_t = log_likelihood(K, scan, top.pose, p, extent)
            ll_g, med_g, plain_g = log_likelihood(K, scan, gt, p, extent)
            print(f"      top mode vs GT: jump {np.linalg.norm(top.pose[:2]-gt[:2]):.2f} m / "
                  f"{math.degrees(abs(wrap(top.pose[2]-gt[2]))):.0f} deg;  median|SDF| (door-weighted) GT={med_g:.3f} top={med_t:.3f};  "
                  f"scan log_lik GT={ll_g:.1f} top={ll_t:.1f}")
            rows.append((name, tag, gt, top, r, med_g, med_t, ll_g, ll_t))
        # DOOR_1 rival probe: polish from poses standing at K's only known doorway (pushed 0.3 m into the room), facing
        # in / out / east / west, and compare their full evidence (scan l + 1/2 log|S| + prior) with the GT-basin mode's.
        bl_p = beyond_lines(subsample(scan, 4 * pa.max_points), pa, np.random.default_rng(20261021))
        ptsp = subsample(scan, 4 * pa.max_points)
        area = polygon_area(K)
        ref = polish(K, ptsp, mu, pa, extent, bl_p)
        ev = lambda m: m.log_lik + 0.5 * log_det3(m.cov) + log_prior_modes(pa, m.pose, m.cov, area)  # noqa: E731
        print(f"  -- DOOR_1 rival probe (arm-a params): evidence = l + 1/2log|S| + log_prior;  GT-basin mode: "
              f"{ev(ref):9.2f} (l={ref.log_lik:.1f})")
        mid = 0.5 * (ap1.a + ap1.b) - 0.3 * n1
        for lab, yw in (("facing into room", math.atan2(-n1[1], -n1[0])), ("facing out the door", math.atan2(n1[1], n1[0])),
                        ("facing east", 0.0), ("facing west", math.pi)):
            pr = polish(K, ptsp, np.array([*mid, yw]), pa, extent, bl_p)
            print(f"      from DOOR_1 {lab:20s} -> ({pr.pose[0]:7.3f},{pr.pose[1]:7.3f},{math.degrees(pr.pose[2]):7.1f}deg) "
                  f"d_GT={np.linalg.norm(pr.pose[:2]-gt[:2]):5.2f} m  l={pr.log_lik:9.2f}  evidence={ev(pr):9.2f}  "
                  f"(vs GT-basin {ev(pr)-ev(ref):+9.2f} nats)")
        plot_scenario(os.path.join(a.out, f"scenario_{name}.png"), name, world, K, [ap1], gt, mu, scan, out["a"], out["b"], zoom)

    print("\n" + "=" * 150)
    print("SUMMARY (top mode of each arm)")
    print(f"{'scenario':12s} {'arm':3s} {'jump m':>7s} {'jump deg':>8s} {'top w':>6s} {'ESS':>5s} {'med@GT':>7s} {'med@top':>7s} "
          f"{'ll@GT':>8s} {'ll@top':>8s}  verdict")
    for name, tag, gt, top, r, mg, mt, lg, lt in rows:
        j = np.linalg.norm(top.pose[:2] - gt[:2]); jd = math.degrees(abs(wrap(top.pose[2] - gt[2])))
        verdict = "JUMPS to rival" if j > 1.0 else "stays at GT"
        print(f"{name:12s} {tag:3s} {j:7.2f} {jd:8.1f} {top.weight:6.3f} {r.ess:5.2f} {mg:7.3f} {mt:7.3f} {lg:8.1f} {lt:8.1f}  {verdict}")
    print(f"\nPNGs in {a.out}   total {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
