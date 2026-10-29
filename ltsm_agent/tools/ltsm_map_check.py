#!/usr/bin/env python3
"""ltsm_map_check.py -- offline consistency check of an ltsm memory dump.

  python3 -I ltsm_map_check.py <ltsm_memory.json>

Prints (1) for every door face (object node named r<k>_door_*) its distance to the
nearest edge of its room's polygon, and (2) for every pair of rooms reachable by two
different passage paths, the disagreement of the two composed room<-room transforms.

Exit 1 if any face is > FACE_WALL_LIMIT from a wall or any loop disagrees by more than
LOOP_TRANS_LIMIT / LOOP_YAW_LIMIT.  These are FLAGGED REPORT limits for a human/CI
reading a finished map, NOT runtime gates and NOT model parameters.

Chaining: T(room_a<-room_b) = A . Rz(pi) . B^-1, with A, B the two matched faces' poses
in their own room frames (the same doorway seen from both sides, facing opposite ways).
"""
import json
import math
import re
import sys
from collections import defaultdict

import numpy as np

FACE_WALL_LIMIT = 0.15   # m
LOOP_TRANS_LIMIT = 0.3   # m
LOOP_YAW_LIMIT = 10.0    # deg
MAX_PATH_EDGES = 6
FACE_RE = re.compile(r"^r\d+_door_")


def val(node, name, default=None):
    a = node["attribute"].get(name)
    return default if a is None else a["value"]


def rt_matrix(link):
    """Latest slot of the RT history ring (rt_head_index = slot written NEXT)."""
    la = link["linkAttribute"]
    t = la["rt_translation"]["value"]
    r = la["rt_rotation_euler_xyz"]["value"]
    n = len(t) // 3
    head = la.get("rt_head_index", {}).get("value", 0) // 3
    k = (head - 1) % n if n else 0
    x, y, z = t[3 * k:3 * k + 3]
    rx, ry, rz = r[3 * k:3 * k + 3]
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]
    return T


def rz_pi():
    T = np.diag([-1.0, -1.0, 1.0, 1.0])
    return T


def yaw(T):
    return math.atan2(T[1, 0], T[0, 0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def seg_dist(p, a, b):
    ab = b - a
    L = float(ab @ ab)
    t = 0.0 if L == 0 else min(1.0, max(0.0, float((p - a) @ ab) / L))
    return float(np.linalg.norm(p - (a + t * ab)))


def main(path):
    d = json.load(open(path))
    syms = d["DSRModel"]["symbols"]
    rt = {}  # child id -> (parent id, T parent<-child)
    for pid, n in syms.items():
        for l in n.get("links", []):
            if l["label"] == "RT" and l["src"] == pid:
                rt[l["dst"]] = (pid, rt_matrix(l))

    def room_pose(nid):
        T = np.eye(4)
        cur = nid
        for _ in range(64):
            if syms[cur]["type"] == "room":
                return cur, T
            if cur not in rt:
                return None, None
            par, M = rt[cur]
            T = M @ T
            cur = par
        return None, None

    faces = {}  # id -> (room id, T room<-face)
    for nid, n in syms.items():
        if n["type"] == "object" and FACE_RE.match(n["name"]):
            r, T = room_pose(nid)
            if r is not None:
                faces[nid] = (r, T)

    bad = False
    print("== door faces: distance to nearest wall of own room ==")
    worst_face = (0.0, None)
    for nid, (r, T) in sorted(faces.items(), key=lambda kv: syms[kv[0]]["name"]):
        room = syms[r]
        px = val(room, "delimiting_polygon_x", [])
        py = val(room, "delimiting_polygon_y", [])
        poly = np.array(list(zip(px, py)), float)
        if len(poly) < 3:
            print(f"  {syms[nid]['name']:<22} room {room['name']}: no polygon")
            continue
        p = T[:2, 3]
        dist = min(seg_dist(p, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))
        flag = dist > FACE_WALL_LIMIT
        bad |= flag
        if dist >= worst_face[0]:
            worst_face = (dist, f"{syms[nid]['name']} ({room['name']})")
        print(f"  {syms[nid]['name']:<22} {room['name']:<10} {dist:7.3f} m {'FLAGGED' if flag else 'ok'}")

    # room graph: edges (room_a, room_b, passage, T room_a<-room_b)
    adj = defaultdict(list)
    for nid, n in syms.items():
        if n["type"] != "metaconcept" or not n["name"].startswith("passage_"):
            continue
        ms = [l["dst"] for l in n.get("links", []) if l["label"] == "match" and l["dst"] in faces]
        for i in range(len(ms)):
            for j in range(len(ms)):
                if i == j:
                    continue
                ra, A = faces[ms[i]]
                rb, B = faces[ms[j]]
                if ra == rb:
                    continue
                adj[ra].append((rb, n["name"], A @ rz_pi() @ np.linalg.inv(B)))

    def paths(src, dst):
        out = []

        def dfs(cur, seen, T, names):
            if len(names) > MAX_PATH_EDGES:
                return
            if cur == dst and names:
                out.append((names, T))
                return
            for nb, pn, M in adj[cur]:
                if nb in seen and nb != dst:
                    continue
                dfs(nb, seen | {nb}, T @ M, names + [pn])

        dfs(src, {src}, np.eye(4), [])
        return out

    print("== passage loops: disagreement of two paths room_a <- room_b ==")
    worst_loop = (0.0, 0.0, None)
    rooms = sorted({r for r, _ in faces.values()}, key=lambda i: syms[i]["name"])
    nloops = 0
    for ia, a in enumerate(rooms):
        for b in rooms[ia + 1:]:
            ps = paths(a, b)
            seen = set()
            uniq = []
            for names, T in ps:  # dedupe by passage sequence (a passage with >2 faces repeats)
                k = tuple(names)
                if k not in seen:
                    seen.add(k)
                    uniq.append((names, T))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    (n1, T1), (n2, T2) = uniq[i], uniq[j]
                    dt = float(np.linalg.norm(T1[:3, 3] - T2[:3, 3]))
                    dy = abs(math.degrees(wrap(yaw(T1) - yaw(T2))))
                    flag = dt > LOOP_TRANS_LIMIT or dy > LOOP_YAW_LIMIT
                    bad |= flag
                    nloops += 1
                    lab = f"{syms[a]['name']}<-{syms[b]['name']}"
                    if (dt, dy) > worst_loop[:2]:
                        worst_loop = (dt, dy, lab)
                    print(f"  {lab:<20} via {'+'.join(n1)} vs {'+'.join(n2)}: "
                          f"{dt:6.3f} m {dy:6.1f} deg {'FLAGGED' if flag else 'ok'}")
    if not nloops:
        print("  (no room pair reachable by two different passage paths)")

    print(f"== worst face {worst_face[0]:.3f} m [{worst_face[1]}] (limit {FACE_WALL_LIMIT}); "
          f"worst loop {worst_loop[0]:.3f} m / {worst_loop[1]:.1f} deg [{worst_loop[2]}] "
          f"(limits {LOOP_TRANS_LIMIT} m / {LOOP_YAW_LIMIT} deg) => {'FAIL' if bad else 'PASS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
