#!/usr/bin/env python3
"""
make_stage_fixture.py — author etc/stage_two_rooms.json, the offline two-room graph.

WHY A FIXTURE. Eviction fires when the robot carries two RT edges, one to the room it is leaving
and one to the room it has just localised in. `room_concept` cannot produce that yet (it creates a
single node literally named "room" and adopts any room-typed node it finds), so the whole
transaction would be untestable until that half lands. This file is the stand-in: a synthetic live
graph, seeded from JSON on a PRIVATE DDS domain, shaped exactly like the real one at the instant of
the hand-over.

    root
     +-- Shadow (robot)
          +-- room_0          the room being left      RT: robot -> room_0
          |    +-- floor
          |    |    +-- wall_0 .. wall_3
          |    |         +-- door_1                    on wall_1, the doorway just crossed
          |    +-- table_1                             a non-wall child, to prove the BFS carries it
          +-- room_1          the room just entered    RT: robot -> room_1

The parent chain (room -> floor -> wall -> door) is NOT invented here: it is what
room_scene_graph.cpp:2089-2160 and door_scene_graph.cpp:140-147 actually build.

Z IS ZERO EVERYWHERE, on purpose. The real walls sit at half the room height; flattening the
fixture into the z=0 plane makes every composition in the test checkable by hand in SE(2), which is
what the seam actually is. Nothing in the eviction code is 2-D, so this constrains the test, not
the code.

Run from the component root:  python3 tools/make_stage_fixture.py
It rewrites etc/stage_two_rooms.json and prints the ground truth the self-test asserts.
"""
import json, math, os

# ── attribute type codes, from cortex/api/dsr_utils.cpp:144-220 ────────────────────────────────
STR, INT, FLOAT, VECF, BOOL, VECB, U32, U64 = 0, 1, 2, 3, 4, 5, 6, 7

def A(t, v):  return {"type": t, "value": v}
def se2(x, y, yaw):
    return {"rt_translation": A(VECF, [x, y, 0.0]),
            "rt_rotation_euler_xyz": A(VECF, [0.0, 0.0, yaw])}

def R(a):     return ((math.cos(a), -math.sin(a)), (math.sin(a), math.cos(a)))
def mul(A_, B):
    (a,b),(c,d) = A_; (e,f),(g,h) = B
    return ((a*e+b*g, a*f+b*h), (c*e+d*g, c*f+d*h))
def apply(M, v):  return (M[0][0]*v[0] + M[0][1]*v[1], M[1][0]*v[0] + M[1][1]*v[1])

class T:
    """Minimal SE(2): (R(yaw), t). Composition is parent <- child, like a DSR RT edge."""
    def __init__(s, x, y, yaw): s.x, s.y, s.yaw = x, y, yaw
    def __mul__(a, b):
        x, y = apply(R(a.yaw), (b.x, b.y))
        return T(a.x + x, a.y + y, a.yaw + b.yaw)
    def inv(s):
        x, y = apply(R(-s.yaw), (-s.x, -s.y))
        return T(x, y, -s.yaw)
    def on(s, p):
        x, y = apply(R(s.yaw), p); return (s.x + x, s.y + y)
    def __repr__(s): return f"(x={s.x:+.6f} y={s.y:+.6f} yaw={s.yaw:+.6f})"

# ── the scene ─────────────────────────────────────────────────────────────────────────────────
ROOM0 = (6.0, 4.0)          # the room being left, centred on its own origin
ROOM1 = (5.0, 4.0)          # the room just entered
DOOR_IN_ROOM0 = (3.0, 0.5)  # on wall_1, the +x wall
ROBOT_IN_ROOM0 = T(2.0, 0.4, 0.0)     # where the robot was, in room_0 coordinates
ROBOT_IN_ROOM1 = T(-1.0, 0.3, 0.6)    # where it is now, in room_1 coordinates -- a real yaw, so
                                      # the seam is not a pure translation

# A DSR RT edge robot->room carries T(robot <- room), which is the INVERSE of the robot's pose in
# the room: exactly what room_scene_graph.cpp:430 computes as t_robot_to_room.
RT_ROBOT_ROOM0 = ROBOT_IN_ROOM0.inv()
RT_ROBOT_ROOM1 = ROBOT_IN_ROOM1.inv()

SEAM = RT_ROBOT_ROOM1.inv() * RT_ROBOT_ROOM0          # T(room_1 <- room_0)
DOOR_IN_ROOM1 = SEAM.on(DOOR_IN_ROOM0)

def rect(w, h):
    return [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]

def walls(poly):
    """(name, mid, yaw, length) per edge — room_scene_graph.cpp:2122-2160."""
    out = []
    for i, p0 in enumerate(poly):
        p1 = poly[(i + 1) % len(poly)]
        d = (p1[0] - p0[0], p1[1] - p0[1])
        L = math.hypot(*d)
        out.append((f"wall_{i}", ((p0[0]+p1[0])/2, (p0[1]+p1[1])/2), math.atan2(d[1], d[0]), L))
    return out

# Real epoch milliseconds, as the live graph carries them. (An earlier version of this file used
# small numbers because the JSON loader looked like it truncated uint64 through a 32-bit
# QVariant::toUInt. That is the EDGE copy of the attribute switch, cortex/api/dsr_utils.cpp:179; the
# NODE copy at :105 already goes through toString().toULongLong() -- commit af03fe6 fixed one twin
# and missed the other. timestamp_creation is a node attribute, so it round-trips correctly.)
# Only the ORDERING is load-bearing: it is what tells eviction which of the two rooms is the new one.
BIRTH0, BIRTH1 = 1_789_200_000_000, 1_789_200_600_000     # room_0 born 10 minutes before room_1
sym, links = {}, {}
def node(nid, name, ntype, attrs, parent=None, level=0):
    # pos_x/pos_y are the DSR viewer's 2-D layout, nothing physical. Spread by level and by the
    # order nodes are declared, so the fixture is readable in the graph window.
    a = {"level": A(INT, level), "pos_x": A(FLOAT, float(-260 + 60 * (len(sym) % 9))),
         "pos_y": A(FLOAT, float(-140 + 70 * level)), **attrs}
    if parent is not None: a["parent"] = A(U64, str(parent))
    sym[str(nid)] = {"id": str(nid), "name": name, "type": ntype, "attribute": a, "links": []}
def link(src, dst, label, attrs=None):
    sym[str(src)]["links"].append({"src": str(src), "dst": str(dst), "label": label,
                                   "linkAttribute": attrs or {}})

node(100, "root", "root", {"color": A(STR, "SeaGreen")}, level=0)
node(200, "Shadow", "robot", {"color": A(STR, "Blue")}, parent=100, level=1)
link(100, 200, "RT", se2(0, 0, 0))

def build_room(base, name, size, birth, robot_rt, with_contents):
    poly = rect(*size)
    node(base, name, "room",
         {"delimiting_polygon_x": A(VECF, [p[0] for p in poly]),
          "delimiting_polygon_y": A(VECF, [p[1] for p in poly]),
          "room_height": A(FLOAT, 3.0),
          "timestamp_creation": A(U64, str(birth)),
          "color": A(STR, "Green")}, parent=200, level=2)
    link(200, base, "RT", se2(robot_rt.x, robot_rt.y, robot_rt.yaw))
    if not with_contents:
        return
    node(base+1, "floor", "floor", {"timestamp_creation": A(U64, str(birth))}, parent=base, level=3)
    link(base, base+1, "RT", se2(0, 0, 0))
    for i, (wname, mid, yaw, L) in enumerate(walls(poly)):
        wid = base + 10 + i
        node(wid, wname, "wall",
             {"width_m": A(FLOAT, L), "height_m": A(FLOAT, 3.0),
              "mesh_path": A(STR, "room_concept/meshes/wall.obj"),
              "timestamp_creation": A(U64, str(birth))}, parent=base+1, level=4)
        link(base+1, wid, "RT", se2(mid[0], mid[1], yaw))
    # The door hangs from wall_1 (the +x wall), its RT expressed IN THAT WALL'S FRAME.
    w1 = walls(poly)[1]
    local = apply(R(-w1[2]), (DOOR_IN_ROOM0[0] - w1[1][0], DOOR_IN_ROOM0[1] - w1[1][1]))
    # ★ ONE JUDGEMENT EPISODE, as door_concept writes it. The stage run harvests this, appends the
    # CSV row, folds it into the passage node's Beta and audits the result -- the whole passage
    # chain, offline. It is an UNCLAIMED crossing (no `passage_outcome`), which is the common case
    # and the one whose folding rule is easiest to get wrong: the crossing itself must NOT fold (the
    # robot only walks through doorways that are already passable), while its `passage_llr` must.
    # llr = ln(10), so the expected posterior is analytic: Beta(1,1) -> Beta(1.638298, 0.936170).
    node(base+30, "door_1", "object",
         {"object_subtype": A(STR, "door"), "width_m": A(FLOAT, 0.9),
          "depth_m": A(FLOAT, 0.05), "height_m": A(FLOAT, 2.0),
          "passage_seq": A(INT, 1),
          "passage_datetime": A(STR, "2026-09-13T10:15:00.000+0200"),
          "passage_crossed": A(BOOL, True),
          "passage_passable": A(BOOL, True),
          "passage_open_prob": A(FLOAT, 0.91),
          "passage_llr": A(FLOAT, 2.302585),
          "passage_clear_span_m": A(FLOAT, 0.98),
          "passage_body_width_m": A(FLOAT, 0.62),
          "passage_open_requested": A(BOOL, False),
          "passage_open_answer": A(STR, ""),
          "passage_outcome": A(STR, ""),
          "passage_duration_s": A(FLOAT, 4.2),
          "passage_episode_cut": A(BOOL, False),
          "timestamp_creation": A(U64, str(birth))}, parent=base+11, level=5)
    link(base+11, base+30, "RT", se2(local[0], local[1], 0.0))
    node(base+40, "table_1", "object",
         {"object_subtype": A(STR, "table"), "width_m": A(FLOAT, 1.2),
          "depth_m": A(FLOAT, 0.8), "height_m": A(FLOAT, 0.75),
          "timestamp_creation": A(U64, str(birth))}, parent=base, level=3)
    link(base, base+40, "RT", se2(-1.0, 0.8, 0.0))

build_room(300, "room_0", ROOM0, BIRTH0, RT_ROBOT_ROOM0, with_contents=True)
build_room(400, "room_1", ROOM1, BIRTH1, RT_ROBOT_ROOM1, with_contents=False)

# NO `current` edge is seeded. The fleet today writes none, and eviction must work from the birth
# stamps alone and then PLACE the first one itself -- the bootstrap path, which is also what a
# fleet starting with a single room hits.

out = os.path.join(os.path.dirname(__file__), "..", "etc", "stage_two_rooms.json")
with open(out, "w") as f:
    json.dump({"DSRModel": {"symbols": sym}}, f, indent=2)
    f.write("\n")

print(f"wrote {os.path.normpath(out)}  ({len(sym)} nodes)")
print()
print("GROUND TRUTH for the self-test (metres, radians):")
print(f"  RT robot->room_0        {RT_ROBOT_ROOM0}")
print(f"  RT robot->room_1        {RT_ROBOT_ROOM1}")
print(f"  seam  T(room_1<-room_0) {SEAM}")
print(f"  door_1 in room_0        ({DOOR_IN_ROOM0[0]:+.6f}, {DOOR_IN_ROOM0[1]:+.6f})")
print(f"  door_1 in room_1        ({DOOR_IN_ROOM1[0]:+.6f}, {DOOR_IN_ROOM1[1]:+.6f})")
