#!/usr/bin/env python3
"""
make_proto_stage_fixture.py — author etc/stage_proto_room.json, the offline PROTO-ROOM graph.

The instant the live passage exists for: the robot has just crossed DOOR_1 out of the apartment and
room_concept has born a PROTO-ROOM for the space beyond it. `current` is still on the apartment.

    root
     +-- Shadow (robot) --current--> room
          +-- room            the apartment (current)            RT: robot -> room
          |    +-- floor   (collapsed=true, as room_concept seeds it)
          |         +-- wall_0 .. wall_3
          |              +-- door_1   on wall_1 (+x), the door just crossed
          |              +-- door_3   on wall_3 (-x), a DISTRACTOR the argmin must not pick
          +-- room_2          PROTO: `proto` self-edge, room_id 2, empty polygon   RT: robot -> room_2
               +-- door_2     door_concept's ENTRY MIRROR of door_1 (parent = room_2), 4 cm off, as a
                              second measurement of the same hole would be

Expected: ONE live passage_0 with match edges to door_1 and door_2; `current` unchanged; no eviction;
the apartment seeded into memory, room_2 not. Then the stage deletes door_2 and the passage must go.

Run from the component root:  python3 tools/make_proto_stage_fixture.py [--promote]

--promote writes etc/stage_promote.json instead: the same scene with the robot 1.2 m past the aperture
(σ 5 cm), so ltsm PROMOTES room_2 on the first cycle, and wall_2 hung from the ROOM rather than the floor
(room_concept's legacy shape) to prove memory re-hangs it under the floor without moving it.
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from make_stage_fixture import A, se2, R, apply, T, rect, walls, STR, INT, FLOAT, VECF, BOOL, U64  # noqa: E402

sym = {}
def node(nid, name, ntype, attrs, parent=None, level=0):
    a = {"level": A(INT, level), "pos_x": A(FLOAT, float(-260 + 60 * (len(sym) % 9))),
         "pos_y": A(FLOAT, float(-140 + 70 * level)), **attrs}
    if parent is not None: a["parent"] = A(U64, str(parent))
    sym[str(nid)] = {"id": str(nid), "name": name, "type": ntype, "attribute": a, "links": []}
def link(src, dst, label, attrs=None):
    sym[str(src)]["links"].append({"src": str(src), "dst": str(dst), "label": label,
                                   "linkAttribute": attrs or {}})
def door(nid, name, parent, level, x, y, birth):
    node(nid, name, "object", {"object_subtype": A(STR, "door"), "width_m": A(FLOAT, 0.9),
                               "depth_m": A(FLOAT, 0.05), "height_m": A(FLOAT, 2.0),
                               "timestamp_creation": A(U64, str(birth))}, parent=parent, level=level)
    link(parent, nid, "RT", se2(x, y, 0.0))

APT = (8.0, 5.0)
DOOR1_IN_APT = (4.0, 0.5)      # on wall_1, +x
DOOR3_IN_APT = (-4.0, -1.0)    # on wall_3, -x
ROBOT_IN_APT = T(4.6, 0.4, 0.3)       # just past door_1
PROMOTE = "--promote" in sys.argv
# The proto-room frame is room_concept's: origin = the crossed aperture centre, +y OUTWARD. s = the robot's y.
ROBOT_IN_R2  = T(-0.8, 1.2, -0.4) if PROMOTE else T(-0.8, 0.1, -0.4)
RT_ROBOT_APT, RT_ROBOT_R2 = ROBOT_IN_APT.inv(), ROBOT_IN_R2.inv()
R2_FROM_APT = RT_ROBOT_R2.inv() * RT_ROBOT_APT                    # T(room_2 <- room)
DOOR2_IN_R2 = R2_FROM_APT.on((DOOR1_IN_APT[0] + 0.04, DOOR1_IN_APT[1]))
BIRTH_APT, BIRTH_R2 = 1_789_200_000_000, 1_789_200_900_000

node(100, "root", "root", {"color": A(STR, "SeaGreen")})
node(200, "Shadow", "robot", {"color": A(STR, "Blue")}, parent=100, level=1)
link(100, 200, "RT", se2(0, 0, 0))
# The base frame promotion reads the footprint from (ROBOT_GEOMETRY.md: body, 0.47 x 0.47 m).
node(201, "body", "body", {"width_m": A(FLOAT, 0.47), "depth_m": A(FLOAT, 0.47)}, parent=200, level=2)
link(200, 201, "RT", se2(0, 0, 0))

poly = rect(*APT)
node(300, "room", "room", {"delimiting_polygon_x": A(VECF, [p[0] for p in poly]),
                           "delimiting_polygon_y": A(VECF, [p[1] for p in poly]),
                           "room_height": A(FLOAT, 3.0),
                           "timestamp_creation": A(U64, str(BIRTH_APT))}, parent=200, level=2)
link(200, 300, "RT", se2(RT_ROBOT_APT.x, RT_ROBOT_APT.y, RT_ROBOT_APT.yaw))
link(200, 300, "current")
node(301, "floor", "floor", {"collapsed": A(BOOL, True),
                             "timestamp_creation": A(U64, str(BIRTH_APT))}, parent=300, level=3)
link(300, 301, "RT", se2(0, 0, 0))
W = walls(poly)
for i, (wname, mid, yaw, L) in enumerate(W):
    node(310 + i, wname, "wall", {"width_m": A(FLOAT, L), "height_m": A(FLOAT, 3.0),
                                  "timestamp_creation": A(U64, str(BIRTH_APT))}, parent=301, level=4)
    # wall_2 hangs from the ROOM in the promote fixture (legacy walls authored before the floor existed);
    # floor->room is the identity, so the same RT reads correctly from either parent.
    wparent = 300 if (PROMOTE and i == 2) else 301
    sym[str(310 + i)]["attribute"]["parent"] = A(U64, str(wparent))
    link(wparent, 310 + i, "RT", se2(mid[0], mid[1], yaw))
for nid, name, wi, p in ((330, "door_1", 1, DOOR1_IN_APT), (331, "door_3", 3, DOOR3_IN_APT)):
    _, mid, yaw, _ = W[wi]
    local = apply(R(-yaw), (p[0] - mid[0], p[1] - mid[1]))
    door(nid, name, 310 + wi, 5, local[0], local[1], BIRTH_APT)

node(400, "room_2", "room", {"delimiting_polygon_x": A(VECF, []), "delimiting_polygon_y": A(VECF, []),
                             "room_id": A(U64, "2"), "room_height": A(FLOAT, 3.0),
                             "timestamp_creation": A(U64, str(BIRTH_R2))}, parent=200, level=2)
# ★ NO rt_covariance HERE: cortex's JSON loader builds every RT link through RT_API from
# rt_translation + rt_rotation_euler_xyz ONLY (cortex/api/dsr_utils.cpp:~256), so any other attribute on an
# RT link is dropped and the edge would silently arrive without a covariance. The promote stage writes σ = 5 cm through RT_API at startup instead
# (SpecificWorker::initialize, Promotion.stage_check).
link(200, 400, "RT", se2(RT_ROBOT_R2.x, RT_ROBOT_R2.y, RT_ROBOT_R2.yaw))
link(400, 400, "proto")
door(402, "door_2", 400, 3, DOOR2_IN_R2[0], DOOR2_IN_R2[1], BIRTH_R2)

out = os.path.join(os.path.dirname(__file__), "..", "etc", "stage_promote.json" if PROMOTE else "stage_proto_room.json")
with open(out, "w") as f:
    json.dump({"DSRModel": {"symbols": sym}}, f, indent=2)
    f.write("\n")
print(f"wrote {os.path.normpath(out)}  ({len(sym)} nodes)")
print(f"  door_2 in room_2 {DOOR2_IN_R2}  -> door_1 in the apartment, 0.04 m off; door_3 is 8 m away")
