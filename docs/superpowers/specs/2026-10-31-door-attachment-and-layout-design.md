# The doorway as the rooms' one attaching point; the layout optimizer's poses used — design

2026-10-31 · branch `feat/room-recognition` (worktree rt-build) · status: **APPROVED (Q1, Q2 decided)**
Relates to: `2026-10-29-room-transition-protocol-design.md` (Rev 3, exit chain), ltsm `layout_optimizer.{h,cpp}` (main,
52149a90, not yet on this branch), `room_scene_graph.cpp::find_recall`.

## 1. Why

A remembered room re-enters the live graph placed THROUGH THE DOOR just crossed. Three live facts say that placement is
a guess today:

| run | observation |
|---|---|
| pingpong16 leg 3 | recalled room_2 drawn ~4.6 m inside room_1; scan vs recalled model misfit 1.1 m (0.01–0.03 when built) |
| pingpong19, every recall | `find_recall`'s "turned face" π flip fired every time (door_2 −2.2°, door_4 +177.8°, door_7/9 +177.7°) |
| pingpong18 | the remembered face of room_1's door moved 25 cm along its wall between two visits |
| pingpong20 | the room-frame vantage 1.4 m off the axis of the door the anchor perceives |

`find_recall` builds the live door's frame from the aperture's two ends `a → b` (`T_int_door`), whose order is not a
convention, and then decides by a CENTROID TEST whether the recalled outline lands on the wrong side and must be turned by
π. For room_2, whose frame sits at its door, the two answers are room_2 and its mirror image inside room_1. One wrong
answer is pingpong16.

Meanwhile ltsm already solves the whole memory's metric layout as a MAP estimate (seam mixture, Manhattan, non-overlap,
shared wall, adjacency — the user's "adjacency, non-intruding, parallelism") every time a new room is archived, and
**nothing uses the result**: the corrected poses are cached for display; write-back was deferred (memory note
ltsm-map-layout-optimizer, "v2"). And that work is not on this branch.

## 2. The design

### D1 — One convention for a door FACE, applied where faces are composed (room_concept, ltsm)

AMENDED 2026-10-31 (user approved): a door node's frame takes its yaw from its WALL (x = the wall tangent, y = x + 90°),
so "+y into its room" holds only for a counter-clockwise outline — pingpong19's room_1 (area −60.4 m², clockwise) and
room_2 (+66.8 m²) put DOOR_1's two faces π apart, which is why the centroid flip fired on every recall. door_concept's
frames are LEFT AS THEY ARE (its leaf / hinge / swing models are defined in them). The convention is applied where faces
are composed: a FACE = the door node's pose in its own room, turned by π if needed so that +y points INTO that room, decided
LOCALLY — the point half the door's width along +y from the aperture centre must lie inside the room's own outline. No
other room is consulted.

Consequence: the two faces F_a (in room a) and F_b (in room b) of ONE doorway are related by a half-turn, always:

    T(a ← b) = F_a · Rπ · F_b⁻¹                         (the attaching point, exact; no other choice exists)

### D2 — Recall placement is that composition (room_concept)

`find_recall` uses the live door's **RT frame** (the node's pose in the current room, D1), not the aperture's `a → b`:

    T_int_recall = T_int_room · F_live · Rπ · F_mem⁻¹

F_live = the live face's pose in the current room (its RT), F_mem = the remembered room's own face of the same doorway in
memory (what ltsm already publishes on the recall's RT, `face ← room`, inverted). The centroid test and its flip are
deleted. A recall whose composition puts the outline overlapping the current room is then a FACT about a wrong face
pairing, logged and refused (entering as a new room), not silently turned.

### D3 — The seam the optimizer sees is the same composition (ltsm)

The optimizer's SEAM measurement for a passage is D1's `F_a · Rπ · F_b⁻¹` with the faces' covariances propagated, replacing
today's chain/mouth choice. Its WALL term (both faces on one wall line, parallel, thickness apart) stays: with D1 it pins
the attaching point; together they make the doorway the rooms' one shared point, as the user asked.

### D4 — The corrected poses are written back and used (ltsm, room_concept)

After each solve ltsm writes the MAP poses T(map ← room) and their Laplace covariances into memory (the dated generation it
already saves). Then, on a recall whose BOTH rooms are in memory (the current room was itself recalled or archived
before), ltsm publishes on the recall node the layout's `T(current ← recalled) = T(map ← current)⁻¹ · T(map ← recalled)`
with its covariance, next to the door's. room_concept fuses the two placements by precision (both Gaussian: the door's from
the two face covariances, the layout's from the Laplace one) and logs their disagreement (Mahalanobis). When the current
room is not yet in memory (the first return out of a new room), the door composition D2 alone places it.

### D5 — Bring the optimizer onto this branch first

Cherry-pick 52149a90 (and its follow-ups touching `layout_optimizer`, `layout_sidecar`, `memory_store`, `topology_*`)
onto `feat/room-recognition`, resolve against today's ltsm changes (owners-delete, recall diagnostics), rebuild, run
`tools/layout_optimizer_test` and `tools/layout_harness` on the pingpong memories.

## 3. What does NOT change

The exit chain (links 1–5), the controller, odometry-only crossing, the open-door gate, the survey-until-established
rule. The TLA model is unaffected (it models identity and order, not geometry).

## 4. Tests

1. door_concept: every face's +y points into its own room (fit, mirror, restore) — unit test on synthetic rooms.
2. room_concept: `find_recall` composition on synthetic pairs, both a→b orders, both rooms' frames anywhere: the recalled
   outline never overlaps the current room; the old centroid flip gone.
3. ltsm: optimizer seams from D1; `layout_optimizer_test` + `layout_harness` on pingpong14/19 memories.
4. Replay check: for pingpong16/19 recalls, recompute D2's placement from the logged factors and compare with ground truth.
5. Live: pingpong (6 round trips) on the pingpong19 controller; grade on ground truth; recall misfit (median |SDF| in the
   30 s after each re-entry) must stay at the "built" level (< 0.1 m) on every recall.

## 5. Order of work

D5 → D1 (+ test 1) → D2 (+ test 2, 4) → D3 → D4 → live (test 5).

## 6. Decisions (user, 2026-10-31)

- Q1. D4 fusion vs replacement — DECIDED: the optimizer's own inlier/outlier mixture judges the seam; the door placement
  stands when the layout calls that seam an outlier.
- Q2. DECIDED: v1 moves rooms only; a face's along-wall position becomes a nuisance variable in v2.
