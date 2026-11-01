# Warm start from a stored LTSM memory — design

Date: 2026-11-01. Author: Claude with the user (pbustos). Status: user decisions recorded; awaiting approval.

## Intent

Every tour today starts with an empty long-term memory: the robot must first learn room_1 and cross into room_2 before
anything new can be tested. To work on the third room (R3), the fleet must start **already knowing room_1 and room_2**
(their layouts, frames, doors and the passage through DOOR_1) and go from room_1 to room_2 to R3 directly.

User's decisions:
- ltsm is told to load a stored two-room memory; room_concept gets `MapMode = "ltsm"`, in which it estimates nothing at
  start-up and **waits until ltsm hands it room_1's layout**.
- For now ltsm loads **room_1** as the live room. The robot starts as now: the tour teleports it to a central place in
  room_1, and **room_concept localises it** against the loaded layout (its global relocalisation). The robot's pose is
  never part of the stored memory — a tour stopped half-way must not leave the robot remembered in another room.
- The doors of room_1 come up **in the working memory as if door_concept had detected them before**: they enter
  door_concept's normal life-cycle (in sight they gather evidence; out of sight they are left alone). ltsm does nothing
  else about them for now.
- **The stored memory is a fixed fixture:** every run starts from the same R1+R2 memory; what a run learns (R3) is not
  written back into it for now.

## Goals

- One switch in room_concept's config (`MapMode = "ltsm"`) and one in ltsm's (the stored memory to load) turn the warm
  start on; `given` / `estimate` / `last` are unchanged.
- After start-up, with no mission sent: `room_1` live, frozen and localised in **its stored frame**; its doors in the
  graph; ltsm's `recall_room_2` under the DOOR_1 door node (the existing RoomRecall, which needs exactly that); the room
  name counter past room_2, so the next new room is `room_3`.
- The first mission can be `cross room_2` (the ping-pong mechanism, unchanged), then exploration from room_2 toward R3.

## Non-goals

- Starting in any room other than room_1, or in a room chosen at run time.
- Recognising where the robot woke up (the 2026-10-22 recognition design covers entry through a never-crossed door,
  not a cold start).
- Writing a run's learning back into the stored memory.
- Any change to the crossing / recall / eviction machinery beyond what start-up needs.

## Anti-goals (a failure even if it "works")

- room_concept guessing a layout or a pose when ltsm has not delivered, or when the global search cannot decide: it
  holds and says so instead.
- A live room_1 in a frame other than the stored one (ltsm matches rooms by name `r<id>_<name>` and doors by PLACE in
  the room's own frame; a different frame silently corrupts both).
- A new room named `room_2` (the remembered room_2 is then taken for it).
- A run that modifies the stored two-room memory.

## What exists (recon 2026-11-01, worktree rt-build)

- **ltsm persistence** (`ltsm_agent/src/memory_store.cpp`, config `[Memory] persist / persist_path / history_dir`):
  loads the newest dated generation in `history_dir`, else `persist_path`; saves a new dated generation (startup seed,
  each eviction, forget, shutdown). One location for read and write.
- **Recall at a crossing** (`ltsm_agent/src/room_recall.cpp`): ltsm creates NO room/floor/wall/door node in the live
  graph. It posts `recall_<room>` (`metaconcept`, `object_subtype "room_recall"`, `room_id`, `delimiting_polygon_x/y` in
  the room's own frame, RT child of the live door, T(face←room)). `room_concept` (`room_scene_graph.cpp` 1135-1183,
  1639-1740, 1832-1855) creates the room node with that name/room_id and adopts the polygon as given
  (`adopt_map_layout_as_given`). A late variant (`post_late`, `recall_pose_prior`) hangs under the robot.
- **room_concept start-up**: given mode creates `room_1` (`dsr_create_room_and_reparent`) from an SVG layout and runs
  the global relocalisation (`bootstrap_initialization_from_lidar`: grid search, or validation of
  `etc/last_robot_pose.txt` when present). A startup sweep deletes existing room nodes unless
  `RoomConcept.PreserveBootstrapRoom`.
- **Room names**: room_concept alone allocates them (`etc/room_name_hwm.txt` + the live graph).
- **door_concept**: deletes every `door*` object node at start-up (`remove_owned_door_nodes`), and later adopts any
  `door*` object node with a room→door RT edge (`DoorFitter::ensure_instance`, `pre_belief_geometry`). It also has its
  own door memory, `etc/door_identities.csv` (faces keyed by room name, restored onto detections); the tours clear it.
- **ltsm seed**: a live `room_1` with `room_id 1` matches the remembered `r1_room_1` → "already remembered", nothing
  copied; evictions then reuse the `r1_` prefix.
- **RoomRecall** posts `recall_room_2` under a live door when that door lies (by place, within half its width) on a
  remembered face of a passage whose other face is in another remembered room — which requires room_1 published in its
  stored frame.

## The design

### 1. The stored memory (a fixture)

A directory holding what one successful run left behind, frozen: the ltsm memory generation (memory graph + layout
sidecar) and whatever else must agree with it (see §4, §5). The first fixture is taken from **pingpong41** (12/12 legs,
memory r1 = 31 walls, r2 = 8 walls, passage through DOOR_1). Each run copies the fixture into ltsm's `history_dir`
(after archiving whatever is there), so ltsm loads it with the existing loader and saves the run's own generations
beside it; the fixture itself is never written. No ltsm persistence code changes for this.

### 2. ltsm: deliver room_1 at start-up

New config `[WarmStart] Room = "room_1"` (absent = off). After loading the memory and once the robot node exists
(robot_concept's, born before the others), ltsm posts **one** recall node for that room in the live graph, the same kind RoomRecall posts at a crossing:
`recall_room_1`, `metaconcept`, `object_subtype "room_recall"`, `room_id` (1), the stored outline
`delimiting_polygon_x/y` in the room's own frame, and a marker that it is a **start-up** recall (it hangs from the
robot node, carries no door transform, and claims no pose). The outline is read
from the same memory attributes RoomRecall reads. ltsm keeps the node until room_concept has created the live `room_1`,
then retires it (the RoomRecall rule "the room is live again"); if room_concept restarts before that, the node is simply
still there. *(Builder-check answers, Claude's call.)*

### 3. room_concept: `MapMode = "ltsm"`

A third map mode. At start-up room_concept still runs its stale-node sweep (a previous crashed run's `room_1` must not
be taken for the delivered one), then creates no room and estimates nothing; it waits for a start-up recall
(`object_subtype "room_recall"` hanging from the robot, start-up marker). Then:
1. creates the live room node with the recall's name and `room_id` (`room_1`, 1), its outline adopted as the given,
   frozen layout in that frame (the same adoption a re-entry uses: `adopt_map_layout_as_given`);
2. localises the robot with the existing global relocalisation against that layout, room-wide, with **no saved pose
   and no hint** (`last_robot_pose.txt` is ignored in this mode — Claude's call: the user wants the robot found, not
   told); if the search cannot decide, the pose is held as undecided and
   nothing downstream acts on it (existing `pose_undecided_` path);
3. names the live room after the recall (`room_1`, room_id 1); the counter comes from the fixture's file (§5);
4. from then on behaves exactly as a recalled room (crossings, hand-overs, recalls unchanged).
While waiting it says so in its status; it never falls back to estimating or to an SVG.

### 4. room_1's doors in the working memory

**door_concept creates them** (user's option (a)), through the path it ALREADY has for a recalled room
(`enter_room_memory` → `restore_remembered_doors`, `specificworker.cpp:3742-3860`): when door_concept has seen ltsm's
`recall_<room>` for the room it enters, it restores that room's doors from its own remembered-door table
(`door_identities.csv` rows WITH a pose), as door nodes of its own, then the normal life-cycle takes over. Two
consequences (revised 2026-11-01 while planning, superseding the first draft):
- the start-up recall carries **no door data**: the fixture includes door_concept's table from the same run
  (pingpong41's ends with `room_1:door_1` at (-0.54, 4.61), yaw 3.13 — DOOR_1, matching ltsm's archive of that face);
- door_concept's entry rule `recalled = announced and not first_resolution` (`door_identity_table.h:416`) gains one
  exception: an announcement by a **start-up recall** counts on the first room too.

### 5. Names

The fixture includes room_concept's own counter file `etc/room_name_hwm.txt` (= 2 for the pingpong41 memory), installed
by the tour. (Revised while planning: carrying the highest `room_id` on the recall would need a new DSR attribute and a
cortex reinstall; the counter file needs nothing new.)

### 6. The tour

A new tour script (or a flag of the ping-pong/three-rooms script) that installs the fixture instead of clearing memory,
sets `MapMode = "ltsm"` / `[WarmStart] Room = "room_1"`, teleports to the usual start pose, and waits for
`room_1` localised + `recall_room_2` posted before the first mission. Its legs: `cross room_2`, then explore room_2 and
cross into R3, and back. DOOR_2 is opened the way the cross mission opens DOOR_1 (DoorControl); the builder checks the
bridge advertises DOOR_2 and, if not, flags it as a separate Webots-side item.

## Left to the builder

- Exact names of the config keys and of the start-up marker attribute.
- Whether `MapMode = "ltsm"` reuses the recall-adoption code path or the late-recall in-place adoption
  (`room_scene_graph_late_recall.cpp`), whichever needs fewer special cases.
- Readiness checks and log lines for the tour.

## Testing

- Offline: room_concept's adoption of a start-up recall (unit level, as the existing recall tests), name counter.
- Live acceptance: from the pingpong41 fixture, after start-up with no mission: room_1 localised (pose vs ground
  truth within 0.10 m / 3°, Claude's call), its doors in the graph, `recall_room_2` under DOOR_1's node, then `cross room_2` succeeds, then a new room is
  named `room_3`. The fixture files are byte-identical after the run.

## Calls made by Claude (for the user to check)

- The robot node is robot_concept's and exists first; ltsm posts the start-up recall after it and keeps it until room_1 is live.
- room_concept still sweeps stale room nodes before waiting.
- Global relocalisation room-wide, no pose hint; `last_robot_pose.txt` ignored in this mode.
- (Revised) The name counter comes from the fixture's `room_name_hwm.txt`; doors from door_concept's own table.
- The fixture is copied into ltsm's `history_dir`, never used in place.
- Acceptance tolerance 0.10 m / 3°.
