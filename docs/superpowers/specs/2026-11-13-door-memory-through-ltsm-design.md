# Door memory through LTSM; refuted doors in a private phantom file

**Status:** BUILT 2026-11-14 (Task 6 done)
**Agents touched:** door_concept, ltsm_agent, cortex (new attribute types; the owner reinstalls).
**Rules this serves:** agents talk only through memories (no hidden world memory in an agent); facts about the
WORLD live in LTSM, facts about an agent's own SENSOR stay private to that agent (owner ruling 2026-11-13).

## 1. Problem

door_concept keeps a second, private memory of the world: `door_concept/etc/door_identities.csv`
(`door_identity_table.h` `IdentityRow`, loaded into `core/place_table.h` `PlaceTable`). When LTSM recalls a room,
door_concept rebuilds every remembered door from that file (`restore_remembered_doors`, specificworker.cpp ~4025),
including doors that are out of sight. LTSM does **not** give door_concept the remembered doors: on recall it posts
only metaconcept notes — start-up recall, late recall, `proposal_*`, `recall_room_N` under an already-live door, and
the `ltsm_topology` JSON (face names, no geometry). Checked 2026-11-13 against `ltsm_agent/src/room_recall.cpp`,
`room_proposal.cpp`, `topology_publish.cpp`, `passage_live.cpp`, `room_recognizer.cpp`.

Two memories of the same doors already disagree in practice:
- the "entered FRESH … reduced to identity only" path exists only to stop the file outliving LTSM;
- `door_identities.csv.corrupt_door22_20261112` sits in `door_concept/etc`;
- the file keys a door by ROOM NAME; LTSM can rename a room (recognition swap) or re-solve its layout.

Meanwhile LTSM already archives each door node with ALL its attributes (`RoomEviction::twin`,
room_eviction.cpp:207–263: copies `src.attrs()` minus parent/level/RT). So whatever door_concept publishes on the
door node is remembered for free; what is missing is (a) the evidence that only the file holds and (b) a way back
from LTSM to door_concept on recall.

## 2. What is world, what is sensor

| Item | Today | Kind | Goes to |
|---|---|---|---|
| aperture pose in the room, width, height, depth | file + node | world | LTSM (already on the node) |
| leaf angle φ, hinge side, swing side | file + node | world | LTSM (already on the node) |
| existence evidence `L_evidence`, its physical part `Sp` | file only | world (belief about it) | **new node attrs** → LTSM |
| crossed (proof for ever) | file only | world | **new node attr** → LTSM |
| last seen / last sighted (wall ms), place σ along the wall, height σ | file only | world (age + precision of the belief) | **new node attrs** → LTSM |
| view ledger (one trial per viewpoint, κ) | file only | world (what has been looked at) | **new node attr** → LTSM |
| visit counter | file only | world | **new node attr** → LTSM |
| refuted doors (`removed_ms`, place, height): "no door here" | file only | **sensor**: the detector lied here | **private phantom file** (§5) |
| change hazard learned from revisits | place table | agent model parameter | stays private (unchanged) |
| detector calibration (`door_detectors.csv`) | private | sensor | stays private (unchanged) |

## 3. Door → LTSM: publish the belief on the node

door_concept writes, on its own door nodes, at the rate it already updates them (no new timer):

| attribute (proposed name) | type | from |
|---|---|---|
| `door_exist_evidence` | float | `PlaceBelief::L_evidence()` |
| `door_exist_physical` | float | `PlaceBelief::Sp()` |
| `door_crossed` | bool | belief `crossed` |
| `door_seen_ms` | uint64 | `last_seen_wall_ms` |
| `door_sighted_ms` | uint64 | `sighted_wall_ms` |
| `door_place_sd` | float | aperture σ along the wall |
| `door_height_sd` | float | fit posterior sd of height |
| `door_view_ledger` | string | `ViewLedger::serialise()` (already one comma-free cell) |
| `door_visit` | int | visit counter |

Each needs a `REGISTER_TYPE` line in cortex `dsr_attr_name.h`; **the owner reinstalls cortex**, then door_concept and
ltsm_agent are rebuilt. Type-attributed access only (`<door_exist_evidence_att>`), never `runtime_checked_*`.
Verify the existing names for width/height/depth/leaf before adding any (reuse, don't duplicate).

LTSM changes nothing to archive them: `twin()` copies them with the node.

## 4. LTSM → door: a "remembered door" note on recall

When LTSM makes a remembered room the robot's room — start-up recall, late recall, or an ACCEPTED recognition
proposal — it posts, for each remembered door face of that room, one note into the live graph:

- type `metaconcept`, subtype `remembered_door`, name `remembered_<face name>`;
- parent = the live room node; RT edge room → note = the face's pose **in that room's own frame** (the recalled room
  is published in its remembered frame, room_recall.cpp:154, so the memory pose is directly usable);
- attributes: every archived door attribute of the face (geometry, leaf, and the §3 evidence), plus `room_id` and
  the face's memory name;
- the note is a FACT carrier, not a door: never type `door`/`object` (door_concept's owned-node sweep deletes those
  by name; same reason `passage_*` and `recall_room_N` are metaconcepts).

Lifetime: LTSM deletes the notes of a room when that room stops being current (eviction) or when door_concept has
adopted them (door_concept sets `door_adopted`; LTSM deletes adopted notes). This mirrors the existing `recall_room_N` / `proposal_*` notes; reuse their helpers
(`RoomRecall::live_name_of`, `RoomEviction::rt_attributes`).

**door_concept on a note.** `restore_remembered_doors` keeps its logic — `plan_restore` (skip a face a live door
already holds, by place), `restored_belief` (age the evidence by the learned hazard), `restore_leaf_phi` (age the
leaf), `seed_restored_instance` — but its INPUT becomes the notes instead of the CSV rows. It creates its own door node
(birth stays door_concept's; lifecycle unchanged), then marks the note adopted by setting `door_adopted` (bool) on it; it never deletes
the note. The `pending_restore_room_` wait for the outline
stays.

**Matching a live door to a face** stays as today: by place (`RoomEviction::remembered_hole`), names are opaque.

## 5. Private phantom file — refuted places, per agent

Owner ruling: refuted detections are a fact about the agent's own detector, so each concept agent keeps them in a
private file. This is the persistence for `MODEL_HISTORY.md` §4 ("Phantom memory"); update §5 of that doc, which
had left persistence open and leaned to a graph attribute.

- **File:** `<agent>/etc/<class>_phantom_memory.csv`, one shared reader/writer in `common/` (e.g.
  `common/phantom_memory/phantom_memory.h`), written with `imbue(std::locale::classic())`, read with
  `std::from_chars`.
- **Key:** LTSM's `room_id` (the stable id on the room node, not its name) + cell in THAT room's frame
  (+ view-bearing bin, for the §4 field). A room name or a global position would go stale when LTSM renames or
  re-solves a room.
- **Row (door, phase 1):** room_id, x, y, yaw, width, height, height_sd, removed_ms — exactly what door_concept's
  removed rows hold today, so the remembered-height fix (c63d56d9) keeps working.
- **Row (general, phase 2 — §4 field, NOT in this spec's build):** room_id, cell, bearing bin, confident-death
  count, exposure, last update ms. Only confident disconfirmations count (§4 attribution rule).
- **Forgetting:** an entry whose `room_id` is no longer in LTSM's memory (`ltsm_topology` room list) is dropped.
  The memory scales birth evidence, never vetoes (§4) — unchanged.

## 6. Migration

1. cortex attribute types (owner reinstall) → rebuild door_concept + ltsm_agent.
2. door_concept publishes §3 attrs (behaviour unchanged; LTSM starts archiving them).
3. ltsm_agent posts §4 notes; door_concept restores from notes **behind a flag** `DoorConcept.RestoreFromLtsm`
   (default false). A/B: notes vs CSV on the same tours.
4. Phantom file (§5 phase 1): move removed rows out of `door_identities.csv` into `door_phantom_memory.csv`;
   a one-off converter maps room names → room_id via the current LTSM memory and drops rows it cannot map.
5. When the flag has passed (§7), flip the default, delete the restorable columns from the identity CSV and the
   "entered FRESH / strip" path. What remains private in door_concept: the phantom file, the hazard, detector
   calibration.

## 7. Acceptance

Unit (door_concept/tests, ltsm_agent/tools):
- note → restored belief equals CSV row → restored belief for the same door (round trip of every §3 field);
- `plan_restore` skips a face a live door holds; a refuted place in the phantom file is not restored nor re-born
  from a weak detection (shower-glass case);
- phantom file round trip under `setlocale(LC_ALL, "")` (comma locale), keyed by room_id; rows of a forgotten room
  are dropped.

Live (threerooms-piso.wbt, explore_tour, flag on vs off, ≥3 tours each):
- on re-entering a remembered room, every remembered door exists within one cycle of the outline loading, including
  doors out of sight; no duplicate door is born on a restored one;
- the GlassBay shower partition is not re-born/removed repeatedly (cf. door_7 in clean60);
- tour pass rate and door-crossing times no worse than the frozen series (clean62–76: 13/15).

## 8. Owner decisions (2026-11-13)

1. Note adoption: door_concept sets `door_adopted` on the note; LTSM (its creator) deletes it.
2. A recalled-but-never-re-seen door's aged evidence IS written back: LTSM archives whatever the node says at
   eviction (no special case).
3. Phase 2 (§4 learnt false-alarm field, all object agents) gets its own spec after this one lands.
