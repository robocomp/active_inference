# Crossing experiment — the room agent must estimate, not relocalise

**Question.** When the robot is driven out of a known room through one of its doors, does room_concept
recognise that it left, birth a proto-room, and switch to ESTIMATING the new space — instead of trying
to relocalise its way back into a room it is no longer in?

This is a **topology and branch test**, not a geometry test. Nothing here scores a layout.

---

## Why it is not obvious

★ **Losing the pose and crossing a doorway look identical from inside the room**: SDF misfit up, pose
degraded. No threshold on misfit can separate them, which is why the discriminator is the door aperture
as a *prior* plus the footprint's position relative to it, and not a gate on the residual.

The failure it prevents is concrete: on 2026-09-14 the localiser lost the pose in the doorway
(`pose_trace` y 3.69 → (−0.42, −3.00) at 23:00:50) and settled wrong, because recovery kept scoring the
scan against the apartment while the robot was already past the aperture.

---

## What was built (2026-09-16)

| piece | where | note |
|---|---|---|
| crossing evidence `p_cross` | `src/door_crossing.h` | logit-sum of `door_open_prob` and P(footprint past the line)·P(in span). ★ NOT a product — a live `door_open_prob` of 0.14 can never reach 0.95 through a product |
| proto-room birth | `room_scene_graph.cpp: step_proto_room` | `room_<k>`, `proto` self-edge, frame = the crossed aperture, frozen |
| **runtime LOCALIZING → SEARCHING** | `room_concept.cpp: begin_room_estimation` | sets the runtime estimate flag, clears `wall_frozen_`, calls `configure_room_estimate()` |
| **room RT retired** | `room_scene_graph.cpp: mark_room_rt_not_current` | `valid=false`, written once; node, doors and ltsm's `current` left alone |
| proto becomes primary child | `write_robot_room_rt` | re-dispatch at the top; the room's edge is never written again |
| gauge protection | `reanchor_map_frame` + `take_pending_reanchor` | the one-shot re-anchor is absorbed into `T_room_proto_` so the published frame never moves |

★ **Recovery is not switched off by hand, and must not be.** It gates on `map_guided_checks_allowed()`
→ `map_ready()` → `map_ready_`, which `configure_room_estimate()` clears. *Entering SEARCHING is
leaving the relocalisation branch.* A second switch would be a place for the two halves to disagree.

Build: clean. `gn_selftest`: ALL PASS (0 failures). `ProtoRoom.Enabled = false` restores the
single-room agent exactly — nothing outside `step_proto_room` sets `room_rt_retired_`.

---

## Setup

- World: `webots-shadow/worlds/piso/tworooms-piso.wbt`. **DOOR_1** — the only door with a room behind
  it; DOOR_0 opens onto bare ground.
- `etc/config.toml`: `MapMode = "given"` (start localised in the apartment), `[ProtoRoom] Enabled =
  true`, `BirthProb = 0.95`. All three already set.
- Driving is **joystick**. The controller cannot plan through a door — the grid marks outside-polygon
  occupied, a `cross` target is rejected `OutsideRoom`, and the door node is an obstacle box.
- Open the door with door_concept's strip button before crossing.

---

## What to read — `tmp/proto_room.csv`

Nothing worth reading goes to the terminal: you are driving. The terminal gets one line per one-shot
transition (`BORN`, `RT RETIRED`, `LOCALIZING -> SEARCHING`) and warnings; **everything else is in the
CSV**, written by the localiser thread at the pose rate.

**One row per pose update, on every path.** An early return is a row with a reason, never a missing
row — so a gap in `ts_ms` means the agent stopped, not that the code took a branch nobody logged.

★ **Absence is `nan`, never 0 or −1.** Every aperture column is a probability, a distance or a sigma,
and 0 is a legal reading of all three — `p_open=0` is "door_concept says shut", which is *data*. A
frame with no aperture offered writes `nan` across them and `n_apert` says why. Assert through the parse.

| group | columns |
|---|---|
| join keys | `ts_ms` (POSE stamp — joins pose_trace and the localiser log), `wall_ms` (joins `tmp/crossing.csv`) |
| event | `event` ∈ {`` , `born`, `reanchor`} |
| pose | `x y theta sig_x sig_y sig_theta` |
| the door's half | `n_apert door p_open` |
| the pose's half | `p_past p_span p_geom s u sig_s sig_t span_w` |
| the decision | `p_cross`, `birth_prob` (the configured level, so the file is self-describing) |
| branch state | `proto_id retired estimating searching map_ready outside_prob` |
| what recovery saw | `sdf_mse pred_sdf_median misfit_raw misfit_weighted iters cond diverged` |
| the gauge | `proto_x proto_y proto_theta` — `T_room_proto_`, so a frame jump is visible directly |

`misfit_raw` is written with **the same expression recovery uses**, not a re-derivation: a column that
computes a quantity its own way cannot refute the code it audits.

### PASS

```
awk -F, 'NR==1||$3!=""' tmp/proto_room.csv        # the transitions
```
- exactly one `born` row, and from it on: `proto_id>0`, `retired=1`, `estimating=1`, `searching=1`,
  `map_ready=0`;
- `misfit_weighted` collapses toward 0 as `outside_prob` → 1 across the doorway;
- no recovery/relocalisation anywhere in the agent log;
- `proto_x/y/theta` **constant** for the whole run, including across any `reanchor` row.

### How it fails, and which file tells you

| what the CSV shows | cause | whose |
|---|---|---|
| `p_cross` plateaus at **0.50** while `p_geom`→1 | `p_open` collapsed (last run 0.028 → 2e-14 → 8e-27); the fuse caps there | **door_concept** — the known blocker |
| `n_apert=0` for the whole approach | no apertures published at all — not a door-belief problem, a plumbing one | door_concept / `door_apertures` |
| `born` present, then `estimating=0` | the branch did not switch | here — `begin_room_estimation` |
| `proto_x/y/theta` step at a `reanchor` row | the gauge move was not absorbed | here — `take_pending_reanchor` |
| `born` while `s` < `r_body` | span/past term | `door_crossing.h` |
| `proto_id` returns to 0 | ltsm `detect()` evicted the 2nd room | **ltsm_agent** — keep eviction off |

★ **The door belief decides this run and it is not in this agent.** Read `p_open` before concluding
anything about the branch logic: if it collapsed, the run says nothing about what was built here.

---

## Known staleness, deliberately not fixed

The camera-calibration channels were handed the **old** room's polygon at startup
(`calib_->set_room_polygon`, `specificworker_startup.cpp`) and `RoomConcept` has no handle on them, so
they keep projecting the apartment until the new polygon closes. Harmless for a topology run; wire it
before trusting any camera-derived number taken past a doorway.
