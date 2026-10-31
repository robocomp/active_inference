# Door crossing (room_1 ↔ room_2 ping-pong): state of the work, for a deep review

2026-10-31 · worktree `active_inference.worktrees/rt-build`, branch `feat/room-recognition` · ~81 commits since 10-30.
User, 10-31: "Two weeks to cross a damn door. It is too much." This document is the honest state, written to be reviewed.

## 1. The task

Webots, robot Shadow (DIFFERENTIAL base). Three rooms; the test is `estimation_scripts/scripts/pingpong_tour.sh`: 6 round trips
room_1 ↔ room_2 through DOOR_1 (DOOR_0, DOOR_2 kept shut), each leg a mission `cross new` (leg 1) or `cross room_k` (door
resolved by ltsm's `recall_room_k` under it), graded on Webots ground truth; `trace_check.py` checks protocol invariants on a
recording of every graph write.

Agents (DSR shared graph, each owns its nodes): controller (mission, door passage, motion), door_concept (doors: detection,
fit, leaf, existence, affordances approach/open/cross, robot-frame "interaction anchor" `door_N@robot`), room_concept (room
layout estimation, localisation = SDF fit + relocaliser, room switch), ltsm_agent (long-term memory: archive left room,
recall, room recognition, layout optimizer), residual_concept (obstacles).

## 2. Results (legs passed / 12)

| run | result | what ended it |
|---|---|---|
| pingpong3–8 | 0,1,3,5,4,5 | protocol bugs (exit chain), route planner "no free point past the door" |
| pingpong9 | 2 | relocaliser flipped pose 177° at a re-entry |
| pingpong10 | 3 | route planner P_post |
| pingpong11 (AIF servo) | 4 | relocaliser 179° flip |
| pingpong13 (deep entry) | 4 | door_concept refit the crossed door (pos 1 m, leaf 85→66°) ⇒ door_wont_open |
| pingpong14 | 6 | robot on the jamb (perceived opening 0.1–0.2 m off) + false THROUGH |
| pingpong16–18 | 1–3 | door open accepted but not moved; recalled room 4.6 m off; servo stalled on withdrawn affordance |
| **pingpong19** | **7** | route planner P_post |
| pingpong20–22 (vantage-only route) | 1 each | design failed; REVERTED |
| pingpong23 | 1 | route planner P_post |
| pingpong24 | 1 (harness) | `cross new` took the shut DOOR_0 (correct crossing, grader wrong) |
| **pingpong25** | **9** | relocaliser took a 90° mode in room_2: room_2's stored outline included a 7.3×2 m band of room_1 seen through the door |
| pingpong26 / 27 | 0 | picker chose the bathroom door (all doors shut); a PHANTOM door_1 on the other wall (door_concept existence p 0.05) was chosen and driven at |

## 3. What is in the code now (what each piece is for)

Protocol (room transition, spec `docs/superpowers/specs/2026-10-29-room-transition-protocol-design.md` Rev 3, TLA+ model
`specs/tla/RoomTransitionExit.tla`, ping-pong config passes): the crossing is a FACT written by the agent that performs it —
`exit` edge chain: (1) controller robot→OLD_FACE at THROUGH; (2) room_concept OLD_FACE→NEW_ROOM on switching; (3) door_concept
NEW_FACE→OLD_FACE (twin); (4) ltsm OLD_ROOM→NEW_ROOM archived; (5) each owner deletes its own nodes. `common/room_transition/exit_fact.h`.

Localisation across the doorway: room_concept odometry-only while a door of its room is being crossed (Lucas Bonilla TFM §5.1),
relocaliser disarmed while the transition is open (cross executing or link 2 alive).

Controller doorway (route mode `[DoorPassage] UseRoute=true`):
- route on the room grid to door_concept's vantage P_pre, then aperture, then P_post (since 60d31084 P_post = door_concept's
  cross target in the door frame, NOT searched in the room grid — Lucas's three intention targets);
- `CrossServo=true`: once at the vantage AND in front of an open door (open-door gate: P(open)≥½, opening seen, s<0, |e|<w/2,
  facing, at vantage) the doorway is driven by an Active-Inference point-ahead law on the robot-frame anchor
  (`controller/src/door_cross_aif.h`; harness 75/75) through the old anchor-servo guard chain (`step_anchor_cross`);
- THROUGH (link 1) = the servo at the cross target (1.3 m past the plane) AND the body past the perceived opening;
- latched anchor keeps the servo going after door_concept withdraws the cross; withdrawn cross before the body is in the
  opening ⇒ back to the approach; escalating escapes (4 levels) then give up "stuck"; `cross new` picks the closest OPEN door.

door_concept: a face backed by a crossing keeps pose and leaf (no refit). UNCOMMITTED (built, unrun): every offer (approach /
open / cross) multiplied by P(exists), so a phantom affords nothing.

Room recall placement (spec `specs/2026-10-31-door-attachment-and-layout-design.md`, D1+D2 done): a recalled room is placed
through the door by the exact two-face composition `F_live · Rπ · F_mem⁻¹` (`common/room_transition/door_face.h`, 384-check
test) — replaced a centroid-test π-flip guess. Live: placements repeat within 5 mm.

Room outlines: room_concept's free space stops at the door plane (46cc262a, geometric ray cut; replaced the probabilistic
filter). Unrun live.

ltsm layout optimizer (MAP over room poses: seams, Manhattan, non-overlap, shared wall, adjacency) cherry-picked (28a27091);
D3 (its seams from the same face composition) and D4 (its corrected poses used) NOT done.

## 4. The pattern the user objects to

Almost every run exposed a new failure in a different place; almost every fix was a new rule added where it showed up. Three
fixes were reverted (vantage-only route). Several mechanisms overlap: the controller's open-door gate and door picker re-judge
what door_concept already judges in its affordance preconditions; the controller holds TWO pictures of the door (room-frame
door node vs robot-frame anchor / measured gap) which disagreed by up to 1.4 m; room outlines were polluted through doors.

## 5. Questions for the reviewer

1. Is the architecture right, or is the problem framed wrongly? What is the SIMPLEST structure that crosses a door reliably
   (Lucas's TFM did 3 door-relative intention targets + odometry-only + exit edge, and it worked)?
2. Which of the mechanisms in §3 are essential, which are patches to remove, which are duplicates?
3. The two-pictures-of-a-door problem: which single representation should drive the crossing?
4. What should be done next, in what order, and what should be deleted?
