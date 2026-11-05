# Exploration tour from scratch — doors as curiosity, crossing as intrinsic motivation

Date 2026-11-05. Base: main = 4f9e1c59. For the ECMR 2027 paper (rooms and passages instantiated by exploration; see
memory ecmr-2027-paper-plan).

## Why

The paper's claim is that rooms and doors are pre-existing concepts instantiated from evidence while the robot explores,
building a long-term representation. Every tour so far started from a fixture memory (warm_start_tour.sh,
three_rooms_door0) and was scripted leg by leg (`LEGS="room_2 room_3 room_1"`). The paper needs a run that starts knowing
nothing, and in which the robot decides by itself when to explore, when to cross and when to come back.

## What the user decided (2026-11-05)

1. Room_concept in **Estimate** mode, the three rooms estimated **from scratch**.
2. After entering a room, the robot runs that room's own affordances (`afford_room`, the room's epistemic/coverage
   planners) — **as they are**; the room's camera-coverage phase is NOT made information-theoretic for this work.
3. **Doors are the curiosity drive** through door_concept's existing epistemic **looks** (door hypotheses, valued in nats).
4. **Crossing is an intrinsic motivation**: a found, not-yet-crossed door is something the robot wants to cross.
5. **Leave a room when it is promoted** (fully estimated), with a **maximum-time fallback**.
6. If a not-yet-crossed door appears **before** the room is promoted, the robot **finishes estimating the room first, then
   crosses**.
7. If the room is done and no not-yet-crossed door is known, the robot **returns to the previous room** (through the door it
   came in by).
8. Back in a room already known, it heads for **another door not yet crossed**.

## Goals

- One autonomous run, three rooms of threerooms-piso.wbt, no fixture, no scripted legs, no manual help: every room estimated
  and promoted, every door instantiated and crossed at least once, the LTSM holding all three rooms and their passages at
  the end.
- A per-run record good enough for the paper: per room — entry time, time to promotion, IoU against ground truth (as the
  warm tours grade it), doors found (when, from where), crossings and returns, the reason each crossing/return was chosen.

## Non-goals / anti-goals

- No change to the room's planners or to door_concept's look valuation (decision 2/3). If they are weak, the paper reports it.
- No scripted target sequence in disguise (no list of doors or rooms anywhere in the run's configuration).
- Not a new crossing chain: the existing door passage (approach → open → cross, C1 local policy) does the crossing; the
  re-aim fix of 4f9e1c59 stays.
- Anti-goal: a run that "succeeds" because a timeout fired at the right moment. Every room leave must be logged with its
  cause (promoted / max time), and max-time leaves count against the run.

## Constraints

- Controller rule (CLAUDE.md, controller_approach_order.h): pragmatic door offers are selected only when a passage names
  them. The new drive must name the door through the existing passage API, not by letting the open contest pick `cross`.
- No thresholds where a model term will do (CLAUDE.md). The drive's choices (which door, when) are a decision over beliefs
  that already exist: P(exists) of a door, the LTSM's crossing record, the room's promotion. The max-time fallback is a
  deliberate, flagged exception (decision 5) and is a config value, not a magic number in code.
- Rates first (memory feedback-rates-before-features): the drive must not drop controller/room/retina below nominal.
- Webots is shared; warn the peer sessions before runs.

## The design

### A. Start state (harness)

A new tour script beside warm_start_tour.sh (it reuses its fleet start/stop, lock, graph recorder and trace_check): room_concept
`MapMode = "estimate"`, LTSM and door identities EMPTY (archived first, as the warm script archives), all doors shut, robot at the
usual start pose in room_1, controller in **AffordancesOnly** plus the new drive enabled. Run ends on the drive's own "done",
or a whole-run time limit (config).

### B. The exploration drive (controller)

A small state machine owned by the controller session, active only when its config flag is on, alongside AffordancesOnly
selection (the room's places and the door looks keep driving whenever the drive has nothing to do):

- **EXPLORING room R** (entered at t_R): the open contest drives. The drive watches: R promoted? (the room node's promotion /
  `room_explored_p`, whichever room_concept publishes for a promoted room), and the set of doors of R.
- **Leave condition** = R promoted, or t − t_R > MaxRoomTime (fallback, logged as such).
- On leave: choose the next door among R's doors that are FOUND (door_concept P(exists) over its own existence-assert level —
  the level door_concept already uses to offer the door's affordances, not a new one) and NOT YET CROSSED (no passage record in
  the LTSM for it), the cheapest by route length. If one exists → start a door passage to it (cross).
  If none → start a door passage back through the door R was entered by (return). In the start room with no entry door and
  no candidate → done.
- **Crossing**: the existing door passage runs; on completion the robot is in room R' → EXPLORING R'. If R' is already promoted
  (a known room), the leave condition holds at once, so the drive chooses again immediately (decision 8).
- **Done** when no room the robot can reach has a found, not-yet-crossed door and the current room is promoted.

### C. The record (for the paper)

One CSV row per drive decision (time, room, cause: promoted | max-time | door-found-after-promotion | return | done, chosen
door, candidates with their P(exists) and crossed flag), plus the existing tour artefacts. A grading script turns it, with
the Webots ground truth, into the per-room table of the Goals.

## Calls made after the builder check (mine — check them)

- **Promotion of crossing-born rooms (risk 1): the code path exists.** room_scene_graph.cpp ~l.1805–1852: crossing into an
  unknown room calls `begin_room_estimation` + `arm_startup_proto` (startup_event `crossing_born`), i.e. the same start-up path
  that promotes on the layout explorer's `finished`; crossing into a REMEMBERED room promotes at once (`promote_startup_proto(…,
  true)`). "Promoted" = room_concept's `startup_promoted_` as published on the room node (the builder names the attribute).
  It has not been run live from an empty LTSM; the spike checks it. If it fails live, fixing room_concept/ltsm so crossing-born
  rooms promote IS in scope (the goal needs it), reported to the user before the plan proceeds.
- **"Crossed" is the drive's own run memory**, keyed by PASSAGE (from_room, to_room, via door face), written when a passage
  completes. A door face counts as crossed if a recorded passage left its room through it; the face the robot ENTERED by (the
  mirror face in the new room) is recorded as crossed at entry and remembered as R's entry door. The LTSM starts empty, so
  nothing else is needed; reading LTSM passage records is not required.
- **"Found"** = door_concept offers the door's `approach` affordance (its offer already carries P(exists) and door_concept's own
  offer rule) — no new level in the controller.
- **Room promoted and no door found** ⇒ return (decision 7); the drive does not wait for further looks. A door found later in a
  revisit is picked up then.
- **Starting a passage**: the drive uses the same internal entry point the Command-mode mission's `cross 'door_X'` uses (it already
  releases the current claim and restricts selection); a running look/place claim is released by that path.
- **Route cost** for choosing among candidates: the door route's length if the existing builder returns one, else straight-line
  distance (logged which).
- **Grading**: room IoU and pose error against Webots ground truth with the existing estimation tooling (proto_room.csv,
  gt_error.csv, the warm tours' door lines); per-room ground-truth polygons from threerooms-piso.wbt as the warm grader uses.

## Left to the builder

The exact state machine's file; the CSV columns; how "entered by" is remembered; MaxRoomTime's default (suggest 5 min, flagged);
the whole-run limit; reading "crossed" from the LTSM vs the controller's own tour journal (test_tour_journal exists).

## Risks to check first (a short spike before the plan is written)

1. **Does Estimate mode estimate and PROMOTE a room entered by a crossing?** Start-up promotion is room_concept's; crossing-born
   rooms were ltsm's (memory startup-proto-room-fleet). If the second room never promotes, the leave condition never holds.
2. **Do door_concept's looks find DOOR_2 / DOOR_0 while their rooms are being explored**, with no remembered faces?
3. **Is "not yet crossed" readable** for a door seen only from one side?
