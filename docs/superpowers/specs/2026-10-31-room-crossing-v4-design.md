# Room crossing v4: the controller crosses locally, the scan decides the room

2026-10-31 · status: APPROVED (user), AMENDED after the offline check (§ Amendment 1) · replaces the deep-entry switch of Rev 3
Sources: the user (2026-10-31, this session); Fable's from-scratch design ("one intent, one switch, scan decides");
Lucas Bonilla's TFM (door-relative targets, odometry-only crossing).

## The design in five steps

1. **At the door.** The robot reaches door_concept's vantage; the door offers `open` (if shut) and `cross`. Unchanged.
2. **Cross on local sensors.** The controller writes the INTENT edge (`interacting` robot → door; exists since 09d880ab)
   and drives through the opening on the door it sees in its own LiDAR (the anchor servo, robot frame, odometry between
   scans, the live-returns guard). No room map is used for this motion.
3. **In as far as the free space allows.** After the opening the controller keeps driving along the door axis while the
   LiDAR shows room for the body ahead, up to "more than a metre" (the user's wish), and stops where it does not
   (a wall, furniture, a closet). No fixed target. With little space past the door it stops there and the room's
   affordances take over (step 4).
4. **room_concept decides the room, by which room explains the scan better.** From the moment it sees the intent edge:
   - the **old room stops learning**: no outline, wall or free-space update while the intent edge lives, so nothing seen
     through the door can enter it; its localiser keeps running, with the through-door points explained by the rival
     (below), not by the old room;
   - it holds a **rival room**, placed through the door (the exact face composition, `door_face.h`):
     - **remembered** — the door has been crossed before AND ltsm has a room behind it (its recall under the door):
       ltsm's room is trusted as the rival (its outline, its identity);
     - **new** — otherwise: room_concept starts estimating the new room (the proto branch) from what it sees through
       the opening, and that estimate is the rival;
   - every scan, the log-likelihood of the scan under each room at the robot's pose (the pose carried continuously
     through the door transform, never reset) is accumulated in a two-state belief {old, rival};
   - **the switch**: when the belief says `rival`, room_concept moves the robot's RT parent to the rival room
     (remembered: the recalled room becomes current; new: the proto room becomes current and keeps being estimated).
     This is the one moment the robot is in the new room.
   - **Little space past the door** (user 2026-10-31): the controller just stops once the body is clear of the opening
     and the free space ahead is used up, and hands the motion to the room's own affordances — the estimation branch's
     epistemic places (afford_room) — which move the robot safely inside the new space while the comparison goes on.
     The same hand-over applies whenever the robot has stopped and the belief is still undecided: more of the new room
     in view, more evidence, and the motion is the room's, not a crossing rule.
5. **Back to normal.** The controller ends the crossing (deletes the intent edge) when the switch has happened AND it has
   stopped; then missions, planner and affordances resume. ltsm archives the room left once the robot's parent has
   moved away from it; nothing waits on the archive.

## What is kept, added, removed

| | |
|---|---|
| KEPT | approach/open affordances, the anchor servo + AIF law + returns guard, the intent edge (link 0), recall placement by face composition (D1/D2), ltsm recall + archive, odometry-only/relocaliser-off while the intent edge lives |
| ADDED | free-space-limited depth (controller); hand-over to the room's affordances (afford_room) when the space ahead is used up or the room is undecided; old room frozen while crossing; rival room + {old, rival} scan comparison; switch = RT parent move (room_concept) |
| REMOVED | controller THROUGH decision and link 1 as the switch trigger; deep-entry latch (`note_deep`, second THROUGH block, `AnchorCrossStandoffM` as a target); switching at a doorway line; legs waiting on ltsm's archive |

## Open points (to settle before code)
- Early in a NEW room the rival's model is thin (a strip of free space through the opening). The comparison must treat
  "unexplained by either" as neutral, not as evidence for the old room. Check offline on recorded crossings
  (pingpong13/29): the belief must flip within ~1 m past the opening and never before the body is through.
- Which parts of the Rev 3 exit chain remain (the door's twin face in the new room is still needed for the way back).
- TLA: the protocol shrinks to intent edge + parent move; model it only if the chain changes.

## Verification
Offline first, on recorded runs: switch time vs ground-truth body position (never before the opening, within ~1 m
after it); the old room's outline before vs after a crossing (unchanged); zero false switches while standing at an
open door. Live: room_1 ↔ room_2, judged per leg on: intended room (GT), robot inside, old outline unchanged,
no relocaliser jump, no agent waiting.

## Amendment 1 (user 2026-10-31, after the offline check) — the switch is the body past the opening

The offline check (`room_concept/estimation_scripts/scripts/crossing_switch_check.py`; pingpong29 + pingpong13, 6 crossings
× {remembered, new} × 5 seeds × 2 noise settings) REFUTED "which room explains the scan better" as the switch: 79/120 pass;
14 switched 0.3–1.6 m BEFORE the door (recall 0.3 m off, pose 0.1 m off), 8 never switched. Cause: standing at an open
door, the returns from the room beyond vote "robot in the rival" though they only show the rival is VISIBLE. Modelled
honestly (each hypothesis = both rooms), both predict the same scan and the only difference left is which side of the
door the body is on.

So step 4's switch is: **the body wholly past the opening, as the controller measures it on the opening it sees in its
own LiDAR** (robot frame). Everything else in steps 1–5 stands, including the trust rule (ltsm's room if the door was
crossed before and a room is remembered behind it, else room_concept estimates a new one) and the hand-over to the
room's affordances.

**The door filter on both sides of the doorway** (user): returns whose ray passes through the opening are "the next
room" and are filtered —
- in the OLD room, by the through-door filter (the door's aperture in the old room's door list; restored by 506b42b4);
- in the NEW room, from the switch on: the new room CARRIES THE CROSSED DOOR. room_concept puts the opening it has just
  crossed into the new room's own door list at the switch (composed into the new frame through the door), held until
  door_concept's face of the doorway replaces it. Without it the new room's door list stays empty until door_concept
  writes its face (switch + DDS + door_concept's cycle), and the rays looking back into the old room are unfiltered.
