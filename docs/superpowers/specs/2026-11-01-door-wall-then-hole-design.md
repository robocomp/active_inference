# The crossing's door: find the wall, then the hole in it

2026-11-01 · status: APPROVED direction (user: "go with 1"), building offline first · door_concept (the interaction anchor)

## Why a fresh approach

The anchor update scores every ray against an opening hypothesis over the full pose (x, y, yaw) at once. In room_1's corridor
stub before DOOR_1 every run found a different pose that explains the rays about as well: the plane 1.5 m short (the free
corridor looks like an opening at any depth, pingpong33), rotated 46 deg onto the stub's side walls once the Helios returns
above the lintel were added (pingpong34 leg 1), 0.75 m to the side. Each fix added a term to the same joint likelihood.

## The approach — two small, separate estimates

1. **The wall (depth, yaw).**
   - Direction: the room's walls are Manhattan and the door lies on one; its direction in the robot frame is known to ~1 deg
     (room outline + the localiser's heading). It is a tight prior on the wall's yaw, not a free parameter, so a side wall
     at 90 deg is not a candidate.
   - Offset: fitted to the points that can only be that wall — the wall above the lintel (Helios; it spans the opening and is
     never hidden by the leaf) and the wall's face beside the opening at any height (bpearl + Helios). Robust 1-D fit of the
     points' distance to a line of the known direction, near the predicted offset.
2. **The hole (lateral centre, width).** The points ON the wall plane (within its thickness) and below the lintel, projected
   along the wall: the opening is the gap of about the door's width nearest the prediction; its edges are the jambs. The open
   leaf is off the plane (perpendicular to it) and the empty corridor has no points on the plane, so neither can fake a jamb
   or an opening.
3. **Close in** (the Helios loses the lintel at ~0.6 m): the wall offset is carried by the robot's odometry; the gap keeps being
   measured while the jambs are in view.
4. **The leaf** is placed afterwards relative to the hinge jamb, as an obstacle, never as a competitor for the wall.

Outputs the same anchor (pose in the robot frame + sigma), so the controller does not change.

## Validation, offline first

pingpong34's frame record (bpearl + Helios rays per frame) with ground truth: run the estimator on every frame of every
anchor (the wall direction from the anchor's creation prior, carried by the record's own motion), and report depth / lateral /
yaw error by distance to the door. Pass: depth and lateral within ~0.05-0.10 m and yaw within a few degrees on room_1's
face (the corridor stub) from 2 m in; no regression on room_2's face. Then live.
