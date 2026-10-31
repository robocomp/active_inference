# Door cross as Active-Inference action control, in the robot frame — design

2026-10-31 · branch `feat/room-recognition` (worktree rt-build) · status: **DRAFT, for review**
Relates to: `2026-10-29-room-transition-protocol-design.md` Revision 3 (the crossing is an action; X1 the controller writes
link 1), `2026-10-20-local-motion-policy-design.md`, `controller/src/controller_anchor_servo.h`.

## 1. Why

pingpong8 (odometry-only room_concept during the crossing, d5d07b06) passed legs 1–5 with no pose jump and a clean
trace. Leg 6 failed in the controller's **room-frame route planner**: `cannot place the route: no free point past the door on
its axis between 0.77 and 1.57 m` (the same failure as pingpong6 leg 6). Every door-route failure since 10-24 has the same
shape — something in the ROOM frame (the grid past the door, the room-frame door, the perceived opening re-expressed through
the room pose) decides how the body goes through a hole it can perceive directly:

| run | failure | room-frame dependency |
|---|---|---|
| pingpong3 leg 1 | `cross_unreachable` rebuilding from inside the doorway | grid planner, P_post |
| pingpong6 leg 6, pingpong8 leg 6 | no free point past the door 0.6–1.6 m | grid cells beyond the door |
| pingpong7 leg 4 | 150 rebuilds, 6–8 m run-ins, 45 s late THROUGH | opening mapped through a jumped room pose |
| same-side route (open) | pre and post both at s < 0 on the room-frame door_2 | room-frame door |

The crossing is an interaction with ONE perceived object. The correct controller for it is the one the user proposed
(10-31): *imagine the goal observation — the door frame behind me — and act to make it true*. That is Active-Inference
action control: action descends the free energy of the predicted observation, which needs ∂o/∂a, the robot-frame interaction
matrix of the opening. It is an on-the-spot visual servo specific to this interaction, and it needs no room frame at all.

## 2. What already exists (do not rebuild it)

`controller/src/controller_anchor_servo.h` (2026-10-18, 849 lines) already servoes on door_concept's robot-frame anchor
`door_N@robot`: e (offset along the wall), s (progress along the normal), heading error, σ_lat from the anchor covariance,
P_fit = Φ((w/2 − half_w − |e|)/σ_lat) as the speed scale, a differential branch (turn toward the desired velocity d), the
measured LiDAR gap as the steering target (10-19), leaf hypotheses, `lidar_free_advance`, the live-return guards. It reached
**6/6 four times on 10-19** (try91, try102–104, try107). It is still wired (`controller_session.cpp` ~4662, chain mode,
`UseRoute=false`).

It was retired on 10-24 for the route (recog_try10): driven as the local policy's Place goals it stalled in the doorway —
"the crossing's pose jump read as a failed prediction ⇒ release ⇒ backing up". **That cause was the room-frame pose jump,
which d5d07b06 removes.** So the servo's retirement reason is gone, and what this design adds is (a) the Active-Inference
form of the law, (b) the differential-base geometry done right, (c) its place inside the route mode, (d) link 1 from it.

## 3. The law

### 3.1 Generative model (robot frame; forward = +y, side = +x)

Hidden state of the interaction: the opening's pose in the robot frame, A = (centre c, wall direction a, normal n through
the opening from the robot's side), width w — door_concept's anchor (mean + covariance), predicted by odometry between
sightings (door_concept already publishes "PREDICTED — carried by odometry" for up to 3 s unseen).

Observation the controller acts on: o = (e, s) of a CONTROLLED POINT P of the body, measured in the anchor frame:
  e = a·(P − c)   lateral offset from the opening's normal line
  s = n·(P − c)   progress through the plane

Goal prior (the imagined observation "the door frame behind me"): e* = 0 with precision Π_e = 1/σ_e², where
σ_e = (w/2 − half_w) is the room the body has on each side (wide opening ⇒ loose, tight ⇒ stiff), and s* = the body wholly
past the plane, with precision Π_s from the anchor's own along-normal σ.

### 3.2 Differential base: control a point AHEAD of the axle

Shadow is differential (v forward, ω; the bridge drops `side`). The axle midpoint's velocity has rank 1 in (v, ω), so a
one-step gradient on the axle cannot reduce a lateral offset — the descent stalls beside the axis or spins (Brockett: no
smooth static feedback stabilises a unicycle POSE). The goal is not a pose, it is a region, and the standard remedy applies:
servo a point P = (0, d) ahead of the axle. Its velocity in the robot frame is

  Ṗ = (−d·ω, v)      ⇒      ∂Ṗ/∂(v, ω) = J = [[0, −d], [1, 0]],   det J = d ≠ 0

so J is invertible: P is fully actuated. d is not tuned: **d = the body's front reach** (`support_radius` along +y, the same
quantity THROUGH and the route already use) — the front is the part that must thread the opening first.

The observation's derivative is the interaction matrix L = ∂(e, s)/∂a = [aᵀ; nᵀ]·J. Free energy of the predicted observation
one control period Δt ahead, F(a) = ½ Σ Π_i (o_i(a) − o_i*)² + ½ aᵀ Π_a a (the action prior: a costly, finite rate), and
action descends it; its fixed point is the closed form

  a* = (Lᵀ Π_o L Δt² + Π_a)⁻¹ Lᵀ Π_o (o* − o) Δt

i.e. a precision-weighted inverse of L. No gains: Π_o comes from the goal prior and the anchor covariance, Π_a from the base's
rate limits (v_max, w_max as 1σ of the action prior). The existing P_fit scaling falls out of it (a wide σ_lat lowers Π on the
along-normal goal relative to the lateral one, so it re-centres before advancing).

As P converges onto the normal line the heading converges to n on its own (the axle trails the front).

### 3.3 Collision: expected free energy over a short horizon (V2, only if needed)

The one-step law ignores the rear swing and the jambs. V1 keeps the existing robot-frame safety net unchanged (live returns
self-filtered by the body, the swept-body caps, `lidar_free_advance`) — they cap, they do not steer. If V1 shows jamb
contact on oblique entries, V2 rolls out a few (v, ω) sequences from a* over ~1 s, predicts the jambs (c ± w/2·a) in the
robot frame along each, and adds the expected contact surprise −log P(no contact) to the score — the local policy's
look-ahead (`Policy.Enabled`), with the cross goal prior as its preference. V2 is not built unless V1 fails for that reason.

### 3.4 Ends

THROUGH = the existing test (body wholly past the latched normal, s_body > r_body, drive_door_route) ⇒ link 1 (unchanged).
The goal prior being met and the crossing fact are the same event. After THROUGH the servo hands back (as the route did
at P_post).

## 4. Where it lives

**Controller agent** (owns action). In route mode (`UseRoute=true`) the route is kept ONLY up to the vantage P_pre (or
skipped when the anchor is already fresh); from the moment the cross is claimed and its anchor is fresh, `drive_door_route`
hands motion to the servo instead of the route follower. No grid lookup past the door, no rebuild on the perceived
opening, no P_post. The route's doorway segment and its rebuild logic stay in the code behind a flag for the A/B (as with
the spline tracker): `[DoorRoute] Doorway = "servo" | "route"`, default servo once validated.

door_concept: unchanged as the perception side (anchor + odometry prediction). One dependency (pingpong7 leg 3): while the
cross is `executing`, door_concept must not withdraw `aff_<door>_cross` on P(reach) = 0 — the body standing in the
opening is not a reason the cross is unreachable. Either door_concept holds the offer while executing (the FACT_HOLD rule
applied to the affordance), or the controller ignores a withdrawal of the affordance it is executing while its anchor is
fresh. Decide in review; the first is the cleaner contract.

room_concept: nothing new (odometry-only during the cross, d5d07b06).

## 5. Tests

1. **Offline closed loop** (`controller/tests/test_anchor_servo_aif.cpp`, header-only law): unicycle kinematics at 20 Hz,
   a 1.0 m opening in a wall, body = Shadow's footprint. Starts: lateral −0.5…+0.5 m, heading −45°…+45°, 0.8–2.0 m before
   the plane, anchor noise at the logged σ. Pass: every start reaches THROUGH, the body polygon never intersects a jamb,
   no reversal. Report time-to-THROUGH and min clearance. Also an anchor-dropout case (3 s unseen, odometry prediction).
2. **Same harness, old law** (`anchor_cross_command`) for the comparison — if the old law already passes, V1 is a
   re-derivation of it in Active-Inference form and the gain is the absence of hand-set τ's.
3. **Live**: pingpong9, 6 round trips, graded on ground truth with trace_check (S7/S8/L1 must stay clean).

## 6. Order of work

1. Harness + current `anchor_cross_command` through it (baseline, no code change in the agent).
2. The AIF point-ahead law, header-only, through the same harness.
3. Wire it as the doorway driver in route mode behind the flag; door_concept "no withdraw while executing".
4. pingpong9.

## 7. Open questions for review

- Q1. d = body front reach — or the LiDAR origin (what sees the jambs)? Front reach is the physical constraint; proposed.
- Q2. Hand-back after THROUGH: stop, or continue to the next mission step's goal? (the 10-22 note "no stop on room entry").
- Q3. Keep the measured LiDAR gap (10-19) as the observation, or door_concept's anchor alone? The gap won 6/6 on 10-19;
  proposed: gap when measured, anchor as its prior (as the servo does today).
