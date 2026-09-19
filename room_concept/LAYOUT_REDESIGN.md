# Room layout estimation: a redesign with a threshold budget

Status: **design, not built.** Written 2026-09-19 after a day of measurement on the live agent and
two independent design reviews. The current estimator works (it holds a 6.00 × 4.00 m rectangle to
2–3 mm while driving, commit `d307c84`) but it carries ~100 tunable decision constants and degrades
as rooms gain structure. This document says what to build instead, what it costs, and — most
importantly — **what would refute it**.

---

## 1. Why a redesign, in one table

The complexity ladder (`WS_TOUR_ROOM=...`, robot driven through the real kinematics, 3 laps):

| room | kind | true verts | IoU | Hausd | walls found |
|---|---|---|---|---|---|
| rect | plain shell | 4 | 0.987 | 0.024 | 4 ✓ |
| c1 | 1 corner column | 6 | 0.982 | 0.039 | 6 ✓ |
| c2 | 2 corner columns | 8 | 0.957 | 0.110 | 7 |
| **c4** | 4 corner columns | 12 | **0.801** | **0.881** | **4** ✗ |
| w1 | 1 mid-wall column | 8 | 0.995 | 0.057 | 6 |
| w2 | 2 mid-wall columns | 12 | 0.965 | **0.783** | 7 |
| c2w1 | 2 corner + 1 mid-wall | 12 | 0.965 | 0.105 | 7 |

It degrades monotonically and collapses at four corner columns. `w2` keeps a good pose
(rmse 0.024) while its Hausdorff blows to 0.78 — the failure is **structural**, not pose.

### The threshold audit (counted, 2026-09-19)

| | |
|---|---|
| `WallMap::Params` fields | 68 (36 float, 24 bool, 8 int) |
| …of those, decision constants | 32 |
| …flagged ⚠ by their own author | 17 |
| hard-coded numeric comparisons in `wall_map.cpp` | 67 |
| `RoomConcept::Params` fields | 147 |

### Every defect found on 2026-09-18/19 was a policing rule misfiring

Not one was a geometry error:

- the existence channel deleted walls below a rectangle with **no floor and no price**, while the
  priced path (`try_down_jumps`) refuses to → the published room became a **triangle with the robot
  outside it**;
- a span rule killed a wall whose bins were at the confidence clamp (+9.2, 12 947 points over 775
  frames) because its trimmed span came to 0.451 m against a 0.500 m floor — it died **49 mm short**,
  which is why a column appears early in a run and then vanishes;
- `heal_order` deletes a near-parallel neighbour (sin < 0.05, i.e. **2.87°**) purely on point count,
  with no evidence test;
- a `re_derive` proposal reached the **published** polygon before the adoption judge ruled on it —
  the layout flipped 4 ↔ 18 vertices on 46 of 2020 frames, at the cadence, while the map held at
  exactly 4 walls;
- twin fusion adds a candidate's line-fit information (~13.9 per point) to a wall's deflated
  accumulator (~0.3 per point) — a **~46× currency mismatch** that inflates confidence several-fold
  in one step.

**All of these exist to police a state space that permits illegal states.** Make the states legal by
construction and the police are unemployed. That is the whole thesis of this document.

---

## 2. Constraints any design must meet

These are measurements, not preferences.

1. **Association fails by common mode.** All segments in a scan share one pose, so their innovations
   share a rank-3 component. At the failure frame all four walls showed the **same −5.8°**
   innovation (spread 0.09°) against a ~6.3° gate: every segment failed in the same frame, and with
   nothing associated nothing could correct the pose — permanently. Any design must remove the
   shared pose error **before** residuals are attributed to structure, or a residual-driven
   extension rule will confidently propose a column to explain a heading error.
2. **There is no scan registration step.** Association runs at the dead-reckoned pose. Cartographer's
   scan-to-map prior is σ = 0.100 m / 0.025 rad; S-Graphs+ floors its odometry edge at 0.1 m /
   0.2 rad; ours asserted **1.4 mm / 1.6 mrad** parked. No published system feeds raw dead reckoning
   as the stiff term.
3. **The data term is 1/5000 of physical.** `WallPointFactor` weighs each return by
   `0.5·inv_var·pda/N_slot`; the code's own publish rescale `k = 2σ_obs²N/σ_sensor² = 5000` proves it.
   Correlated returns must not collapse σ — the honest form is line-to-line with the fit's own
   information **saturated by a common mode**.
4. **Degeneracy is not a concern here.** A 4×6 room ray-cast has condition number **1.68**
   (closed form `atan(W/L)/atan(L/W)`). Spend no effort on it.
5. **Contracts.** ~20 Hz; publish an ordered polygon (`delimiting_polygon_x/y`, read by ~16 agents)
   with a per-corner σ driving the publish decision; eventually represent L rooms, columns/alcoves,
   and a 45° chamfer.

---

## 3. The state: a union of axis-aligned boxes

In the **rectilinear map frame** the layout is

```
R  =  ∪ Bᵢ  \  ∪ Cⱼ
```

positive boxes (main room, L-extension, alcove) minus negative boxes (a free-standing column). A box
is four wall lines — the same wall landmarks we already estimate — and an attached box **shares
lines with its parent** (an alcove adds 3 offsets, an L-step 2), so the parameter count matches the
turn-word alternative while the representation stays trivially valid.

**θ₀ is deleted, not pinned.** The map frame *is* the rectilinear frame. This removes the variable,
the gauge factor, the re-anchor and the θ₀ prior in one move. The first pose's yaw becomes a variable
estimated by registration; the translational gauge is pose 0 held constant for ever.

### What is an identity rather than a rule

- **Manhattan** — no variable can express another angle.
- **Closure** — a union of closed boxes is a closed region. `order`, `heal_order` and
  `repair_if_crossing` exist because free (φ, d) lines can become parallel-adjacent; boxes cannot.
- **Validity** — the only inequality is each box's positive width, and `width → 0` *is* the removal
  event, priced like any other edit rather than policed by `heal_order`'s point count.
- **Corner σ** — each vertex is `(d_x, d_y)`, so `Σ_v` is a 2×2 sub-block of the offset covariance
  plus the flatness term. Exact; `corner_model_sigma` disappears.

This is also SLIBO-Net's slicing-box representation — the published way to write a rectilinear room
so that Manhattan, closure and validity are identities.

⚠ **Correction to an earlier claim.** S-Graphs' room vertex is *not* `[ρx, ρy, wx, wy]`. The released
code (`src/g2o/edge_room.cpp:188-228`) shows `EdgeRoom4Planes` constrains the room **centre only** —
no width, no orthogonality, no closure; widths appear solely as a detection threshold (t_w = 0.5 m).
Its rooms are *detected* by an ESDF skeleton rule set, and its output is a centre plus four plane
ids, not an ordered polygon. **Adopted verbatim it stops at rung 1 of the ladder by construction.**
Take its skeleton (keyframes, wall landmarks, pose↔wall edges) — which we already have — and its
finite room as a structural *primitive*; do not take its room detector, its room edge or its weights.

---

## 4. The pose: register first, with a data-driven regulariser

Adopt **Kinematic-ICP** (Guadagnino et al., ICRA 2025, arXiv 2410.10277; code `kiss-icp/kinematic-icp`):

```
G(T̂ ⊞ Δu) = χ(T̂ ⊞ Δu) + (1/β_t)·Δx²        β_t = χ(T_{t−1} O_t)
```

`β_t` is the map cost **evaluated at the wheel-odometry prediction**: when odometry and scan
disagree — which is exactly a shared pose error — β is large and the prior dissolves; when they
agree, odometry holds translation. It regularises **translation only**, leaving heading to the
LiDAR, because wheel heading is the noisier channel and heading is the best-observed DOF in a
rectangular room. Zero tuned numbers, and it answers constraints 1 and 2 together.

Scale comes from **KISS-ICP** (Vizzo et al., RA-L 2023): `σ_t` is the RMS of the motion model's own
deviation, `τ_t = 3σ_t` is the correspondence distance and `κ_t = σ_t/3` the Geman-McClure scale. The
association distance becomes **3× the measured prediction error** — replacing `rfe_huber_delta`,
`assoc_chi2`'s width, and the leave-one-out gate built on 2026-09-18.

`χ` already exists: the polygon SDF with gradient (`SdfFactor`, `room_gn_solver.cpp`).

---

## 5. Structure admission: Bayes factor > 1, nothing else

A point is **unexplained** when its posterior responsibility for {inside, outside} exceeds every
edge's. Unexplained points — *after registration*, so a pose error cannot masquerade as structure —
are re-segmented by RANSAC and accumulate as proposals. Edits are box add / box remove.

```
ΔL = Δ log p(scans | R′) − Δ log p(scans | R)
     − Σ_new offsets log(span / σ_post)          (Laplace/Occam per new parameter)
     − log(#attachment sites)                     (where the box could have gone)
     >  0
```

Every hypothesis is **re-registered** under itself before scoring; otherwise a pose error buys a box.
Removal is the same test reversed, so a wall cannot vanish unpriced — the triangle becomes
unreachable.

**The margin is BF > 1, i.e. 0 nats.** Today's `birth_nats = 4.605` (Jeffreys "decisive") is tuned
hysteresis. Flapping is not a reason to add a margin: the same window and the same re-registration
are used in both directions, so the sign only flips when new data flips it — a legitimate revision.

Sanity: from a rectangle, one step costs `log C(6,1) = 1.79` nats for the word plus ~5.7 nats per new
offset ≈ **13.2 nats**, against today's hand-tuned `2 × edge_code_nats = 14.6`. The tuned value was
within 1.4 nats of the principled one — which is why the current economy half-works.

---

## 6. The threshold budget

| kind | count | what |
|---|---|---|
| **Physical** (datasheet / tape measure) | 6 | σ_s (range noise), σ_d,flat and σ_φ,flat (wall flatness), z-band (2), r_max |
| **Measured online** | 0 constants | β_t, σ_t/τ_t/κ_t, mixture proportions π |
| **Compute budgets** (change speed, not decisions) | ~9 | point cap, voxel/downsample, window length K, RANSAC confidence + cap, γ, δ_min, τ₀ |
| **External contract** | 1 | per-corner σ publish bar (set by consumers) |
| **Canonical** | 1 | BF > 1 (0 nats) — not a free number |

**≈17 constants, 0 tuned decision constants**, against ~100 today. Retired with their jobs: Manhattan
σ and off-prior, assoc/merge χ², obs σ, Huber δ, birth nats, birth min frames, the whole existence
ledger (bins, p_det, clamps, span floor), stub thickness/free-behind, level-2 constants, corner model
σ, rect prior σ's, gauge σ's, splice tolerances, eigenvalue caps, ZUPT σ, the command prior, and the
common-mode gate built on 2026-09-18.

Flagged honestly as *published but asserted*: KISS-ICP's `δ_min = 0.1 m` and `τ₀ = 2 m`.

---

## 7. What it cannot do

- **Non-rectilinear walls** (30°, curved) are not representable until the alphabet grows; a 30° wall
  is reported as unexplained, never as structure. This is the price of making Manhattan an identity
  and it must be visible in the publish status.
- **A 45° chamfer** needs a corner-cut primitive (or classes mod 8); design it in from day one as an
  index into a direction table so it is a later config change, not a redesign.
- **Doorways** are not structure: through-door returns are the outside-leak component, and doors
  belong to `door_concept`. Extents become *derived* (an edge is the segment between its vertices),
  so `s_min/s_max`, per-bin existence, `try_spur_wraps` and level 2 all lose the state they lived in
  — and with it all four of the misfires in §1.
- **Given mode** becomes the same code with edits disabled: registration only, not a separate branch.
- **Every constant tuned on the current bench must be re-baselined.** The 50-room IoU tables, the
  adoption-judge comparisons, `trial_adoption`'s 0.9333 — all describe a machine that will no longer
  exist. Carrying them over is the single biggest risk in the project.

---

## 8. Migration, graded at every step

Each step is gradeable on the existing bench: `WS_REPLAY=<tmp/wall_input_*.bin>` replays real
recorded LiDAR+odometry and prints association-over-time deciles, births/deaths, wall count, worst
corner σ, recovered side lengths (truth 4.00 × 6.00) and map-frame rotation; `WS_TOUR_ROOM=` walks
the ladder.

1. **Honest weights, in the existing code.** Line-to-line factor with Σ_flat; odometry at its
   calibrated covariance; ZUPT, command prior and the LOO gate off.
   *Grade:* association deciles stay ~100% **without** the gate; map rotation ≤ its posterior σ.
   *Refuted if* the gate is still needed — then the common mode is not in the pose.
2. **Registration** (Kinematic-ICP objective, our SDF as χ, β from data; KISS σ_t for scale).
   *Grade:* rmse, map rotation, and that the pose follows a translation.
   *Refuted if* the pose still does not follow.
3. **Layout as boxes**, rectangle only, over the current wall landmarks.
   *Grade:* sides within 2σ of 4.00 × 6.00; side-length **NIS ~ χ²₂** (calibration, not accuracy).
   *Refuted if* NIS is systematically > 1, or init tilts on a concave first scan.
4. **Box add/remove with BF > 1.**
   *Grade:* **the ladder must be monotonic** — rect → c1 → c2 → c4 and w1 → w2 → c2w1. Count
   structure changes per 1000 frames on a *static* room.
   *Refuted if* the ladder has a hole, or the change rate exceeds what evidence noise predicts.
5. **Delete** `order`, splice, heal, repair, existence, extents, level 2, spur wraps, re-anchor,
   gauge factors, `edge_code_nats`. *The bench must not move.*

### A test that must pass before any of it is believed

**Inject a +5° bias into every dead-reckoned yaw in the replay. Zero structure proposals may fire.**
If any box is proposed, the re-registration discipline has a hole and the design is refuted — that
is the exact failure mode that produced the triangle.

---

## 9. Honest counterweight

The current estimator, as of `d307c84`, holds a rectangle to 2–3 mm while driving and publishes on
100% of frames. This redesign is justified by the *ladder*, not by the rectangle. If the cheapest
steps (1 and 2 above) recover the ladder on their own, **the rewrite is not justified by this
failure** and should be reconsidered — that is the outcome to hope for, not to defend against.
