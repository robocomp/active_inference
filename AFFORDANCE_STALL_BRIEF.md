# The robot does not reach its viewpoints — problem brief

State as of 2026-08-19. Written for someone picking this up cold. Everything below is measured;
where it is inference it says so. Two agents are involved: `room_concept` (producer of exploration
viewpoints) and `controller` (consumer that drives there).

## The one-line blocker

**The controller claims an affordance and then does not drive toward it.**

```
afford_room EXECUTING (controller-claimed) — target (-1.50,1.62) d=5.46m best=5.47m no_progress=4.5s/25s
afford_room EXECUTING (controller-claimed) — target (-1.50,1.62) d=5.47m best=5.47m no_progress=8.9s/25s
```

`best` is the closest approach ever achieved on this target. It never improves. The robot holds the
claim for the full `ExecStallTimeout = 25 s` without closing, room abandons, and the cycle repeats.

## The metric that matters (and the one that misled us all day)

**Use closure efficiency**: net reduction in distance-to-target, measured only within periods where
the target is constant, divided by path travelled. Path length alone reads "moving" while the robot
circles its goal — this hid the failure through several reports.

Last long run: **33% overall, 0% across the final ten minutes** (70 episodes, 118.5 m travelled,
38.9 m net closure). `d_target` essentially never drops below ~0.5–0.6 m on any approach.

## Established, with numbers

- **The base is fine.** Room localizer, independent of the controller: `corr(vel_rot, dtheta/dt) = +0.856`,
  slope **+0.760** over 9916 samples. Rotation is executed correctly, ~76% of commanded rate delivered.
- **Room's target is stable.** Churn fell from 21% of rows to ~1–2%. Over 1478 rows the producer moved
  the raw target >5 cm **12 times**; the controller moved it **7 times**.
- **Enormous rotation demand.** **1611 deg commanded in 47.9 s** of final-metre approach (~3.4 real
  revolutions after the delivery factor). `cmd_rot` at its cap on **65%** of cycles, `cmd_adv == 0` on 26%.
- **The standpoint repair is permanent, not exceptional.** `fix_held = 1` on **94%** of rows;
  the driven target differs from the published one by >5 cm on **55%** of rows.
- **Gain churn breaks target identity.** `pub_gain` observed at **1.368 → 1.441 → 1.903 within ~1 s**.
  The controller's `same_target_instance` treats `epistemic_gain` differences above **1e-3** as a NEW
  target instance, so every cycle replans from scratch. Fix agreed but not yet landed: exclude gain from
  identity — it is a priority, not an identity. `publish_target` already refuses to rewrite while the
  consumer holds the claim (`current_active && current_pending`), so this is safe.
- **Terminal wedging is gone.** Across 84 min in 3 runs: longest stall 7.5 / 7.5 / 7.7 s, **zero episodes
  over 10 s**, against a 74 s unrecoverable pin before. The robot always recovers — it just gets nowhere.

## Ruled out (do not re-chase)

- **room_concept ping-ponging on one cell.** Measured clean: 10 distinct well-spaced cells, one per
  completion, `tgt == pub`, never a repeat. The apparent loop was the consumer failing every approach.
- **`refresh_belief` gating on Satisfied.** Inert in the live config — `BeliefForgetTime` is absent from
  `config_apartamento.toml` (the live file) so forgetting is disabled.
- **Footprint model vs live ESDF disagreeing.** They agree: r=0.943 with a constant 0.273 m offset,
  exactly the body half-extent. Different reference points, not a conflict.
- **Rotation sign inversion.** A logging artefact only; see traps below.

## Open

1. **Why the controller does not drive on a claimed target.** The blocker. Prime suspect is the gain
   churn above (replanning every cycle ⇒ no accumulated progress), unconfirmed.
2. **Why ~3.4 revolutions per final metre**, on a `Policy::Reach` target that has no final facing.
3. **Room's score is degenerate again, at the other end.** After ~29 min every cell is ~1700 s old, so
   `neg = log1p(age/tau)` converges to 2.64–2.74 for all of them; at `w_ior_drive=0.5` that is ~1.37 and
   swamps `marg_fim <= 0.53`. Live consequence: a cell with **`marg_fim=0.0000` outranks one with 0.5313**.
   Selection degenerates to round-robin by least-recently-attempted. An unbounded drive term swamping a
   bounded information term is the underlying shape.
4. **Two-cell oscillation** with a 25 s period (= `ExecStallTimeout`), alternating between a cell 1.04 m
   away and one 5.45 m away, closing on neither. Downstream of (1): room abandons because nothing moves.

## Traps that cost real time

★ **A CSV column name is a claim, not a definition. Read the line that writes it.** Three failures today:
- `yaw_err_deg` is `0` on 100% of rows — written as `has_value() ? ... : 0.f`, and a Reach target never
  sets it. **0 means NOT APPLICABLE**, not "no error".
- `min_esdf_m` is robot-centre-referenced; `clear_now_m` is footprint-referenced. Comparing them, or
  comparing a pose-specific value against a run-mean, manufactures a fake 370x disagreement.
- `rob_facing_deg` is `remainder(theta + pi/2, 2pi)`, whose sign convention does not match `cmd_rot`.
  Correlating them yields −0.44 and the false conclusion that the robot turns backwards.

★ **Do not judge a fix on a short window.** This failure degenerates over ~25 minutes. Runs looked
healthy for their first 2–5 minutes several times and then collapsed. Report on the **tail** separately
from the aggregate: one run read 33% overall while its final ten minutes were flat 0%.

★ **`approach_diag.csv` is truncated at every run start.** Archive it before restarting or the
comparison is lost.

## Good instruments

- `controller/approach_diag.csv` — `raw_tgt_x/y` + `fix_held` answer "who moved the target" locally,
  with no cross-log alignment needed. Best diagnostic added today.
- `room_concept/tmp/sdf_localizer/log_*.csv` — cols 97–103 carry `aff_outcome`, completions, the
  planner's target and the published target. Note `pub_tx/pub_ty` are stamped on every publish
  ATTEMPT including declined ones, and `pub_ok` is true even on the no-op path — neither means the
  node was rewritten.
- `controller/stall_events.csv` — verdict histogram (`wedge` / `spin` / `throttle_stall`).

## What changed today, and what it bought

room: attempts recorded in their own register instead of being stamped into the visit grid (they were
false observations); attempt-IoR multiplies the reward terms; level-triggered retire fires once per
arming. controller: the standpoint repair no longer relocates onto the robot's own footprint, and a
held repair is re-tested at the heading it was frozen at.

Bought: terminal wedge eliminated, churn 21% -> ~1-2%, first non-zero `Satisfied` outcomes of the day
(~25% across long runs), completion rate 1/83 s -> 1/45 s. Did not buy: the robot still does not
arrive. All of it is UNCOMMITTED.
