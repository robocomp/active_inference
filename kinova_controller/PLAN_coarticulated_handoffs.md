# Plan — Downstream legs + coarticulated handoffs (per-via preference field)

**Status:** proposed, not started. **Date:** 2026-06-13.
**Goal:** turn the zero-terminal-velocity STOPS between `place / lower / retreat` into
pass-through vias via the per-via-precision preference field. Keep the lift a deliberate
hard STOP (grasp integrity); let transit/retreat flow.

**Locked decisions (from the user):**
- **Retreat scope = Cartesian phases only.** Flow within place/lower/retreat; retreat blends
  toward a return standoff, then the existing joint-space homing takes over. Do NOT (in this
  plan) rework the retreat→home→re-track boundary / the outer `Phase` machine.
- **Precision source = per-segment `c_k` + floors.** Each via's pass-throughness is driven by
  its own segment confidence `c_k = Π_m[k]/(Π_m[k]+Π_s)`, with a hard FLOOR on constraint-gated
  vias so a skilled agent can't over-cut them. This is the "per-via precision" the prototype
  memory identifies as the next step (see [[preference-field-prototype]]).

---

## Why (the gap, precisely)

The 2-via Gaussian-mixture preference FIELD is already fully implemented
(`efe_gradient.cpp:284-310`, `EFEParams::use_field`/`prec_current`/`prec_next`/`prec_ref`/
`field_overlap`). λ_c = exp(−Π_c/prec_ref) is the "pass-throughness": high Π ⇒ √-to-stop
(parity with a discrete leg), low Π ⇒ cruise-through bend toward the next via.

The gap is the **precision SOURCING**. `efe_drive` (`pick_and_place_fsm.cpp:492-500`) feeds the
field ONE global skill-interpolated precision:

```cpp
params.prec_current = field_prec_stop_ + skill_c()*(field_prec_pass_ - field_prec_stop_);
```

This is the prototype's documented failure mode: a single global precision can't keep the
constraint-gated lift via safe (`prec_pass=1` over-cuts → bottle-rise confirm misses → retries,
26 stops / 10.8 s / 18/20) AND let the free vias cruise (`prec_pass=5` is safe but stoppier than
the hand-coded blend, 17.8 vs 12.2 stops). Vias have **heterogeneous** precision needs ⇒ per-via.

## Current downstream chain (what stops where)

`run_grasp_phases`, `pick_and_place_fsm.cpp`:

| Phase | line | target | `blend_next` | terminal behavior |
|---|---|---|---|---|
| `Lifting` | 1328 | `lift_target_` | `place_hover_` | rise-confirm GATE (not a kinematic stop); `hard_level_hold` |
| `PlaceMoving` | 1393 | `place_hover_` | `place_pos_` | coarticulated via (blend rounds it) |
| `PlaceLowering` | 1408 | `place_pos_` | `nullopt` | **HARD STOP** — set-down + settle (upright) |
| `PlaceReleasing` | 1432 | `place_pos_` | `nullopt` | dwell (gripper opens — unavoidable stop) |
| `PlaceRetreating` | 1450 | `retreat_target_pos_` | `nullopt` | **HARD STOP** → joint-space homing to rest |

So `lift→hover→place` already coarticulates through the committed blend path
(`Controller.blend_radius=0.04`, the production path; `eb48ed0`). The dead stops left to convert:
**set-down arrival**, the **release dwell** (keep — physical), and the **retreat** (flow it toward
a return standoff instead of dead-stopping before homing).

Field vs blend selection in `efe_drive`: field only if `use_preference_field_` AND `blend_next`
set; else the coarticulation-blend path if `blend_radius>0` and inside the zone; else √-to-stop.

## Config anchors

`etc/config.toml` Controller: `use_preference_field=false` (prototype OFF), `blend_radius=0.04`,
`field_prec_pass=1.0`, `field_prec_stop=30.0`, `field_prec_ref=6.0`, `field_overlap=0.06`.
`etc/config_blendab_field.toml` is the field A/B config (`use_preference_field=true`,
`field_prec_pass=5.0`).

---

## Phases

### Phase 0 — Baseline capture
Run committed `config.toml` (field off, blend on), N≈10 cold-start episodes. Record whole-episode
time, stop-count, success. Parity reference. **Interleave** any later A/B (sim degrades across
runs — the old pilot was order-confounded, N=2, inconclusive, kept out of the paper).

### Phase 1 — Per-via precision plumbing
Add an optional per-call **`via_prec_pass`** to `efe_drive` (the skilled / pass-through precision
FOR THIS via), replacing the global interpolation for that call. Default = `field_prec_pass_`
so nothing changes until a phase opts in. The field then interpolates
`field_prec_stop → max(via_prec_pass, via_floor)` by `skill_c()` (= `c_seg(cur_seg_)`). cbuild.

### Phase 2 — Classify & wire the downstream vias
Source each via's precision from its segment `c_k`, floor the constraint-gated ones:
- `Lifting` via → **high floor** (grasp integrity; the rise-confirm must fire — this is exactly
  the over-cut the floor prevents). Effectively unchanged behavior, now principled.
- `PlaceMoving` hover via → **low** (free transit; passes through, as it already does via blend).
- `PlaceLowering` set-down → **medium floor** (upright set-down is constraint-gated; skill
  loosens the arrival but can't over-cut).
- `PlaceRetreating` → **low** + give it a `blend_next` toward the return standoff so the retreat
  FLOWS instead of dead-stopping before homing.
- `PlaceReleasing` dwell → leave as a stop (gripper must physically open).

### Phase 3 — Enable field, verify parity → emergence
`use_preference_field=true` in a test config. (1) novice: parity with the blend baseline (same
stops, same success). (2) as `c_k` rises: free vias pass through, floored vias still stop, success
preserved. Measure stop-count + whole-episode time reduction.

### Phase 4 — A/B + decide production path
Interleaved field-vs-blend cold-start. If per-via field matches/beats blend with fewer stops and
equal success → promote to production; else keep blend as production, field documented as the
realized prototype. Sync `EFE_CONTROLLER_MATH.md §4`, [[preference-field-prototype]], and the
paper's set-down-flow sentences (`main.tex` ~L452/L483) if the numbers change.

## Risks / notes
- **Lift over-cut** (prototype failure): the floor on the lift via is the fix — verify the
  rise-confirm still fires every episode.
- **Retreat→homing boundary** crosses into joint-space `moveJointsWithAngle` (`pick_and_place_fsm
  .cpp:1471`). Truly flowing it would need the return as a Cartesian via too — OUT of scope here
  (user chose Cartesian-phases-only); the retreat just blends toward a return standoff.
- Build with **cbuild**; run with **dangerouslyDisableSandbox**; stop with **SIGTERM not -9**.

## Related (separate work item, NOT this plan)
**Open-loop look-up reduction.** The confidence-driven observation sampling (every-cycle look-up
→ ~3 per episode as skill rises) lived in `1a8d011` (`specificworker.cpp`, `period = 1 +
round(c·(skilled_sample_period−1))`, belief/observation fusion) and was DROPPED in the FSM
refactor `dbddb07` — only vestigial state survives (`belief_grasp_`, `cycles_since_obs_`,
`obs_count_rep_`, declared+reset, never read; `run_tracking` re-observes every cycle at
`pick_and_place_fsm.cpp:254`). Not in the current paper. Worth restoring as a follow-up: it
decouples perception rate from control rate (a real look-up caps a real rig at ~20 Hz, not 60 Hz),
making the open-loop / model-precision story concrete. See [[skill-learning-open-loop]].

**Both this flow work AND the look-up/open-loop work are the two contributions of the planned RA-L
follow-up paper.** Full argument — incl. the `v·τ ≤ β·Δ` lag-safety coupling (speed is licensed by
lag) and the closed→open-loop bootstrap that is safe-because-slow at the start — is in
`drive/Papers/Papers Ongoing/2026/RAL-Kinova-followup/PAPER_NOTES.md`.
