# P1 registered predictions: parked test and A/B/A visit

Status: REGISTERED BEFORE ANY RUN (shared-engine plan rev 2, Task P1). Build under test: `feat/shared-engine` at 6eaeb138 (the countable binary that `parked_test.sh` launches). Nothing below was fitted to a measurement. Every bound is either derived here from manifest and config values, or is a formula that the report fills in with the run's logged ρ and Σ. Where no bound can be derived, the test reports failing, and that is said below.

The code that evaluates these predictions:
- `room_concept/estimation_scripts/scripts/aba_model.py`: the model; `aba_model.py registered` prints every number in this document;
- `parked_report.py`: checks (i) and (ii), and says which hypothesis the data favour;
- `aba_report.py`: check (iii);
- `aba_geometry.py`: ground truth, standpoints and frames.

## 1. Objects and ground truth

World: `threerooms-piso.wbt` (mtime 2026-11-22 19:29). The ground truth is the extent of the mesh surfaces the likelihood fits (spec "Graded quantities" plus the AuRo A3 amendment). θ = (cx, cy, H, w, h, yaw), where h is the depth and R(yaw)·ŷ points from the front to the back. The back is the outward normal of the nearest room-polygon edge. For all three objects it is world +y, the north wall.

| DEF | class | centre (world) | w | h (depth) | H | source |
|---|---|---|---|---|---|---|
| NIGHTSTAND_R | nightstand | (−4.003, 13.216) | 0.5046 | 0.5087 | 0.6155 | mesh body box (`tools/gt_mesh/mesh_extents.py`) |
| BED_1 | bed | (−5.183, 12.391) | 1.766 | 2.09–2.12 (an interval) | 0.72 | AuRo A3: frame ∪ blanket width, frame including the headboard, blanket top; centred on the mattress centre |
| TOILET_1 | toilet | (−0.553, 12.931) | 0.5890 | 0.7383 | 1.0126 | mesh footprint of bowl + tank. H is not graded (Δ_H = 0) and is reported only. |

At run time `aba_visit.sh` reads the live Webots pose (`webots_object.py get`) and stores it in `aba_plan_<DEF>.json`.

Frame check on objects_replay1: the chain world → room_concept internal (fit per visit) → published room frame (gauge) maps the bed's truth to (−0.313, 3.084). The bed-depth diagnosis measured the bed's true centre in that frame as (−0.305, 3.074). The two agree to about 1.3 cm.

## 2. Standpoints (the rule, then what it gives)

**Nominal.**
- A is 1.5 m in front of the object's front face, on the front normal through the centre, so the range to the centre is z0 = h/2 + 1.5.
- B is 45° round the centre at the same range. On an arc the |Δz| term of ρ is then zero, and the prediction rests on the bearing change.

**Feasibility.** A standpoint p is feasible when the robot's disc lies inside the room polygon and clear of every footprint in the room, the object's own included. The disc radius is r = 0.3252 + 0.10 = 0.4252 m:
- 0.3252 m is the circumscribed body radius (`ROBOT_GEOMETRY.md`). The robot must turn in place to face the object.
- 0.10 m is the controller's `RouteSafetyMargin` (`controller/etc/config.toml`).

This is a geometric test of collision, not a tuning threshold.

**Chosen range.**
- z is the range nearest z0, on a 1 cm grid, at which A and at least one B side are both feasible.
- If both B sides are feasible, the one with more clearance is used.
- The range stays the same at A and B. ρ is never gated: the report uses whatever the logged path gives.

| object | z0 | z chosen | moved | A (world) | B (world) | why it moved |
|---|---|---|---|---|---|---|
| NIGHTSTAND_R | 1.754 | 2.200 | +0.45 | (−4.003, 11.016) | (−2.448, 11.660), side + | The bed's frame comes within 0.30 m of the front normal. A is first clear of its foot corner at z ≥ 2.20 m. The − side lies over the bed. |
| BED_1 | 2.560 | 2.480 | −0.08 | (−5.183, 9.911) | (−6.937, 10.637), side − | The south wall and the DOOR_1 strip at y ≈ 9.48 |
| TOILET_1 | 1.869 | 2.530 | +0.66 | (−0.553, 10.401) | (1.236, 11.142), side + | At z0, B+ falls inside the bathtub (x 0.74–1.67, y 11.56–13.46) and B− outside the west wall (x −1.61). The nearest range with a feasible B is 2.53 m. |

The robot faces the object at every standpoint: after `go_to`, a `move turn_deg=` computed from its Webots pose. Footprints the parser could not read are FloorLight, Cabinet and the three monitors (templated PROTOs). The nearest of them is 2.48 m from any standpoint (`aba_model.py registered` prints each distance).

**Planned ρ for A → B** (`common/ai_belief/viewpoint.h`, ported in `aba_model.py`):
- Helios ring gap g = 0.0392 rad, as the agents log it at start.
- The controller's path lies between the chord and the arc.

| object | path | Δφ (rad) | Δz (m) | w⊥ at B (m) | ρ_cam | ρ_L |
|---|---|---|---|---|---|---|
| nightstand | chord / arc | 0.785 / 0.785 | 0.335 / 0.003 | 0.716 | 6.6e-3 / 7.7e-3 | 2.0e-9 |
| bed | chord / arc | 0.785 / 0.785 | 0.378 / 0.004 | 2.748 | 0.18 / 0.21 | 2.0e-9 |
| toilet | chord / arc | 0.785 / 0.785 | 0.385 / 0.004 | 0.939 | 1.2e-2 / 1.4e-2 | 2.0e-9 |

So B is a nearly independent view for the camera (except on the bed, whose angular half-width is large) and an independent one for the LiDAR. ρ_chain is not predictable: it depends on the localiser's corrections along the path. It is logged, and the report uses the logged value.

## 3. The model (every input named)

For each θ component k, every measured view j of channel c ∈ {cam, L} sees y_jc = θ_k + b_jc + e_j + s.

**The terms of y_jc:**
- **b_jc, the shared sensor error.** Its marginal is v_c. It is AR(1) across views, with the logged `rho_cam` / `rho_L`.
  - v_cam = 0.0289² on cx, cy, H, w and h. This is `kZedYoloBoundaryStdM`, measured (F1.6).
  - v_cam on yaw = 0.03², from `AI2CommonModeYawStd`.
  - v_L = 0.03² on cx, cy and yaw; 0.02² on H, w and h. These are `AI2CommonMode{Pos,Size,Yaw}Std` in all three class configs, the engine's own `sig2_L`.
- **e_j, the pose chain.** It is shared by both channels of a view. Its marginal comes from the logged `spose_*` and `lever_*` (the chain_B rows):
  - cx: spose_x² + lever_y²·spose_psi²;
  - cy: spose_y² + lever_x²·spose_psi²;
  - yaw: spose_psi².

  It is AR(1) with the logged `rho_chain`. The x–ψ cross-covariance is not logged and is left out.
- **s, the shape latent.** ρ = 1, marginal d = Δ²/12, where Δ is the manifest's `[shape] layer_spread_m`.
  - **PC** (one latent per channel, which is what the engine implements: kSc, kSL): s_cam and s_L are independent.
  - **S** (one shared latent): the same s for both channels.

**Posterior.** A parked view pins y. Its frames are dense, and their fresh share is about 0 while parked. So σ_k² = (1/P0_k + 1ᵀC⁻¹1)⁻¹.

P0 is the class prior from the manifest:
- H: `[prior.height] std_m`;
- w and h: `[prior.footprint] std_m`;
- yaw: `[prior.alignment] std_deg` (3°);
- cx and cy: none.

| class | Δ (cx, cy, H, w, h, yaw) m | P0 std H / w / h (m) |
|---|---|---|
| nightstand | 0.025, 0.025, 0, 0.04, 0.05, 0 | 0.10 / 0.06 / 0.06 |
| bed | 0.015, 0.015, 0.10, 0.20, 0.03, 0 | 0.10 / 0.20 / 0.20 |
| toilet | 0.06, 0.06, 0, 0.25, 0.12, 0 | 0.15 / 0.08 / 0.08 |

**Two view sets bracket the live σ.** Both use the logged ρ.
- **hi:** the dwells only (A1, B, A2), one view each. The ρ into a dwell is the product of every logged ρ since the previous dwell ended. Views taken while driving are dropped, so σ_hi is an upper bound.
- **lo:** every update is its own view. A new view starts whenever a logged ρ is below 1. Each driving update counts as a fully informative parked view, so σ_lo is a lower bound.

**Not in the model.** These are stated, not absorbed into any tolerance; a failure they cause is reported as a failure:
- cross-DOF coupling;
- the once-entered wall terms (stand-off on the wall-normal position and the depth h; room-axis on yaw);
- the birth position covariance on cx and cy, which is not logged. Its absence biases the predicted cx, cy σ upward.

**Tolerances.** All come from the solver's stop rule (`gn_tol_sigma` = 0.1, the derivation in `common/ai_belief/tests/toy_model.h`):
- one reported σ against an analytic value: 5 %;
- a ratio of two reported σ's: 10 %;
- nats: ln(1.1) per DOF, so ±6 ln 1.1 = ±0.572 nats on a six-DOF sum.

## 4. Predictions

### (i) The plateau at A, over the second half of the A1 dwell (60 s)

**Flat.** |σ_end/σ_mid − 1| ≤ 10 % + (√(1 + n q0/σ_mid²) − 1).
- n is the number of predict steps in the half.
- q0 is the permanence prior per step: rate · `AI2AgeNominalDtS` · scale².
- On the position DOFs q0 is 1.2e-9 (nightstand), 7.8e-10 (bed) and 1.6e-12 m² (toilet), so the growth allowance is below 1e-3 at σ ≈ 1 cm.

**Floor (the shape latent).** σ_A1 ≥ floor × (1 − 5 %), with the class prior included. A DOF with Δ = 0 has no floor check.
- PC: floor = (1/P0 + n_c·12/Δ²)^−½. With no prior and two channels this is Δ/√24.
- S: floor = (1/P0 + 12/Δ²)^−½. With no prior this is Δ/√12.

| object | DOF | Δ/√12 | Δ/√24 | floor PC (prior incl.) | floor S (prior incl.) |
|---|---|---|---|---|---|
| nightstand | cx, cy | 0.0072 | 0.0051 | 0.0051 | 0.0072 |
| nightstand | w | 0.0115 | 0.0082 | 0.0081 | 0.0113 |
| nightstand | h | 0.0144 | 0.0102 | 0.0101 | 0.0140 |
| bed | cx, cy | 0.0043 | 0.0031 | 0.0031 | 0.0043 |
| bed | H | 0.0289 | 0.0204 | 0.0200 | 0.0277 |
| bed | w | 0.0577 | 0.0408 | 0.0400 | 0.0555 |
| bed | h | 0.0087 | 0.0061 | 0.0061 | 0.0087 |
| toilet | cx, cy | 0.0173 | 0.0122 | 0.0122 | 0.0173 |
| toilet | w | 0.0722 | 0.0510 | 0.0430 | 0.0536 |
| toilet | h | 0.0346 | 0.0245 | 0.0234 | 0.0318 |

**Level.** σ_A1 ∈ [σ_lo (1 − 5 %), σ_hi (1 + 5 %)], computed by the report from the logged ρ and chain.

**Nominal values** below assume both channels measure, the planned ρ, and no chain. The run's chain (spose ≈ 5 cm in objects_replay1) raises cx, cy and yaw well above these:

| object | DOF | σ_A PC | σ_B PC | σ_A2 PC | σ_A S | σ_B S | σ_A2 S |
|---|---|---|---|---|---|---|---|
| nightstand | cx/cy | 0.0214 | 0.0156 | 0.0131 | 0.0220 | 0.0164 | 0.0140 |
| nightstand | H | 0.0162 | 0.0116 | 0.0095 | (same) | | |
| nightstand | w | 0.0177 | 0.0140 | 0.0125 | 0.0191 | 0.0158 | 0.0145 |
| nightstand | h | 0.0186 | 0.0152 | 0.0138 | 0.0206 | 0.0177 | 0.0166 |
| nightstand | yaw | 0.0197 | 0.0144 | 0.0120 | (same) | | |
| bed | cx/cy | 0.0210 | 0.0157 | 0.0132 | 0.0213 | 0.0160 | 0.0135 |
| bed | H | 0.0257 | 0.0235 | 0.0225 | 0.0315 | 0.0298 | 0.0292 |
| bed | w | 0.0433 | 0.0419 | 0.0414 | 0.0575 | 0.0566 | 0.0562 |
| bed | h | 0.0176 | 0.0136 | 0.0119 | 0.0185 | 0.0147 | 0.0131 |
| bed | yaw | 0.0197 | 0.0150 | 0.0126 | (same) | | |
| toilet | cx/cy | 0.0242 | 0.0192 | 0.0172 | 0.0271 | 0.0228 | 0.0211 |
| toilet | H | 0.0163 | 0.0116 | 0.0095 | (same) | | |
| toilet | w | 0.0447 | 0.0439 | 0.0436 | 0.0543 | 0.0540 | 0.0538 |
| toilet | h | 0.0280 | 0.0259 | 0.0252 | 0.0346 | 0.0332 | 0.0328 |
| toilet | yaw | 0.0197 | 0.0145 | 0.0120 | (same) | | |

The two hypotheses separate most on the layered sizes:
- bed w: 4.3 cm (PC) against 5.8 cm (S);
- bed H: 2.6 against 3.2 cm;
- toilet w: 4.5 against 5.4 cm;
- toilet h: 2.8 against 3.5 cm.

On the nightstand they differ by under 2 mm.

### (ii) The drop at B

σ_B/σ_A1 ∈ [min(r_lo, r_hi)(1 − 10 %), max(r_lo, r_hi)(1 + 10 %)], where r is the model's ratio under the logged ρ. In addition, σ_B must fall in its own level band.

Nominal ratios (PC), from the table:
- DOFs limited by the sensor latents drop to 0.72–0.82 (nightstand cx 0.73, w 0.79, h 0.82; bed h 0.77; toilet cx 0.79).
- The shape-limited ones barely move: bed H 0.91, toilet h 0.93, bed w 0.97, toilet w 0.98.

This is the latent prediction: a second view cannot remove a ρ = 1 layer error.

### (iii) A-again: promise and delivery

**Net delivery of a segment.** This is ½ Σ_k ln(σ_k² at the claim / σ_k² at the dwell's end), from the logged σ. Its predicted band is the same sum computed from the lo and hi model σ's, giving [min, max] ± 0.572 nats.

Nominal predictions:

| object | B, PC | A2, PC | B, S | A2, S |
|---|---|---|---|---|
| nightstand | 1.72 | 0.96 | 1.57 | 0.85 |
| bed | 1.23 | 0.72 | 1.13 | 0.66 |
| toilet | 1.20 | 0.65 | 1.04 | 0.56 |

**The derived A2 band is NOT "small"**, unlike the spec's wording ("a small promise and a small delivery"). The engine's ρ is a function of the PATH since the channel's last update, not of the standpoint. Returning A → B → A gives a correlation of ρ_AB·ρ_BA ≈ 0 between the A1 and A2 shared errors, so the engine treats A2 as a fresh third view, worth about half of B.
- If the live net delivery at A2 comes out near 0, the simulator's systematic error is a function of the VIEWPOINT, which the path-ρ model cannot represent.
- That would be evidence against the engine's ρ model, not a pass.
- The registered band stays as derived.

**Checks:**
- net_B in band_B;
- net_A2 in band_A2;
- the promise at the A2 claim in band_A2;
- net_A2 ≤ net_B. For an AR(1) shared error a third view adds no more than the second: (1 − ρ)² ≥ 0.

### Hypothesis decision

Each check in (i)–(iii) is evaluated under PC and under S.
- **The σ trace tests the implementation against its own model**, which is PC. If the logged σ follows PC and fails S, that confirms the engine computes what it claims. It does not show that the world has per-channel layers.
- **The independent reading is the ground-truth error** at the end of the last dwell. Over the (object, DOF) pairs with Δ > 0, ln N(e; 0, σ_S²) − ln N(e; 0, σ_PC²), where σ_X is the logged σ times the model ratio σ_X/σ_PC. A positive value favours S. With three objects and one error per DOF this is weak evidence, and the report says so.

## 5. Tests that report failing because no bound applies

- **The promise at a claim is the engine's `predicted_gain` at the offer's OWN standpoint**, the one its planner chose (offers.csv). The engine is never asked for the gain at A or at B: no such query exists. The report prints the offer standpoint's distance to the scripted one and tests the logged promise against the A2 band anyway. A promise for another standpoint that falls outside the band is reported as a FAIL.
- **`delivered.csv` summed over a visit is not the visit's net entropy change.** Each row is one update's drop from the Σ that update started from. Σ also grows between updates (predict and ageing steps outside `update()`), and the sum never subtracts that growth.
  - On objects_replay1's bed, 20 s windows sum to 11.6, 13.4 and 8.6 nats, where the net marginal change was 1.24 nats.
  - The report prints the sum beside the net change and tests only the net change.
  - This matters for AuRo test (b) as amended ("delivered = net entropy change"): the sum of `delivered.csv` is not that quantity.
- **Model omissions** (cross-DOF coupling, wall terms, the birth position prior) have no derived tolerance. A level or drop failure they cause stays a failure.

## 6. What objects_replay1 can and cannot say

`parked_report.py objects_replay1` parses the run and lists its parked stretches, which are runs of updates with every logged ρ exactly 1. The run has no scripted A/B/A, so **no pass/fail comes from it**. Its stretches are 0–9 s long (the robot drove a tour). It only shows that the parsing and the extraction run.

## Addendum 2026-11-25, before any run: the promise now comes from the scripted standpoint

This addendum follows a coordinator ruling on P1 concern 3. Every number in sections 1–4 above is unchanged. The bands, the floors, the planned ρ, the nominal σ's and the nominal deliveries are the same as in commit 890ff64a.

**What changed.**
- **The promise.** Section 5's first bullet is withdrawn: the promise at a claim is no longer the planner's offer for its own standpoint.
  - The agent now answers a gain query for a GIVEN standpoint. It calls `EpistemicPlanner::gain_at`, which is the computation of its offers: the same engine `predicted_gain`, cap, ρ path, camera with detector envelope, obstacles and Helios rays.
  - The query and `compute()` share `standpoint_gain()` and `offer_inputs()`.
  - `gain_query_test` checks that at the planner's own standpoint the query equals the proposal's `promised_nats` bit for bit (1.04201961 against 1.04201961 nats). It also checks that a look from the parked, already-seen front standpoint promises less than one from the unseen side (0.0033 against 1.03 nats).
  - The agent logs every answer in `gain_query.csv`, together with its update serial. When the offer cache holds a proposal of that serial, the same row also carries that proposal's `promised_nats` beside the gain recomputed at its standpoint: a live bit-for-bit check.
- **When the promise is asked.** `aba_visit.sh` asks at each claim (A1, B, A2), before the robot moves. The request is a line appended to the agent's `etc/gain_query.txt`, in the published room frame. Its yaw is the heading that faces the object, in the engine's math convention (atan2, 0 = +x), the same convention as `rc::nbv::face_viewpoint`.
- **New check at A1, B and A2:** |promise − net| ≤ 6 ln(1.1) = 0.572 nats. This is the stop-rule tolerance already used for the bands.
  - The promise is ONE hypothetical update's JOINT θ entropy drop. The net change is the visit's MARGINAL sum, and it includes the updates taken while driving to the standpoint.
  - The difference between the two (the change in the components' mutual information, and the driving views) has no derived tolerance. A failure it causes is reported as a failure.
- **The A2 promise check is kept** (the promise must fall in band_A2), now on the query's promise. **The same check is added at B** (the promise must fall in band_B).
- The planner's own offer is still printed at each claim, as context only, with its distance from the scripted standpoint.

**What did not change.** The registered expectation for A2 still applies: the derived band is not "small", for the path-ρ reason given in section 4 (iii). Both the promise and the net change at A2 are predicted to be about half of B's.
