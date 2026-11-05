# room_concept relocaliser — redesign from scratch (2026-11-01)

Status: implemented; legacy deleted (2026-11-05). A/B results in the plan's ledger (`docs/superpowers/plans/2026-11-02-relocaliser-rivals.md`,
tasks 7/7b/7c). Amended 2026-11-04: §3 likelihood, §4.2 prior, §5, §6, §7 and the §9 hot-path bar (each marked "Amended").
Original author: design pass for the user's verdict "the relocaliser is a piece of garbage".
Sources read: `memory/relocaliser-diagnosis-11-01.md`, `room_concept_reloc.cpp` (991 lines), `room_concept_reloc_modes.cpp` (282),
`room_concept_reloc_bootstrap.cpp` (376), `reloc_search.cpp` (953), `published_pose.h` (133), `room_scene_graph_late_recall.cpp` (252),
and the `[reloc]` lines of `estimation_scripts/crossing/warm19..warm27`.

## 1. Intent

The tracker (SDF localiser, early exit, sdf ~0.03 m) IS the localiser. The relocaliser must do one thing: **tell apart "my pose is wrong" from
"the scene is not what the map says", and only in the first case move the pose, paying for the distance**. Everything else it does today
(three-state HMM, learnt tracking emission, hold episodes, anchor retirement by information, clamp) is machinery around a decision that was
never given the right evidence. The new design is one posterior over a handful of pose hypotheses, scored by the SAME likelihood the tracker
uses, with a prior that makes jumps expensive. "Lost" is not a state; it is posterior mass on a rival hypothesis.

## 2. Failures it must make impossible (with evidence)

Baseline, counted from the recorded logs (`[reloc] lost|seed: moved` lines; gt_error.csv has ~45 ms rows):

| run | pose moves | searches | refused ("no pose explains") | duration |
|---|---|---|---|---|
| warm21 | 8 | 19 | 5 | ~11 min (≈ 44 moves/h) |
| warm25 | 63 | 66 | 2 | ~2.6 min (runaway, clamp trusting pred=0) |
| warm26 | 2 | 16 | 10 | ~16 min |
| warm27 | 11 | 27 | 11 | ~6.6 min (≈ 100 moves/h) |

Almost every search is triggered by `P(lost)=0.50x` at `s` of 0.02-0.05 m, i.e. while the scan fits as well as it ever does:
`warm27: P(lost)=0.504 at s=0.023 m`, `P(lost)=0.502 at s=0.024`, `P(lost)=0.501 at s=0.027 m — mixture search`; warm21 flips a healthy pose
(sdf .015) by 5.86 m / 179° at +253 s and 5.24 m / 176° at +449 s.

| # | Failure | Root cause in current code | What makes it impossible here |
|---|---|---|---|
| F1 | One bad frame triggers a search; trigger sits at 0.50x | `update_lost_belief`: LOST and MISMATCH share `log_broad_emission`, so any misfit splits mass 50/50; trigger `p_lost >= 0.5` | There is no LOST-vs-MISMATCH belief. The only thing that can move the pose is a **rival's log Bayes factor against the incumbent on the same scan**; a mismatch lowers both likelihoods equally and cancels (§4.3). |
| F2 | 4-6 m / 180° twin accepted for ~5 nats | `mode_posterior` prior `(1-pl)N(inc)+pl·U`, `pl` ≈ 0.5 | Prior of a rival is `π_kid / V` with a **fixed hazard** π_kid = `RelocHazard` (1e-3/frame), never scaled by an inferred P(lost). Price ≈ 16 nats (§4.2). |
| F3 | Published pose runs away (−268, −458; cov 290-539 m) | `clamp_correction` took `pred` = default (0,0,0) | The clamp is deleted. `pred` is part of the tracker's filter state (`std::optional`, absent after `RoomModel::reset` → no motion term is formed). A jump is priced in the prior, not clamped. |
| F4 | Twin accepted on the first Move | single-frame decision, `explains_like_tracking` only checks big median | A rival is accepted only on **accumulated** log-odds over novel scans (SPRT band ±ln 19, existing flagged constant), after paying the Occam price. A twin that fits no better than the incumbent has ΔlogL ≈ 0 per frame and stays at −16 nats forever. |
| F5 | Entry-area seeds vanish when the door anchor retires; whole-room search afterwards | `rt_.door_anchor` lifetime + `anchor_information` retirement logic | The entry area is not a separate object: it is the **initial incumbent prior** (§5). It decays by itself as the tracker's covariance shrinks. Nothing to retire. |
| F6 | Adoption after a swap leaves 0.07-0.26 m residual | single-scan forced search then tracker continues on the K-layout from a coarse mode | Multi-scan fit of the last N scans against K's SDF inside the prior mode's covariance (§6). |

## 3. Generative model

State: robot pose `x = (x, y, θ)` in the current room frame. Observation `z_t` = LiDAR wall-band points (the same ~hundreds of points the tracker
subsamples). Control `u_t` = odometry/IMU delta with its covariance (as today).

**Hypotheses** (a finite mixture, size ≤ 1 + K_max, K_max = 4):
- `H0` (incumbent): the tracker's Gaussian `N(μ_t, P_t)`, P from the SDF Gauss-Newton Hessian + odometry growth (what `UpdateResult.covariance` is now).
- `H_r`, r = 1..K: rival poses `(x_r, Σ_r)` carried rigidly by the same odometry delta each frame. Each has a single scalar of state: log-odds `λ_r` against H0.
- Implicit `H_kid` = "kidnapped / unknown": uniform over `V = area(room) · 2π` (or `V_entry`, §5). It is never a state; it is the *prior* that rivals are drawn from.

**Likelihood (shared by all hypotheses, so mismatch cancels).** *Amended 2026-11-04 (tasks 7b, 7c; the user's decision (a)).* The per-point
SDF band `(1-ε)·N(sdf_i(x); 0, σ²) + ε·U` of the original design charged off-map structure the TRUE pose sees (furniture, an unlisted
opening, an ajar door listed open) uniform clutter, while a pose standing at a wall opened the cell at its feet and explained the same returns
as "the next room" with lines fitted to the scan: on warm44 that pose won near-ties (2 false moves, 94 cm, cov 4.1 m²). ℓ is now a
**beam model on the room map** (`reloc_search.cpp`, `BeamTerm`), per return along its ray from the sensor at the hypothesised pose, z the
measured range, z* the range to the first mapped wall (ray cast on the polygon), σ_r = σ/|cos α| (σ = the wall-band σ, per-wall common-mode
offset `RelocWallOffsetSigma` marginalised from each mode's covariance as before), λ = `Params::unmapped_rate`:
- p_short(z) = λ·e^{−λz} on [0, z*): an unmapped surface meets the beam first (surfaces at λ per metre of ray);
- p_hit(z) = e^{−λz*}·N(z; z*, σ_r²): nothing in the way, the beam returns from the wall;
- p_beyond(z) = e^{−λz}·[N(z; z_line, σ_r'²) + λ] on (z*, z2), + e^{−λz2}·N(z; z2, σ_r2²): only through an OPEN cell; the same beam model in
  the next room, whose walls are lines fitted to the scan (each paying its Occam marginal), ending at z2 = the next mapped wall on the ray
  (this room's wall seen from behind). A return past z2 crossed a second mapped wall: clutter only;
- p_rand(z) = 1/L: ε = `RelocOutlierFrac`, L the room's extent.
Shut wall: `(1−ε)(p_short + p_hit) + ε/L`; open cell: `(1−c)·[(1−ε)(p_short + p_beyond) + ε/L] + c·shut`, c = P(leaf | open).
*Not adopted*: giving in-room short returns the scan's fitted lines as unmapped planar surfaces (symmetric with the next room) widened
the single-frame margins on warm44 but explained synthetic uniform clutter as lines (clutter30: 4 false moves, 6.6 m; mirrored room 3 flips).
**The doors are a UNION (task 7b, 77ed4e95)**: each listed door is shut, or open where listed, or (`list_wrong`) open anywhere; a cell's beams
are evidence once however many doors open it, `ℓ = Σ_i shut_i + log Σ_{c_1..c_K} Π_k prior_k(c_k)·Π_{C∈∪c_k} e^{Δ_C}`, exact by forward-backward
over the cells with beams. *Task 7c*: the boundary cells are cut at each listed door's mouth (the mouth is one cell, so a listed door explains
all its beams), and a door "anywhere" is a WINDOW one door wide (`opening_cell_m`) of consecutive quarter-door cells of one edge, the list_wrong
mass uniform over windows — the door's position marginalised at ¼ of its width instead of snapped to a 1 m grid (the grid's alignment had
decided reloc_selftest S). The recursion's state is (doors placed, cells still open by an earlier window): 2^K·(L+1) states; verified
against brute-force enumeration to 1e-6 nats.
λ is a flagged world constant: 0.1 /m (measured 0.02–0.16 /m per recorded truth frame, 0.085 pooled). The lattice (proposal only) still scores
the SDF raster; every mode, rival and ℓ comparison uses the beam model.

**Priors:**
- H0: `N(pred_t, P_pred)`, pred = previous posterior ⊕ odometry delta (no scan). This is the *odometry-predicted view*: the evidence of "consistency over motion" is the
  tracker's own innovation `μ_t − pred_t` under `P_pred`, which is already inside H0's marginal (§4.1).
- H_r at birth: `π_kid / V` (uniform; `π_kid` = `RelocHazard` per frame, a rate of the world, not a threshold).
- Between frames: `λ_r` is floored at its birth value (a kidnap can happen at any moment, so a long-refuted rival can come back at the price of a fresh one).

**Where the pose is weak:** `P_t` is a function of geometry (near a wall the scan constrains one direction; corridors constrain two of three). It enters twice, with no
extra parameter: H0's Laplace volume grows (cheaper to have been wrong along the unconstrained axis), and a rival that lies within H0's weak axis is simply H0's own
posterior mass (its prior under `N(pred,P)` is high, so its Bayes factor never accumulates a jump). The 'near a wall → weaker' fact therefore *reduces* false jumps along
the wall and keeps the pose rigid across it, which is what the geometry supports.

## 4. Inference: how "lost" is inferred

### 4.1 Marginal evidence (Laplace, both sides)
For hypothesis `h` with mode `x_h`, Hessian `Σ_h⁻¹`:
```
ln M_h = ℓ(x_h) + ln p(x_h | h) + ½ ln|2π Σ_h|
```
H0: `p(x|H0)=N(x;pred,P_pred)`. Rival: `p(x|H_r)=π_kid/V`.

### 4.2 Log Bayes factor of a rival against the incumbent (birth)
```
λ_r(0) = ln π_kid/(1-π_kid)  +  [ℓ(x_r) − ℓ(x_0)]  +  ln( ½|2πΣ_r| ^½ / V )  −  ln N(x_0; pred, P_pred) − ½ ln|2πΣ_0|
```
Order of magnitude (apartment room 40 m², Σ_r ≈ (0.02 m)²(0.02 m)²(0.01 rad)²): `ln(V/√|2πΣ|)` ≈ 9, `ln(1/π_kid)` = 6.9 ⇒ **≈ 16 nats up front**
before any Δℓ; a 180° twin of a healthy pose therefore needs the scan to prefer it by 16 nats, and a twin explains the scan *as well as* the incumbent, not better
(Δℓ ≈ 0). Gate-free: this is the Occam / prior-volume term; if the room is bigger or the pose tighter the price grows by itself.

**Amended 2026-11-04 (implemented in `room_concept_reloc_rivals.cpp`): a BOUNDED MIXTURE PRIOR.** The `−ln N(x_0; pred, P_pred)` term above is
unbounded: an overconfident P_pred priced a 5 cm tracker correction at 2·10⁴ nats. Every hypothesis now shares ONE prior over the pose,
`p(x) = (1−π_kid)·N(x; pred, P_pred) + π_kid / V` (pred, P_pred = the odometry prediction or the entry anchor, §5), and in Laplace form
`λ_r = [ℓ_r − ℓ_0] + [ln p(x_r) + ½ ln|2πΣ_r|] − [ln p(x_0) + ½ ln|2πΣ_0|]`. For x_r far from the prediction and x_0 near it this IS the
formula above; it differs only where §4.2 was silent (a rival AT the prediction while the tracker jumped off it; a tracker that jumped by itself),
and the incumbent can never be priced below `ln(V/π_kid)`.

### 4.3 Sequential update (per novel scan)
```
λ_r(t) = max( λ_r(t-1) + w_t · clip_ε[ ℓ_r(z_t) − ℓ_0(z_t) ] ,  λ_r(birth) )
```
- Both ℓ are the SAME robust likelihood on the SAME points ⇒ open doors, people, furniture, beyond-door returns hit both and cancel. This is the whole answer to F1.
- `w_t ∈ [0,1]` = view novelty (existing `view_change`): a parked robot's identical view counts once. Kept, it is correct and already in the code. A kidnap while
  parked still moves the view ⇒ w→1.
- `clip_ε` = per-scan LLR lower-bounded by `ln ε` (heavy-tailed mixture `(1-ε)·L + ε·U`, already used in `accumulate_late_evidence`), so one outlier scan cannot
  decide anything. No hard clamp.
- Evidence "moves with the robot": a rival is carried by the same odometry, so a wrong rival keeps losing as the view changes, a right one keeps winning.

### 4.4 Decision (Bayes, flagged)
Switch incumbent to `r` when `λ_r ≥ +ln 19` (SPRT band α=β=0.05, the existing flagged constant of `accumulate_late_evidence`). Equal-cost, sequential, justified in the
code already; the PRIMARY protection against flips is the 16-nat birth price and Δℓ=0 for twins, not the band. On a switch the old incumbent becomes a rival with `λ=-λ_r`
(honest Bayes), the published covariance gets the mixture spread `Σ_k w_k d_k d_kᵀ` (`held_spread`, kept; it is already the right object) while any rival has non-negligible
posterior weight. Real kidnap: Δℓ per novel scan is tens to hundreds of nats ⇒ accepted on the first or second proposed scan; recovery = proposer latency (≤ one search, §7) + 1-2 frames.

"Lost" reported to others (`pose_undecided_`, the controller's refusal) = posterior mass on rivals `Σ_r e^{λ_r}/(1+Σ e^{λ_r})`, a number, not a state.

### 4.5 What proposes rivals (never decides)
A background proposer (§7) runs the existing mixture search (`reloc_search`: yaws from wall structure, robust likelihood, Laplace modes) on a deep-copied scan and returns
up to K modes. It only **adds rivals** (with their `λ_r(0)` as in §4.2) to the main-thread list. It cannot move the pose. Healthy twin ⇒ proposed, scored at −16, expires. This
removes the need for *any* "should I search?" belief: search effort is a scheduling question, the jump is an evidence question.

## 5. Entry area after a door hand-over

On hand-over through a door (`note_passage_handover`), the new room's tracker is **initialised with** `N(μ_entry, Σ_entry)`:
- `μ_entry` = the doorway pose in the new room's frame (door node already published; the room-to-room RT the controller used), heading = door normal into the room
  (±π for the reversed crossing handled as a second component only if the crossing direction is unknown; it usually is known).
- `Σ_entry`: lateral = door width/2 (uniform along the aperture, moment-matched), longitudinal = robot clearance + odometry growth since the hand-over, heading σ ≈
  door-normal spread from the controller's alignment (a few ×10°) + odometry growth. Volume `V_entry ≈ w·d·Δθ ≈ 0.9 m × 1.5 m × 1.6 rad ≈ 2` vs `V_room·2π ≈ 250`.
This **is the incumbent's prior**, not a separate anchor object: it widens P (the tracker's early-exit/GN simply has more to do for a few frames) and shrinks by itself as scans
arrive. Consequently (a) rival births inside the entry ellipsoid are H0's own mass, never jumps; (b) rival births outside it pay the full ln(V/…) price; (c) a rival search
domain is `V_entry` for the first seconds (proposer samples ±3σ_entry only) and widens with odometry growth: the "entry area and heading, not the whole apartment" requirement is
the **proposer's sampling prior = H0's prior**, so the cost is ≈ 100x smaller than a whole-room search (§7) and the Occam price ≈ 4.8 nats cheaper for the correct entry pose.
No retirement logic exists: there is nothing to retire.

**Amended 2026-11-04 (as implemented): the entry prior prices BIRTHS only.** The door anchor is H0's prior in the bounded mixture of §4.2
(`pred`, `P_pred` = the best anchor component, `V = min(area·2π, V_entry)`, V_entry the ±3σ box of the anchor), so a rival inside the entry
area is H0's own mass and one outside it pays the full price. It does NOT restrict the proposer: `search()` has no sampling-only hook, so
the proposer still searches the whole room (`Proposer::submit`'s `sampling_cov` is unused). The anchor still retires by information
(tracker ≥ anchor in nats, body out of the doorway).

## 6. Adoption after a recognition swap ("this new room is remembered room_1")

Today: one forced search (`run_late_recall_search`) on one scan, the tracker continues from a mode, residual 0.07-0.26 m.
Proposal (kept structure of `seed_reloc_prior` + verdict; changed fit):
1. ltsm's `T(K←P)` gives the prior mode `N(T·μ, Σ_T)` in K's frame, `Σ_T` from the recogniser (layout-fit residual), not 0.
2. Fit the pose with the **last M scans** (ring of ~1 s, M≈10-20 keyframes with plain-SLAM relative poses, already available since the survey ran) jointly against K's wall SDF: one
   Gauss-Newton on 3 DOF, multi-scan Hessian ⇒ rotation and translation constrained by many viewpoints. Cost one-off ~10-20 ms × M subsampled points, off the hot path.
3. The verdict keeps accumulating `λ` (K explains vs does not), starting from the recogniser's log-odds (existing), by the SAME Bayes factor machinery (§4.3) with H_K = the fit and
   H_notK = broad emission. That folds `accumulate_late_evidence` and the lost-belief emission into one function.
   *Amended 2026-11-04 (f7cfccb1): N_eff.* The fit treats every point of every keyframe as independent, so N keyframes of one view (a parked robot)
   multiplied the information by N. Each keyframe is worth `1 / Σ_j exp(−½ d_ijᵀ B⁻¹ d_ij)` (B = the likelihood's basin: σ_sdf in translation,
   σ_sdf / median range in rotation), and the Laplace information is scaled by `N_eff / N` before the Gaussian fusion with the prior: a parked
   ring is no tighter than one scan, views a basin apart count fully. A late fit is dropped on a foreign pose jump or a room change. Each
   keyframe's returns keep their own sensor origin in the beam model (task 7c).
4. After acceptance the tracker's H0 = the multi-scan fit with its covariance (not a point). The 0.07-0.26 m residual is first diagnosed (§9 open question 3): if it is K's
   layout error (K remembered with that offset), no pose can remove it and it must be reported as such; if it is pose error, the multi-scan fit removes it.

## 7. Compute budget (nominal 20 Hz = 50 ms per cycle)

| Part | Where | Cost | Notes |
|---|---|---|---|
| Incumbent ℓ(z), Laplace terms | already computed by the tracker | 0 | reuse `UpdateResult` |
| Rival scoring (K ≤ 4 × ≤128 subsampled points, bilinear SDF, plus carry by odometry) | hot path | ~0.02-0.05 ms | 4×128 grid lookups |
| Sequential update λ_r, switch test, mixture spread | hot path | < 0.01 ms | O(K) |
| Proposer (reloc_search mixture search) | **worker thread, 1 in flight, deep-copied scan** | measured 8-60 ms typical, 220 ms worst (warm logs) | never on the 50 ms path; result is *applied* on the next cycle as rival births |
| Proposer cadence | scheduler | 2 Hz while no rival has `λ > λ(birth) + ln 19` of progress, continuous while one has (a rival is plausibly winning); every novel entry-area scan for the first seconds after hand-over | a duty cap (≤ 25 % of one core) is a scheduling constant, flagged; it cannot move the pose |
| Entry prior initialisation | once per hand-over | µs | closed form |
| Multi-scan adoption fit | worker, once per swap | ~10-30 ms | off hot path |
| Total added to the hot path | | < 0.1 ms | vs 50 ms cycle |

*Amended 2026-11-04: the hot-path bar is **p99 ≤ 5 ms** (relaxed by the user from 1 ms).* Measured (task 7c, tape replay), rival scoring with the
doors' union and the beam model costs 0.66–1.1 ms p50 and 1.6–2.6 ms p99 per frame (max 5.3 ms), not the ~0.05 ms estimated above: every ℓ
is a ray cast per point on the polygon, the cells' forward-backward and the BeyondLines bookkeeping.

Note the current mixture search runs synchronously inside `relocalisation_step` (`grid_search_initial_pose`, 8-222 ms), i.e. it can itself miss the 20 Hz budget.

## 8. What is deleted

From `room_concept_reloc.cpp`/`_modes.cpp`/`room_concept.h`:
- `update_lost_belief`, `log_tracking_emission`, `log_broad_emission`, `belief_`, `reset_lost_belief`, `set_belief_mismatch`, `lost_logodds_`, `track_mu_ln_s_`/`track_var_ln_s_`
  learning, `reloc_track_memory_frames`, `RelocTrackSMedian`, `RelocTrackLogSigma` (the 3-state HMM and its learnt emission; ~100 lines + params).
- `relocalise_from_belief` and the `p_lost >= 0.5f` trigger; `pending_episode_p_lost_` as the prior's scale; `rt_.hold_open` episode machinery (rivals replace it).
- `mode_posterior`'s `(1-pl)N + pl·U` prior, the `explains_like_tracking` median gate, `last_search_confirmed_incumbent_`/`last_search_moved_` bookkeeping (≈ 280-line file shrinks to the rival list).
- `rt_.door_anchor`, `step_door_anchor`, `anchor_information`/`gaussian_information` retirement block (~40 lines in `relocalisation_step`), `passage_prior.h` anchor mixture in `log_prior`
  (replaced by the entry prior as H0's prior).
- `published_pose.h::clamp_correction` and its kinematic limits (pose jumps are priced, not clamped); keep `substitute_survey`.
- `late-recall` single-scan `run_late_recall_search` + `late_pending_` hostage logic is replaced by §6 (the verdict function stays, once, shared with rivals).
- Config: `RelocHazard` stays (the one rate), `RelocOutlierFrac`, `RelocWallOffsetSigma`, `RelocBodyClearance` stay; the rest of the Reloc* keys go.
Kept: `reloc_search.cpp` as the proposer (likelihood, apertures, Laplace modes), `bootstrap` for t0 seeding, `held_spread`.

## 9. Offline evaluation plan (on recorded runs with ground truth)

Data: warm19..warm27 (`gt_error.csv`: gt vs est pose at ~45 ms, sdf_mse, pred_x/y/theta; `gt_pose.tsv`; `door_events.csv`; `[reloc]` lines). Existing tools:
`tools/reloc_selftest.cpp` (300 injected kidnaps), `reloc_handover_replay.cpp`, `scan_slam_replay.cpp`.
Prerequisite (step 0): the warm runs do not appear to store raw LiDAR scans (only `anchor_frames.bin` around hand-overs). Add a binary scan ring dump (wall-band points + odom delta
per frame, ~2 k points × 20 Hz ≈ 0.5 MB/s, gzip) in room_concept and re-record 3-4 crossing runs. Until then the real-data part is limited to counting baseline decisions in the
logs and replaying hand-over anchors.

Metrics:
1. **False relocalisations per hour while healthy**: count published-pose jumps > max(3σ_pred, 5 cm) in frames where `|gt−est| < 5 cm` just before and after (gt_error.csv). Baseline from the table
   in §2 (≈ 44-100/h on warm21/27, runaway on warm25).
2. **Recovery time after a real kidnap**: inject teleports in the scan replay (poses ≥ 1 m and 90°/180° away, also during motion and parked), time from the teleport to published pose within 10 cm / 5°.
   Baseline from `reloc_selftest` (right basin 300/300 vs 235/300 of the old search; per-event latency not recorded).
3. **Worst pose error** (max |gt−est| over the run, and p99), and the max published covariance.
4. **Twin test**: replay healthy segments in the *mirrored* room (180° point symmetry) and verify zero flips; and a genuine kidnap to the twin must produce a published covariance spread, not a confident flip.
5. **Hand-over test**: for each door crossing in warm19-27, pose error at +0.5 s, +2 s, +5 s after the hand-over, and the number of proposer calls (compare the entry-area search volume).
6. **Adoption test**: every recognition swap in the logs: residual (sdf_mse, pose-vs-gt) at +1 s, +5 s; split into pose error vs layout error by comparing against gt.
7. **Cost**: hot-path µs per frame (rival scoring), worker ms, frame-time p99 vs 50 ms.

**Acceptance bar** (proposed, the user decides): (1) 0 false relocalisations in all healthy replays (≥ 1 h cumulated) and in the mirrored-room test; (2) recovery median ≤ 0.5 s, p95 ≤ 1.5 s, success
≥ 99 % of ≥ 300 injected kidnaps (not worse than the old 300/300); (3) worst pose error on warm19-27 replays ≤ the best old run (warm22/23/24 were quiet), no covariance above 1 m²; (4) hot-path
added time ≤ 5 ms p99 (amended 2026-11-04, was < 1 ms), no frame over 50 ms attributable to relocalisation; (5) adoption residual ≤ 3 cm for pose-attributable error; (6) the deleted code stays deleted (net negative line count).

## 10. Risks and open questions for the user

1. **The sequential λ needs Δℓ calibrated on real scans.** If ε/σ are mis-set the 16-nat price is wrong by a factor; the offline test with real scans (step 0) is the check. The number of
   effectively independent points per scan (common-mode wall errors) matters: Woodbury handles per-wall offsets but scan points are still correlated; a conservative effective-N should be measured.
2. **Truly symmetric rooms**: a real kidnap to an exact twin is undetectable by any scan; the design publishes the mixture spread and waits for a distinguishing view (door, furniture). The controller then
   needs to accept a bimodal pose for that time. Acceptable? (today it is forced to pick one.)
3. **Adoption residual cause is not yet diagnosed** (I did not run it). Could be K's remembered layout offset (unfixable by pose) rather than pose error. The §6 fit helps only the second case.
4. **Constants that stay**: SPRT band ln 19 (α=β=0.05, existing and flagged), `RelocHazard` = 1e-3/frame (a world rate), the proposer duty cap ≤ 25 % of a core and 2 Hz base cadence (scheduling,
   can't move the pose), K_max = 4 rivals. Nothing is a trigger on a pose or misfit value.
5. **Rival carry by odometry**: pure dead reckoning for ≤ few seconds; a rival that survives longer is refreshed by the proposer (re-detected rivals merge by same-mode test, which needs a
   merge rule — I'd use "same Laplace mode" = Mahalanobis under Σ_r+Σ_new, a threshold in disguise; flag).
6. **Step 0 (scan recording) is a prerequisite** for any believable real-data evaluation; it costs a few hundred MB per hour.
7. **Cutover**: this replaces three interacting files; suggestion is to land it behind `RelocEnabled`-style flag with the old path unchanged for A/B, then delete (the user's "code and code and code" concern
   argues for removing the old path in the same series once §9 passes).
