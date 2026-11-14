# door_concept: one model for every mask, birth and removal by comparison, looks that settle hypotheses

Date: 2026-11-14. Status: APPROVED by the owner 2026-11-14 ("Review ok"). Nothing coded.
Decisions marked **(PI)** are the owner's, from the 2026-11-14 conversation. Everything else is the designer's call and can
be changed by the owner.
Builds on: `2026-11-05-structure-from-evidence-design.md` §3.2 (birth by comparison, typical-size prior),
`2026-11-07-door-existence-model-design.md` (one existence belief; largely in code now, see §2),
`2026-11-09-undecided-candidate-look-design.md`, `2026-11-10-room-door-belief-design.md` (room_door_count.h).
Recon used (spot-checked against main 87b6018a): memory files `door-recon-11-14-gates.md` (157 gates and constants,
birth to removal) and `door-recon-11-14-looks.md` (the look code and what it lacks).

## 1. Intent

**(PI)** "A well founded door agent with no patches and the minimum number of thresholds." Every decision door_concept
takes about a door (is there one, which door explains this mask, is it still there, where should the robot look) falls
out of one generative model. Hard switches added after the fact to fix one failure are removed, or kept only with a
written reason why no model term can do their job.

The owner's picture of the case that drives this **(PI)**:
- An open door can be hard to detect. When door_concept gets a mask of door height and a few centimetres deep, that is
  a leaf seen side-on. It looks for more evidence: LiDAR returns passing out of the room's outline next to the leaf's
  hinge end. Together they make "an open door there" a strong hypothesis.
- The door is created in the wall of its FRAME (the wall the hinge is in), not in a wall parallel to the leaf.
- Near a door that already exists, the leaf is evidence for that door: it confirms the door or counts against it,
  depending on what the door's model (leaf angle, hinge side, swing side) predicts.
- Edge case: two doors in different walls, very close (a corner), one open and its leaf hiding the other. The leaf must
  be contrasted with the LiDAR opening to decide which model it fits.
- When the situation stays ambiguous, an epistemic look gathers more information. **(PI)** The observing pose is the one
  that most reduces the current uncertainty, so it depends on the situation, not on a fixed recipe.

**(PI)** Scope includes every part of the 11-05 design: the typical door-size prior, the wall-band prior, the per-wall
cost of a second door, and birth decided by comparing "exists" against "does not exist".

## 2. What exists today (main 87b6018a)

Already built, kept and built on:
- **Birth is already a comparison.** Each tracker candidate carries a `PlaceBelief`; a detection gets `birth_evidence = 1`
  iff `decide(L)` says the place exists (`core/door_existence.cpp` ~425-435). The shared tracker's frame streak is set to 1,
  so it no longer decides.
- **Place prior** `core/place_prior.h`: wall-band location term (one-sided at the proto-room frontier) and per-wall base
  rate (0.4 for the first door on a wall, 0.01 for each further one).
- **Existence channels** (`core/existence_update.cpp`, `core/view_ledger.h`): mask support, ZED, geometry, Ricoh, with
  a view ledger that counts a view, not a frame, as one trial.
- **Room door count** `core/room_door_count.h`: exact posterior over how many doors a room has.
- **LiDAR opening evidence** for a known door: `door_opening_edges.h` (the through-span of an opening, registered to its
  wall), `door_wall_gap.h` (wall then hole, for crossing). room_concept's `doorway_necks.h` treats an unrecognised
  doorway as a hypothesis on the outline side.
- **Looks**: three producers (live door `aff_door_<id>` in `epistemic_planner.cpp`; remembered door
  `aff_door_look_<name>` in `door_look.h`; undecided candidate `aff_door_look_place_<key>`, specificworker.cpp ~4440).
  Delivered as ObjectAffordance nodes, ranked by AffordanceManager (`gain − λ·dist`,
  common/affordance_manager/affordance_manager.cpp ~860-975). The delivery path is reused as is.

Not done:
- The size prior in `etc/config.toml` is still the Webots fit prior (1.00 ± 0.06 m wide, 2.10 ± 0.08 m high). The typical
  prior (0.85 ± 0.12, 2.05 ± 0.06) is only a comment (line ~333).
- `etc/door_detectors.csv` has never been fitted: every detector rate is a compiled default with n = 0.
- No look scores competing explanations; each asks one yes/no or one Gaussian question about one door. Visibility for
  looks is camera only (common/nbv `visible_fraction`): no LiDAR prediction, no predicted mask, hinge and swing ignored.

The ten hard switches that overrule or bypass the comparison (from the recon, the ten most patch-like):

| # | Switch | Where |
|---|---|---|
| S1 | Swept-arc disc + aperture-quad birth veto, radius half_w + leaf_w + half_t, hinge-agnostic | core/door_core.cpp ~742-765 |
| S2 | "Claimed slice" birth veto + greedy one-mask-to-one-door evidence assignment | core/door_core.cpp ~656-663, ~778-797 |
| S3 | Fixed separation `birth_min_sep` 0.70 m + `MergeOverlap` 0.20; merge removes a door outside L, keeping the one with more matched frames | common/instance_tracker.h ~240, ~308; core/door_core.cpp ~343-366 |
| S4 | Truncation cliff `trunc_frac > 0.10`: the frame becomes "no trial", no support sample; checked in two places | core/door_core.cpp ~187; core/existence_update.cpp ~21-29, ~54; core/door_fit.cpp ~883-892 |
| S5 | Per-wall base rate switched on an integer count ordered by birth; candidates counted differently; a second door-count prior elsewhere | core/place_prior.h ~89; core/door_existence.cpp ~373-375, ~546-549; specificworker.cpp ~4517 |
| S6 | Ricoh counts hits only, never a miss (`Ricoh.OpenDetectRel` / `ricoh_detect_scale` are dead) | core/view_ledger.h ~92, ~135; core/existence_update.cpp ~74-78 |
| S7 | Geometry may not refute: `lg = max(0, lg)` on an unfitted μ1 = μ0 + 2s | core/view_ledger.h ~142 |
| S8 | One detection per candidate carries the verdict; others `birthable = false`; judged only on a new frame | core/door_existence.cpp ~296, ~431-433 |
| S9 | "Is this mask an existing door" decided by a Mahalanobis gate (9.0) and a metric gate (0.40 m) in five places | core/existence_update.cpp ~122; core/door_existence.cpp ~309-355; core/door_core.cpp ~647, ~684 |
| S10 | Restore skips and radii (same-hole ~1.85 m, refutation dates, lift radius), `kFitWithoutFreshDataFrames = 5` | door_identity_table.h ~318-352, ~618, ~667; core/place_table.h ~125 |

Also in the recon, `clutter_fraction_leaf_marginal` (door_belief.cpp ~112-165) marginalises leaf angle and hinge but
uses ONE swing side. That is what removed a restored, open door in chat_e2e18 and re-born it on the leaf plane.

## 3. The model

### 3.1 Vocabulary: labelled Bernoullis

door_concept is written as a labelled multi-Bernoulli filter, in the same terms the objects session uses for chairs
(2026-11-14), so the two can be merged later. **(PI)** Separate for now, merged later.
- Each door and each undecided candidate is a labelled Bernoulli: an existence probability r (kept as log-odds L) and a
  state (place on its wall, width, height, leaf angle φ, hinge side, swing side). Labels are door_concept's door names.
  Remembered doors restored from LTSM notes are labelled Bernoullis with the prior their note gives.
- Each frame brings a set of measurements: camera masks (ZED, Ricoh) and the LiDAR scan.
- A detection probability p_D(x, pose) comes from the existing visibility model (frustum, walls and furniture, range,
  truncation).
- Clutter has an explicit density κ (spurious door-class masks per unit image area or per view).
- New doors appear with a birth density b(x) = wall-band prior × typical-size prior × per-wall base rate (§3.4).

### 3.2 One responsibility per mask

Every mask m is explained by exactly one of:
1. door i's aperture (frame or open span), for each door or candidate i near it;
2. door i's leaf, marginalised over φ × hinge × **both** swing sides;
3. a new door (the birth density);
4. clutter (κ).

The responsibilities ρ(m → ·) are the normalised products of each cause's likelihood and prior (each existing cause
weighted by its door's r and p_D). They replace, in one computation:
- the swept-arc veto (S1) and the claimed-slice veto (S2): a mask a known door's leaf or aperture explains gives little
  responsibility to "new door", so its birth contribution is small, continuously, and only as far as that door's actual
  φ/hinge/swing posterior explains it;
- the five hard "is this an existing door" gates (S9): the gates become the likelihood g(m | door i), Gaussian in the
  existing fit's terms;
- the greedy assignment (S2) and the one-detection-per-candidate rule (S8): every mask contributes to every cause in
  proportion to its responsibility;
- the single-swing-side leaf marginal in `clutter_fraction_leaf_marginal`.

Existence update: the exact Bernoulli update with clutter, marginalised over associations — not "an assigned mask
proves the door" (r ← 1). For one door and a set of masks M, r' = r·(1 − p_D + p_D·Σ_m g(m)/κ(m)) / (1 − r·p_D +
r·p_D·Σ_m g(m)/κ(m)), with the association shared between doors by the responsibilities. A miss with p_D = 0 (not looked
at, occluded) changes nothing. The view ledger's rule (a view, not a frame, is one trial) stays: frames of one view are
correlated and must not be counted as independent evidence.

The door's continuous state (designer's call): the existing fit stays the per-door state estimator. Hinge side × swing
side is a discrete mixture of four branches, each with the existing leaf-angle posterior (`door_phi_posterior.h`); the
branch weights are updated by the leaf likelihoods. The responsibilities weight what each mask contributes to the fit.
No new continuous estimator is introduced.

Ambiguous clusters (masks and doors that the gates connect; e.g. the corner with two doors) keep a small set of weighted
joint hypotheses instead of one best assignment (k-best within the cluster). This set is also what the epistemic look
scores (§4). The cluster boundary is computed from the likelihoods (a mask whose likelihood under a door is negligible
against κ does not join it), not from a fixed radius.

### 3.3 Every channel two-sided

- **Ricoh counts misses** (S6): a view in which the Ricoh should have seen the door and did not counts against it, by its
  p_D. The dead config is either used or removed.
- **Geometry may refute** (S7), once its two likelihoods (door vs no door) are fitted from recorded runs. Until fitted,
  its weight is the honest one of an unfitted channel, not a one-sided clamp.
- **Truncation is a loss of information, not a cliff** (S4): a truncated view constrains size less (the size likelihood
  widens with the fraction cut off) and keeps whatever it measures about place and existence. One function, one place.
- **The detector rates are fitted** from recorded runs into `etc/door_detectors.csv` (hit rate, false-alarm rate per
  sensor and range band) before the series that judges this work. Fitting is part of this work.
- **LiDAR opening** is a channel of its own: for a hypothesised aperture on a wall, the likelihood of the scan under
  "opening here" (returns pass beyond the room outline along the aperture's span; the leaf meets the wall only at its
  hinge) against "solid wall here". Built from `door_opening_edges.h`'s through-span likelihood, generalised from "a
  known door" to "a hypothesised aperture".

### 3.4 Priors

- **Typical-size prior (PI, values approved 11-05):** width 0.85 ± 0.12 m, height 2.05 ± 0.06 m. It enters birth as the
  complexity term: the KL of the door's fitted size estimate against this prior, plus uniform along the wall. It replaces
  the Webots fit prior in the birth and existence path. Never score a raw mask's height against it (the `size_w` note in
  door_core.cpp ~455-537 records that this blocked births); a truncated view pays no size cost.
- **Wall band (PI: "doors are in walls, an extremely hard prior"):** as in `place_prior.h`, carried as one term at birth
  and after. Its σ carries the wall's own uncertainty (a real door was 0.2 m inside the estimated outline in recog_proto5).
  A provisional room's outline is a frontier, not walls (the existing one-sided treatment).
- **Per-wall base rate (PI):** a second door on one wall is very rare except in a corridor. **(PI, 2026-11-14) The two
  rates are learned from memory, seeded with 0.4 / 0.01.** Each is a Beta belief: P(a wall has a door) and P(a wall with a
  door has another). The seed is today's values weighted as five walls of experience (Beta(2, 3) and Beta(0.05, 4.95));
  the counts come from LTSM's remembered rooms in the graph (memory twins of walls and doors; agents talk only through
  memories, so no file), excluding the room the robot is in, so a room's own doors never set their own prior. The rate
  used is the Beta mean; with an empty memory behaviour is today's. A per-metre rate was considered and rejected: it
  cannot make a second door on a short wall as rare as the owner's prior. Corridors are handled later, when rooms have a
  type. What changes (S5): the cost does not depend on birth order. It is computed from the
  expected number of OTHER doors on that wall, Σ r_j, from the same beliefs, so removing a door frees the other of the
  cost, and an ambiguous wall does not pay in full. (Designer's call on the circularity: each cycle reads the other doors'
  r from the previous cycle, one sweep, no fixed-point iteration, so the result is order-independent; inside a k-best
  cluster the joint hypothesis carries the count exactly.) One door-count prior: `room_door_count.h` is the only one; the
  second one (specificworker.cpp ~4517) goes.
- A wall is not a polygon edge (room_1's outline has 32 edges): walls come from the existing `wall_hypotheses`.

### 3.5 Birth and removal by comparison, both ways

- A candidate is born when F(exists) < F(not), i.e. L > 0 with the priors of §3.4 inside L. Removal is the same
  comparison read the other way. No hysteresis band (two thresholds). The unequal cost of a false door and a missed door
  is already stated once as a decision loss in `core/door_decision.h`: C_keep and C_delete in seconds of the robot's
  experience (compiled defaults from the logged tours, 30 s and 88.25 s, persisted in etc/door_costs.csv; a door being
  crossed has an infinite delete cost). It stays as is; the line follows from it. (Designer's call: keep today's loss, so
  flag-off births are identical and the comparison changes only in its terms.)
- A newly born door starts from the evidence that bore it, not from a fixed `Existence.BirthLogodds` (1.0). A restored
  door keeps its restore prior.
- **Duplicates and merges (S3).** Two labels at one hole are two hypotheses competing for the same masks and the same
  LiDAR opening. Their responsibilities split the evidence, the per-wall base rate charges the second, and the weaker one
  falls below the line by comparison. The fixed 0.70 m separation and the merge (a second removal authority outside L)
  go. This is also the fix expected for dm12 (an entry-mirror door created 0.17 m from a live crossed door): the crossed
  door explains the masks, the mirror candidate gets almost no responsibility and pays the second-door cost.
  `common/instance_tracker` is not edited (other agents use it): door_concept sets its separation, in its own config and
  only under the flag, to the minimum that keeps the tracker's own bookkeeping sound, and decides duplicates itself.
- **Restore (S10):** a remembered door is a labelled Bernoulli with its note's prior; whether a live mask is that door
  is its responsibility, not a radius. The dated-refutation rule from 04d256d7 stays (it is about memory, not evidence).
  The frame counter `kFitWithoutFreshDataFrames` is replaced by the view ledger's notion of a fresh view.
- Amend CONCEPT_AGENT_LIFECYCLE.md §1 and door_concept's CREATE row: a door is born by comparison inside door_concept;
  the shared tracker only holds candidates.

### 3.6 The leaf seen side-on

A mask about door-high and a few centimetres deep is a leaf seen edge-on far more often than a door frame. Its
likelihood under "leaf of a door with hinge at this end" is computed for each end of the mask:
- The hinge end must touch a wall (the wall-band prior); that wall is the FRAME wall **(PI)**.
- The aperture lies along the frame wall, starting at the hinge, on either side (hinge left or right), width ≈ the leaf's
  length (with the size prior), φ ≈ the angle between the leaf and the frame wall.
- The LiDAR opening channel (§3.3) scores each of these apertures: returns passing out of the outline along the
  aperture's span support it.
- If an existing door's leaf posterior explains the mask, the responsibility goes to that door: its existence, φ, hinge
  and swing are updated. If no door explains it and the opening is seen, "new door in the frame wall" wins and the door
  is born there. If the opening is solid wall, clutter or a hidden-door hypothesis wins.
- The corner case: the hypotheses "open door A in wall 1 whose leaf this is" and "door B in wall 2 hidden behind it" are
  in one cluster (§3.2). Only A predicts an opening at its hinge; B predicts its own aperture's opening in wall 2. The
  LiDAR channel separates them where the scan sees either opening; where it sees neither, the cluster stays ambiguous and
  the look of §4 is offered.

### 3.7 The existence score that swings (chat_e2e19)

Restored door_2's geometry channel swung ±2..5 nats per update with alternating sign (door_concept.log 18093-19862 in
that run), withdrawing a crossing.

**Diagnosis (from `door_existence_log.csv`, door_2, 3764 rows; `tools/existence_swing.py`; the replay tape does not exist
yet).** The suspected cause (two explanations picked in turn) is REFUTED; the swing is the model's own scale moving, not the
measurement. Evidence, by CSV line (header = line 1):

- `geo_x` (the leaf contour's edge support, in units of the frame reference) is steady across a swing: lines 2029-2036
  (cycles 10174-10181) have geo_x = 1.53, 1.48, 1.52, 1.53, 1.40, 1.43, 1.54, 1.38, while geo_dl goes +1.65, +1.18, -2.18,
  -0.64, +5.03, -4.59, -0.34, +3.71. It is not alternating between two values.
- What alternates is `geo_s`, the sd of the control contours' support: 0.031, 0.020, 0.082, 0.252, 0.015, 0.116, 0.206, 0.020
  (a 17x jump frame to frame), with geo_mu0 wandering 0.74-0.97 and geo_mu1 = geo_mu0 + 2 geo_s following it. geo_dl is large
  and positive exactly when s is tiny (line 2033: s = 0.0146, +5.03; line 2036: s = 0.0195, +3.71; line 2039: s = 0.0144,
  +3.39) and negative when s is large (line 2034: s = 0.116, -4.59; line 2037: -2.87).
- Why: `ViewLedger::add_geometry` keeps only the running mean of x and OVERWRITES mu0/mu1/s2 with the latest frame's, and
  `recompute` re-scores the whole view's aggregate with them: lg = wb((xb-mu0)^2 - (xb-mu1)^2)/(2 s2) = 2 wb ((xb-mu0)/s - 1)
  because mu1 = mu0 + 2s. So lg scales as 1/s, and s is a sample standard deviation of only a few control contours
  (view_ledger.h:139-143; existence_update.cpp:225-232), a very noisy estimate. The delta geo_dl is therefore mostly
  "the view's accumulated evidence re-priced at today's s", not new evidence, and its sign alternates with s.
- The same line pattern appears in bursts at lines 715-743 (cycles 8862-8890, p 0.68 -> 0.22 -> 0.90 -> 0.35 within two
  cycles), 1159-1165, 2009-2038 and 18534-18626 of the log; 100 sign changes among 664 nonzero geo_dl rows of door_2. Line
  18622 (cycle 9312) adds an outlier frame: geo_x = 6.4 with s = 2.4, mu0 = 2.3 (a degenerate control set), geo_dl = -2.15
  undoing the +2.15 of the previous line.
- sup/zed channels in those rows are near zero or constant, as the owner noted; the existence swing is the geometry channel
  alone (L moves 3.44 -> -0.98 across lines 2033-2034).

**What the logs cannot decide.** The log holds the last frame's x, mu0, mu1, s, not the view's running mean xb, its weight wb
or the control count k, so the exact lg per row cannot be recomputed from it, and whether k = 2 (one degree of freedom)
dominates is inferred, not seen. The tape must check: (1) k per frame and the controls' individual supports; (2) that
replaying with s2 held at its running pooled value removes the alternation; (3) the outlier frames (line 18622).

**Fix, and where it is carried.** Not the §3.2 hinge marginalisation. The geometry likelihood must carry the uncertainty of
(mu0, s) itself instead of treating a k-control sample sd as exact: each frame is added to the view as its own term (or the
view keeps pooled sufficient statistics of mu0 and s2 across frames, with a Student-t / inverse-chi-square marginal of
its k-1 degrees of freedom), so the score is additive and no frame's s re-prices the others. This is a new task for the
controller to add to the plan (geometry channel of Task 6's ledger work, or its own task before it); Task 6 is not the place
unless it already rewrites `add_geometry`.

## 4. Epistemic look: the pose that settles the current hypotheses

**(PI)** The look is the pose that most reduces the current uncertainty, so it is computed from the situation.

- **What is uncertain.** The weighted hypothesis set of each ambiguous cluster (§3.2), including the null and clutter
  members, plus the existence of every undecided candidate and restored door. One set, H, with weights w(h).
- **Candidate poses.** Room-wide reachable poses (the existing standpoint and stand-off machinery, `door_standpoint.h`,
  common/nbv obstacles), not a fixed fan around one door.
- **Forward model.** For each hypothesis h and pose q: the predicted masks (each door's aperture and leaf rendered with
  its φ/hinge/swing, occluded by walls, furniture and the other doors' leaves) and the predicted LiDAR returns (opening or
  wall along each hypothesised aperture). Hinge and swing geometry enter here; today's look visibility ignores them.
- **Score.** Expected information gain: EIG(q) = H(w) − E_{z~p(z|q)} H(w | z), computed over the discrete set by
  enumerating the predicted outcomes (each predicted observation is mostly "opening seen / not seen" and "mask of this
  shape / none", so the sum is small). Detection and clutter rates come from the fitted detector table.
- **Delivery.** One producer of epistemic door looks, publishing the existing ObjectAffordance node type with the gain in
  nats; AffordanceManager ranks it against travel as it does today (`gain − λ·dist`). The three current producers become
  this one, or are kept where they ask a question H does not contain (e.g. D-optimal place refinement of a born door), with
  the reason written down. No new cortex attribute is expected (designer's call): the existing ObjectAffordance gain
  carries the EIG in nats, and the hypothesis set stays inside door_concept. If one proves necessary it goes through
  REGISTER_TYPE and the owner's cortex reinstall, flagged in the plan as a blocking step.
- **On arrival** the observations update H through §3.2, and the look is re-scored. A look whose hypotheses have been
  settled is withdrawn by its gain falling, not by a rule.

## 5. The fate of each switch

| # | Fate |
|---|---|
| S1, S2, S8, S9 | Replaced by the responsibilities of §3.2. |
| S3 | Replaced by §3.5's competition; the tracker's separation set to the minimum its bookkeeping needs (reason written). |
| S4 | Replaced by truncation as widened size likelihood (§3.3). |
| S5 | Replaced by Σ r_j on the wall; one count prior (§3.4). |
| S6, S7 | Two-sided channels with fitted rates (§3.3). |
| S10 | Responsibility for live masks; the dated refutation stays, with its reason (memory, not evidence). |

Every other gate in the 157-item recon that this work touches is either removed or carries a one-line reason in the code
for why no model term does its job. Physical model parameters (sensor σ, wall thickness, door leaf thickness) are not
thresholds and stay.

## 6. Code shape

- The pure filter math (Bernoulli update with clutter, responsibilities over a cluster, k-best joint hypotheses within a
  cluster, discrete EIG) lives in ONE self-contained header in door_concept (no door, DSR or Qt dependencies), written so
  it can later move to `common/` and be checked against the objects session's version. Door knowledge (wall priors, leaf
  model, LiDAR opening) stays in door files.
- `common/instance_tracker` is not edited.
- One door_concept config flag selects the new model (off = today's behaviour) while the A/B series runs. After the owner
  accepts the series, the flag and the old path are removed: the end state has no flag.
- Rates before features: door_concept stays at its nominal rate; the k-best set and the EIG are sized so they do.

## 7. Success **(PI, "that's the bar")**

1. The live cases pass: chat_e2e18 (open leaf seen side-on: the restored door is confirmed, no re-birth on the leaf
   plane), chat_e2e19 (restored door's existence does not swing; the crossing is not withdrawn), dm12 (no mirror
   duplicate of a crossed door).
2. The series does not drop below today's results **(PI, pass rule approved 2026-11-14)**: the same 15-run three-room
   series (reference 13/15, c63d56d9) passes at least 13/15; the from-scratch tours find every true door; phantom births
   are not higher than with the flag off; agent rates stay nominal.
3. Every one of the ten switches is removed, or kept with a written reason why no model term can do its job.

Tests (door_concept/tests; nearest existing: test_door_core_tracker.cpp, test_door_core_existence.cpp,
test_door_ricoh_windows.cpp, test_room_door_count.cpp), at least:
- Flag off: births identical to today.
- A parked phantom: a long stare cannot manufacture evidence; absence keeps it out. A miss with p_D = 0 changes nothing.
- Good views need fewer observations than poor ones.
- 0.80 m and 1.00 m doors both born; a 3 m candidate needs more evidence; a truncated view pays no size cost.
- L at birth equals the accumulated evidence, not 1.0; a restored door keeps its prior.
- A candidate far off any wall is not born; a real door 0.2 m off the estimated wall still is; a door seen in a
  provisional room before its wall is estimated is not killed.
- A second door on a wall needs more evidence, yet strong evidence still bears it; removing one door frees the other of
  the cost; an ambiguous wall does not pay in full; the result does not depend on birth order.
- A side-on leaf mask with an opening at its hinge end bears a door in the frame wall, not in the leaf's plane.
- A side-on leaf mask that an existing open door's leaf explains (either swing side) raises that door's existence and
  bears nothing.
- The corner case: open door A's leaf hiding door B in the adjacent wall; with A's opening seen, A wins; with neither seen,
  the cluster stays ambiguous and a look with positive EIG exists from a pose that sees one opening.
- Two labels at one hole: the weaker falls below the line by comparison (dm12's geometry).
- EIG: a pose that sees nothing relevant scores ≈ 0; a pose that sees the discriminating opening scores ≈ H(w).
- Replays of the recorded inputs of chat_e2e18, chat_e2e19 and dm12, once recorded (below).

**Data (checked 2026-11-14).** The three runs (room_concept/estimation_scripts/crossing/{chat_e2e18,chat_e2e19,dm12}) have
logs (door_existence_log.csv, door_filter.csv, door_events.csv, crossing.log, gt_pose.tsv; dm12 also door_decisions.bin)
but NO DoorCore input tape: `RecordInputs = false` in door_concept/etc/config.toml. Designer's calls: §3.7's diagnosis
starts from door_existence_log.csv; then the three cases are re-run live once with `RecordInputs = true` (CLAIM / FREE on
Webots), on today's code, to produce the replay tapes the tests and the fitting use. Ground-truth labels for fitting come
from the Webots world's true door poses. The detector table is fitted on Webots now and refitted on the real robot when it
is available; the typical-size prior stays (a 1.0 m Webots door sits 1.25 σ from 0.85 m and must be born — a test above).

## 8. Goals, non-goals, anti-goals

Goals: §1 and §7.
Non-goals: merging with the objects session's chair filter (later, **(PI)**); editing `common/instance_tracker`, cortex,
the controller's crossing, or room_concept's doorway necks; door actuation; the retina startup race; the alcove phantom.
Anti-goals: a new threshold, gate or clamp where a model term exists; a live case passed by special-casing it; rates below
nominal; a door created in the wall parallel to a leaf.

## 9. Constraints

Type-attributed DSR attributes only (never `runtime_checked_*`); new attributes go through cortex's REGISTER_TYPE and the
owner reinstalls cortex. Queued DSR signals only. `std::from_chars` to read files, `imbue(std::locale::classic())` to
write them. `and/or/not`. C++23. Builds `make -C build -j8` after `pgrep -a cc1plus`. SIGTERM only. Webots is shared:
CLAIM / FREE with the chat and objects sessions before live runs. Work in a worktree on a feature branch; commit locally,
push and merge only when the owner asks. Any paper (AuRo) run must pin the pre-11-14 libdsr_api.so or be reported as a
post-freeze build.

## 10. Left to the builder

- The exact form of g(m | aperture) and g(m | leaf) in terms of the existing fit, and of κ.
- The size of the k-best set per cluster and of the EIG's candidate-pose set, within the rate budget.
- How the LiDAR opening channel is factored out of `door_opening_edges.h`.
- Which recorded runs fit the detector table and the geometry likelihoods, and the fitting script.
- The order of the work (a suggestion: diagnose §3.7 and fit the detector table first; then §3.2-§3.5 behind the flag;
  then §3.6; then §4), and how it splits into tasks.

## 11. Owner rulings in this document

All questions are answered. The designer's calls the owner should check while reading: keep today's decision loss (§3.5);
previous-cycle r for the per-wall cost (§3.4); four-branch hinge × swing mixture over the existing fit (§3.2); tracker
separation only under the flag (§3.5); no new cortex attribute (§4); re-run the three cases live with RecordInputs on
before building, fit the detector table on Webots now and refit on the real robot later (§7, Data).

## 12. Errata found while planning (2026-11-14, designer's rulings against the code at 87b6018a)

- §3.5 "starts from the evidence that bore it": already true. `Existence.BirthLogodds` is retired and refused at load
  (door_config_retired.h); a newborn copies the candidate's belief (door_core.cpp ~844-850). Kept as a test only.
- §3.5 "no hysteresis band": `decide()` (core/door_decision.h) has two lines, birth at L > 0 and removal at
  L < −ln(C_delete/C_keep) ≈ −1.08. Both follow from the stated loss, so they are not hand-set thresholds; they stay. What
  this design forbids is any further hand-set band.
- §3.2 "four-branch mixture": it already exists as `LeafChannel` (door_leaf_channel.h, 4 `PhiHmm` filters, one per
  hinge × swing). The work reuses it; the gap is that `clutter_fraction_leaf_marginal` and the birth/existence paths
  ignore the swing branches.
- §3.4 "walls from wall_hypotheses": walls come from `wall_groups(polygon, σ)` (core/place_prior.h), the one place that
  joins collinear polygon edges into walls. `wall_hypotheses` (ricoh_door_birth.h) is a per-ray lookup.
- §3.4 "the second door-count prior": specificworker.cpp ~4517 calls room_door_count.h with a literal geometric prior
  (mean 2, cap 8). It is replaced by `count_prior_from_rooms` fed from memory (same reader as the per-wall rates).
- §3.4 typical-size prior: `Door.Prior*` also drives the FIT (door_fit.cpp and others) and the crossing tests. The typical
  prior gets its own keys (`Existence.TypicalWidthM/Std`, `Existence.TypicalHeightM/Std`) used only by the birth and
  existence path, as the spec says. The fit's prior is unchanged in this work.
- §3.5 merge: `MergeOverlap` is door_concept's own merge (door_core.cpp ~341), not the tracker's; the separation is
  `Tracker.BirthMinSepM`, set in DoorCore::associate. Both change only under the flag; common/instance_tracker untouched.
