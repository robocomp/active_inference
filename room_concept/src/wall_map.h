/*
 *  wall_map.h — the room as a CLOSED POLYGON MODEL: a cyclic order of wall landmarks, refined by
 *  evidence and changed only by splice JUMPS (model-first redesign, 2026-08-31).
 *
 *  WHAT THE MODEL IS
 *  -----------------
 *  Not a bag of walls. The state is an ordered CCW cycle of wall landmarks — Hesse lines
 *  n(φ)·p = d, the 2-D analogue of S-Graphs+ plane landmarks (Bavle et al., arXiv 2212.11770) —
 *  whose consecutive intersections ARE the polygon. Closure, contiguity and simplicity are
 *  properties of the state space, not tests applied afterwards. The first version kept a free wall
 *  set and derived the polygon from adjacency votes; live it grew hundreds of junk landmarks,
 *  because a free-floating wall costs nothing and the structural constraints never pushed back on
 *  the evidence. Here there is nowhere to put junk: every wall is an edge of the room.
 *
 *  INITIALISATION IS A PRIOR ON SHAPE AND SIZE
 *  -------------------------------------------
 *  The first scan's oriented bounding box seeds a RECTANGLE (initialize_rect): four walls, cyclic
 *  order, θ₀ from the box. rect_prior_sigma_* is that prior's strength — strong enough to anchor
 *  the yaw from frame 1 (with nothing absolute, odometry yaw drift re-bred rotated copies of every
 *  wall: the hairball), weak enough that the data moves every side.
 *
 *  EVIDENCE, LIKELIHOOD, PRIOR — as in every concept agent
 *  ------------------------------------------------------
 *  Segments (wall_segmenter) associate to the model's edges by a Mahalanobis test on (φ, d) that
 *  includes segment noise, edge uncertainty, pose uncertainty and a model-error term (map_sigma —
 *  the corner detector's lesson: without it a converged edge refuses its own re-observations and
 *  twins are born). Associated points become the solver's wall factors; the hierarchical Manhattan
 *  prior (θ₀ + per-edge class) constrains angles softly — a chamfer keeps k = −1 and no factor.
 *
 *  STRUCTURE CHANGES ARE JUMPS, NOT BIRTHS
 *  ---------------------------------------
 *  Unexplained segments accumulate as CANDIDATES; a candidate that earns a decisive ΔF (clutter
 *  comparison + Occam + class cost, > birth_nats, seen from ≥2 poses) proposes a SPLICE:
 *    - notch/extension on a host edge E: [E, jog, C, jog, E] (E appears twice; the room grows an
 *      alcove, gains the second SPACE seen through a wide opening, or loses a pillar-sized bite);
 *    - corner cut with a neighbour: [E, jog, C, N] or [P, C, jog, E] — how an OBB rectangle becomes
 *      the L-shaped truth — and obliquely without the jog: [E, C, N] (a chamfer, k = −1).
 *  A variant commits only if the resulting cycle builds a simple CCW polygon. No valid splice ⇒ no
 *  change, counted in splice_rejected — a refusal must never be mistaken for "never asked".
 *
 *  THE STEP-BACK OPERATOR (existence, per extent bin)
 *  --------------------------------------------------
 *  Every beam crossing an edge's extent testifies about the BIN it crossed: ON supports, BEYOND
 *  refutes (×P(detect)), SHORT is occlusion and holds. Dead end bins shrink the extent; an edge
 *  whose testable span is gone dies at −birth_nats — symmetric with birth, never a timer — and is
 *  spliced OUT, the cycle healing by collapsing parallel neighbours. (Whole-wall odds once killed a
 *  real wall that had merely lost a stretch: the residual layer's per-bin lesson, one level up.)
 *
 *  `information` is the carried (φ, d) precision the solver applies as a prior each solve. It grows
 *  only when a window slot is DROPPED (gn::absorb_wall_observations) — the live window's factors
 *  already enter the solve, so carrying them too would count them twice.
 */
#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <array>

#include "wall_segmenter.h"

namespace rc::wallmap
{
    struct Params
    {
        float manhattan_sigma_rad = 2.0f * static_cast<float>(M_PI) / 180.f;  // σ_ε of the room↔wall factor
        float manhattan_off_prior = 0.1f;   // prior mass of "this wall obeys no class" (a chamfer)
        // ── MODEL error of an edge estimate between VIEWS (multi-ring spread, band mixing heights,
        // residual calibration). Without it the innovation holds only white noise, and once an edge's
        // carried precision reaches millimetres every re-observation fails the gate (measured live:
        // 490 walls from six, all twins). The corner detector's map_sigma lesson. Enters association,
        // candidate matching and the merge — never the solver factors.
        float map_sigma_d       = 0.04f;    // m
        float map_sigma_phi_rad = 1.0f * static_cast<float>(M_PI) / 180.f;
        float assoc_chi2  = 5.991f;         // χ²₂ @95% — segment↔edge and segment↔candidate gate ⚠
        float merge_chi2  = 5.991f;         // χ²₂ @95% — two edges statistically one ⚠
        float obs_sigma   = 0.05f;          // m — σ_obs of the wall point factor (RoomConcept.ObsSigma)
        float huber_delta = 0.15f;          // m — its Huber knee (RoomConcept.HuberDelta)
        float sensor_range = 15.f;          // m — extent of the uniform prior on d (Occam term)
        float birth_nats  = 4.605f;         // ln 100 — decisive Bayes factor ⚠ decision constant
        int   birth_min_frames = 2;         // a jump needs a second view (tracker-only birth)
        int   max_candidates = 32;
        float publish_corner_sigma = 0.06f; // m — every derived corner must be this sharp to publish
        // ── Model-first initialisation: the OBB rectangle prior on shape and size ────────────────
        // HONEST about what an OBB knows: on a non-convex cloud (an L) PCA tilts the box by 15-20°,
        // and a 5° prior then REFUSED the true walls' segments at the gate — the sides never rotated
        // onto the truth and the real walls were spliced in as extra edges instead (harness-caught).
        float rect_prior_sigma_d       = 0.50f;  // m — per-side offset prior strength
        float rect_prior_sigma_phi_rad = 15.0f * static_cast<float>(M_PI) / 180.f;
        // A candidate span end within this of a host edge's end "reaches the corner" — a corner-cut
        // splice instead of a notch. ⚠ a tolerance, tied to extent noise at range.
        float splice_end_tol = 0.5f;        // m
        // Stub jumps: a free-standing interior wall (spur) the boundary wraps around — spliced as
        // the THREE-wall sequence [E, near-face, tip, far-face, E]. Selection against the notch
        // family is no longer a fragile tie-break: the two classes differ over exactly ONE
        // observable region (the strip BEHIND the candidate face), so the global ΔF comparison
        // reduces to free-grid evidence there — free cells CONNECTED to the robot's own free
        // component assert the wrap, occupied cells refute it, and stale disconnected free space
        // (a sealed-off room change: the grid never forgets) is silent. See try_splice.
        bool  enable_stub_jumps = true;
        float stub_thickness = 0.12f;       // m — thin-wall prior thickness of a stub; evidence refines it
        // ⚠ decision constant: net connected-free fraction behind the face above which the room
        // provably wraps (only stub variants offered) and below which it provably does not (only
        // boundary variants offered).
        float stub_free_behind_min = 0.25f;
        // ── ORDER-QUANTIZED GROWTH (user design 2026-09-01), PRICED BY CODE LENGTH (2026-09-02) ──
        // The map starts as the minimal Manhattan polygon (the rectangle) and grows in EVEN order
        // jumps (+2 corner-cut/direct, +4 notch/spur-wrap). Each jump pays the DESCRIPTION LENGTH
        // of the lines it names (Rissanen 1978, two-part code; the model-complexity term of
        // Floor-SP): a Manhattan edge costs ln(4) for its class plus ln(2·sensor_range / cell) for
        // its offset at the resolution the data are encoded at (the grid cell) — 7.3 nats at 15 m
        // and 8 cm, i.e. 14.6 per +2 edges where the bench had settled on a hand-tuned 15. An
        // off-class edge (an odd, chamfer jump) also names its angle at the same resolution,
        // ln(2π·sensor_range / cell), instead of ln 4. The jump is accepted only when the evidence
        // it NEWLY EXPLAINS over the region where the two polygons disagree (jump_delta_nats)
        // exceeds that length. See WallMap::edge_code_nats(). What the code length cannot derive
        // is how many NEW lines a move names: a replacement swaps one line for another (1, the
        // noise floor the free version lacked: measured 0.865 → 0.460 with 0); a spur wrap names
        // its tip cap and its mirror (2; the mirror's offset is the wall thickness). ⚠ structural
        // counts, swept on the bench.
        int   replace_code_edges = 1;
        int   wrap_code_edges    = 2;
        // ── LEVEL 2 — the RESIDUAL PASS (user design 2026-09-02). The coarse cycle above estimates
        // the room's main lines and lawfully leaves small features whole: a pillar on a wall, an
        // alcove. On the PUBLISHED COPY only (never in the live map — level 1 stays soft and local),
        // the residual zones — connected matter INSIDE the polygon, connected free space OUTSIDE
        // it, each attached to one edge — propose a rectangular step of that edge, priced by the
        // code length of the edges it adds and paid by the same grid evidence as a level-1 jump.
        // Recomputed at every publish, so a level-1 correction simply moves the zones under it.
        bool  enable_level2 = true;
        // The cluster's cell bounding box only LOCATES the feature; its three degrees of freedom —
        // the two ends along the wall and the depth — are then fitted to the returns themselves.
        // For an axis-aligned step the three decouple into three one-dimensional robust fits (the
        // median of the returns owned by each of the three faces), so no joint optimisation is
        // needed. Without this the rectangle is quantised to the 8 cm cell and padded half a cell
        // beyond it: measured, a 0.60 x 0.38 m pillar came out 0.88 x 0.53.
        bool  level2_fit = true;
        // The smallest feature level 2 will keep, in METRES rather than cells. It used to be two
        // cells (16 cm) applied AFTER the fit, which threw away every spur the fit had just measured
        // correctly at 10-16 cm — the population test found 2 spurs in 100. The floor is physical:
        // nothing thinner than the thinnest wall the model represents.
        float level2_min_m = 0.07f;
        // How far a residual cell must lie from every published edge before it counts. Measured at
        // 1.0 and 1.5 cells: 1.0 lets more of a shallow feature through but also lets a clean wall's
        // own surface noise form clusters — three synthetic rooms grew spurious steps (notch
        // Hausdorff 0.020 -> 0.141 m). 1.5 stands; the size floor above is what unlocked the small
        // features, not the clearance.
        float level2_clear_cells = 1.5f;
        // ── FORWARD-MODEL REFEREE (Thrun 2003, "Learning occupancy grid maps with forward sensor
        // models"; built 2026-09-03 as a REFEREE, deciding nothing yet). Every stored beam is
        // scored against a polygon by the likelihood of its measured range given the range the
        // polygon predicts along its ray (first edge hit): a mixture of a hit near the prediction,
        // an unexpected short return, and a random return. Every judge site logs, next to its own
        // verdict, the Δ log-likelihood the forward model assigns the same trial; the bench scores
        // both against the truth. ⚠ mixture weights and σ are the model's parameters — to be
        // LEARNED by EM from the robot's own data, as in the paper; hand values until then.
        bool  forward_referee = false;      // store beams (cheap: ~20 B each)
        bool  referee_log     = false;      // ALSO score every judged decision (minutes per seed under churn)
        float beam_w_hit   = 0.80f;
        float beam_w_short = 0.10f;         // w_rand = 1 − w_hit − w_short
        float beam_sigma_m = 0.03f;         // hit width — range noise plus the polygon's own error
        std::size_t beam_store_max = 800000;   // ring: ~1100 frames × 720 rays on the bench
        // Standing structure pays RENT: a down-jump is accepted when the evidence AGAINST removal
        // (grid delta plus the removed walls' net existence-bin nats) is smaller than the code
        // length it refunds — Occam pushes the polygon down through evidence-neutral structure (a
        // pocket over unobserved space). The two-part code refunds the FULL length (1.0); a
        // fraction below 1 is a switching cost — the [keep·cost, cost] hysteresis band that once
        // stopped commit/death flapping (~1000 cycles per run in the pre-existence-bin era). It
        // measured INERT on the bench at 0.5 / 0.7 / 0.85 / 1.0 (2026-09-02), so the principled
        // value stands; lower it only against measured flapping.
        float order_keep_fraction = 1.0f;
        // SURRENDER RULE INSIDE ADOPTION (user design 2026-09-01): a wall backed by OBSERVED
        // existence support (bins above the birth seed) may not be dropped by a contour adoption
        // for free. This is a VETO, not a price — a global trade cannot protect a small structure
        // inside a big-gain adoption (measured: a 26-bin wrapped face ≈ 120 nats against
        // thousands of area nats). ⚠ threshold ≈ two fully-observed bins.
        float adopt_surrender_nats = 9.f;
        // A splice pays the code length of the lines it names. It does NOT yet pay for the
        // observed existence support it ERASES when its new cycle drops a wall the old one held —
        // the term the re-derivation judge has carried since 2026-09-03. Measured consequence:
        // on the apartamento a +2 splice bought 217 free cells for 866 nats and swallowed 61
        // matter cells of the flat's long partition for 237, and the partition was three metres
        // shorter for ever after. See WallMap::try_splice.
        // MEASURED HARMFUL, default OFF: charging a splice the full existence evidence of a wall
        // its new cycle drops makes legitimate REPLACEMENTS unaffordable — a replacement erases the
        // wall it replaces by construction, and the line-proximity test that lets the re-derivation
        // judge recognise "the new cycle still holds this line" is too tight for a splice that
        // rebuilds the line a little further out. Paired on 7 random rooms: IoU 0.957 -> 0.819 with
        // polygons that no longer close at all. Kept because the ASYMMETRY is real and worth
        // revisiting with a looser held-test; the term itself, as written, is not affordable.
        bool  splice_surrender = false;
        // ADOPTION JUDGE (measured 2026-09-03, IoU seed 7/1001/424242): 0 = grid-IoU margin 0.02 +
        // health waiver + surrender veto (the incumbent, 0.951/0.969/0.943); 1 = the one energy
        // (grid + support in − surrender − code length, 0.926/0.969/0.918). adopt_repair splices
        // self-crossings out of the re-derived cycle before judging it: 0.936/0.928/0.582 under the
        // incumbent, 0.940/0.793/0.919 under the energy. Both agree BETTER with the single-step
        // truth (does this step raise IoU?) and produce WORSE maps: a repaired or +0.01 cycle that
        // creates thinly supported walls is adopted, they die, the cycle flaps. The margin and the
        // veto are hysteresis; a self-crossing is a symptom of a wrong contour, not a defect to
        // patch. Kept switchable for the bench (WS_ADOPT_JUDGE, WS_ADOPT_REPAIR).
        // 2 = adopt ANY closed cycle (bench only: what the free-space contour itself contains,
        // with no judge in the way — the batch-at-saturation experiment).
        //
        // RE-MEASURED 2026-09-04 on the full 50 random rooms, paired, and the 3-seed reading above
        // was hiding the shape of it. Mean IoU 0.9219 -> 0.9279, MEDIAN 0.9565 -> 0.9510, 20 rooms
        // better against 21 worse — a wash, until you split by how healthy the incumbent was:
        //     baseline below 0.90 (9 rooms):  0.756 -> 0.841   (+0.085)
        //     baseline 0.95 and up (31):      0.967 -> 0.948   (−0.020)
        // The energy judge RESCUES THE TAIL and TAXES THE HEALTHY. Room 32 — which finished 36.5°
        // out with 99% of itself observed — goes 0.517 -> 0.956, Hausdorff 3.607 -> 0.379 m; so
        // that whole-room tilt was this judge refusing the correction, not the direction estimator
        // (see Params::theta0_posterior). Rooms 38, 45 and 22 follow it. Against that, room 28 goes
        // 0.879 -> 0.561, and the apartamento — healthy on all three seeds — goes 0.908/0.975/0.945
        // to 0.817/0.975/0.899 with deaths exploding 17 -> 119 on seed 7.
        //
        // WHY it is wrong on a healthy map: the grid is not the truth. Grazing beams carve real
        // surfaces fresh-free, so a cycle that explains the GRID better can be worse than the room,
        // and the 0.02 IoU margin is hysteresis against exactly that. The energy judge has no such
        // floor and keeps swapping marginally-better cycles. Neither is right: the margin should be
        // in NATS, scaled to what the accumulated per-cell log-odds error can explain, rather than
        // a fixed slice of an IoU that cannot see a 0.13 m partition. That variance is not tracked
        // today; it is the next piece of work here.
        // 3 = the energy, with a margin sized to the GRID'S OWN ERROR: adopt iff dE exceeds
        // adopt_sigma_k standard deviations of dE. Judge 0's margin is a fixed slice of an overlap
        // that cannot see a thin wall; judge 1 has no margin at all over a grid that lies. This one
        // asks the only question that survives both objections — is the improvement larger than the
        // grid's own uncertainty can explain?
        int  adopt_judge  = 0;
        // Confidence level of that margin, in standard deviations. Not a magic cutoff: the same
        // convention as the segmenter's chi2 levels, and the quantity it multiplies is measured
        // from the grid rather than chosen.
        //
        // MEASURED 2026-09-04 on the 50 rooms, and judge 3 LOSES to both of the others: mean IoU
        // 0.9154 against 0.9219 (judge 0) and 0.9279 (judge 1), 15 rooms better and 28 worse, floor
        // 0.349. It gets the worst of both — it does not protect a healthy map as well as judge 0
        // (0.953 vs 0.967 over 31 rooms) and does not rescue a sick one as well as judge 1 (0.761
        // vs 0.841 over 9).
        //
        // WHY, and this is the useful part. Room 32 finishes 0.956 under judge 1 and 0.349 under
        // judge 3, and the two runs are identical for thirteen decisions. The fourteenth is
        //     dE = 2.5 nats, sigma = 33.4, IoU 0.467 -> 0.745, ADOPT (judge 1)
        // — a step worth two and a half nats against a thirty-three nat standard deviation, i.e.
        // statistically indistinguishable from noise, which raises the overlap by 0.28 and unlocks
        // the next one (dE = 22719, IoU 0.035 -> 0.860). Judge 3 refuses it, precisely because the
        // margin is doing its job, and the room never escapes.
        //
        // So the margin is right about the statistics and wrong about the policy, for a reason that
        // is now nameable: ESCAPING A WRONG TOPOLOGY TAKES A SEQUENCE OF STEPS, SOME OF WHICH ARE
        // INDIVIDUALLY WORTHLESS. A per-step significance test forbids exactly that, and no
        // single-step judge — however well calibrated — can fix room 32; judge 1 only manages it by
        // being permissive enough to random-walk there. (The same run also shows the grid energy
        // scoring the RIGHT cycle at −777 nats while it raised IoU 0.458 -> 0.748, so the energy is
        // not a reliable single-step guide here either.) What this points at is a judge over
        // SEQUENCES rather than steps — a trial adoption kept only if it survives — which is the
        // reversible-jump proposal from the literature review, or the batch pass at saturation.
        float adopt_sigma_k = 2.f;
        // THE SEQUENCE JUDGE. A cycle the incumbent judge refuses, but which is closed and explains
        // more grid than it costs, is adopted ON TRIAL: the refused-from state is snapshotted, the
        // challenger runs the map for trial_frames, and it is kept only if it still explains the
        // grid better THEN — on evidence that did not exist when it was proposed. Otherwise the
        // snapshot is restored. This is the only form of judge that can pass a step which is
        // individually worthless: room 32 escapes through one worth 2.5 nats against a 33-nat
        // standard deviation, which no per-step test can distinguish from noise.
        bool  trial_adoption = true;
        // How long the challenger has to prove itself. Its scale is set by how long it takes the
        // robot to re-observe the region the two cycles disagree about, which at these speeds and
        // this sensor range is tens of frames, not hundreds.
        int   trial_frames = 40;
        // The health waiver (see the re-derivation judge) lets a challenger past the IoU margin
        // when the incumbent's own corners are too blunt to publish. Priced: the corner sharpness
        // it buys must beat the code length of the edges it adds, both in nats. Unpriced, ANY
        // improvement in the last decimal licensed the adoption and the map churned.
        bool adopt_waiver_priced = true;
        bool adopt_repair = false;
        // ── STRICT MANHATTAN (user directive 2026-09-01): this stage estimates the room's MAIN
        // LINES only. Every polygon wall carries a Manhattan class; an off-axis candidate (chamfer,
        // oblique clutter line) may never splice into the cycle — it stays in the candidate bank as
        // the input of a later chamfer-refinement postprocessing stage. Small details are already
        // priced out by the order-jump economy. ⚠ gate: how far off the nearest Manhattan direction
        // a candidate may lie and still count as structural (measured oblique junk sits 17° off).
        bool  manhattan_strict = true;
        float manhattan_gate_rad = 10.f * static_cast<float>(M_PI) / 180.f;
        float manhattan_gain = 1.f;         // bench-only scale on the in-loop Manhattan factor (review #4 test)
        // theta0 from a global mixture search over every observed direction, instead of the first
        // scan's bounding box refined by a mean inside the current class assignment. See
        // WallMap::update_theta0 for the model. OFF by default and this is the measured reason:
        // it does estimate the direction better and sooner — five frames instead of six, and it
        // cannot be trapped half a class out — but arriving sooner lets STRUCTURE changes in on
        // frame 1, before there is enough evidence to judge them, and the structure machinery is
        // what actually sets the score. Paired on the random rooms it lost 0.034 IoU, 7 rooms of
        // the 8 compared. The tilted bounding box was accidentally acting as a delay. Keep the
        // estimator; the thing to fix first is upstream of it.
        bool  theta0_posterior = false;
        bool  debug_splice = false;         // diagnostic prints from try_splice (bench use)
        // ── GLOBAL re-derivation cadence (re_derive): the escape hatch from a wrong local topology
        // runs on a slow clock, or sooner when local jumps are visibly stuck (rejections pile up).
        // Shared by the bench and the live agent so the bench grades what the agent executes.
        // ⚠ cadence constants — chosen on the bench, not derived.
        bool  enable_rederive          = true;
        int   rederive_every_frames    = 40;
        int   rederive_after_rejections = 30;
        // ── Existence (the step-back operator), per extent bin — see the header comment ──────────
        float exist_refute_pdet = 0.5f;     // P(detect): weight of a pass-through vs a support ⚠
        float exist_bin_m       = 0.25f;    // m — extent bin width (spatial resolution of refutation)
        // ── ONE CURRENCY FOR LINE CLAIMS (review #10, 2026-09-02). Every claim about MATTER ON A
        // LINE — a stub face entering the cycle, a spur wrap's overshoot, the support a down-jump
        // or a contour adoption surrenders — is priced in the SAME nats: per extent bin, at most one
        // nat per frame (one frame = one observation), net of the birth seed, saturating at
        // birth_nats of evidence. Candidates accrue it in Candidate::bins; a wall born from a
        // candidate inherits it. The per-point gain (56k nats for one 12k-point candidate) decides
        // BIRTH only and buys nothing. AREA claims — every boundary splice — keep paying in the
        // grid's cell log-odds, which already reward an edge placed on matter (matter released to
        // the exterior). MEASURED (12 variants, 3 seeds): folding the two into one scalar loses
        // 0.02-0.15 IoU every way it was tried — adding the line term to boundary splices double
        // counts the grid's matter reward, excluding on-line matter from the grid removes it, and a
        // per-frame grid rule starves adoption — so the split is by CLAIM TYPE, not by operator.
        // ⚠ The 15-nat toll stays a bench-calibrated constant (≈ 3 confirmed bins of line per +2
        // edges); an MDL description-length cost is the principled replacement (review #10).
    };

    struct WallLandmark
    {
        std::uint64_t id = 0;
        int   k = -1;                       // Manhattan class, −1 ⇒ no room↔wall factor
        float phi = 0.f, d = 0.f;           // map frame; n(φ) points INTO the room
        Eigen::Matrix2f information = Eigen::Matrix2f::Zero();   // carried (φ, d) precision
        // THE PRIOR (review #5, 2026-09-03), in information form (Λ, μ): what the DROPPED slots
        // said the wall is. Zero at birth — the birth-time `information` above is the candidate's
        // line-fit precision from points still in the live window, and a prior built from it
        // anchors the wall at its birth estimate (measured: joint-solve loss 0.03 → 39, IoU
        // 0.969 → 0.889). `information` keeps its gating and corner-sigma roles; only prior_info
        // enters the solver, and only from gn::absorb_wall_observations. Until now the prior was
        // re-centred on the wall's current estimate every solve — damping, not a prior. Fused at
        // merges and transformed at re-anchor like the wall itself.
        Eigen::Matrix2f prior_info = Eigen::Matrix2f::Zero();
        Eigen::Vector2f prior_mu = Eigen::Vector2f::Zero();
        float manhattan_var = 0.f;          // σ_ε² in force for this wall's room factor (0 ⇒ off)
        float room_factor_dF = 0.f;         // converged cost of that factor, nats (diagnostic)
        bool  has_extent = false;
        float s_min = 0.f, s_max = 0.f;     // tangent coordinates of the observed extent
        int   frames_seen = 0;
        int   points_seen = 0;
        std::int64_t last_seen_ms = 0;      // last frame a segment associated to this edge
        std::int64_t born_ms = 0;           // creation time — the freshness reference for evidence about this edge
        // Existence log-odds (nats), PER extent bin of width Params::exist_bin_m; bins_s0 is bin 0's
        // lower edge. exist_lodds is the summary (max over bins). Seeded at birth with birth_nats; a
        // bin dies below −birth_nats; dead END bins shrink the extent; no testable span ⇒ death.
        float exist_lodds = 0.f;
        std::vector<float> exist_bins;
        float bins_s0 = 0.f;

        Eigen::Vector2f normal() const { return linefit::normal_of(phi); }
        Eigen::Vector2f tangent() const { return linefit::tangent_of(phi); }
        linefit::Line2D line() const { linefit::Line2D l; l.normal = normal(); l.d = d; return l; }
    };

    /// A line no edge explains, accumulated across frames until the model comparison proposes a jump.
    struct Candidate
    {
        float phi = 0.f, d = 0.f;
        Eigen::Matrix2f information = Eigen::Matrix2f::Zero();
        int   frames = 0;
        int   npts = 0;
        float gain = 0.f;                   // Σ point_gain_nats of its points about the fused line (nats) — BIRTH only
        float s_min = 0.f, s_max = 0.f;
        // Line support in the one currency: per extent bin (width Params::exist_bin_m from bins_s0),
        // at most one nat per frame, saturating at birth_nats. What a splice may SPEND; a wall born
        // from this candidate inherits it on top of its seed.
        std::vector<float> bins;
        float bins_s0 = 0.f;
        float evidence() const { float e = 0.f; for (float b : bins) e += b; return e; }
        int   this_frame_seg = -1;          // segment index that updated it THIS frame (−1 none)
        std::int64_t first_ms = 0, last_ms = 0;
    };

    /// One slot's observation of one edge, in the ROBOT frame of that slot. Consumed by
    /// gn::WallPointFactor. Points are copied (not indexed) because old slots are subsampled.
    struct WallAssoc
    {
        std::uint64_t wall_id = 0;
        Eigen::Matrix<float, Eigen::Dynamic, 2> pts;
        Eigen::VectorXf weights;            // range/incidence weights, mean 1 over the slot
        float pda = 1.f;                    // association posterior
    };

    struct Corner
    {
        std::uint64_t wall_a = 0, wall_b = 0;   // consecutive edges of the order
        Eigen::Vector2f p = Eigen::Vector2f::Zero();
        bool  inferred = false;             // kept for the viewer contract
        float sigma = 0.f;                  // largest σ of the corner position (m), from the edges' information
    };

    struct Polygon
    {
        std::vector<Eigen::Vector2f> verts;      // CCW; edge i runs from verts[i] to verts[i+1]
        std::vector<std::uint64_t>   wall_of_edge;
        std::vector<Corner>          corners;    // corners[i] is verts[i]
        bool closed = false;
        bool publishable = false;
        std::vector<int> crossing_edges;         // edge indices involved in self-crossings (repair input)
        float worst_corner_sigma = 0.f;
        std::string status;                      // human-readable: why not closed/publishable
    };

    /// Why a jump was just committed — the line to read when the map misbehaves.
    struct BirthInfo
    {
        std::uint64_t id = 0;               // the new C edge
        float phi = 0.f, d = 0.f;
        int   npts = 0, frames = 0;
        float dF = 0.f;
        std::uint64_t nearest_wall = 0;     // host edge of the splice
        float nearest_chi2 = -1.f;          // (CSV contract; −1 when not computed)
        int   seg = -1;                     // segment index that completed the jump (z attribution)
    };

    struct FrameResult
    {
        std::vector<int>   seg_to_wall;     // per segment: index into walls, or −1
        std::vector<float> seg_pda;
        std::vector<WallAssoc> assoc;       // for the newest slot (robot frame)
        int births = 0;                     // committed splices
        std::vector<BirthInfo> births_info;
        int splice_rejected = 0;            // qualified candidates no valid splice could place
        struct DeathInfo { std::uint64_t id = 0; float lodds = 0.f; int frames_seen = 0; int points_seen = 0; };
        int deaths = 0;
        std::vector<DeathInfo> deaths_info;
        int twins_fused = 0;                // candidates that turned out to BE an existing edge
        int segments_associated = 0;
        int merged = 0;
        int candidates = 0;
        int unexplained_points = 0;
        std::vector<int> seg_to_candidate; // per segment: candidate index or −1 (display)
    };

    class WallMap
    {
    public:
        /// Projection EM at the map layer: re-estimate theta0 (weighted mean class-corrected
        /// orientation — the shared-angle M-step) and project every classified wall onto its class
        /// axis about its own extent centre. Call after each solve; the published polygon is then
        /// exactly Manhattan by construction while the solver stays soft.
        void enforce_manhattan();
        /// The OUTPUT-stage projection: the same operation on a COPY, returning its polygon.
        /// The model and solver stay soft (six mutating variants measurably degraded pose and
        /// structure); only what is PUBLISHED is exactly Manhattan.
        Polygon manhattan_polygon() const;
        /// One stored range measurement in the map frame (forward-model referee).
        struct Beam { Eigen::Vector2f o, d; float r; };
        std::vector<Beam> beams;
        std::size_t beam_next_ = 0;
        int frames_observed_ = 0;
        /// A judged structure change, both polygons in the map frame, with the incumbent judge's
        /// evidence and cost, and the forward model's Δ log-likelihood and the code-length cost it
        /// would apply. The bench compares both verdicts with the truth.
        struct Decision
        {
            const char* site = "";
            int frame = 0;
            bool accepted = false;
            float evidence = 0.f, cost = 0.f;   // the incumbent judge: accepted ⇔ evidence > cost
            float fwd_dll = 0.f, fwd_cost = 0.f; // the forward referee's verdict ⇔ fwd_dll > fwd_cost
            std::vector<Eigen::Vector2f> cur, trial;
        };
        mutable std::vector<Decision> decisions;
        /// log p(z | polygon) of one beam: ray-cast to the first edge, then the beam mixture.
        float beam_loglik(const Beam& b, const std::vector<Eigen::Vector2f>& poly) const;
        /// Σ over beams crossing the disagreement box of [log p(z | trial) − log p(z | cur)].
        float forward_delta(const std::vector<Eigen::Vector2f>& cur, const std::vector<Eigen::Vector2f>& trial) const;
        /// Log one decision with both judges' verdicts (no-op unless Params::forward_referee).
        void referee(const char* site, bool accepted, float evidence, float cost, float fwd_cost,
                     const std::vector<Eigen::Vector2f>& cur, const std::vector<Eigen::Vector2f>& trial) const;
        /// LEVEL 2 (Params doc): decorate a published polygon with the rectangular steps that its
        /// residual zones pay for. Pure: reads the grid, returns a new polygon.
        Polygon decorate(const Polygon& pub) const;
        Params params;

        /// An outline adopted on trial, with the state to put back if it does not survive. The
        /// GRID is deliberately not part of it: sensor evidence is never rolled back, and the whole
        /// point is that the verdict is passed on a later, better grid than the proposal was.
        struct Trial
        {
            bool open = false;
            int  frames_left = 0;
            int  opened_at = 0;
            float dE_at_open = 0.f;
            std::vector<WallLandmark>  walls;        // the incumbent, as it was
            std::vector<std::uint64_t> order;
            std::vector<Candidate>     candidates;
        };
        Trial trial_;
        /// Decide an open trial: keep the challenger iff it still explains the CURRENT grid better
        /// than the snapshot would, else restore the snapshot. Called once the window runs out.
        void resolve_trial();

        bool  theta0_born = false;
        float theta0 = 0.f;
        float theta0_information = 0.f;
        /// One frame's PRECISION SNAPSHOT — how firm the room's estimate has become, in the units
        /// each quantity is actually held in. The freezing is not one number: the direction, the
        /// individual walls, the corners they imply and the existence of each wall all settle on
        /// their own schedules, and a failure usually shows as one of them freezing early.
        struct Precisions
        {
            int   frame = 0;
            float theta0_deg = 0.f;        // the room's reference direction
            float theta0_disp_deg = 0.f;   // spread of the observed directions about it
            float theta0_gate_deg = 0.f;   // half-width the Manhattan gate is running at
            int   n_walls = 0, n_cand = 0, n_order = 0;
            float sigma_phi_deg = 0.f;     // median over the cycle's walls
            float sigma_d_m = 0.f;         // median over the cycle's walls
            float corner_sigma_m = 0.f;    // worst corner of the cycle — the publishable test
            float corner_sigma_med_m = 0.f;// median corner: separates one bad corner from all of them
            int   corners_over_bar = 0;    // corners above corner_sigma_max (how many block publication)
            int   n_corners = 0;
            float exist_nats = 0.f;        // median wall existence log-odds
            float class_err_deg = 0.f;     // median |wall − its class|: how Manhattan it really is
        };
        Precisions precisions() const;
        /// Re-derive theta0 from the accumulated direction evidence (see the .cpp for the model).
        void update_theta0();
        // Direction evidence, accumulated from every SEGMENT ever observed, as a histogram over the
        // QUADRUPLED angle (which folds the four Manhattan classes onto one circle). Weighted by
        // each segment's own angular precision. orient_off_ carries re-anchor rotations so the
        // histogram never has to be re-binned. Map frame.
        static constexpr int kOrientBins = 180;          // 2 deg in the quadrupled angle = 0.5 deg in theta0
        std::array<double, kOrientBins> orient_hist_{};
        double orient_off_ = 0.0;
        std::vector<WallLandmark> walls;
        std::vector<Candidate>    candidates;
        /// CORNER RESIDUE: oblique segments the model explains as its own segmenter corner
        /// artifacts. Never associated, never spliceable — but preserved, because a real CHAMFER has
        /// the same signature and accumulates here persistently at its corner. The postprocessing
        /// stage (chamfers, small details) reads this ledger; the main-lines stage never does.
        std::vector<Candidate>    corner_residue;
        // ── FREE-SPACE EVIDENCE: a coarse log-odds grid the scans build directly ─────────────────
        // Every beam traverses free space and ends on matter; that is observed, not inferred. The
        // grid answers two questions the wall set cannot: WHERE the room's free region actually is
        // (the global re-derivation traces its contour), and where the FRONTIERS are (free cells
        // touching unknown — the epistemic explorer's targets, and the honest "am I done" test).
        struct FreeGrid
        {
            // 8 cm: a cell must be smaller than the thinnest wall the model must keep, or grazing
            // traversals through partially-occupied cells outvote endpoint hits and the beams CARVE
            // THROUGH thin interior walls (measured: 38 of 42 spur-face cells marked free at 15 cm).
            float cell = 0.08f;
            float x0 = 0.f, y0 = 0.f;
            int nx = 0, ny = 0;
            std::vector<float> lodds;             // + occupied, − free, 0 unknown; clamped ±4
            // Endpoint returns per cell, counted SEPARATELY from the odds: a return localises matter
            // in the cell near-certainly, while a traversal of a partially occupied cell (a thin wall
            // shares its cell with air) is weak and explicable. A saturating scalar is ORDER-DEPENDENT
            // — a cell driven deep-free by corridor traffic before its first frontal view could never
            // climb back — so returns are kept as their own evidence and three of them assert matter
            // regardless of how many beams grazed past.
            std::vector<unsigned short> hits;
            // Timestamp (ms) of the last FULL-WEIGHT free marking per cell (−1 = never). Freshness
            // is evidence: a beam traversing a cell freely AFTER a wall face came into existence
            // proves the room continues behind that face (the stub discriminator's question); free
            // log-odds alone cannot say WHEN the cell was last actually reachable — the grid never
            // forgets, so a sealed-off room change keeps stale free cells for ever.
            std::vector<std::int64_t> free_ms;
            bool ready() const { return nx > 0; }
            int  idx(int i, int j) const { return j * nx + i; }
            bool in(int i, int j) const { return i >= 0 and i < nx and j >= 0 and j < ny; }
            Eigen::Vector2f at(int i, int j) const { return {x0 + (i + 0.5f) * cell, y0 + (j + 0.5f) * cell}; }
            void init(const Eigen::Vector2f& centre, float half_span);
            void mark(const Eigen::Vector2f& origin, const std::vector<Eigen::Vector2f>& pts_map,
                      std::int64_t ts_ms);
            bool is_occupied(int i, int j) const
            { return in(i, j) and (hits[static_cast<size_t>(idx(i, j))] >= 3 or lodds[static_cast<size_t>(idx(i, j))] > 1.5f); }
            bool is_free(int i, int j) const
            { return in(i, j) and lodds[static_cast<size_t>(idx(i, j))] < -1.f and hits[static_cast<size_t>(idx(i, j))] < 3; }
            bool is_unknown(int i, int j) const
            { return in(i, j) and std::abs(lodds[static_cast<size_t>(idx(i, j))]) < 0.5f and hits[static_cast<size_t>(idx(i, j))] < 3; }
        };
        FreeGrid fgrid;
        /// Free cells adjacent to unknown cells — where the map still has questions (map frame).
        std::vector<Eigen::Vector2f> frontiers() const;
        /// WEAKLY-HELD MATTER: cells with 1–2 endpoint returns (suspected, unconfirmed) or with
        /// returns that grazing passes still contest. Thin interior walls live or die here — they
        /// are confirmed only if something goes and LOOKS at them frontally, so they are epistemic
        /// targets in their own right, not merely map state.
        std::vector<Eigen::Vector2f> weak_matter() const;
        /// Connected free component of the grid containing `seed_map` (dilated 4-conn flood: a free
        /// cell whose 8-neighbourhood holds matter is wall surface, not passage). Empty when the
        /// grid is not ready or no floodable seed exists nearby. Used by re_derive() and by the
        /// stub discriminator in try_splice.
        std::vector<char> free_component(const Eigen::Vector2f& seed_map) const;
        /// Description length of one polygon edge in nats (Params doc, "PRICED BY CODE LENGTH"):
        /// class + offset at the grid's resolution over the sensed span; an off-class edge names
        /// its angle too. The one number every order jump, wrap and down-jump is priced in.
        float edge_code_nats(bool manhattan_class) const;
        /// GLOBAL re-derivation: trace the free region's contour, snap its runs to the evidence
        /// lines (walls ∪ candidates, Manhattan preferred), and ADOPT the resulting cycle iff it
        /// explains the observed free space better than the current one. The escape hatch from a
        /// wrong local topology that greedy jumps cannot leave (measured: a 107k-point wall homeless
        /// through 5140 rejections). Returns true when the cycle was swapped.
        bool re_derive(const Eigen::Vector2f& robot_map);

        // THE MODEL: wall ids in cyclic CCW order. A wall may appear twice (a notch splits its host
        // line into two runs). The polygon is these edges intersected in order; there is no other
        // topology state — no votes, no walk. Public so the harness can wound it.
        std::vector<std::uint64_t> order;

        /// Model-first initialisation: a CCW quad (the scan's OBB) — the prior on room shape and
        /// size. θ₀ is born from it; everything after is refinement plus splice jumps.
        void initialize_rect(const std::vector<Eigen::Vector2f>& rect_ccw);

        /// The per-frame pipeline: existence first (step-back), then segment↔edge association
        /// (many-to-one, Mahalanobis+PDA), candidate accumulation, qualified candidates proposing
        /// splices. `pose`/`pose_cov` is the slot's CURRENT estimate; weights empty ⇒ ones.
        FrameResult observe(const wallseg::Result& seg, const std::vector<Eigen::Vector2f>& pts_robot,
                            const Eigen::VectorXf& weights, const Eigen::Vector3f& pose,
                            const Eigen::Matrix3f& pose_cov, std::int64_t timestamp_ms);

        /// Fuse edges that have become statistically indistinguishable. Returns how many.
        int merge_indistinguishable();

        /// The model's polygon: the ordered edges intersected. Never throws.
        Polygon build_polygon() const;
        Polygon build_from(const std::vector<std::uint64_t>& ord) const;

        /// One-shot gauge change: p' = R(−rot)·(p − c). Transforms every edge, candidate and θ₀.
        void reanchor(const Eigen::Vector2f& c, float rot);

        const WallLandmark* find(std::uint64_t id) const;
        WallLandmark*       find(std::uint64_t id);
        int index_of(std::uint64_t id) const;

        struct ClassChoice { int k = -1; float eps = 0.f; float cost = 0.f; };
        ClassChoice classify(float phi) const;

        static Corner intersect_walls(const WallLandmark& a, const WallLandmark& b, bool inferred);

        /// Segment (robot frame) → map-frame (φ, d) and the 2×3 Jacobian w.r.t. the pose.
        static void to_map(float phi_r, float d_r, const Eigen::Vector3f& pose,
                           float& phi_m, float& d_m, Eigen::Matrix<float, 2, 3>& H);

    private:
        std::uint64_t next_id_ = 1;
        /// A qualified candidate becomes a JUMP: a notch on a host edge, or a corner cut with a
        /// neighbour. Committed only if the resulting cycle builds a simple CCW polygon. Returns the
        /// new C edge's index in walls, or −1 (rejected — counted, never silent).
        int  try_splice(const Candidate& c, FrameResult& fr, std::int64_t ts);
        /// SPUR WRAP from the model's own residual: an order wall whose existence-SOLID extent
        /// overshoots its polygon corner into the interior is matter the current topology cannot
        /// explain (a spur's near face rides collinear on a mapped wall, so no candidate ever
        /// exists for it). Wrap the boundary around the evidence: the wall runs on to the end of
        /// its solid bins, a perpendicular tip cap T there, a mirror face M returns —
        /// [.., W, T, M, ..]. Gated on connected free space observed behind the overshoot (the
        /// room provably wraps); a wrong wrap dies by the step-back operator. ≤ 1 per frame.
        int  try_spur_wraps(FrameResult& fr, std::int64_t ts);
        /// DOWN-JUMPS — the symmetric half of order-quantized growth: propose removing order
        /// entries (zero-evidence walls and duplicate rides), accept when the evidence surrendered
        /// (jump_delta plus removed walls' existence-bin nats) is worth less than the refunded
        /// order cost × keep_fraction. Evidence-neutral structure (a pocket over unobserved space)
        /// unfolds; supported structure pays its rent and stays. ≤ 1 per frame.
        int  try_down_jumps(FrameResult& fr, std::int64_t ts);
        /// Is `w` the verified FAR FACE of a wrapped thin wall — an anti-parallel observed twin at
        /// thin separation AND connected-free space observed on w's outward side? (A phantom twin
        /// 0.2 m behind a boundary wall has exterior/unknown there.) Needs comp_cache_ fresh.
        bool mirror_backed(const WallLandmark& w) const;
        /// Honest uncertainty of θ0: the worst class disagreement among CONFIRMED order walls
        /// (≥500 points), or the OBB prior's 15° while nothing is confirmed. The strict-Manhattan
        /// gate and the class cost widen by this, so an unconfirmed tilted θ0 cannot lock the
        /// true walls out (the tilted-prior lesson, measured a second time).
        float theta0_sigma() const;
        /// Evidence a trial polygon newly explains vs the current one, in grid-log-odds nats,
        /// evaluated ONLY over the region where the two interior claims differ (bounding box of
        /// unmatched vertices ± margin). Matter cells the trial releases from the interior count
        /// FOR it; FRESH connected free cells it releases count AGAINST it; stale free (free_ms
        /// before `fresh_ref_ms`) and unknown are silent — a sealed room change must stay payable.
        /// Grid nats the trial explains over the current cycle, where the two disagree. When
        /// `var_out` is given it also returns the VARIANCE of that number: each contested cell is a
        /// two-valued outcome — the cell is really matter with probability sigmoid(l) and really
        /// free otherwise — so its contribution has variance (a−b)²·p(1−p) and the total is their
        /// sum. It needs no new constant: a cell the grid is sure about contributes almost nothing,
        /// an unknown cell contributes its full swing, and a claim spanning many uncertain cells is
        /// correspondingly less trustworthy than the same number of nats over a few certain ones.
        float jump_delta_nats(const Polygon& cur, const Polygon& trial, std::int64_t fresh_ref_ms,
                              float* var_out = nullptr) const;
        /// Remove a dead edge from the order and HEAL the cycle (collapse parallel neighbours).
        void splice_out(std::uint64_t id);
        void heal_order();
        /// A self-crossing cycle is IMPOSSIBLE — persistent crossing is itself decisive refutation of
        /// the weakest edge involved (a wrong commit that beams alone cannot undo: they support it
        /// obliquely from the real walls it was mis-spliced onto). Counts consecutive crossing frames
        /// and removes the least-supported crossing edge after a short persistence.
        // Splice out the weakest edge of a self-crossing cycle after 5 persistent frames; `immediate`
        // skips the persistence (for a copy with no history). Returns true if a wall was removed.
        bool repair_if_crossing(bool immediate = false);
        int  crossing_frames_ = 0;
        /// Robot position (map frame) at the last observe() — the seed of the stub discriminator's
        /// connectivity flood, whose result is cached per frame timestamp.
        Eigen::Vector2f last_pose_xy_ = Eigen::Vector2f::Zero();
        std::vector<char> comp_cache_;
        std::int64_t comp_cache_ts_ = -1;
        WallLandmark make_wall(float phi, float d, const Eigen::Matrix2f& info, float exist_seed,
                               std::int64_t ts);
        /// Give NEW (extent-less) edges their extent from the polygon they now bound.
        void seed_extents_from_polygon();
        void reclassify_all();
        void update_existence(const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& pose,
                              FrameResult& fr);
    };
} // namespace rc::wallmap
