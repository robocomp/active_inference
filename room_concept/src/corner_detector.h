#pragma once

#include <vector>
#include <cmath>
#include <optional>
#include <Eigen/Dense>
#include "line_fit.h"

namespace rc {

/// Detects room corners in lidar scans using model-guided partitioning.
///
/// Pipeline:
///   1. Project model corners (world frame) into robot frame.
///   2. For each predicted corner, gather lidar points within a search radius.
///   3. Partition the neighbourhood into two groups using the known wall
///      directions from the polygon model.
///   4. Fit a line (PCA) to each group and intersect.
///   5. Accept the detection if it passes angle and distance quality gates.
class CornerDetector
{
public:
    // ===== Configuration =====
    struct Params
    {
        float search_radius       = 1.5f;   // meters around predicted corner to gather points
        int   min_points_per_line = 3;       // minimum points per wall group
        float ransac_threshold    = 0.06f;   // inlier band width for optional outlier rejection
        float max_match_distance  = 1.5f;   // meters — LEGACY metric association cap. Superseded by the
                                            // Mahalanobis gate (assoc_chi2); kept only as a coarse
                                            // pre-filter on the gather so the cost matrix stays small.
        float min_corner_angle    = 25.0f;   // degrees — kept for MODEL-corner selection only (set_model_corners)
        float max_corner_angle    = 155.0f;  // degrees — kept for MODEL-corner selection only (set_model_corners)
        float max_orientation_dev = 20.0f;   // degrees — LEGACY hard gate (unused now; kept for config back-compat)

        // ── Graded-covariance detection (replaces the old hard rej_angle/rej_orient gates) ──
        // A detection is no longer discarded when its geometry is marginal; instead it carries a
        // per-detection 2×2 information matrix Λ_det (robot frame) whose precision shrinks smoothly
        // with wall-fit scatter, orientation deviation, and intersection shallowness. The RFE loss
        // consumes Λ_det anisotropically, so a marginal corner contributes weakly and a clean
        // asymmetric corner (the notch) contributes strongly — no thresholds, covariance → SDF pose.
        float wall_band     = 0.35f;   // meters — perpendicular tolerance gathering wall points around the
                                       // PREDICTED wall line. Must exceed the chronic model misfit (~0.32 m
                                       // here) or the true wall points fall outside the band and NO detection
                                       // forms (the real reason corners almost never fired). Was hardcoded 0.12.
        float base_sigma    = 0.04f;   // meters — corner detection noise floor σ0 (per-wall).
        float orient_tau_deg = 20.0f;  // degrees — smooth orientation-trust scale: ori_scale = exp(−(dev/τ)²).
                                       // dev=τ → 37% weight, 2τ → 2%. Replaces the 20° hard cut.

        // ── Exclusion: "two corners cannot occupy the same physical space" ──
        // The Hungarian below only guarantees each candidate OBJECT is claimed once; two model corners
        // with overlapping search discs can still each synthesise their OWN candidate at (nearly) the
        // same physical point, and both survive → duplicated/near-coincident corners in the UI and a
        // double-counted corner factor in the loss. The fix is not a metric radius but a statistical
        // identity test: two detections are the SAME physical corner when their separation is not
        // resolvable given their own uncertainties, i.e. the Mahalanobis distance under the combined
        // covariance S = Σ_a + Σ_b falls inside a χ²₂ confidence region. Precise detections a few cm
        // apart stay separate (a real notch); vague ones half a metre apart fuse. Fusion is
        // information-weighted (Λ = Λ_a + Λ_b), so no evidence is thrown away.
        float merge_chi2 = 5.991f;     // χ²₂ at 95% — separation below this ⇒ statistically one corner.
                                       // 0 disables the exclusion test.
        float merge_prior_sigma = 0.0f;// meters — σ of the weak "corner lies inside the search disc"
                                       // prior added to Λ_det before inverting, so an unconstrained
                                       // (rank-1) detection gets a LARGE but finite covariance instead
                                       // of an infinite one. 0 ⇒ use search_radius.

        // ── Association: Mahalanobis gate + ambiguity-weighted precision ──────────────────────────
        // A metric cap (max_match_distance) cannot express "which model corner produced this
        // detection". Once the layout gained chamfers and pillars, 51 model-corner pairs sit closer
        // together than that cap, so a ~0.3 m pose error lets a detection be claimed by the WRONG
        // corner; the factor then confidently pulls the pose onto a bogus correspondence (the jumps).
        // Two model-level fixes replace it:
        //  (a) the Hungarian cost IS the squared Mahalanobis distance under the INNOVATION covariance
        //      S = Σ_det + H·P_pose·Hᵀ, gated at assoc_chi2. The gate widens when the pose is
        //      uncertain and tightens when it is sharp — no fixed metre radius anywhere.
        //  (b) each accepted corner's precision is scaled by the POSTERIOR PROBABILITY that its
        //      association is the right one (PDA-style normalisation over every in-gate model corner).
        //      A detection two corners could equally explain contributes ~nothing instead of a
        //      confident wrong pull. This is what "keep only the N best corners" reaches for, done as
        //      precision rather than a cap: unambiguous corners keep full weight, aliasing ones mute
        //      themselves, and nothing has to be discarded by rank.
        float assoc_chi2 = 5.991f;     // χ²₂ @95% — association gate on the innovation covariance.
        float map_sigma  = 0.06f;      // meters — MODEL error: how far the traced SVG layout sits from
                                       // the real wall. Without it S would hold only sensor noise
                                       // (σ≈0.04 m ⇒ a 0.10 m gate), and the layout's own ~0.3 m misfit
                                       // would reject every corner. It belongs in the generative model:
                                       // the map is a hypothesis, not ground truth. Raise it if the
                                       // "gate=" counter dominates; the PDA weight then keeps the now-
                                       // ambiguous corners from voting confidently, so the failure mode
                                       // is graceful (weak corners) rather than wrong (swapped ones).
                                       // 0.10 → 0.06 from LIVE DATA 2026-07-20: chi2_mean was 0.33
                                       // (honest ≈ 1) and real residuals were 0.016–0.129 m, nowhere near
                                       // the 0.32 m the old wall_band comment assumed. Also acts as the
                                       // LOWER bound on the per-corner gather band (see detect()): a band
                                       // tighter than the map error can never gather the true wall.
        float assoc_min_weight = 0.0f; // floor on the association posterior (0 = pure PDA). Raise only
                                       // if you want ambiguous corners to retain some pull.

        // ── Landmark admissibility: does the MAP actually assert this vertex? ──────────────────────
        // Prior to inference, and not a gate on it. map_sigma is the layout's own positional error, so
        // a jog whose adjacent walls are not long compared to it is not a feature the trace can claim
        // exists — the SVG cannot distinguish a 6 cm step from a straight wall, and two model corners
        // that close cannot be resolved by a sensor at base_sigma either, so they alias each other by
        // construction (they are what feeds merged_coincident and drags mean_assoc_prob down).
        // Excluding them is a MODEL-fidelity correction of the same kind as not modelling the skirting
        // board — the vertex stays in polygon_ and in the SDF, it just stops being a point landmark.
        // Scaled by map_sigma rather than set in metres so it tracks the map error it derives from.
        // 0 disables. Live layout (apartamento, map_sigma=0.06): 3σ=0.18 m sits inside a clean bimodal
        // gap — 8 trace artefacts (0.062–0.149 m) drop, the 24 real corners (≥0.315 m) all survive.
        float min_wall_map_sigmas = 3.0f;

        // ── Landmark RETIREMENT by observed information ────────────────────────────────────────────
        // The rule above asks whether the MAP asserts a vertex. This asks whether the ROBOT can ever
        // measure it — a different question, and live data (23400 frames, apartamento) showed geometry
        // alone cannot answer it: v27 and v28 are adjacent vertices of the SAME pillar with identical
        // 0.333 m walls, yet v28 yields λ_min≈107 while v27 is a permanent coin flip at λ_min≈1.6.
        // Which corners actually compete depends on visibility along the driven trajectory, not on the
        // layout, so no static geometric rule separates them (v5's nearest neighbour is 0.11 m away and
        // it is still the best landmark in the room — because that neighbour is always occluded).
        //
        // So retire on evidence: track each corner's information yield — the smallest eigenvalue of the
        // Λ_det the loss actually consumes, which is already scaled by the association posterior, so a
        // corner is penalised for aliasing AND for shallowness through one number. A corner whose yield
        // stays negligible is one this room-and-robot combination cannot observe informatively.
        //
        // The bar is expressed as a σ, in units of map_sigma, so it stays physical: a landmark must pin
        // its own position along its WEAKEST axis at least this well, otherwise it says less than the
        // polygon already does and has no business voting. λ_bar = 1/(min_yield_map_sigmas·map_sigma)².
        // Live: 5σ = 0.30 m ⇒ λ_bar ≈ 11, inside the observed bimodal gap (worst good corner 46.4,
        // best bad one 1.6) with ~4× margin either side. 0 disables retirement.
        float min_yield_map_sigmas = 5.0f;
        // Leak of the running yield estimate, per MATCHED frame: yield ← (1-leak)·yield + leak·λ_min.
        // 0.02 at ~20 Hz ⇒ ~2.5 s time constant. Leaky ON PURPOSE — a retired corner keeps being
        // detected and recovers by itself if the robot later drives where it becomes observable.
        float yield_leak = 0.02f;
        // Matched frames to observe before retirement may fire — an EVIDENCE BUDGET, not a delay.
        // Counted in matches rather than in frames or seconds on purpose: a corner can only mislead
        // the loss on a frame where it actually matched, so this is exactly "how many uninformative
        // votes are tolerated before we stop listening", and it costs the same regardless of how often
        // the corner fires. Keep it SMALL. It was 100 and that was a design error: the corners most
        // worth retiring are the ones that match rarely, so the delay scaled inversely with how bad
        // the corner was — live, the worst aliaser in the room (v11, assoc_prob 0.20, matching 0.86%
        // of frames) needed ~10 minutes to reach 100 matches, while a merely-mediocre corner retired
        // in seconds. 10 is ample given the seeded estimate and the ~28× gap between the worst kept
        // corner and the best retired one.
        int   yield_warmup = 10;
        // Hysteresis. Retirement LATCHES: a corner is retired when its yield falls below λ_bar and is
        // released only once it climbs back above yield_release_factor·λ_bar. Without the gap a corner
        // sitting near the bar toggles every few frames, which is not merely ugly — each toggle adds
        // and removes a factor from the loss, so the pose sees a landmark set that changes under it.
        // These corners are strongly viewpoint-dependent (live: v27's running yield went 0.09 → 185 as
        // the robot reached a view where that pillar face is observable), so crossings are frequent and
        // genuine; the band decides how much improvement counts as "the view really did get better".
        // 1.0 disables the hysteresis, restoring a single bar.
        float yield_release_factor = 2.0f;
    };

    // ===== Output types =====

    struct CornerMatch
    {
        int    model_index;         // index into the ORIGINAL polygon vertex list
        Eigen::Vector2f detected;   // detected position (robot frame, meters)
        Eigen::Vector2f predicted;  // predicted position (robot frame, meters)
        Eigen::Vector2f model_world;// model corner world position (for display)
        float  distance;            // ||detected - predicted||
        float  angle_deg;           // angle between the two fitted lines
        Eigen::Matrix2f covariance; // 2×2 detection uncertainty (robot frame) — legacy, display only
        Eigen::Matrix2f information;// 2×2 graded precision Λ_det = Σ_L (ori_scale_L/σ_L²) n_L n_Lᵀ (robot frame),
                                    // ALREADY scaled by assoc_prob below. Rank-1 when the two walls are
                                    // near-parallel (shallow corner → the bisector direction is left
                                    // unconstrained). This is what the loss uses.
        float  assoc_prob = 1.f;    // posterior probability that this detection belongs to model_index
                                    // rather than to another in-gate model corner. 1 = unambiguous.
        float  assoc_chi2_val = 0.f;// squared Mahalanobis distance of the winning association (display).
        // ── Retirement (see Params::min_yield_map_sigmas) ────────────────────────────────────────
        // A suppressed match is still DETECTED, still carries its numbers, and is still returned — it
        // simply must not enter the loss. Kept in `matches` rather than dropped so the viewer can show
        // it as retired and the stats can keep watching it recover; silently vanishing landmarks are
        // how a detector starts looking broken.
        bool   suppressed = false;  // retired: informative yield never materialised
        float  yield = 0.f;         // running λ_min estimate (1/m²) behind that decision
        float  runnerup_chi2 = 1e9f;// squared Mahalanobis distance of the BEST RIVAL model corner for
                                    // this same detection. Large ⇒ the correspondence is unambiguous;
                                    // close to assoc_chi2_val ⇒ a coin flip, which is what makes the
                                    // pose jump. This is the number that identifies WHICH corners alias.
        int    n_rivals = 0;        // how many OTHER model corners were inside the gate for this same
                                    // detection. ★ THE ASSOCIATION'S INPUT, LOGGED BESIDE ITS VERDICT:
                                    // assoc_chi2_val alone cannot be audited, because the gate leaves
                                    // every over-gate pair INFEASIBLE and so it is TRUNCATED to
                                    // [0, assoc_chi2] by construction — an emitted match can never
                                    // exceed the bound, and its distribution therefore says nothing
                                    // about how often the gate was right. n_rivals and runnerup_chi2
                                    // are what the gate had to choose BETWEEN, and 0 rivals means
                                    // there was no choice to get wrong.
    };

    /// ── THE ONE "IS THIS CORNER MATCHED?" RULE, SHARED BY EVERY VIEW ────────────────────────────
    /// The 2-D canvas and the camera overlays both decide whether to draw a corner. When they each
    /// carried their own version of that decision they disagreed about the same state — the canvas
    /// showed only believed corners while the overlay showed everything — and two views of one
    /// estimator arguing is a bug report waiting to be filed against the estimator.
    ///
    /// Display-only: nothing here gates detection, association or the loss. A corner that fails this
    /// is still tracked, still in `matches`, and still available to any view that asks for all of
    /// them (the camera window's All/Matched button does exactly that).
    ///
    /// The two conditions, and why each is the number it is:
    ///   1. matched to a model vertex at all, with a finite position;
    ///   2. `assoc_prob >= 0.5` — the PDA posterior that this detection came from `model_index`
    ///      rather than a rival in-gate vertex. Below a coin flip the corner is not matched to the
    ///      vertex it claims to be; it is merely the best of several guesses. This is the LiDAR
    ///      counterpart of the RGB path's `pi_vis`, tested at the same 0.5.
    ///
    /// There is deliberately NO chi2 test here, because the association gate already bounds
    /// `assoc_chi2_val` upstream and a repeat of it could never fire:
    ///   • corner_detector.cpp:531 leaves every (model corner, detection) pair whose squared
    ///     Mahalanobis distance exceeds `Params::assoc_chi2` (5.991 = χ²₂ @95%) INFEASIBLE;
    ///   • solve_hungarian drops infeasible pairs when it extracts the assignment
    ///     (corner_detector.cpp:166), so an over-gate pair is never assigned;
    ///   • corner_detector.cpp:580 is the ONLY writer of `assoc_chi2_val`, and it writes that
    ///     same gated cost — so every emitted match carries a value in [0, assoc_chi2].
    /// `assoc_chi2` is a compiled-in constant: no config key reaches it (room_config.cpp loads the
    /// other CornerDetector::Params fields, not this one), so the `assoc_chi2 <= 0` escape hatch
    /// cannot be taken by a running agent either. Should that ever change, this rule stops bounding
    /// the residual — which is the association gate's job, not the viewer's.
    [[nodiscard]] static bool matched_for_display(const CornerMatch &m) noexcept
    {
        return m.model_index >= 0 and m.detected.allFinite() and m.assoc_prob >= 0.5f;
    }

    /// Running totals since construction — the per-frame DetectionResult resets every detect(), and
    /// fourteen samples cannot settle a calibration question (χ²₂/2 has unit variance, so the s.e. on
    /// a mean of 14 is 0.27). These accumulate over the whole tour.
    struct TourStats
    {
        double nis_pre_sum = 0.0; int nis_pre_n = 0; int nis_pre_over = 0;
        double nis_acc_sum = 0.0; int nis_acc_n = 0;
        double s_det_sum = 0.0, s_pred_sum = 0.0, s_map_sum = 0.0; int s_terms_n = 0;
        // ori_scale = exp(−(dev/τ)²) multiplies each line's INFORMATION, so it divides the covariance:
        // C = Λ⁻¹/ori. It is the last hand-set term in the corner path, and after removing the
        // base_sigma double count the detection σ was still 0.156 m against 0.049 m residuals — an
        // excess of about 2.5x in variance, which an ori of 0.3-0.6 would produce exactly. Logged
        // before being touched, because the two previous culprits were guessed wrong.
        double ori_sum = 0.0; int ori_n = 0; double ori_min = 1.0;
        [[nodiscard]] float ori_mean() const { return ori_n ? static_cast<float>(ori_sum / ori_n) : 0.f; }
        // ── EVERY INPUT TO THE PROPAGATION, so one tour can settle where the excess lives ─────────
        // corner sigma^2 is built from each line's (phi,d) covariance through a Jacobian whose phi
        // column is the LEVER ARM |t.p_corner| — the corner's tangent coordinate from the robot
        // origin. So the phi contribution to corner position is roughly sigma_phi * lever, and a
        // corner 4 m out turns a 1 deg line-angle error into 7 cm. These record each factor on its
        // own: the fit's sigma_phi/sigma_d, what it was fitted from (points, scatter), the lever,
        // and the corner's opening angle whose 1/sin amplifies all of it. Guessing cost three tours.
        double sphi_sum = 0.0, sd_sum = 0.0, npts_sum = 0.0, resid_sig_sum = 0.0, lever_sum = 0.0;
        int    line_n = 0;
        double angle_sum = 0.0; int angle_n = 0;
        [[nodiscard]] float sphi_deg()   const { return line_n ? static_cast<float>(sphi_sum / line_n) : 0.f; }
        [[nodiscard]] float sd_m()       const { return line_n ? static_cast<float>(sd_sum / line_n) : 0.f; }
        [[nodiscard]] float npts_mean()  const { return line_n ? static_cast<float>(npts_sum / line_n) : 0.f; }
        [[nodiscard]] float resid_sig()  const { return line_n ? static_cast<float>(resid_sig_sum / line_n) : 0.f; }
        [[nodiscard]] float lever_m()    const { return line_n ? static_cast<float>(lever_sum / line_n) : 0.f; }
        [[nodiscard]] float angle_deg()  const { return angle_n ? static_cast<float>(angle_sum / angle_n) : 0.f; }
        [[nodiscard]] float nis_pre_mean() const { return nis_pre_n ? static_cast<float>(nis_pre_sum / nis_pre_n) : 0.f; }
        [[nodiscard]] float nis_acc_mean() const { return nis_acc_n ? static_cast<float>(nis_acc_sum / nis_acc_n) : 0.f; }
        /// Standard error of nis_pre_mean under the χ²₂/2 null (unit variance per sample).
        [[nodiscard]] float nis_pre_se() const { return nis_pre_n ? 1.f / std::sqrt(static_cast<float>(nis_pre_n)) : 0.f; }
        [[nodiscard]] float s_det_sigma()  const { return s_terms_n ? static_cast<float>(s_det_sum  / s_terms_n) : 0.f; }
        [[nodiscard]] float s_pred_sigma() const { return s_terms_n ? static_cast<float>(s_pred_sum / s_terms_n) : 0.f; }
        [[nodiscard]] float s_map_sigma()  const { return s_terms_n ? static_cast<float>(s_map_sum  / s_terms_n) : 0.f; }
    };

    /// Totals since construction; see TourStats. Read-only to callers, folded in by detect().
    [[nodiscard]] const TourStats& tour_stats() const { return tour_; }
    void reset_tour_stats() { tour_ = TourStats{}; }


    /// ── ONE ROW PER CORNER EVALUATION — the raw record, not a moment of it ───────────────────
    /// Three separate hypotheses about the corner covariance have now been aimed at the wrong term,
    /// every one of them from a tour MEAN. A mean over a heavy-tailed per-line quantity says nothing
    /// about the typical line, and no further accumulator recovers a distribution from its first
    /// moment. So this carries the whole state of one candidate at the association gate — both lines'
    /// inputs, both their covariances, the Jacobian's amplifiers, the three terms of S and the
    /// innovation — and room_concept dumps it verbatim. Every remaining question about the
    /// calibration is then arithmetic on a file instead of another tour.
    struct CornerProbe
    {
        int   model_index = -1;
        int   propagated  = 0;   // 1 = Jacobian propagation, 0 = rank-1 fallback (is the new path live?)
        int   over_gate   = 0;
        float det_x = 0.f,  det_y = 0.f;    // detected intersection, robot frame
        float pred_x = 0.f, pred_y = 0.f;   // predicted model corner, robot frame
        float nu_x = 0.f,   nu_y = 0.f;     // innovation = detected − predicted
        float d2 = 0.f;                     // Mahalanobis² against S = S_det + S_pred + S_map
        float sdet_xx = 0.f, sdet_xy = 0.f, sdet_yy = 0.f;
        float sprd_xx = 0.f, sprd_xy = 0.f, sprd_yy = 0.f;
        float smap_xx = 0.f, smap_yy = 0.f;
        float angle_deg = 0.f;              // corner opening angle
        float sin_theta = 0.f;              // |det M| — the 1/sin(θ) amplification of the propagation
        // Per adjacent line, slot 0 = incoming edge, 1 = outgoing.
        int   npts[2]      = {0, 0};    // EFFECTIVE count Σw the fit was made of
        int   nraw[2]      = {0, 0};    // returns the gather band offered; npts/nraw = responsibility kept
        int   rival[2]     = {-1, -1}; // polygon edge that took the most responsibility from this fit
        float rival_share[2] = {0.f, 0.f}; // its share of everything taken — 1 means a single culprit
        float ori[2]       = {0.f, 0.f};    // exp(−(dev/τ)²), the orientation-trust divisor on C
        float resid_sig[2] = {0.f, 0.f};    // √resid_var — per-point perpendicular scatter (the σ fed in)
        float s_mean[2]    = {0.f, 0.f};    // mean tangent coord t·p, measured from the ROBOT ORIGIN
        float s_std[2]     = {0.f, 0.f};    // along-wall spread — what C(φ,φ) actually lives on
        float s_span[2]    = {0.f, 0.f};    // max − min tangent coord: the segment's real extent
        float lever[2]     = {0.f, 0.f};    // |t·p_corner| — the φ column of the Jacobian
        float c00[2] = {0.f, 0.f}, c01[2] = {0.f, 0.f}, c11[2] = {0.f, 0.f};   // line covariance on (φ, d)
    };

    struct DetectionResult
    {
        std::vector<CornerMatch> matches;
        std::vector<CornerProbe> probes;   // one per candidate reaching the gate; see CornerProbe
        int corners_in_fov = 0;
        int corners_detected = 0;
        int corners_accepted = 0;
        // Diagnostic counts — where do in-FOV corners die? With graded covariance most of these are no
        // longer hard rejections but "soft" events kept for observability.
        int rej_occluded = 0;  // ★ model corner NOT visible from the robot (a wall/notch occludes the sight
                               // line) → excluded BEFORE detection: an unreachable corner must not be matched,
                               // must not enter the loss, and must not count toward the early-exit decision.
        int rej_fewpoints = 0; // ★ FORMATION failure: gather grabbed < min_points_per_line on a wall → no
                               // detection formed at all (fires BEFORE the gates). If this dominates, the
                               // wall_band is too tight vs the model misfit — widen it. This is the counter
                               // added to confirm the gather-band hypothesis.
        int rej_dist = 0;      // detection failed the Mahalanobis gate against its OWN model corner
        float mean_assoc_prob = 1.f; // mean association posterior over accepted corners. Well below 1
                               // ⇒ the layout is aliasing (corners closer together than the pose
                               // uncertainty can resolve) and the corner channel is muting itself.
        float min_assoc_prob  = 1.f; // worst single association this frame (0.5 = perfect coin flip).
        // ── Residual distribution over ACCEPTED corners: measures the real model misfit, so map_sigma
        //    can be set from data instead of guessed. resid_max ≫ resid_mean ⇒ one corner is an outlier.
        float resid_mean = 0.f;      // mean ‖detected − predicted‖ (m)
        float resid_max  = 0.f;      // worst ‖detected − predicted‖ (m)
        float resid_chi2_mean = 0.f; // mean whitened residual (χ² units) — 1.0 ⇒ map_sigma is honest,
                               // ≪1 ⇒ map_sigma too large (gate needlessly loose, aliasing invited).
        // ── NIS: is the corner channel's covariance the right SIZE? ───────────────────────────────
        // NIS = νᵀ S⁻¹ ν with ν the innovation (detected − predicted) and S = Λ_det⁻¹ + pred_cov +
        // map_cov — the same S the association gate uses. Divided by its 2 degrees of freedom, an
        // honest covariance averages 1.0: above ⇒ overconfident (or a real bias/misassociation),
        // below ⇒ we are discarding information we have.
        // ⚠ MEASURED BEFORE THE GATE, ON PURPOSE, AND FOR TWO REASONS. The gate is a cut on this very
        // quantity, so an accepted-only mean is truncated by construction. But the larger bias is the
        // ASSIGNMENT: nis_acc_* is built from the Hungarian cost of the CHOSEN pairing, i.e. a minimum
        // over pairings, and an argmin is biased low however wide the gate is. Measured on a live
        // tour with nothing rejected at all — 0 of 14 over the gate — pre read 0.51 and accepted 0.33,
        // a 35% gap with zero truncation. So: nis_pre_* is the CALIBRATION statistic, counting every
        // detection against the model corner it was formed from; nis_acc_* diagnoses assignment
        // quality only and must never be read as a covariance check.
        double nis_pre_sum = 0.0;    // Σ NIS/dof over all gate evaluations
        int    nis_pre_n = 0;
        int    nis_pre_over = 0;     // how many exceeded assoc_chi2 (i.e. would be/were rejected)
        double nis_acc_sum = 0.0;    // Σ over accepted matches (argmin-biased — see above)
        int    nis_acc_n = 0;
        // ── WHICH TERM MAKES S TOO BIG? ───────────────────────────────────────────────────────────
        // S = pos_cov(Λ_det) + pred_cov + map_cov. A NIS below 1 says the SUM is too large; it cannot
        // say which of the three is at fault, and they have different owners — the detector's own
        // propagated covariance, the pose covariance, and map_sigma (a model-error constant last set
        // from data against the OLD constant-σ detector). Mean per-axis σ of each term, in metres.
        double s_det_sum = 0.0, s_pred_sum = 0.0, s_map_sum = 0.0;
        int    s_terms_n = 0;
        double ori_sum = 0.0; int ori_n = 0; double ori_min = 1.0;   // orientation-trust weight actually applied
        [[nodiscard]] float ori_mean() const { return ori_n ? static_cast<float>(ori_sum / ori_n) : 0.f; }
        // ── EVERY INPUT TO THE PROPAGATION, so one tour can settle where the excess lives ─────────
        // corner sigma^2 is built from each line's (phi,d) covariance through a Jacobian whose phi
        // column is the LEVER ARM |t.p_corner| — the corner's tangent coordinate from the robot
        // origin. So the phi contribution to corner position is roughly sigma_phi * lever, and a
        // corner 4 m out turns a 1 deg line-angle error into 7 cm. These record each factor on its
        // own: the fit's sigma_phi/sigma_d, what it was fitted from (points, scatter), the lever,
        // and the corner's opening angle whose 1/sin amplifies all of it. Guessing cost three tours.
        double sphi_sum = 0.0, sd_sum = 0.0, npts_sum = 0.0, resid_sig_sum = 0.0, lever_sum = 0.0;
        int    line_n = 0;
        double angle_sum = 0.0; int angle_n = 0;
        [[nodiscard]] float sphi_deg()   const { return line_n ? static_cast<float>(sphi_sum / line_n) : 0.f; }
        [[nodiscard]] float sd_m()       const { return line_n ? static_cast<float>(sd_sum / line_n) : 0.f; }
        [[nodiscard]] float npts_mean()  const { return line_n ? static_cast<float>(npts_sum / line_n) : 0.f; }
        [[nodiscard]] float resid_sig()  const { return line_n ? static_cast<float>(resid_sig_sum / line_n) : 0.f; }
        [[nodiscard]] float lever_m()    const { return line_n ? static_cast<float>(lever_sum / line_n) : 0.f; }
        [[nodiscard]] float angle_deg()  const { return angle_n ? static_cast<float>(angle_sum / angle_n) : 0.f; }
        [[nodiscard]] float s_det_sigma()  const { return s_terms_n ? static_cast<float>(s_det_sum  / s_terms_n) : 0.f; }
        [[nodiscard]] float s_pred_sigma() const { return s_terms_n ? static_cast<float>(s_pred_sum / s_terms_n) : 0.f; }
        [[nodiscard]] float s_map_sigma()  const { return s_terms_n ? static_cast<float>(s_map_sum  / s_terms_n) : 0.f; }
        [[nodiscard]] float nis_pre_mean() const
        { return nis_pre_n > 0 ? static_cast<float>(nis_pre_sum / nis_pre_n) : 0.f; }
        [[nodiscard]] float nis_acc_mean() const
        { return nis_acc_n > 0 ? static_cast<float>(nis_acc_sum / nis_acc_n) : 0.f; }
        // Rival statistics. An accepted corner either HAS a competing model corner inside the gate or it
        // does not; averaging the two cases is meaningless (the old version averaged the INFEASIBLE
        // sentinel and reported "583334", which only ever encoded "58% had no rival"). Report the count
        // separately and average χ² over the contested ones only.
        int   corners_with_rival = 0;   // accepted corners that have ≥1 competing model corner in gate
        float runnerup_chi2_mean = 0.f; // mean rival χ² over THOSE only. Close to the winner's χ² ⇒ coin flip.
        // Mean convexity agreement of the detections rejected on convexity: ≈0 ⇒ they were shallow/noisy
        // wedges (the gate is firing on ambiguity, not on a real flip); ≈−1 ⇒ genuine 180° disagreement.
        float convex_rej_agree_sum = 0.f;   // summed; divide by rej_convex
        [[nodiscard]] float convex_rej_agree_mean() const
        { return rej_convex > 0 ? convex_rej_agree_sum / static_cast<float>(rej_convex) : 0.f; }
        int soft_orient = 0;   // orientation trust ori_scale < 0.05 (heavily downweighted, NOT discarded)
        int rej_convex = 0;    // convexity sign mismatch — KEPT as a hard gate (topological disambiguator
                               // for rot180: |dir·model_dir| is 180°-blind, only convexity breaks the tie)
        int rej_noninformative = 0; // ★ association posterior 0/NaN, or Λ_det not finite / rank-0 after
                               // weighting ⇒ the match carries NO information and was DROPPED rather
                               // than emitted as a zero-precision factor. A zero-precision observation
                               // constrains nothing yet still enters the Hessian, which is how a
                               // singular Hessian (cond_num 1e8) turned into a NaN pose on 2026-07-21.
        int rej_unassigned = 0;// survived to candidate but lost the 1-to-1 Hungarian assignment
        int merged_coincident = 0; // ★ candidates absorbed by the exclusion test — two model corners
                               // synthesised detections at the same physical point (overlapping search
                               // discs) and were fused into one. If this is chronically high the model
                               // polygon has corners closer together than the detector can resolve.
        int model_dup_dropped = 0; // model corners dropped at set_model_corners() because they coincide
                               // with an already-kept vertex (degenerate/repeated polygon vertices).
        int model_short_wall_dropped = 0; // ★ vertices refused LANDMARK status at set_model_corners()
                               // because their adjacent walls are too short for the map to assert them
                               // (see Params::min_wall_map_sigmas). They remain in the polygon and in
                               // the SDF; only their point-landmark channel is off.

        // ── Per-model-corner attribution (indices into the ORIGINAL polygon vertex list) ───────────
        // Aggregate rej_* counters say WHAT is failing but not WHICH corner, which is what decides
        // whether a given pillar earns its keep. Pair these with the accepted corners in `matches`
        // (each carries model_index, assoc_prob, runnerup_chi2, information) for the full picture:
        // in_fov but rarely in matches ⇒ the corner costs work and returns nothing.
        std::vector<int> in_fov_indices;    // corner was visible and a detection was attempted
        std::vector<int> occluded_indices;  // corner was in range but occluded — NOT its own fault
        int corners_suppressed = 0;         // matches retired this frame by the yield rule
    };

    // ===== Interface =====

    explicit CornerDetector() = default;
    explicit CornerDetector(const Params& p) : params_(p) {}

    void set_model_corners(const std::vector<Eigen::Vector2f>& polygon_vertices);

    /// `pose_cov` is the CURRENT 3×3 SE(2) pose covariance (x, y, θ; world frame). It enters the
    /// association gate through the innovation covariance S = Σ_det + H·P·Hᵀ, so a poorly-localized
    /// robot associates permissively and a sharply-localized one refuses distant corners. Passing
    /// zero reduces the gate to detection noise alone.
    /// NOT const: each call folds this frame's evidence into the per-corner information-yield belief
    /// that drives retirement (see Params::min_yield_map_sigmas).
    DetectionResult detect(const std::vector<Eigen::Vector3f>& lidar_points,
                           float robot_x, float robot_y, float robot_theta,
                           const Eigen::Matrix3f& pose_cov = Eigen::Matrix3f::Zero(),
                           float max_range = 15.0f);

    Params& params() { return params_; }
    const Params& params() const { return params_; }

    /// Polygon indices refused landmark status for having walls shorter than the map can assert.
    /// Valid after set_model_corners(); reported once at startup so the exclusion is never silent.
    const std::vector<int>& short_wall_dropped_indices() const { return short_wall_dropped_; }

private:
    Params params_;
    TourStats tour_;   // running totals over the whole run (see tour_stats())

    /// Full room polygon (world frame) — retained for the ray-cast occlusion/visibility test so an
    /// occluded corner (behind a wall or the notch step) is excluded before detection.
    std::vector<Eigen::Vector2f> polygon_;

    /// Model corner with its two adjacent wall directions.
    struct ModelCorner
    {
        Eigen::Vector2f position;       // world frame
        Eigen::Vector2f edge_in_dir;    // unit direction of wall arriving at this corner
        Eigen::Vector2f edge_out_dir;   // unit direction of wall leaving this corner
        float convexity_sign;           // sign of edge_in × edge_out (positive = CCW turn)
        float wall_in_length;           // length of the incoming wall (prev→curr)
        float wall_out_length;          // length of the outgoing wall (curr→next)
        int original_index;             // index in original polygon
    };
    std::vector<ModelCorner> model_corners_;

    /// How many polygon vertices were rejected as coincident with an already-kept model corner.
    int model_dups_dropped_ = 0;

    /// Polygon indices refused landmark status by the min_wall_map_sigmas admissibility rule.
    std::vector<int> short_wall_dropped_;

    /// Per-model-corner information-yield belief, parallel to model_corners_ (see
    /// Params::min_yield_map_sigmas). Reset by set_model_corners; updated by detect().
    // samples = frames this corner MATCHED. `retired` LATCHES (see Params::yield_release_factor):
    // it is the state the hysteresis carries between frames, not a per-frame recomputation.
    struct Yield { float lambda_min = 0.f; int samples = 0; bool retired = false; };
    std::vector<Yield> yield_;
    /// original polygon index → slot in model_corners_/yield_, for attributing a match back.
    std::vector<int>   slot_of_original_;

    /// 2D line in Hesse form. Shared with the wall segmenter — see line_fit.h.
    using Line2D = linefit::Line2D;

    /// PCA line fit — returns nullopt if fewer than min_points. Forwards to linefit (one definition).
    static std::optional<Line2D> fit_line_pca(const std::vector<Eigen::Vector2f>& pts, int min_points)
    { return linefit::fit_line_pca(pts, min_points); }

    /// Intersect two lines.  Returns nullopt if (nearly) parallel.
    static std::optional<Eigen::Vector2f> intersect(const Line2D& a, const Line2D& b,
                                                    float* angle_deg = nullptr)
    { return linefit::intersect(a, b, angle_deg); }
};

} // namespace rc
