#pragma once

#include <vector>
#include <optional>
#include <Eigen/Dense>

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
    };

    struct DetectionResult
    {
        std::vector<CornerMatch> matches;
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

    /// 2D line: normal · p = d   (normal is unit length)
    struct Line2D
    {
        Eigen::Vector2f normal;
        float d;
        float resid_var = 0.f;   // λ_min / N — mean squared perpendicular scatter of the fitted points
                                 // about the line (≈ sensor noise for a clean wall; large for a cluttered
                                 // gather). Inflates σ_L so a poorly-fit wall is trusted less.
        int   npts = 0;          // number of points the line was fit to
        Eigen::Vector2f direction() const { return Eigen::Vector2f(-normal.y(), normal.x()); }
    };

    /// PCA line fit — returns nullopt if fewer than min_points.
    static std::optional<Line2D> fit_line_pca(const std::vector<Eigen::Vector2f>& pts,
                                               int min_points);

    /// Intersect two lines.  Returns nullopt if (nearly) parallel.
    static std::optional<Eigen::Vector2f> intersect(const Line2D& a, const Line2D& b,
                                                     float* angle_deg = nullptr);
};

} // namespace rc
