/*
 * hood_belief.h  —  AI2 hood belief (see CONCEPT_AGENT_RECIPE.md, HOOD notes)
 *
 * The SIMPLEST box concept: a hood is ONE free-standing, floor-anchored cuboid — no legs, no top
 * slab, no w↔h symmetry, no round/square shape selection, no footprint-moment. ONE generative primitive
 * (a solid box occupying z∈[0,H]) + a uniform clutter component, soft per-point responsibilities (a
 * mixture, no hard min()/threshold), inferred by the shared recursive variational-Laplace Gaussian filter
 * carrying a FULL 6×6 covariance. Pure Eigen, no torch, no DSR — unit-testable in isolation (self_test()).
 *
 * State θ = [cx, cy, H, w, h, yaw]  (6 DOF ≡ the recipe's [cx,cy,yaw,w,d,h]):
 *   cx,cy = room-frame centre XY   ·   H = height of the box TOP above the floor
 *
 * ★VERTICAL ANCHORING — a hood HANGS, it does not stand. Cloned from refrigerator, whose box spans
 * z ∈ [0, H] because a fridge rests on the floor, so one number fixed both its extent and its placement.
 * A range hood's underside sits ~1.55 m up with nothing beneath it, so the box spans z ∈ [H − extent, H].
 * H keeps its meaning (top above floor) and the ONLY thing that changes is the bottom: 0 → H − extent.
 *
 * ★WHY extent IS A PARAMETER AND NOT A 7th DOF. It would be more general to estimate the vertical extent,
 * and it would also be unidentifiable: a hood's underside is a crisp edge against the hob gap (LiDAR
 * strikes it, the mask's lower boundary is clean) while its TOP merges into the wall or leaves the frame
 * entirely at any usable stand-off. Estimating a DOF the data cannot resolve is how the size oscillation
 * bugs in this lineage started. So extent carries a prior and a stated uncertainty instead — and if a
 * viewpoint ever does resolve the top edge, promoting it to a DOF is a contained change from here.
 *   w = width (local X)            ·   h = depth (local Y)   ·   yaw = rotation about +Z.
 * The box is asymmetric (w≠h resolves the 90° swap), so there is NO w↔h mode accumulator and NO canonical
 * fold — canonicalize() is a no-op. The honest 180° front/back yaw ambiguity is NOT folded; it is left in
 * the reported σ_yaw. n_prims()==1. Static furniture: transition() = I.
 */

#pragma once

#include <array>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>

#include "../../common/ai_belief/recursive_laplace.h"   // shared predict/MAP/Woodbury engine
#include "../../common/ai_belief/lidar_ray_factor.h"     // shared YOLO-independent LiDAR first-hit factor

namespace rc
{

// 2D footprint second-moment of a point band (centroid + principal axis + full extents). Kept as a small,
// model-independent geometry helper for the fitter's optional birth seed; NOT part of the fit itself
// (n_prims()==1, no footprint-moment factor). ok=false when too few points fell in the band.
struct FootprintMoment
{
    bool  ok = false;
    int   n = 0;
    float cx = 0.0f, cy = 0.0f;                  // 2D centroid (m)
    float ext_major = 0.0f, ext_minor = 0.0f;    // full extents (m), major ≥ minor
    float phi = 0.0f;                            // major-axis angle (rad), atan2 in (−π/2, π/2]
};

// The nearest room wall, as seen by THIS frame (ported from cabinet_concept, simplified — a fridge is a
// SINGLE box, so there is NO run / wall-id / collinear-merge / segment machinery). Supplied by the fitter
// from the room polygon (room_concept's `delimiting_polygon_x/y`, a NOMINAL model — no circularity between
// "the wall" and the fitted fridge). ok=false ⇒ the wall factor is inert this frame (free-standing).
struct WallRef
{
    bool            ok      = false;
    Eigen::Vector2f p       = Eigen::Vector2f::Zero();   // any point on the wall line (room frame, m)
    Eigen::Vector2f n       = Eigen::Vector2f::UnitX();  // UNIT normal, pointing INTO the room
    float           sigma_m = 0.02f;                     // wall position uncertainty (m)
};

// Appearance-based FRONT (door) cue: the room-frame direction the DOOR faces + a confidence in [0,1]. Emitted
// by HoodProjection::detect_front (project the FITTED box into the live ZED RGB, score vertical-edge /
// door-ness energy per visible face, pick the winner) and consumed by HoodBelief::resolve_front. Kept in
// the pure-Eigen belief header (no OpenCV/DSR) so both the belief resolver and the projection unit share it.
struct FrontCue
{
    float bearing_rad = 0.0f;   // room-frame yaw of the winning face's OUTWARD normal (direction the door faces)
    float confidence  = 0.0f;   // (max−second)/(max+eps) door-ness margin ∈ [0,1]; high when one face clearly wins
    // ★Room-frame bearing FROM THE FRIDGE TO THE OBSERVER. Not the same thing as bearing_rad, and the
    // distinction is the whole point: bearing_rad is a property of the OBJECT (which way the door faces),
    // while this is where the look came FROM. front_acc_ needs the latter to tell a genuinely new view
    // from the same view repeated — keying novelty on the cue direction would suppress a fresh viewpoint
    // that happens to agree. NaN ⇒ unknown ⇒ no novelty weighting (the legacy sum).
    float view_bearing_rad = std::numeric_limits<float>::quiet_NaN();
};

// Fitted hood state θ = [cx, cy, H, w, h, yaw] (room frame). See the file header for the field map.
struct HoodBeliefState
{
    float cx = 0.0f, cy = 0.0f, H = 1.70f, w = 0.60f, h = 0.60f, yaw = 0.0f;

    // Inert chart flag (kept so the fitter's `use_quotient` assignment compiles). A single asymmetric box
    // needs no symmetry-quotient optimisation chart, so the plain (identity) chart is the only path.
    inline static bool use_quotient = false;

    Eigen::Matrix<float, 6, 1> vec() const
    { return (Eigen::Matrix<float, 6, 1>() << cx, cy, H, w, h, yaw).finished(); }
    static HoodBeliefState from_vec(const Eigen::Matrix<float, 6, 1>& v)
    { return {v(0), v(1), v(2), v(3), v(4), v(5)}; }
};

// Belief parameters: the observation/mixture model, priors, and the transition + common-mode covariances.
// All are measurement-model stds/precisions (never σ-floors or gates). Several fields are carried INERT so
// the shared fitter's parameter-fill compiles unchanged; they have no effect on this single-box model and
// are noted as such.
struct HoodBeliefParams
{
    // Observation model
    float sigma_base_m    = 0.03f;   // base on-surface noise std (m); R = σ² (+ motion_var + …) per point
    float clutter_frac    = 0.10f;   // ε: prior weight of the uniform clutter component
    float clutter_scale_m = 0.15f;   // a point further than ~this from the box surface is likely clutter
    float model_sigma_m   = 0.010f;  // residual model/surface std (m) added in quadrature (R floor)

    // Priors (broad; only break the empty-cloud degeneracy)
    float prior_pos_std  = 0.30f;    // position prior std (m) on cx,cy
    float prior_size_std = 0.30f;    // (legacy/inert now that the size prior is split into footprint vs height)
    float prior_yaw_std  = 0.60f;    // broad yaw prior std (rad)

    // ── FOOTPRINT + HEIGHT prior (a standard fridge footprint ≈ 0.60×0.60 is TIGHT; HEIGHT varies a lot) ──
    // A Gaussian prior on the footprint DOFs (w,h≡depth) is strong (small std), so a partial front-only view
    // cannot let the unobserved depth float; the HEIGHT prior is deliberately BROAD (data-driven). These also
    // split the previously-uniform size entry of prior_cov_diag() into footprint (tight) vs height (broad).
    float prior_footprint_m   = 0.60f;   // mean of the w and h (depth) prior (m)
    // ★A HOOD IS NOT SQUARE IN PLAN, and one prior for both axes was a refrigerator fact. A fridge is
    // ~0.60 x 0.60 so a single tight footprint prior is right for it; a hood is ~0.90 ALONG the wall
    // and ~0.50 OUT from it. Pulling depth to the width's mean pushed the back face h/2 too far
    // back, which is measured directly as a wall gap: back_centre = c + (h/2)·outward, so every
    // centimetre of excess depth is half a centimetre of penetration the flush prior must undo —
    // and it is scaled by exp(−(gap/reach)²), so the further out it starts the weaker the pull.
    // Defaults to prior_footprint_m so any agent that IS square keeps its previous behaviour.
    float prior_depth_m = -1.0f;   // <0 ⇒ use prior_footprint_m (square footprint)
    float prior_footprint_std = 0.08f;   // TIGHT → strong footprint prior (m); λ = 1/std² folded into accumulate_extra
    float prior_height_m      = 2.05f;   // mean of H = the hood TOP above the floor (m)
    // Vertical extent of the hood body (m): the box spans [H − this, H]. NOT estimated — see the header
    // note. σ ≈ 0.12 m across domestic models, which is the honest uncertainty this constant stands in for.
    float vertical_extent_m   = 0.50f;
    float prior_height_std    = 0.30f;   // anchor std (m); stops the box top floating above the cloud (accumulate_extra 2a)
    // DEPTH-OBSERVABILITY prior (front-only-view depth-collapse fix): depth (h) is identifiable only when the cloud
    // spans the depth extent (front AND back seen). A front-only view is a thin ly-slab whose many points still drag
    // depth to the clamp (size common-mode caps σ, not the mean). Extra depth precision applied ∝ (1 − observed
    // depth-extent/footprint) → single-face view holds depth at the footprint prior; relaxes once the back is seen.
    float depth_unobs_precision = 1500.0f;  // extra 1/m² when depth is unobserved (single face). 0 = OFF
    float depth_obs_band_m      = 0.10f;    // ly-spread (m) below which the cloud counts as a single depth face
    // TOP anchor: the box top is floor-anchored, so a "free" upper boundary — extending it ABOVE the cloud costs
    // nothing (empty surface unpenalised), so it ratchets up on the over-segmentation junk tail above the fridge
    // (observed: H→2.37 for a 1.9 m mask). A firm TWO-SIDED anchor pins H to the observed robust cloud top
    // (frame.z_top_obs, the p97 top which sits BELOW the junk tail; the tail is separately faded via per-point R in
    // the fitter). Data still sets the LOWER bound (front points force H ≥ real top). Inert when z_top_obs < 0. 0 = OFF.
    float top_no_float_precision = 10000.0f; // 1/m² firm two-sided anchor pinning H → observed robust cloud top
    float top_no_float_margin_m  = 0.02f;   // upward allowance (m) added to the observed top as the anchor target

    // ── "IS THIS REALLY A FRIDGE?" plausibility + short-height prior (model-evidence mis-detection filter) ──
    // A fridge is SQUARE-ish in footprint (w≈h≈0.60) and TALL (H≈1.4–2.0 m). A YOLO mis-detection (e.g. a
    // ~70 cm elongated cabinet mislabelled "hood") contradicts these shape priors → it is poorly
    // explained by the fridge model. These parameters drive a CONTINUOUS plausibility score (no hard cut) and a
    // SOFT one-sided short-height prior. See fridge_plausibility() / accumulate_extra() height term.
    float plaus_aspect_scale      = 0.15f;   // aspect = |w−h|/(w+h); aspect_ok = exp(−(aspect/scale)²) → 1 square
    float plaus_size_scale        = 0.15f;   // size_ok = exp(−((w−fp)²+(h−fp)²)/(2·scale²)), fp = prior_footprint_m
    float plaus_height_min        = 1.20f;   // logistic centre (m): height_ok ≈ 0.5 here, →1 tall, →0 below ~1 m
    float plaus_height_soft       = 0.15f;   // logistic softness (m) of the height_ok falloff
    float plaus_fe_ref            = 2.0f;    // "healthy fridge fit" free-energy reference (NEEDS LIVE TUNING)
    float plaus_fe_scale          = 1.0f;    // fit_ok = exp(−max(0, FE−fe_ref)/fe_scale)
    // The ALTERNATIVE hypothesis "some other furniture-shaped box", against which the fridge hypothesis is
    // scored (fridge_log_evidence_ratio). Broad on footprint size and uninformative on aspect/height, so it is
    // the fridge hypothesis's tightness that earns the positive evidence — not an arbitrary 0.5 boundary.
    float plaus_alt_size_scale    = 0.60f;   // alternative footprint std (m) around prior_footprint_m
    // Short-height PRIOR (accumulate_extra): a one-sided GN factor nudging H up when H < plaus_height_min, with
    // precision GROWING as H falls further below (λ = gain·(deficit/soft)). Continuous covariance, NOT a clamp:
    // fitting a 70 cm cloud as a fridge then fights this prior → worse fit → lower plausibility. 0 = OFF.
    float plaus_height_prior_gain = 2000.0f; // precision (1/m²) per unit (deficit/soft) below plaus_height_min

    // ── WALL-FLUSH + WALL-PARALLEL factor (accumulate_wall; ported from cabinet_concept) ─────────────────
    // A fridge's BACK face rests against a room wall; the mask sees only front/sides, so depth (h) is
    // partial-view-degenerate. The precision is the marginal of a 2-component mixture {flush against this
    // wall, free-standing}: λ_flush = wall_precision·exp(−(gap/wall_reach_m)²)/(1/wall_precision+σ_wall²).
    // The exponential IS the flush component's Gaussian posterior weight given the back-to-wall gap — a
    // CONTINUOUS covariance, not a proximity gate: a genuine mid-room fridge finds no wall in range and the
    // factor fades to nothing (depth stays honestly wide). 0 = OFF.
    float wall_precision          = 400.0f;   // 1/m² at zero gap (≈1/σ² with σ≈5 cm)
    float wall_reach_m            = 0.15f;    // gap scale over which the flush hypothesis loses its weight
    float wall_parallel_precision = 200.0f;   // on (width-axis · wall-normal): back face parallel to the wall — 0 = OFF
    // WALL NO-CROSS (one-sided): resist the back face penetrating PAST the wall into the wall/exterior. Active only
    // when the gap goes negative (crossed) or within wall_no_cross_margin_m; precision GROWS with penetration
    // (λ = wall_no_cross_precision · how-far-past), leaving flush (gap≈0) untouched. Strong; continuous, no clamp.
    float wall_no_cross_precision = 2000.0f;  // 1/m² per m of penetration past the wall (>> wall_precision). 0 = OFF
    float wall_no_cross_margin_m  = 0.0f;     // interior margin (m) inside which the no-cross term activates
    // ── WALL as a COMPETING EXPLANATION (per-point mixture component) ────────────────────────────────────
    // A room wall is a surface another concept (room_concept) has already inferred, so a mask point lying on it
    // is ALREADY EXPLAINED and must not be re-explained as fridge surface. This is explaining-away, NOT an
    // exclusion zone: the wall enters the SAME per-point mixture as box/clutter, so (i) a wall point's box
    // responsibility collapses ⇒ it exerts no GN pull, and (ii) the marginal likelihood is already high there
    // WITHOUT a fridge ⇒ box_log_evidence_gain ≈ 0 ⇒ the hypothesis earns nothing. A real fridge is untouched:
    // its visible front face stands ~h off the wall plane, so those points stay with the box.
    // wall_explain_frac is the component's PRIOR weight — it should track how much the room is trusted; the
    // fitter scales it by the room's own confidence, so bad localization returns the fridge hypothesis its
    // freedom rather than silently vetoing detections. 0 = OFF (no wall component).
    // Door CLEARANCE prior (door_clearance_logprior): nats of preference, at full flush, for the door mode
    // pointing straight INTO the room over one pointing straight into the wall. Scaled by the flush mixture
    // weight, so it vanishes for a genuine mid-room fridge. 0 = OFF. A precision, not a gate.
    float door_clearance_gain = 3.0f;
    // ── INFERRED VOLATILITY (MODEL_HISTORY.md §3): "how fast does THIS object change?" ──────────────────
    // Q stops being a per-frame constant and becomes exp(ω), a per-DOF latent inferred from how much the
    // belief actually MOVES. This is the history stage: with a fixed Q, evidence is discarded exactly as
    // fast as it is acquired (σ_steady = (q/i)^¼ — a ratio of two RATES, with elapsed time absent), so hours
    // of touring leave the belief where seconds would, and an unobserved object's σ walks back to the prior
    // in ~6 minutes. Inferring ω fixes BOTH: retention τ ≈ Σ·exp(−ω) grows AND the floor (exp(ω)/i)^¼ falls.
    // Stiffness is then EARNED by evidence rather than imposed by a clamp, and the hyperprior pull-back keeps
    // it recoverable — a genuinely moved fridge drives ω back up and adapts in seconds.
    bool  volatility_infer   = false;   // OFF ⇒ Q is the constant process_std_* exactly as before
    float volatility_lr      = 0.02f;   // gradient step on ω per MEASURED cycle
    float volatility_sigma   = 2.0f;    // hyperprior width σ_ω: how far evidence may pull ω from ω₀
    float volatility_omega_min = -18.0f; // numerical floor on ω (exp(−18) ≈ 1.5e-8 m²) — not a tuning knob
    float wall_explain_frac  = 0.25f;   // π_wall: prior weight of "this point belongs to a known room wall"
    float wall_explain_sigma_m = 0.05f; // extra std (m) on the wall plane: room-polygon + localization slack

    // Temporal transition (predict): rigid + static ⇒ small process noise per frame.
    float process_std_m   = 0.005f;  // cx,cy,H,w,h per-frame process std (m)
    float process_std_yaw = 0.01f;   // yaw per-frame process std (rad)

    // Per-frame COMMON-MODE error (the error SHARED by all points of one mask; does NOT average out). The
    // frame's information SATURATES at this covariance (Woodbury), so N correlated points can't collapse σ.
    float common_mode_pos_std  = 0.03f;  // shared position error (m); pose-chain cov adds to it
    float common_mode_size_std = 0.02f;  // shared size error H,w,h (m)
    float common_mode_yaw_std  = 0.03f;  // shared yaw error (rad)

    // Optimiser
    int   gn_iters = 4;              // Gauss-Newton iterations per frame
    float fd_eps   = 1e-3f;          // finite-difference step for SDF Jacobians (m / rad)
    float jac_slope_clamp = 5.0f;    // clamp per-DOF SDF slope (guards a seam-point info spike; recipe §2)

    // ── INERT compatibility fields (set by the shared fitter; no effect on this single-box model) ────────
    float top_thickness = 0.03f;     // legacy (no top slab here)
    float leg_radius    = 0.025f;    // legacy (no legs here)
    float pixel_sigma_over_f     = 0.0015f;   // (footprint-residual path — OFF)
    float depth_sigma0_m         = 0.006f;
    float depth_sigma_range_coef = 0.004f;
    bool  footprint_residual     = false;
    float depth_bias_std         = 0.015f;
    float depth_scale_std        = 0.010f;
    float coverage_precision  = 0.0f;
    float coverage_robust_c_m = 0.15f;
    float free_space_precision = 0.0f;
    float footprint_moment_precision = 0.0f;
    float footprint_moment_completeness_gain = 0.0f;
    float footprint_moment_min_completeness  = 0.02f;
};

// One fitted frame's evidence: room-frame points and per-point measurement variance R (m²). Empty R ⇒ base
// σ² for every point. The chain/range/motion common-mode variances are folded into common_mode_inv_diag.
struct HoodFrame
{
    std::vector<Eigen::Vector3f> points;
    std::vector<float>           R;                 // per-point measurement variance (m²); empty ⇒ σ_base²
    Eigen::Vector3f              cam_origin = Eigen::Vector3f::Zero();  // (inert: footprint-residual path OFF)
    bool                         has_rays   = false;                    // (inert)
    std::vector<float>           point_azim;                            // (inert)
    float chain_cov_xx   = 0.0f;   // extra shared position variance (m²) from the pose chain + range + motion (cx)
    float chain_cov_yy   = 0.0f;   // ...                                                                      (cy)
    float chain_cov_yaw  = 0.0f;   // extra shared yaw variance (rad²): range + grazing-view cap + ego-motion
    float chain_cov_size = 0.0f;   // extra shared SIZE variance (m²) on H,w,h: range freezes afar + ego-motion
    float moment_extra_var = 0.0f; // (inert: no footprint-moment factor on a single asymmetric box)

    // Robust (high-quantile) observed cloud TOP z (m); the box top must not FLOAT above it (top-no-float factor).
    // <0 ⇒ not supplied this frame (factor inert). See accumulate_extra (2d).
    float z_top_obs = -1.0f;

    // The nearest room wall for the wall-flush / wall-parallel factor (see WallRef). ok=false ⇒ inert.
    WallRef wall;

    // YOLO-INDEPENDENT LiDAR channel: range returns that fall on the box (room frame, sensor origin +
    // endpoints). Sphere-traced against THIS belief's own SDF by the shared factor. precision==0 ⇒ skipped.
    rc::ai::LidarRays lidar;
    // Additional per-device ray-sets (e.g. the low "bpearl" LiDAR); each keeps its OWN origin.
    std::vector<rc::ai::LidarRays> lidar_extra;
};

// The hood generative model wired onto the shared rc::ai recursive-Laplace engine. Supplies only the
// model hooks the engine calls (box SDF, responsibilities, Jacobian, constraints, and the Q/F/prior/common-
// mode diagonals) + the optional LiDAR range factor. N = 6, static (transition = I), one primitive.
class HoodBelief
{
public:
    static constexpr int N = 6;
    using State = HoodBeliefState;
    using Frame = HoodFrame;

    HoodBelief() = default;
    HoodBelief(const HoodBeliefState& s, const HoodBeliefParams& p) : state_(s), params_(p)
    { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); }

    const HoodBeliefState&     state()      const { return state_; }
    const Eigen::Matrix<float, 6, 6>&  covariance() const { return Sigma_; }
    const HoodBeliefParams&    params()     const { return params_; }
    void set_state(const HoodBeliefState& s) { state_ = s; }
    void set_params(const HoodBeliefParams& p) { params_ = p; }

    // Room interior reference (polygon centroid). Used ONLY to pick which of the two depth faces is the
    // BACK face (the one whose outward normal points AWAY from the interior) — the face driven onto the wall.
    void set_room_interior(const Eigen::Vector2f& c) { room_interior_ = c; has_room_interior_ = true; }

    // Monitor instrumentation for the wall factor.
    float last_wall_gap()    const { return dbg_wall_gap_; }     // back face → wall signed gap (m)
    float last_wall_lambda() const { return dbg_wall_lambda_; }  // applied flush precision (1/m²)
    float last_wall_resp()   const { return dbg_wall_resp_; }    // mean per-point WALL responsibility (0..1):
                                                                 // →1 ⇒ the cloud is a wall, not a fridge
    // Stage the wall the per-point mixture explains with, WITHOUT running an update. Only self_test needs this
    // (production stages it via update(frame)); it lets the test score clouds against a known wall directly.
    void set_frame_wall_for_test(const WallRef& w) { point_wall_ = w; }
    // Centre of the BACK face (the one against the wall): (cx,cy) + (h/2)·outward, outward = the depth axis
    // sign pointing AWAY from the room interior. Public so the fitter can query the nearest wall at it.
    Eigen::Vector2f back_centre(const HoodBeliefState& s) const;

    // ── Inference (delegated to the shared engine) ────────────────────────────
    // Stash this frame's wall BEFORE the engine runs: the per-point mixture's wall component (explaining away)
    // needs it, and mixture_unnormalized is a const hook the engine calls with no access to the frame.
    // ★A degenerate Σ is SELF-SEALING and must never survive a cycle. Σ = 0 (or NaN/inf) makes the tracker's
    // Mahalanobis gate S = P + R²I collapse, so NO detection can ever associate again, so the belief can never
    // be updated, so Σ can never recover — the instance freezes at whatever garbage state it held and sits
    // there forever. Live (etc/ai2_log.csv 20:27): a volatility blow-up drove Q to inf, Σ through the inverse
    // to exactly 0 with σ_yaw = inf, and the instance took 3 measured frames out of 946 before freezing as a
    // 0.126 m slab at −102°. Re-seeding from the prior costs one cycle of confidence and breaks the trap.
    // (Belongs in common/ai_belief eventually — every agent on this engine has the same exposure.)
    void sanitize_covariance()
    {
        bool bad = not Sigma_.allFinite();
        for (int i = 0; i < N and not bad; ++i) bad = not (Sigma_(i, i) > 0.0f);
        if (not bad) return;
        Sigma_.setZero();
        Sigma_.diagonal() = prior_cov_diag();
        omega_ = omega0_; vol_n_ = 0.0f; vol_s_.setZero();   // the volatility estimate is suspect too
        std::printf("HoodBelief: DEGENERATE Σ (non-finite or non-positive diagonal) — re-seeded from prior\n");
    }

    float update(const HoodFrame& frame)
    {
        point_wall_ = frame.wall;
        sanitize_covariance();
        const Eigen::Matrix<float, 6, 1> before = state_.vec();
        const float fe = ai::update<N>(*this, state_, Sigma_, prior_mean_, frame);
        sanitize_covariance();
        // ω is inferred ONLY on a cycle that actually took a measurement. A predict-only cycle has δθ = 0,
        // which under the update below would read as "this object never moves" and drive ω down — the same
        // repetition-is-not-independence error that has bitten this codebase four times.
        update_volatility(state_.vec() - before);
        return fe;
    }
    // Per-DOF log-volatility ω (Q = exp ω) and its hyperprior mean ω₀ — the accumulated experience.
    const Eigen::Matrix<float, 6, 1>& log_volatility() const { return omega_; }
    // Retention time constant per DOF: τ = Σ_ii / exp(ω_i) frames — "how long this DOF remembers".
    Eigen::Matrix<float, 6, 1> retention_frames() const;
    void  predict()                               { sanitize_covariance(); ai::predict<N>(*this, Sigma_, state_, prior_mean_); }
    void  inflate_for_age(float dt_s, float dt_nominal_s)
    { ai::inflate_for_age<N>(*this, Sigma_, state_, prior_mean_, dt_s, dt_nominal_s); }
    Eigen::Matrix<float, 6, 6> predicted_information(const std::vector<Eigen::Vector3f>& pts, float R) const
    { return ai::predicted_information<N>(*this, state_, pts, R); }

    // ── Generative-model hooks (called by the engine; also used as the SDF API) ─
    float sdf_box(const Eigen::Vector3f& p, const HoodBeliefState& s) const;
    float sdf_prim(const Eigen::Vector3f& p, const HoodBeliefState& s, int prim) const;   // prim == 0
    float sdf_compound(const Eigen::Vector3f& p, const HoodBeliefState& s) const { return sdf_box(p, s); }
    Eigen::Matrix<float, 6, 1> sdf_jacobian(const Eigen::Vector3f& p, const HoodBeliefState& s, int prim) const;
    // Soft responsibilities: [box, clutter, wall] (sum = 1) at measurement variance R. The engine reads only
    // index 0 (n_prims()==1); the rest exist so the marginal likelihood is honest about what else explains a point.
    std::array<float, 3> responsibilities(const Eigen::Vector3f& p, const HoodBeliefState& s, float R) const;
    // Un-normalised mixture components [box, clutter, wall] + their sum (= marginal likelihood p(point|model)).
    float mixture_unnormalized(const Eigen::Vector3f& p, const HoodBeliefState& s, float R,
                               std::array<float, 3>& u) const;
    // Mean per-point NLL of the SAME mixture with the BOX COMPONENT REMOVED — i.e. how well the scene is
    // explained WITHOUT positing a fridge here. mean_energy() minus this is the log Bayes factor for "there is
    // a fridge here", the quantity birth/existence evidence should use: a mask lying on a known wall is already
    // fully explained, so the factor is ≈0 no matter how well a box happens to fit it.
    float mean_energy_no_box(const std::vector<Eigen::Vector3f>& pts, float R) const;
    // log p(points | wall+clutter+box) − log p(points | wall+clutter) ≥ 0-ish. Large ⇒ the fridge hypothesis
    // genuinely adds explanatory power; ≈0 ⇒ something already in the scene model accounts for these points.
    float box_log_evidence_gain(const std::vector<Eigen::Vector3f>& pts, const HoodBeliefState& s,
                                float R) const
    { return mean_energy_no_box(pts, R) - mean_energy(pts, s, R); }
    // Mean per-point mixture NEGATIVE LOG-LIKELIHOOD (clutter INCLUDED) = the honest free energy (rises with
    // misfit, unlike the engine's surface-only return). Used for the published/logged FE + convergence.
    float mean_energy(const std::vector<Eigen::Vector3f>& pts, const HoodBeliefState& s, float R) const;
    void  apply_constraints(HoodBeliefState& s) const;
    void  canonicalize(HoodBeliefState&) const {}   // asymmetric box: no symmetry fold

    int   gn_iters() const { return params_.gn_iters; }
    int   n_prims()  const { return 1; }             // one box (clutter is the +1 mixture component)
    float sigma2()   const { return params_.sigma_base_m * params_.sigma_base_m + params_.model_sigma_m * params_.model_sigma_m; }
    Eigen::Matrix<float, 6, 6> transition() const { return Eigen::Matrix<float, 6, 6>::Identity(); }  // static
    Eigen::Matrix<float, 6, 1> process_noise_diag() const;
    Eigen::Matrix<float, 6, 1> prior_cov_diag() const;
    Eigen::Matrix<float, 6, 1> common_mode_inv_diag(const HoodFrame& frame) const;
    // Extra GN factor (engine calls it via C++23 requires): the YOLO-independent LiDAR first-hit range factor.
    void accumulate_extra(const HoodBeliefState& s, const HoodFrame& f,
                          Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const;

    // ── Reported uncertainty / orientation-mode shims ────────────────────────────
    // The per-point fit has NO discrete w↔h mode (the box is asymmetric), so resolve_orientation / flip_evidence /
    // mode_posterior stay trivial (kept so the shared fitter/scene-graph compile). The GENUINE remaining ambiguity
    // is which way the DOOR faces — a square-ish, wall-flush fridge leaves the yaw discrete-ambiguous. That is
    // resolved by APPEARANCE (resolve_front, below), and its residual entropy is folded into the reported σ_yaw.
    float mode_posterior()    const { return 0.0f; }
    float flip_evidence()     const { return 0.0f; }
    bool  resolve_orientation(const std::vector<Eigen::Vector3f>&, float, float = 1.0f) { return false; }

    // ── FRONT (door) yaw resolver — sequential Bayes over the discrete door-facing modes ─────────────
    // Fold the appearance FrontCue (HoodProjection::detect_front) into a per-instance accumulator over the
    // discrete yaw candidates the footprint/wall leave (0/90/180/270). Each cycle, add w·(front_dir_k · cue_dir)
    // to each candidate's evidence — w = confidence·evidence_weight — clamped to ±kFrontClamp so a settled belief
    // can still RECANT if the fridge is physically turned. Adopt the argmax mode (rotate yaw; swap w↔h + Σ rows/
    // cols on a 90°/270° adoption). A clearly RECTANGULAR footprint is only 180°-ambiguous — the 90°/270° modes
    // would swap the fitted w↔h and fight the geometric fit + wall-flush, so they are excluded (geometry-derived
    // candidate restriction, NOT a belief threshold). Mirrors ChairBelief::resolve_orientation. Returns true iff
    // it adopted a new orientation. Door face = local −Y.
    bool  resolve_front(const FrontCue& cue, float evidence_weight = 1.0f);
    // Adopt the best door mode from the accumulated appearance evidence PLUS the geometric clearance prior,
    // with NO new appearance cue. Call every cycle: it is the only path that can fix a fridge whose door faces
    // its own wall, which detect_front can never report (it needs 2 visible faces and never scores the back).
    bool  resolve_front_geometric();
    // Per-mode log-prior from door CLEARANCE: a door cannot open into the wall the fridge is attached to.
    // Recomputed from current geometry wherever the mode is scored — a PRIOR, never accumulated.
    std::array<float, 4> door_clearance_logprior() const;
    // Posterior over the 4 discrete front modes = softmax(front_acc_) with disallowed modes zeroed (see
    // allowed_modes). Front diagnostics: the winning mode's posterior mass (→1 once resolved) and its index.
    std::array<float, 4> front_posterior() const;
    float front_confidence() const;   // max posterior mass (peakedness of the door decision)
    int   front_mode()       const;   // argmax mode index (0 = current believed door direction)

    // REPORTED yaw variance: within-mode Σ(5,5) PLUS the variance of the discrete door-mode offset Δ_k = k·π/2
    // under the front posterior. Undecided (accumulator ~0) → a wide, honest σ_yaw; resolved → collapses to Σ(5,5).
    float yaw_marginal_var()  const;
    Eigen::Matrix<float, 6, 6> covariance_reported() const;

    // ── Diagnostic getters (no footprint-moment / coverage / vacate on this model → all zero) ────────────
    int   last_vacate_beams()  const { return 0; }
    int   last_coverage_pts()  const { return 0; }
    int   last_moment_pts()    const { return 0; }
    float dbg_yaw_after_points() const { return state_.yaw; }
    float dbg_yaw_after_moment() const { return state_.yaw; }
    float dbg_moment_aniso()     const { return 0.0f; }
    float dbg_moment_r_yaw()     const { return 0.0f; }
    float dbg_moment_dyaw()      const { return 0.0f; }
    float dbg_moment_ext_major() const { return 0.0f; }
    float dbg_moment_ext_minor() const { return 0.0f; }
    float dbg_moment_phi()       const { return 0.0f; }
    int   dbg_moment_pts()       const { return 0; }

    // 2D footprint second-moment of the points whose z falls in [z_lo, z_hi]. Geometry helper for the
    // fitter's optional birth seed (gated OFF by default); pure, no state.
    static FootprintMoment footprint_moment(const std::vector<Eigen::Vector3f>& pts, float z_lo, float z_hi);

    // ── "IS THIS REALLY A FRIDGE?" plausibility (model-evidence mis-detection filter) ─────────────────
    // Continuous plausibility ∈ (0,1] that this fitted box is a fridge, as a product of soft factors (aspect,
    // footprint size, height, data-fit) — NOT a hard threshold. ~1 for a proper 0.6×0.6×1.75 low-FE fridge,
    // →0 for an elongated/short one or a poor fit. `fe` = the belief's clutter-inclusive mean_energy. The
    // static overload is pure (self_test / birth-candidate use); the member evaluates the current state.
    static float fridge_plausibility(const HoodBeliefState& s, float fe, const HoodBeliefParams& p);
    float fridge_plausibility(float fe) const { return fridge_plausibility(state_, fe, params_); }
    // Shape LOG-EVIDENCE RATIO (nats): log p(θ | fridge) − log p(θ | other furniture), using NORMALISED
    // densities for both hypotheses. This — not `plausibility − 0.5` — is the quantity that may be summed into
    // an existence log-odds: fridge_plausibility is an UN-normalised product of exponentials whose 0.5 level is
    // arbitrary, so a perfectly ordinary 0.77×0.63×1.52 fridge scored 0.30 and was driven to deletion. Positive
    // ⇒ the fridge hypothesis explains this shape better than generic furniture. See [[hood-table-geometry-churn]].
    static float fridge_log_evidence_ratio(const HoodBeliefState& s, const HoodBeliefParams& p);
    float fridge_log_evidence_ratio() const { return fridge_log_evidence_ratio(state_, params_); }
    // Birth-candidate plausibility from a raw mask cloud (BEFORE any fit): footprint aspect + size from the
    // 2-D principal extents, and a rough height from the cloud z-range. Scales the tracker's birth evidence so
    // an elongated/short candidate never accumulates enough to birth. fit term = 1 (no fit yet). Pure/static.
    static float candidate_plausibility(const std::vector<Eigen::Vector3f>& pts, const HoodBeliefParams& p);
    // Soft SINGLETON inhibition (a kitchen has ~one fridge). Given every instance's accumulated plausibility
    // evidence + existence probability, returns each instance's per-cycle existence-logodds delta:
    //   shape_i   = gain·tanh(plaus_evidence_i / clamp)                       (+ support / − decay by shape)
    //   penalty_i = inhibition·Σ_{j≠i} p_exists_j·[plaus_evidence_j > plaus_evidence_i]   (stronger inhibit weaker)
    //   delta_i   = shape_i − penalty_i
    // SOFT + bounded: a genuinely strongly-supported second fridge survives; the weaker mis-detection decays.
    static std::vector<float> singleton_existence_deltas(const std::vector<float>& plaus_evidence,
                                                         const std::vector<float>& p_exists,
                                                         float gain, float inhibition, float clamp);

    // ── Verification ──────────────────────────────────────────────────────────
    static bool self_test();

private:
    // Wall-flush + wall-parallel structural factor (ported from cabinet_concept). Inert if f.wall.ok==false.
    void accumulate_wall(const HoodBeliefState& s, const HoodFrame& f,
                         Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const;
    // Posterior weight of the "flush against this wall" mixture component: exp(−(gap/reach)²). 0 ⇒ inert.
    float flush_weight(const HoodBeliefState& s, const HoodFrame& f) const;

    // Per-point WALL mixture component (explaining away): its available prior weight this frame, and its
    // un-normalised density at p. Both 0 when no room polygon is known ⇒ the mixture is box+clutter as before.
    float wall_component_frac() const;
    float wall_component(const Eigen::Vector3f& p, float R, float pw) const;

    // Which of the 4 discrete door modes {0,90,180,270} the current footprint leaves open. A rectangular
    // footprint (|w−h|/(w+h) > kRectAspect) allows only {0,180} (a 90° swap would fight the fitted w↔h);
    // a near-square one allows all four. Shared by resolve_front and yaw_marginal_var so both agree.
    std::array<bool, 4> allowed_modes() const;
    // Adopt door mode k: rotate yaw by k·π/2 and, for k∈{1,3}, swap w↔h in the state AND swap Σ rows/cols 3↔4
    // (and prior_mean_ 3↔4) so the covariance stays consistent with the relabelled axes.
    void apply_mode_rotation(int k);

    // Door-facing accumulator: per-candidate accumulated appearance evidence (higher = more door-evidence),
    // relative to the current mode (front_acc_[0] re-baselined to the max on each adoption). All-zero = undecided.
    static constexpr float kRectAspect  = 0.10f;   // |w−h|/(w+h) above which the footprint is treated rectangular
    std::array<float, 4>   front_acc_ = {0.0f, 0.0f, 0.0f, 0.0f};
    // Mode-evidence budget already spent per observer-bearing bin, exactly as ChairBelief does for its
    // 4-way yaw modes. Without it front_acc_ SUMS the same static appearance evidence every frame, so the
    // door-mode confidence grows with dwell time rather than with information — the defect this fleet keeps
    // rediscovering (see CONCEPT_AGENT_INVARIANTS.md, mistake I).
    static constexpr int kFrontViewBins = 24;          // 15° bins around the fridge
    std::array<float, kFrontViewBins> front_view_spent_{};
    float front_view_budget_ = 3.0f;                   // same scale as ChairBeliefParams::view_budget

    HoodBeliefState    state_;
    HoodBeliefParams   params_;
    Eigen::Matrix<float, 6, 6> Sigma_ = Eigen::Matrix<float, 6, 6>::Identity();  // posterior covariance
    Eigen::Matrix<float, 6, 1> prior_mean_ = Eigen::Matrix<float, 6, 1>::Zero(); // transition prior mean

    Eigen::Vector2f room_interior_     = Eigen::Vector2f::Zero();   // polygon centroid: picks the BACK face
    bool            has_room_interior_ = false;
    // This frame's nearest wall, stashed by update() so the const mixture hook can use it as a competing
    // explanation. ok==false (no room polygon) ⇒ the wall component is inert and the mixture is box+clutter.
    WallRef         point_wall_;
    // Inferred per-DOF log-volatility and its hyperprior mean. omega_ starts AT omega0_, so with
    // volatility_infer off the model is bit-identical to the constant-Q original.
    Eigen::Matrix<float, 6, 1> omega_  = Eigen::Matrix<float, 6, 1>::Zero();
    Eigen::Matrix<float, 6, 1> omega0_ = Eigen::Matrix<float, 6, 1>::Zero();
    bool                       omega_init_ = false;
    // Conjugate sufficient statistics for the volatility estimate: discounted observation count and per-DOF
    // discounted sum of squared steps. Q = (n₀·Q₀ + S)/(n₀ + n) — a weighted average, so it cannot diverge.
    float                      vol_n_ = 0.0f;
    Eigen::Matrix<float, 6, 1> vol_s_ = Eigen::Matrix<float, 6, 1>::Zero();
    void update_volatility(const Eigen::Matrix<float, 6, 1>& dtheta);
public:
    // self_test only: drive one volatility step directly, so the test exercises the production update.
    void update_volatility_for_test(const Eigen::Matrix<float, 6, 1>& d) { update_volatility(d); }
private:
    mutable float   dbg_wall_gap_    = 0.0f;
    mutable float   dbg_wall_lambda_ = 0.0f;
    mutable float   dbg_wall_resp_   = 0.0f;   // mean wall responsibility over the last scored cloud
};

}  // namespace rc
