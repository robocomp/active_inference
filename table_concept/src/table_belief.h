/*
 * table_belief.h  —  AI2 table belief (see TABLE.md)
 *
 * A from-scratch, active-inference-faithful replacement for the accreted fit/belief path. ONE compound
 * generative model {top slab + 4 derived legs + clutter} with SOFT per-point responsibilities (a mixture,
 * no hard min()/threshold), inferred by a recursive Gaussian (variational-Laplace) filter that carries a
 * FULL 6×6 covariance. Pure Eigen, no torch, no DSR — unit-testable in isolation (see self_test()).
 *
 * State θ = [cx, cy, H, w, h, yaw]  (6 DOF). Legs are DERIVED (rigidly attached at the outer edge), not a
 * free DOF: 4 cylinders at (±(w/2−LEG_RADIUS), ±(h/2−LEG_RADIUS)), length H−TOP_THICKNESS. Legs stay in
 * the SDF so leg observations inform w/h/yaw.
 */

#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>

#include "../../common/ai_belief/recursive_laplace.h"   // shared predict/MAP/Woodbury engine
#include "../../common/ai_belief/lidar_ray_factor.h"     // shared YOLO-independent LiDAR first-hit factor

namespace rc
{

// 2D footprint second-moment of a top-band point cloud (yaw + full extent from the tabletop's own shape).
//
// Fields: 2D centroid, principal-axis angle (major axis, wrapped to (−π/2, π/2]), and the FULL extents of the
// equivalent uniform rectangle (√(12·λ) of the inertia-tensor eigenvalues). ok=false when too few points fell
// in the band. Model-independent (the only model input is the z-band that selects the tabletop plane) — this
// is the global statistic that breaks the clutter-escape trap.
struct FootprintMoment
{
    bool  ok = false;
    int   n = 0;
    float cx = 0.0f, cy = 0.0f;   // 2D centroid (m)
    float ext_major = 0.0f, ext_minor = 0.0f;   // full extents (m), major ≥ minor
    float phi = 0.0f;             // major-axis angle (rad), atan2 in (−π/2, π/2]
};

// Fitted table state θ = [cx, cy, H, w, h, yaw] (room frame; H = tabletop height, canonical w≥h).
struct TableBeliefState
{
    float cx = 0.0f, cy = 0.0f, H = 0.75f, w = 1.0f, h = 0.6f, yaw = 0.0f;

    Eigen::Matrix<float, 6, 1> vec() const { return (Eigen::Matrix<float, 6, 1>() << cx, cy, H, w, h, yaw).finished(); }
    static TableBeliefState from_vec(const Eigen::Matrix<float, 6, 1>& v)
    { return {v(0), v(1), v(2), v(3), v(4), v(5)}; }
};

// Belief parameters: fixed geometry, the observation/mixture model, priors, and the transition +
// common-mode covariances. All are measurement-model stds/precisions (never σ-floors or gates).
struct TableBeliefParams
{
    // Geometry (fixed, not fitted)
    float top_thickness = 0.03f;
    float leg_radius    = 0.025f;

    // Observation model
    float sigma_base_m    = 0.03f;   // base on-surface noise std (m); R = σ² (+ motion_var + …) per point
    float clutter_frac    = 0.10f;   // ε: prior weight of the uniform clutter component
    float clutter_scale_m = 0.12f;   // a point further than ~this from every surface is likely clutter

    // Coverage / traction factor: grow-only extent pull from clutter-ceded on-plane mask points. 0 = OFF.
    //
    // (EXISTENCE_BELIEF_PLAN.md, table_1.png.) On-plane mask points the mixture ceded to the CLUTTER component
    // still exert a robust, GROW-ONLY pull on the top slab, so a model UNDER-covering a large mask is pulled
    // out to explain it instead of parking with the excess dumped to clutter for free (the escape-valve).
    // Weight = coverage_precision·(top-plane compat)·(clutter resp)·Cauchy(outside dist). Self-bounded
    // (sdf→0 when covered), on-plane-gated (off-plane contamination ignored), reclaims exactly the escaped
    // points. The honest anti-contamination bound is the free-space carve; this is the robust interim.
    float coverage_precision  = 0.0f;
    float coverage_robust_c_m = 0.15f;

    // Free-space / VACATE factor: shrink-only counter-force that BOUNDS the grow-only coverage term. 0 = OFF.
    //
    // (EXISTENCE_BELIEF_PLAN.md Step 4.) A LiDAR beam that passes THROUGH the top-slab z-band (endpoint beyond
    // the far face) proves that crossing is empty, so its midpoint (inside the slab, sdf_top < 0) exerts a
    // robust pull that retreats the boundary. Occupy-where-masked (coverage) + vacate-where-through (this)
    // settle the extent where camera and LiDAR AGREE — coverage can no longer inflate onto clutter. Reads the
    // frame's LiDAR endpoints/origin (same sweep the range factor uses); saturates via the engine common-mode.
    float free_space_precision = 0.0f;

    // Footprint SECOND-MOMENT factor: yaw/extent from the top surface's own 2D shape. 0 = OFF. (tables_5.png.)
    //
    // The per-point SDF mixture cannot rotate/resize a table whose dense mask is bigger or yawed than the
    // current box: interior top points have ∂sdf/∂yaw≈∂sdf/∂w≈0 (flat face), and the outboard rim/leg points
    // that DO carry the signal fall outside the box → the mixture routes them to CLUTTER (zero gradient). The
    // escape is a GLOBAL statistic: the top-band cloud's 2D centroid + inertia tensor is a sufficient statistic
    // of the filled-rectangle model. Its principal-axis angle measures yaw, its eigen-extents (full = √(12·λ))
    // measure w,h — folded as a linear Gaussian factor on (w,h,yaw) into the SAME GN normal equations (so the
    // engine's range-driven common-mode caps its per-frame authority — a far view still can't rotate a
    // converged table; no manual gate). yaw precision scales with the extent ANISOTROPY (ext_major−ext_minor)/
    // (sum): a near-square footprint carries no orientation info → 0 pull (resolve_orientation owns that flip).
    // NOT applied to cx,cy (a partial view's centroid is biased; the mixture already fixes position).
    float footprint_moment_precision = 0.0f;

    // Priors
    float prior_size_std = 0.30f;  // broad size prior std (m) on w,h,H — only breaks the empty-cloud degeneracy
    // Temporal transition (predict): rigid + static ⇒ small process noise per frame.
    float process_std_m   = 0.005f;
    float process_std_yaw = 0.01f;

    // Per-frame COMMON-MODE error: the error SHARED by all points of one mask (localization + mask
    // boundary + deprojection), which does NOT average out over points. The frame's information
    // SATURATES at this covariance (marginalised, Woodbury), so N≈10⁴ correlated points can no longer
    // collapse σ — the posterior stays calibrated. These are measurement-model stds, not a σ floor.
    float common_mode_pos_std  = 0.03f;  // shared position error (m); chain_cov (pose) adds to it
    float common_mode_size_std = 0.02f;  // shared size error w,h,H (m)
    float common_mode_yaw_std  = 0.03f;  // shared yaw error (rad)

    // Optimiser
    int   gn_iters = 4;            // Gauss-Newton iterations per frame
    float fd_eps   = 1e-3f;        // finite-difference step for SDF Jacobians (m / rad)
};

// One fitted frame's evidence: room-frame points and per-point measurement variance R (m²). Pass an empty
// R vector to use the base σ² for every point.
struct TableFrame
{
    std::vector<Eigen::Vector3f> points;
    std::vector<float>           R;        // per-point measurement variance (m²); empty ⇒ σ_base² for all
    float chain_cov_xx = 0.0f;             // extra shared position variance (m²) from the pose chain (cx)
    float chain_cov_yy = 0.0f;             // ...                                                      (cy)
    float chain_cov_yaw = 0.0f;            // extra shared yaw variance (rad²) — grows with view range so a
                                           // distant, vague mask cannot rotate a converged table
    float moment_extra_var = 0.0f;         // SHARED per-frame variance (m²) added to the footprint-moment
                                           // measurement: ego-motion corruption + a gentle range term. Makes the
                                           // moment ACCUMULATE over frames (a moving/near-far view backs off)
                                           // instead of snapping the belief to each frame's noisy footprint.
    float chain_cov_size = 0.0f;           // extra shared SIZE variance (m²) on w,h,H — grows with view range so
                                           // a distant, vague mask cannot RESHAPE/inflate a converged table (the
                                           // per-frame info SATURATES lower with range → geometry freezes; only
                                           // existence/occupancy survives afar). Caps the coverage grow-term too.

    // YOLO-INDEPENDENT LiDAR channel: range returns that fall on the table (room frame, sensor origin +
    // endpoints). Sphere-traced against THIS belief's own SDF by the shared factor. precision==0 ⇒ skipped.
    rc::ai::LidarRays lidar;
};

// The table generative model wired onto the shared rc::ai recursive-Laplace engine. The Bayesian math
// (predict / GN-MAP / Woodbury) lives in common/ai_belief/recursive_laplace.h; this class supplies only
// the table-specific MODEL hooks the engine calls (SDF, responsibilities, Jacobian, constraints, the
// canonical w≥h fold, and the Q/F/prior/common-mode diagonals). N = 6, static (transition = I).
class TableBelief
{
public:
    static constexpr int N = 6;
    using State = TableBeliefState;
    using Frame = TableFrame;

    TableBelief() = default;
    TableBelief(const TableBeliefState& s, const TableBeliefParams& p) : state_(s), params_(p)
    { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); }

    const TableBeliefState&            state()      const { return state_; }
    const Eigen::Matrix<float, 6, 6>&  covariance() const { return Sigma_; }
    const TableBeliefParams&           params()     const { return params_; }
    void set_state(const TableBeliefState& s) { state_ = s; }
    void set_params(const TableBeliefParams& p) { params_ = p; }

    // ── Discrete orientation mode (near-square yaw disambiguation, TABLE.md) ──────────────
    // The 6×6 Σ carries only the WITHIN-mode yaw width (~1°). For a near-square footprint the two
    // classes [(w,h,ψ)] and [(h,w,ψ)] (a w↔h swap ≡ a 90° rotation) have near-equal data energy, an
    // ambiguity a unimodal Gaussian cannot hold — so the per-frame MAP used to SNAP 90° between them.
    // resolve_orientation() is a sequential Bayesian comparison of the two modes; mode_posterior() is
    // p of the ALTERNATIVE (swapped) mode; the REPORTED yaw variance inflates Σ(5,5) by p(1−p)(π/2)²
    // so a still-ambiguous table reports an honest ~45° until an orbit resolves it.
    float mode_posterior()  const;                          // p(alternative mode) = σ(−flip_evidence_)
    float yaw_marginal_var() const;                         // Σ(5,5) + p(1−p)(π/2)²  (rad²)
    Eigen::Matrix<float, 6, 6> covariance_reported() const; // Σ with the yaw marginal folded into (5,5)
    float flip_evidence()   const { return flip_evidence_; }
    // Mean per-point data energy (NLL proxy) at an ARBITRARY state — for the mode-comparison hypothesis test.
    float mean_energy(const std::vector<Eigen::Vector3f>& pts, const TableBeliefState& s, float R) const;
    // One sequential-comparison step over the discrete mode {current, w↔h-swapped}: accumulate the
    // per-frame log-evidence (E_swap − E_now) and adopt the lower-accumulated-energy mode (boundary at zero
    // accumulated evidence = the MAP over the mode; NO tuned threshold). On a flip, swaps w↔h in the state
    // AND rows/cols 3↔4 of Σ. Returns true iff it flipped this call. Call once per fresh frame after update().
    // evidence_weight ∈ [0,1] scales this frame's contribution to the mode accumulator — the fitter passes a low
    // value while the robot is moving (ego-motion-corrupted / split masks must not vote on the discrete mode).
    bool  resolve_orientation(const std::vector<Eigen::Vector3f>& pts, float R, float evidence_weight = 1.0f);

    // ── Inference (delegated to the shared engine) ────────────────────────────
    // Per-point recursive-Laplace update, then the footprint second-moment measurement fused as a separate
    // channel with its own (range-robust) covariance — the DOF (yaw/extent) the flat-top per-point mixture
    // structurally cannot supply (no-op unless footprint_moment_precision > 0).
    float update(const TableFrame& frame)
    {
        const float e = ai::update<N>(*this, state_, Sigma_, prior_mean_, frame);
        dbg_yaw_after_points_ = state_.yaw;   // DIAGNOSTIC: yaw after the per-point GN-MAP, before the moment channel
        apply_footprint_moment(frame);
        dbg_yaw_after_moment_ = state_.yaw;   // DIAGNOSTIC: yaw after the footprint-moment fusion
        return e;
    }
    void  predict()                       { ai::predict<N>(*this, Sigma_, state_, prior_mean_); }
    // Age the belief with NO measurement: Σ ← FΣFᵀ + Q·(dt/dt_nominal), mean held. The fitter calls this when
    // a table's mask stream is stale/dead so Σ grows on the agent's clock instead of freezing (see TABLE).
    void  inflate_for_age(float dt_s, float dt_nominal_s)
    { ai::inflate_for_age<N>(*this, Sigma_, state_, prior_mean_, dt_s, dt_nominal_s); }
    Eigen::Matrix<float, 6, 6> predicted_information(const std::vector<Eigen::Vector3f>& pts, float R) const
    { return ai::predicted_information<N>(*this, state_, pts, R); }

    // ── Generative-model hooks (called by the engine; also used as the SDF API) ─
    float sdf_top(const Eigen::Vector3f& p, const TableBeliefState& s) const;
    float sdf_leg(const Eigen::Vector3f& p, const TableBeliefState& s, int k) const;
    float sdf_compound(const Eigen::Vector3f& p, const TableBeliefState& s) const;   // min over 5 prims (diag)
    float sdf_prim(const Eigen::Vector3f& p, const TableBeliefState& s, int prim) const;
    Eigen::Matrix<float, 6, 1> sdf_jacobian(const Eigen::Vector3f& p, const TableBeliefState& s, int prim) const;
    // Soft responsibilities: [top, leg0..3, clutter] (sum = 1) at measurement variance R.
    std::array<float, 6> responsibilities(const Eigen::Vector3f& p, const TableBeliefState& s, float R) const;
    // Un-normalised mixture components + their sum (= marginal likelihood p(point|model)); shared by
    // responsibilities() and the free-energy readout (mean_energy = mean −log sum). Public so the fitter can log F.
    float mixture_unnormalized(const Eigen::Vector3f& p, const TableBeliefState& s, float R,
                               std::array<float, 6>& u) const;
    void  apply_constraints(TableBeliefState& s) const;
    void  canonicalize(TableBeliefState& s) const;   // once after the GN loop: canonical w≥h fold

    int   gn_iters() const { return params_.gn_iters; }
    int   n_prims()  const { return 5; }             // top slab + 4 legs (clutter is the +1 mixture comp)
    float sigma2()   const { return params_.sigma_base_m * params_.sigma_base_m; }
    Eigen::Matrix<float, 6, 6> transition() const { return Eigen::Matrix<float, 6, 6>::Identity(); }  // static
    Eigen::Matrix<float, 6, 1> process_noise_diag() const;
    Eigen::Matrix<float, 6, 1> prior_cov_diag() const;
    Eigen::Matrix<float, 6, 1> common_mode_inv_diag(const TableFrame& frame) const;
    // Extra evidence folded into the GN normal equations (engine calls it if present): the YOLO-independent
    // LiDAR first-hit range factor. Sphere-traces this belief's own SDF, so the shared call is used unchanged.
    void accumulate_extra(const TableBeliefState& s, const TableFrame& f,
                          Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const;

    // ── Monitor instrumentation ───────────────────────────────────────────────
    // Counts from the LAST accumulate_extra call (converged state): how many free-space beams actually
    // exerted a VACATE pull and how many on-plane points were reclaimed by the COVERAGE term this frame.
    // 0 when the respective precision is OFF. Read by the EvidenceMonitor after update().
    int last_vacate_beams()  const { return dbg_vacate_beams_; }
    int last_coverage_pts()  const { return dbg_coverage_pts_; }
    // Top-band point count the footprint-moment factor used last frame (0 when the factor is OFF or starved).
    int last_moment_pts()    const { return dbg_moment_pts_; }

    // DIAGNOSTIC (rogue-mask yaw attribution): yaw immediately after the per-point pass and after the moment
    // fusion, plus the moment channel's last anisotropy / applied yaw-variance / requested yaw move / observed
    // footprint extents — so the fitter can attribute a per-cycle yaw jump to a specific channel. No fit effect.
    float dbg_yaw_after_points() const { return dbg_yaw_after_points_; }
    float dbg_yaw_after_moment() const { return dbg_yaw_after_moment_; }
    float dbg_moment_aniso()     const { return dbg_moment_aniso_; }
    float dbg_moment_r_yaw()     const { return dbg_moment_r_yaw_; }
    float dbg_moment_dyaw()      const { return dbg_moment_dyaw_; }
    float dbg_moment_ext_major() const { return dbg_moment_ext_major_; }
    float dbg_moment_ext_minor() const { return dbg_moment_ext_minor_; }

    // 2D footprint second-moment of the points whose z falls in [z_lo, z_hi] (the tabletop band). Shared by
    // the per-frame moment factor (accumulate_extra) and the birth seed (the fitter's lazy init) so both use
    // identical math. Pure geometry, no state — the trap-breaking global statistic.
    static FootprintMoment footprint_moment(const std::vector<Eigen::Vector3f>& pts, float z_lo, float z_hi);

    // ── Verification ──────────────────────────────────────────────────────────
    static bool self_test();

private:
    float leg_half_height(const TableBeliefState& s) const
    { return 0.5f * std::max(0.0f, s.H - params_.top_thickness); }
    Eigen::Vector2f leg_center_local(const TableBeliefState& s, int k) const;

    // Footprint second-moment measurement fusion (called by update() after the per-point pass). Separate
    // channel, own covariance, outside the per-point common-mode — see the .cpp for the range-robustness rationale.
    void apply_footprint_moment(const TableFrame& frame);

    TableBeliefState           state_;
    TableBeliefParams          params_;
    Eigen::Matrix<float, 6, 6> Sigma_ = Eigen::Matrix<float, 6, 6>::Identity();  // posterior covariance
    Eigen::Matrix<float, 6, 1> prior_mean_ = Eigen::Matrix<float, 6, 1>::Zero();  // transition prior mean
    float                      flip_evidence_ = 0.0f;  // accumulated log-evidence FOR the current mode vs the w↔h swap
    mutable int                dbg_vacate_beams_ = 0;   // free-space beams that fired a vacate pull (last accumulate)
    mutable int                dbg_coverage_pts_ = 0;   // on-plane points reclaimed by coverage (last accumulate)
    mutable int                dbg_moment_pts_ = 0;      // top-band points used by the footprint-moment factor
    // DIAGNOSTIC only (rogue-mask yaw attribution) — written by update()/apply_footprint_moment(), never read
    // by the fit. Let the fitter split a per-cycle yaw jump across the per-point / moment / flip channels.
    float                      dbg_yaw_after_points_ = 0.0f;  // yaw after the per-point GN-MAP pass (rad)
    float                      dbg_yaw_after_moment_ = 0.0f;  // yaw after the footprint-moment fusion (rad)
    float                      dbg_moment_aniso_     = 0.0f;  // last footprint anisotropy (major−minor)/(major+minor)
    float                      dbg_moment_r_yaw_     = 0.0f;  // last moment yaw measurement variance used (rad²)
    float                      dbg_moment_dyaw_      = 0.0f;  // last moment requested yaw move myaw−yaw (rad)
    float                      dbg_moment_ext_major_ = 0.0f;  // last observed footprint major extent (m)
    float                      dbg_moment_ext_minor_ = 0.0f;  // last observed footprint minor extent (m)
};

}  // namespace rc
