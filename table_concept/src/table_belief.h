/*
 * table_belief.h  —  AI2 table belief (see TABLE_FIT_AI2.md)
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

struct TableBeliefState
{
    float cx = 0.0f, cy = 0.0f, H = 0.75f, w = 1.0f, h = 0.6f, yaw = 0.0f;

    Eigen::Matrix<float, 6, 1> vec() const { return (Eigen::Matrix<float, 6, 1>() << cx, cy, H, w, h, yaw).finished(); }
    static TableBeliefState from_vec(const Eigen::Matrix<float, 6, 1>& v)
    { return {v(0), v(1), v(2), v(3), v(4), v(5)}; }
};

struct TableBeliefParams
{
    // Geometry (fixed)
    float top_thickness = 0.03f;
    float leg_radius    = 0.025f;

    // Observation model
    float sigma_base_m  = 0.03f;   // base on-surface noise std (m); R = σ² (+ motion_var + …) per point
    float clutter_frac  = 0.10f;   // ε: prior weight of the uniform clutter component
    float clutter_scale_m = 0.12f; // a point further than ~this from every surface is likely clutter

    // Coverage / traction (EXISTENCE_BELIEF_PLAN.md, table_1.png): on-plane mask points the mixture ceded to
    // the CLUTTER component still exert a robust, GROW-ONLY pull on the top slab, so a model UNDER-covering a
    // large mask is pulled out to explain it instead of parking with the excess dumped to clutter for free (the
    // escape-valve). Weight = coverage_precision·(top-plane compat)·(clutter resp)·Cauchy(outside dist).
    // Self-bounded (sdf→0 when covered), on-plane-gated (off-plane contamination ignored), reclaims exactly the
    // escaped points. 0 = OFF. The honest anti-contamination bound is the free-space carve; robust interim.
    float coverage_precision  = 0.0f;
    float coverage_robust_c_m = 0.15f;

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

    // ── Discrete orientation mode (near-square yaw disambiguation, TABLE_FIT_AI2.md) ──────────────
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
    bool  resolve_orientation(const std::vector<Eigen::Vector3f>& pts, float R);

    // ── Inference (delegated to the shared engine) ────────────────────────────
    float update(const TableFrame& frame) { return ai::update<N>(*this, state_, Sigma_, prior_mean_, frame); }
    void  predict()                       { ai::predict<N>(*this, Sigma_, state_, prior_mean_); }
    // Age the belief with NO measurement: Σ ← FΣFᵀ + Q·(dt/dt_nominal), mean held. The fitter calls this when
    // a table's mask stream is stale/dead so Σ grows on the agent's clock instead of freezing (see TABLE_FIT_AI2).
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

    // ── Verification ──────────────────────────────────────────────────────────
    static bool self_test();

private:
    float leg_half_height(const TableBeliefState& s) const
    { return 0.5f * std::max(0.0f, s.H - params_.top_thickness); }
    Eigen::Vector2f leg_center_local(const TableBeliefState& s, int k) const;

    TableBeliefState           state_;
    TableBeliefParams          params_;
    Eigen::Matrix<float, 6, 6> Sigma_ = Eigen::Matrix<float, 6, 6>::Identity();  // posterior covariance
    Eigen::Matrix<float, 6, 1> prior_mean_ = Eigen::Matrix<float, 6, 1>::Zero();  // transition prior mean
    float                      flip_evidence_ = 0.0f;  // accumulated log-evidence FOR the current mode vs the w↔h swap
};

}  // namespace rc
