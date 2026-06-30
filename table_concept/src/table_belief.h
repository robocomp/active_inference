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

    // ── Inference (delegated to the shared engine) ────────────────────────────
    float update(const TableFrame& frame) { return ai::update<N>(*this, state_, Sigma_, prior_mean_, frame); }
    void  predict()                       { ai::predict<N>(*this, Sigma_, state_, prior_mean_); }
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
};

}  // namespace rc
