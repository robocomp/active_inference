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
};

class TableBelief
{
public:
    TableBelief() = default;
    TableBelief(const TableBeliefState& s, const TableBeliefParams& p) : state_(s), params_(p) { init_prior_cov(); }

    const TableBeliefState&            state()      const { return state_; }
    const Eigen::Matrix<float, 6, 6>&  covariance() const { return Sigma_; }
    const TableBeliefParams&           params()     const { return params_; }
    void set_state(const TableBeliefState& s) { state_ = s; }
    void set_params(const TableBeliefParams& p) { params_ = p; }

    // ── Generative model ──────────────────────────────────────────────────────
    // Signed distance to the top slab and to leg k (k=0..3) for a room-frame point at the given state.
    float sdf_top(const Eigen::Vector3f& p, const TableBeliefState& s) const;
    float sdf_leg(const Eigen::Vector3f& p, const TableBeliefState& s, int k) const;
    // Min over all 5 surface primitives (diagnostic / clutter test).
    float sdf_compound(const Eigen::Vector3f& p, const TableBeliefState& s) const;

    // Soft responsibilities for a point: [top, leg0..3, clutter] (sum = 1), given measurement variance R.
    std::array<float, 6> responsibilities(const Eigen::Vector3f& p, const TableBeliefState& s, float R) const;

    // ── Inference ───────────────────────────────────────────────────────────────
    // Predict (inflate Σ by process noise), then run gn_iters Gauss-Newton EM updates on the frame.
    // Returns the mean per-point free energy (negative log-likelihood proxy) after the last step.
    float update(const TableFrame& frame);
    // The predict step alone (used when a frame is bias-gated): Σ ← Σ + Q.
    void predict();

    // ── Verification ──────────────────────────────────────────────────────────
    // Synthetic self-test: build a known table, sample top+legs+clutter, fit from a perturbed init, and
    // assert (a) Jacobians match finite differences, (b) params recovered, (c) clutter gets low
    // responsibility, (d) Σ is finite & SPD. Prints a report. Returns true on PASS.
    static bool self_test();

private:
    void  init_prior_cov();
    float leg_half_height(const TableBeliefState& s) const
    { return 0.5f * std::max(0.0f, s.H - params_.top_thickness); }
    // local-frame leg-centre XY for corner k (signs), at the outer-edge inset = leg_radius.
    Eigen::Vector2f leg_center_local(const TableBeliefState& s, int k) const;
    // ∂SDF_prim/∂θ via central finite differences (prim: 0=top, 1..4=leg k-1).
    Eigen::Matrix<float, 6, 1> sdf_jacobian(const Eigen::Vector3f& p, const TableBeliefState& s, int prim) const;
    float sdf_prim(const Eigen::Vector3f& p, const TableBeliefState& s, int prim) const;
    void  apply_constraints(TableBeliefState& s) const;

    TableBeliefState           state_;
    TableBeliefParams          params_;
    Eigen::Matrix<float, 6, 6> Sigma_ = Eigen::Matrix<float, 6, 6>::Identity();  // posterior covariance
    bool                       have_prior_ = false;
    Eigen::Matrix<float, 6, 1> prior_mean_;     // transition prior mean (previous posterior mean)
};

}  // namespace rc
