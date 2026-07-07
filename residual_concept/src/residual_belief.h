/*
 * residual_belief.h  —  AI2 residual (null-concept) belief
 *
 * The active-inference belief for the RESIDUAL concept: the null / catch-all model at the bottom of the
 * concept hierarchy that explains any LiDAR return no specialised agent (table/chair/bottle/human) has
 * claimed. Aligned with table/chair/bottle_concept: ONE generative model (a single ORIENTED FOOTPRINT BOX
 * + a uniform clutter component) with SOFT per-point responsibilities (a mixture, no hard min()/threshold),
 * inferred by the shared recursive variational-Laplace Gaussian filter that carries a FULL 5×5 covariance.
 * Pure Eigen, no torch, no DSR — unit-testable in isolation (see self_test()). This is the ONLY fit path.
 *
 * State θ = [cx, cy, yaw, w, d]  (5 DOF): the in-plane oriented footprint. A residual cluster has NO shape
 * prior (that is exactly what the specialists provide), so the model is the coarsest localizable envelope —
 * an oriented rectangle — fit to the cluster's floor projection. The vertical extent (cz, height) is NOT
 * fitted here: it is anchored directly from the cluster's z-range by the fitter (support step), the same
 * way bottle anchors cz. Downstream, the AUTHORITATIVE navigation geometry is the cluster's convex hull
 * (Layer B); this box belief is the tracking/uncertainty envelope (Layer A) that feeds the instance-tracker
 * association gate and the published RT covariance.
 *
 * The SDF is 2D (ignores z): it fits the footprint. A box has a w↔d / yaw±90° symmetry, folded in
 * canonicalize() exactly as table does. STATIC (transition() = I): a moved obstacle is not tracked by a
 * velocity DOF — it is re-derived every cycle from the residual re-clustering, and predict_stale() only
 * widens the position block so the tracker gate re-associates across a look-away. Shape (w,d,yaw) is treated
 * as RIGID + STATIC (small process noise, as in table/chair/bottle) so EVIDENCE ACCUMULATES: the posterior
 * precision grows every frame and the box resists change in proportion to the evidence already gathered —
 * that is what stops the shape jittering as the underlying cluster re-merges. Fission ("table+4 chairs →
 * 4 chairs") is NOT a size-drift here: when a specialist carves the cluster the supporting POINTS disappear,
 * which is sustained counter-evidence the belief yields to (via the dissolve/tracker path), not a standing
 * per-frame covariance leak on the extent.
 */

#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>

#include "../../common/ai_belief/recursive_laplace.h"   // shared predict/MAP/Woodbury engine

namespace rc
{

struct ResidualBeliefState
{
    float cx = 0.0f, cy = 0.0f, yaw = 0.0f, w = 0.40f, d = 0.40f;

    Eigen::Matrix<float, 5, 1> vec() const
    { return (Eigen::Matrix<float, 5, 1>() << cx, cy, yaw, w, d).finished(); }
    static ResidualBeliefState from_vec(const Eigen::Matrix<float, 5, 1>& v)
    { return {v(0), v(1), v(2), v(3), v(4)}; }
};

struct ResidualBeliefParams
{
    // Geometry (physical floors/ceilings on the footprint extents; a residual can be small (a mug) or large
    // (a wall segment), so the size prior is broad — NO informative shape prior for an unknown object).
    float min_size = 0.05f;
    float max_size = 4.0f;

    // Observation model (LiDAR return noise; a residual cluster is a raw range cloud, not a YOLO mask).
    float sigma_base_m    = 0.03f;   // base on-surface noise std (m); R = σ² (+ chain + …) per point
    float clutter_frac    = 0.15f;   // ε: prior weight of the uniform clutter component
    float clutter_scale_m = 0.12f;   // a point further than ~this from the box edge is likely clutter

    // Priors (broad; only break the empty-cloud degeneracy — a null concept commits to no size/shape).
    float prior_pos_std   = 0.50f;   // position prior std (m) on cx,cy
    float prior_yaw_std   = 1.57f;   // orientation prior std (rad) — essentially uninformative
    float prior_size_std  = 0.50f;   // size prior std (m) on w,d — broad

    // Temporal transition (predict): a physical obstacle is RIGID + STATIC, so shape process noise is SMALL
    // and evidence ACCUMULATES across frames (posterior precision grows → per-frame gain on w/d/yaw falls →
    // the shape only moves when new evidence rivals the accumulated evidence). Same regime as table/chair/
    // bottle. Fission is handled by the dissolve/tracker path (a specialist carving the cluster makes the
    // POINTS vanish = sustained counter-evidence), NOT by a standing per-frame covariance leak on the size.
    float process_std_pos_m  = 0.01f;
    float process_std_yaw    = 0.01f;
    float process_std_size_m = 0.005f;  // ≈table/bottle; a residual envelope refines as more is seen, but
                                        //   growth is deliberate (evidence-driven), not per-frame chasing

    // Per-frame COMMON-MODE error: the error SHARED by all points of one cluster (localization + range +
    // deprojection), which does NOT average out over points. The frame's information SATURATES at this
    // covariance (marginalised, Woodbury), so N correlated points can't collapse σ. Measurement-model stds,
    // NOT a σ floor. The pose-chain cov (robot localization) adds to the position block.
    float common_mode_pos_std  = 0.03f;  // shared position error (m)
    float common_mode_yaw_std  = 0.20f;  // shared orientation error (rad)
    float common_mode_size_std = 0.03f;  // shared size error w,d (m)

    // Optimiser
    int   gn_iters = 5;              // Gauss-Newton iterations per frame
    float fd_eps   = 1e-3f;          // finite-difference step for SDF Jacobians (m)
};

// One fitted frame's evidence: room-frame points and per-point measurement variance R (m²). Pass an empty
// R vector to use the base σ² for every point. No silhouette / lidar-ray channels — the whole cloud is the
// range evidence, so the plain point-SDF likelihood is the entire model (the simplest of the concept agents:
// no accumulate_extra at all).
struct ResidualFrame
{
    std::vector<Eigen::Vector3f> points;
    std::vector<float>           R;             // per-point measurement variance (m²); empty ⇒ σ_base²
    float chain_cov_xx = 0.0f;                  // extra shared position variance (m²) from the pose chain (cx)
    float chain_cov_yy = 0.0f;                  // ...                                                      (cy)
};

// The residual generative model wired onto the shared rc::ai recursive-Laplace engine. The Bayesian math
// (predict / GN-MAP / Woodbury) lives in common/ai_belief/recursive_laplace.h; this class supplies only the
// residual-specific MODEL hooks the engine calls (SDF, responsibilities, Jacobian, constraints, symmetry
// fold, and the Q/F/prior/common-mode diagonals). N = 5, static (transition = I), a single primitive
// (n_prims = 1). NOTE: no accumulate_extra — the point-SDF likelihood is the whole model.
class ResidualBelief
{
public:
    static constexpr int N = 5;
    using State = ResidualBeliefState;
    using Frame = ResidualFrame;

    ResidualBelief() = default;
    ResidualBelief(const ResidualBeliefState& s, const ResidualBeliefParams& p) : state_(s), params_(p)
    { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); }

    const ResidualBeliefState&         state()      const { return state_; }
    const Eigen::Matrix<float, 5, 5>&  covariance() const { return Sigma_; }
    const ResidualBeliefParams&        params()     const { return params_; }
    void set_state(const ResidualBeliefState& s) { state_ = s; }
    void set_params(const ResidualBeliefParams& p) { params_ = p; }

    // ── Inference (delegated to the shared engine) ────────────────────────────
    float update(const ResidualFrame& frame) { return ai::update<N>(*this, state_, Sigma_, prior_mean_, frame); }
    void  predict()                          { ai::predict<N>(*this, Sigma_, state_, prior_mean_); }
    // Stale-frame predict: inflate ONLY the position block of Σ (so the tracker gate widens across a
    // look-away and the re-acquired detection associates) and leave yaw/size tight. F = I (static).
    void  predict_stale()
    {
        const Eigen::Matrix<float, N, 1> q = process_noise_diag();
        for (int j = 0; j < 2; ++j) Sigma_(j, j) += q(j);   // cx, cy only
        prior_mean_ = state_.vec();
    }
    Eigen::Matrix<float, 5, 5> predicted_information(const std::vector<Eigen::Vector3f>& pts, float R) const
    { return ai::predicted_information<N>(*this, state_, pts, R); }

    // ── Generative-model hooks (called by the engine; also used as the SDF API) ─
    float sdf_box(const Eigen::Vector3f& p, const ResidualBeliefState& s) const;
    float sdf_prim(const Eigen::Vector3f& p, const ResidualBeliefState& s, int prim) const;   // prim == 0
    Eigen::Matrix<float, 5, 1> sdf_jacobian(const Eigen::Vector3f& p, const ResidualBeliefState& s, int prim) const;
    // Soft responsibilities: [box, clutter] (sum = 1) at measurement variance R.
    std::array<float, 2> responsibilities(const Eigen::Vector3f& p, const ResidualBeliefState& s, float R) const;
    void  apply_constraints(ResidualBeliefState& s) const;
    void  canonicalize(ResidualBeliefState& s) const;   // box w↔d / yaw±90° symmetry fold (as table)

    int   gn_iters() const { return params_.gn_iters; }
    int   n_prims()  const { return 1; }             // one box (clutter is the +1 mixture component)
    float sigma2()   const { return params_.sigma_base_m * params_.sigma_base_m; }
    Eigen::Matrix<float, 5, 5> transition() const { return Eigen::Matrix<float, 5, 5>::Identity(); }  // static
    Eigen::Matrix<float, 5, 1> process_noise_diag() const;
    Eigen::Matrix<float, 5, 1> prior_cov_diag() const;
    Eigen::Matrix<float, 5, 1> common_mode_inv_diag(const ResidualFrame& frame) const;

    // ── Verification ──────────────────────────────────────────────────────────
    static bool self_test();

private:
    ResidualBeliefState        state_;
    ResidualBeliefParams       params_;
    Eigen::Matrix<float, 5, 5> Sigma_ = Eigen::Matrix<float, 5, 5>::Identity();  // posterior covariance
    Eigen::Matrix<float, 5, 1> prior_mean_ = Eigen::Matrix<float, 5, 1>::Zero();  // transition prior mean
};

}  // namespace rc
