/*
 * bottle_belief.h  —  AI2 bottle belief (mirrors table_concept/TABLE.md, table_belief.h)
 *
 * The active-inference belief for the bottle, aligned with table_concept and chair_concept: ONE
 * generative model (a single vertical cylinder + a uniform clutter component) with SOFT per-point
 * responsibilities (a mixture, no hard min()/threshold), inferred by the shared recursive
 * variational-Laplace Gaussian filter that carries a FULL 5×5 covariance. Pure Eigen, no torch, no DSR —
 * unit-testable in isolation (see self_test()). This is the ONLY fit path.
 *
 * State θ = [cx, cy, cz, radius, height]  (5 DOF, matches BottleState::to_array()). No yaw: a cylinder is
 * rotationally symmetric about its vertical axis, so there is no orientation DOF and no canonical fold —
 * canonicalize() is a no-op and n_prims() == 1 (the simplest of the three concept agents).
 *
 * NOTE (movable object): this belief is STATIC (transition() = I), mirroring table/chair's shipped AI2. A
 * moved bottle is re-acquired via the instance-tracker gate widening on stale predicts (predict_stale()
 * inflates the position covariance only). A CV-coupled AI2 transition (velocity DOFs) is a separate extension.
 */

#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>

#include "../../common/ai_belief/recursive_laplace.h"   // shared predict/MAP/Woodbury engine
#include "../../common/ai_belief/lidar_ray_factor.h"     // shared YOLO-independent LiDAR first-hit factor

namespace rc
{

struct BottleBeliefState
{
    float cx = 0.0f, cy = 0.0f, cz = 0.85f, radius = 0.035f, height = 0.20f;

    Eigen::Matrix<float, 5, 1> vec() const
    { return (Eigen::Matrix<float, 5, 1>() << cx, cy, cz, radius, height).finished(); }
    static BottleBeliefState from_vec(const Eigen::Matrix<float, 5, 1>& v)
    { return {v(0), v(1), v(2), v(3), v(4)}; }
};

struct BottleBeliefParams
{
    // Geometry (physical floors, shared with BottleModel)
    float min_radius = 0.005f;
    float min_height = 0.02f;

    // Observation model
    float sigma_base_m    = 0.02f;   // base on-surface noise std (m); R = σ² (+ motion_var + …) per point
    float clutter_frac    = 0.10f;   // ε: prior weight of the uniform clutter component
    float clutter_scale_m = 0.08f;   // a point further than ~this from the surface is likely clutter

    // Priors (broad; only break the empty-cloud degeneracy)
    float prior_pos_std   = 0.30f;   // position prior std (m) on cx,cy,cz
    float prior_size_std  = 0.03f;   // size prior std (m) on radius,height

    // Temporal transition (predict): rigid + static ⇒ small process noise per frame. SIZE (radius,height) is a
    // CONSTANT physical property, so its process noise is ~0 — once converged it STICKS and can't be re-inflated
    // by a sparse/contaminated frame (the marginal-LiDAR-range radius oscillation). Position keeps process_std_m.
    float process_std_m      = 0.005f;   // cx,cy,cz per-frame process std (m)
    float process_std_size_m = 0.001f;   // radius,height per-frame process std (m) — tiny: size doesn't change

    // Per-frame COMMON-MODE error: the error SHARED by all points of one mask (localization + mask
    // boundary + deprojection), which does NOT average out over points. The frame's information SATURATES
    // at this covariance (marginalised, Woodbury), so N correlated points can't collapse σ. Measurement-
    // model stds, not a σ floor.
    float common_mode_pos_std  = 0.02f;  // shared position error (m); pose-chain cov adds to it
    float common_mode_size_std = 0.01f;  // shared size error radius,height (m)

    // Optimiser
    int   gn_iters = 4;              // Gauss-Newton iterations per frame
    float fd_eps   = 5e-4f;          // finite-difference step for SDF Jacobians (m); cm-scale object
};

// One fitted frame's evidence: room-frame points and per-point measurement variance R (m²). Pass an empty
// R vector to use the base σ² for every point.
struct BottleFrame
{
    std::vector<Eigen::Vector3f> points;
    std::vector<float>           R;             // per-point measurement variance (m²); empty ⇒ σ_base²
    float chain_cov_xx = 0.0f;                  // extra shared position variance (m²) from the pose chain (cx)
    float chain_cov_yy = 0.0f;                  // ...                                                      (cy)
    float chain_cov_size = 0.0f;                // extra shared SIZE variance (m²) on radius,height — grows with
                                                // ego-motion so a moving frame can't reshape the bottle (mirrors
                                                // table's chain_cov_size; the anti-RESHAPE common-mode channel)

    // Occluding-contour silhouette: the RGB-mask's left/right edge rays (room frame). A ray tangent to the
    // vertical cylinder ⇒ perpendicular distance(axis, ray) = radius. This is the ONLY term that pins the
    // depth-degenerate radius of a symmetric cylinder from a one-sided depth cloud (see bottle-sdf-depth-bias).
    Eigen::Vector2f              sil_cam_xy = Eigen::Vector2f::Zero();   // camera centre (room, horizontal)
    std::vector<Eigen::Vector2f> sil_dirs;                              // edge-ray directions (need not be unit)
    float                        sil_precision = 0.0f;                  // total tangent weight (0 = off)

    // YOLO-INDEPENDENT LiDAR channel: range returns that fall on the bottle (room frame, sensor origin +
    // endpoints). Constrains cx,cy IN METRIC DEPTH + radius along the viewing ray — the direction the mask
    // SDF is blind to. Shape-agnostic (sphere-traces this belief's own SDF); rays.precision = 0 ⇒ off.
    rc::ai::LidarRays            lidar;

    // Range de-rating (>= 1): inflates the per-frame COMMON-MODE covariance (position AND size), the quantity
    // that actually caps how far one frame can move the mean. Set to the camera→object range fade so a distant
    // object is left ~untouched (its radius holds) and a close one refines. 1 = full precision.
    float                        precision_scale = 1.0f;
};

// The bottle generative model wired onto the shared rc::ai recursive-Laplace engine. The Bayesian math
// (predict / GN-MAP / Woodbury) lives in common/ai_belief/recursive_laplace.h; this class supplies only
// the bottle-specific MODEL hooks the engine calls (SDF, responsibilities, Jacobian, constraints, and the
// Q/F/prior/common-mode diagonals). N = 5, static (transition = I), a single primitive (n_prims = 1).
class BottleBelief
{
public:
    static constexpr int N = 5;
    using State = BottleBeliefState;
    using Frame = BottleFrame;

    BottleBelief() = default;
    BottleBelief(const BottleBeliefState& s, const BottleBeliefParams& p) : state_(s), params_(p)
    { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); }

    const BottleBeliefState&           state()      const { return state_; }
    const Eigen::Matrix<float, 5, 5>&  covariance() const { return Sigma_; }
    const BottleBeliefParams&          params()     const { return params_; }
    void set_state(const BottleBeliefState& s) { state_ = s; }
    void set_params(const BottleBeliefParams& p) { params_ = p; }

    // ── Inference (delegated to the shared engine) ────────────────────────────
    float update(const BottleFrame& frame) { return ai::update<N>(*this, state_, Sigma_, prior_mean_, frame); }
    void  predict()                        { ai::predict<N>(*this, Sigma_, state_, prior_mean_); }
    // Stale-frame predict: a bottle's SIZE is constant while unseen — only its POSITION becomes uncertain.
    // Inflate ONLY the position block of Σ (so the tracker gate widens across a look-away and the
    // re-acquired detection associates) and leave radius/height tight (so re-acquisition can't loosen the
    // learned size → transient radius spikes). F = I (static), so this is Σ_pos += Q_pos.
    // q_scale re-weights the injected position noise. Default 1.0 = one nominal frame's Q (the historic
    // per-unseen-cycle behaviour). The fitter passes dt/dt_nominal when AI2AgeNominalDtS>0 so Σ_pos grows by
    // the REAL elapsed time on the agent's clock (measurement-age → covariance) — a dead ZED feed then reads
    // downstream as a growing position covariance, not a frozen confident pose. Size stays tight by design.
    void  predict_stale(float q_scale = 1.0f)
    {
        const Eigen::Matrix<float, N, 1> q = process_noise_diag();
        const float k = std::max(0.0f, q_scale);
        for (int j = 0; j < 3; ++j) Sigma_(j, j) += k * q(j);   // cx, cy, cz only
        prior_mean_ = state_.vec();
    }
    Eigen::Matrix<float, 5, 5> predicted_information(const std::vector<Eigen::Vector3f>& pts, float R) const
    { return ai::predicted_information<N>(*this, state_, pts, R); }

    // ── Generative-model hooks (called by the engine; also used as the SDF API) ─
    float sdf_cylinder(const Eigen::Vector3f& p, const BottleBeliefState& s) const;
    float sdf_prim(const Eigen::Vector3f& p, const BottleBeliefState& s, int prim) const;   // prim == 0
    Eigen::Matrix<float, 5, 1> sdf_jacobian(const Eigen::Vector3f& p, const BottleBeliefState& s, int prim) const;
    // Soft responsibilities: [cylinder, clutter] (sum = 1) at measurement variance R.
    std::array<float, 2> responsibilities(const Eigen::Vector3f& p, const BottleBeliefState& s, float R) const;
    // Mean per-point mixture NEGATIVE LOG-LIKELIHOOD (clutter INCLUDED) = the true −log marginal, the honest
    // free energy: F = mean_p −log(u_cyl + u_clut). Unlike the engine's surface-only responsibility-weighted
    // energy (which reads ≈0 on a bad fit because misfit points route to clutter and contribute ~0 — TABLE.md
    // §3, recipe invariant #1), this RISES with misfit, so a drifted/inflated fit is HIGH F, not 0. Used for
    // the convergence gate and the published/logged FE.
    float mixture_nll(const std::vector<Eigen::Vector3f>& pts, const BottleBeliefState& s, float R) const;
    // Mean clutter responsibility over `pts` (fraction of the cloud the cylinder can't explain). The HONEST
    // "explains none of its data" signal that replaces the old energy==0 all-clutter sentinel for divergence.
    float clutter_fraction(const std::vector<Eigen::Vector3f>& pts, float R) const;
    void  apply_constraints(BottleBeliefState& s) const;
    void  canonicalize(BottleBeliefState&) const {}   // rotationally symmetric cylinder: no fold
    // Extra GN factor called by the shared engine (via C++23 requires): the occluding-contour silhouette
    // tangent term dist(axis, ray) = radius. Folds w·JJᵀ / -w·J·r into the frame's (Id, bd). Constrains
    // cx, cy (perpendicular to each ray) and radius — the direction depth is blind to. No-op if disabled.
    void  accumulate_extra(const BottleBeliefState& s, const BottleFrame& f,
                           Eigen::Matrix<float, 5, 5>& Id, Eigen::Matrix<float, 5, 1>& bd) const;

    int   gn_iters() const { return params_.gn_iters; }
    int   n_prims()  const { return 1; }             // one cylinder (clutter is the +1 mixture component)
    float sigma2()   const { return params_.sigma_base_m * params_.sigma_base_m; }
    Eigen::Matrix<float, 5, 5> transition() const { return Eigen::Matrix<float, 5, 5>::Identity(); }  // static
    Eigen::Matrix<float, 5, 1> process_noise_diag() const;
    Eigen::Matrix<float, 5, 1> prior_cov_diag() const;
    Eigen::Matrix<float, 5, 1> common_mode_inv_diag(const BottleFrame& frame) const;

    // ── Verification ──────────────────────────────────────────────────────────
    static bool self_test();

private:
    // Unnormalized mixture terms [cylinder, clutter] at p (likelihood×prior). Shared by responsibilities()
    // (normalizes) and mixture_nll() (sums for the −log marginal).
    std::array<float, 2> mixture_unnorm(const Eigen::Vector3f& p, const BottleBeliefState& s, float R) const;

    BottleBeliefState          state_;
    BottleBeliefParams         params_;
    Eigen::Matrix<float, 5, 5> Sigma_ = Eigen::Matrix<float, 5, 5>::Identity();  // posterior covariance
    Eigen::Matrix<float, 5, 1> prior_mean_ = Eigen::Matrix<float, 5, 1>::Zero();  // transition prior mean
};

}  // namespace rc
