/*
 * common/ai_belief/recursive_laplace.h  —  shared AI2 inference engine (header-only)
 *
 * SHARED across every concept agent (table/chair/bottle/residual/…): the model-independent Bayesian core
 * they all delegate to, so the math lives ONCE and can't diverge per-agent.
 *
 * The model-independent core of the AI2 belief proven in table_concept (see table_concept/TABLE.md):
 * a recursive variational-Laplace Gaussian filter — predict (F·θ, Σ←FΣFᵀ+Q) → Gauss-Newton MAP on the
 * FULL data information → posterior Σ via exact within-frame common-mode (Woodbury) saturation. Each
 * concept agent (table/chair/bottle) supplies only its generative MODEL; the Bayesian math lives here
 * ONCE so it can't diverge or be copied wrong (e.g. the "saturation in the GN mean step breaks
 * recovery" trap is fixed here for everyone). See ../../AI2_MIGRATION_PLAN.md.
 *
 * A Model `m` (the agent's belief object passed by reference) must expose, with N = #DOFs:
 *   using State;  using Frame;                         // State has vec()/from_vec(); Frame.points,.R
 *   int   gn_iters() const;
 *   int   n_prims()  const;                            // # SDF primitives (mixture components − clutter)
 *   float sigma2()   const;                            // base R² when Frame.R is empty
 *   float sdf_prim     (const Eigen::Vector3f&, const State&, int prim) const;
 *   Eigen::Matrix<float,N,1> sdf_jacobian(const Eigen::Vector3f&, const State&, int prim) const;
 *   auto  responsibilities(const Eigen::Vector3f&, const State&, float R) const;   // indexable, n_prims()+1
 *   void  apply_constraints(State&) const;             // per-GN-iter bounds
 *   void  canonicalize     (State&) const;             // ONCE after the loop (e.g. w≥h fold); no-op ok
 *   Eigen::Matrix<float,N,N> transition()          const;   // F (identity = static; CV block = movable)
 *   Eigen::Matrix<float,N,1> process_noise_diag()  const;   // Q diagonal
 *   Eigen::Matrix<float,N,1> prior_cov_diag()      const;   // Σ₀ diagonal
 *   Eigen::Matrix<float,N,1> common_mode_inv_diag(const Frame&) const;  // Σc⁻¹ diagonal (base+chain+range)
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

namespace rc::ai
{

// ─── Predict (transition + process noise · age the stale belief) ─────────────────────────────────

// Predict: inflate Σ by the transition + process noise; record the transition-prior mean. θ is unchanged
// for a static model (F = I); a movable model's F carries the constant-velocity coupling.
//
// q_scale re-weights the per-step process noise Q. Default 1.0 = one nominal step (the historic
// behaviour; update() below relies on this default so all existing call sites are unchanged). Q is
// authored per-frame, so q_scale is how many nominal frames' worth of Q to inject — the mechanism
// inflate_for_age() uses to integrate Q over real elapsed time when measurements are late or absent.
template <int N, class Model>
void predict(Model& m, Eigen::Matrix<float, N, N>& Sigma,
             const typename Model::State& state, Eigen::Matrix<float, N, 1>& prior_mean,
             float q_scale = 1.0f)
{
    if (Sigma.isZero())
        Sigma.diagonal() = m.prior_cov_diag();
    const Eigen::Matrix<float, N, N> F = m.transition();
    Sigma = F * Sigma * F.transpose();
    Sigma.diagonal() += std::max(0.0f, q_scale) * m.process_noise_diag();
    prior_mean = state.vec();
}

// Age the belief with NO measurement — the "stale sensor" term. Consumers call this every control
// tick on their OWN clock (which keeps running when the sensor is dead), EVEN WHEN NO FRESH FRAME
// ARRIVED, passing the wall-clock seconds since the belief was last touched and the stream's nominal
// inter-frame period. It runs only the predict half of the filter — F·Σ·Fᵀ then +Q·(dt/dt_nominal) —
// holding the mean while the posterior covariance grows monotonically with the age of the evidence.
//
// This is the generative-model form of a dead stream: NO gate, NO emergency flag. A silent sensor is
// just evidence getting old, so Σ → ∞ at a rate set by Q; downstream consumers that read Σ (e.g. the
// controller's target-in-frame covariance) then back off continuously, in proportion to staleness,
// instead of tripping a discrete watchdog. When the stream recovers, the next update() tightens Σ again.
//
// Exact for a static model (F = I: table/chair). For a constant-velocity F the covariance growth is
// right but the mean is propagated by a single nominal F rather than F(dt); over long gaps a movable
// model should carry the CV extrapolation in its own transition() — noted, not needed for the static
// concepts this prototype targets first.
template <int N, class Model>
void inflate_for_age(Model& m, Eigen::Matrix<float, N, N>& Sigma,
                     const typename Model::State& state, Eigen::Matrix<float, N, 1>& prior_mean,
                     float dt_seconds, float dt_nominal_seconds)
{
    const float q_scale = dt_nominal_seconds > 1e-6f
                              ? dt_seconds / dt_nominal_seconds     // frames-worth of Q over real time
                              : 1.0f;                               // no nominal rate known → one step
    predict<N>(m, Sigma, state, prior_mean, q_scale);
}

// ─── Update (GN-MAP mean · Woodbury common-mode saturation of Σ) ─────────────────────────────────

// One recursive update on a fresh frame. MEAN: Gauss-Newton MAP using the FULL data information (fast,
// unbiased). COVARIANCE: the per-frame common-mode saturation (Woodbury) — I_eff = Id − Id(Id+Σc⁻¹)⁻¹Id
// → Σc⁻¹ as Id→∞, so N correlated points can't collapse σ. The saturation touches ONLY Σ, never the mean.
// Returns the mean per-point data energy (NLL proxy).
template <int N, class Model>
float update(Model& m, typename Model::State& state, Eigen::Matrix<float, N, N>& Sigma,
             Eigen::Matrix<float, N, 1>& prior_mean, const typename Model::Frame& frame)
{
    using State = typename Model::State;
    using VecN  = Eigen::Matrix<float, N, 1>;
    using MatN  = Eigen::Matrix<float, N, N>;

    predict<N>(m, Sigma, state, prior_mean);
    if (frame.points.empty())
        return 0.0f;

    const float sig2 = m.sigma2();
    const auto  R_of = [&](std::size_t i) -> float
    { return (i < frame.R.size() and frame.R[i] > 0.0f) ? frame.R[i] : sig2; };

    const MatN P0 = Sigma.inverse();
    VecN theta = state.vec();
    const VecN Sc_inv = m.common_mode_inv_diag(frame);
    const MatN Scinv  = Sc_inv.asDiagonal();
    const int  P = m.n_prims();

    MatN Id; VecN bd;
    const auto accumulate = [&](const VecN& th)
    {
        const State s = State::from_vec(th);
        Id.setZero(); bd.setZero();
        for (std::size_t i = 0; i < frame.points.size(); ++i)
        {
            const float R = R_of(i);
            const auto  r = m.responsibilities(frame.points[i], s, R);
            for (int prim = 0; prim < P; ++prim)
            {
                const float w = r[prim] / R;
                if (w < 1e-6f) continue;
                const float d = m.sdf_prim(frame.points[i], s, prim);
                const VecN  J = m.sdf_jacobian(frame.points[i], s, prim);
                Id.noalias() += w * (J * J.transpose());
                bd.noalias() += -w * J * d;
            }
        }
        // Optional model-provided extra factors (e.g. bottle's occluding-contour silhouette, which pins the
        // depth-degenerate radius a symmetric cylinder's SDF cannot). Folded into the SAME GN normal
        // equations so they inform the mean AND the posterior Σ. Detected via C++23 requires, so a model
        // WITHOUT the hook (table/chair) is completely unaffected — no signature, no call.
        if constexpr (requires { m.accumulate_extra(s, frame, Id, bd); })
            m.accumulate_extra(s, frame, Id, bd);
    };

    for (int it = 0; it < m.gn_iters(); ++it)
    {
        accumulate(theta);
        // Schur-consistent within-frame common-mode marginalisation: apply the SAME saturation operator
        // (I − Id(Id+Σc⁻¹)⁻¹) to BOTH the curvature (→ Ieff) and the gradient (→ beff), so the GN MEAN sees
        // the same saturated information as the posterior Σ. The earlier code paired the capped Ieff-implied
        // step with the FULL gradient bd — a per-frame ML refit that let per-frame SHARED bias (viewpoint
        // mask under-segmentation, exactly what Σc models) move the mean at full gain → the position wander.
        // b = Id·δθ_Newton ⇒ beff = Ieff·δθ_Newton, so when the prior is weak this is still the full Newton
        // step (recovery preserved); damping only engages once the prior is competitively strong.
        const MatN IdMinv = Id * (Id + Scinv).inverse();
        const MatN Ieff   = Id - IdMinv * Id;
        const VecN beff   = bd - IdMinv * bd;
        const VecN dtheta = (P0 + Ieff).ldlt().solve(P0 * (prior_mean - theta) + beff);
        if (!dtheta.allFinite()) break;
        theta += dtheta;
        State s = State::from_vec(theta);
        m.apply_constraints(s);
        theta = s.vec();
    }
    State s = State::from_vec(theta);
    m.apply_constraints(s);
    m.canonicalize(s);          // once: model's representation fold (e.g. continuity fold over the symmetry)
    state = s;

    accumulate(state.vec());    // data information at the converged state
    const MatN IdMinv = Id * (Id + Scinv).inverse();
    const MatN Ieff   = Id - IdMinv * Id;
    Sigma = (P0 + Ieff).inverse();

    double energy = 0.0;
    for (std::size_t i = 0; i < frame.points.size(); ++i)
    {
        const float R = R_of(i);
        const auto  r = m.responsibilities(frame.points[i], state, R);
        for (int prim = 0; prim < P; ++prim)
        {
            const float d = m.sdf_prim(frame.points[i], state, prim);
            energy += 0.5 * r[prim] * d * d / R;
        }
    }
    return static_cast<float>(energy / static_cast<double>(frame.points.size()));
}

// ─── Epistemic: predicted Fisher information (next-best-view scorer) ──────────────────────────────

// Predicted Fisher information (N×N) that observing `pts` would add at variance R: I = Σₚ (1/R) J Jᵀ,
// J = ∂sdf/∂θ of the nearest primitive at p. Used by the epistemic Σ-based next-best-view scorer. NOTE:
// deliberately NOT common-mode-saturated — the NBV needs the RELATIVE, per-DOF-anisotropic information to
// rank faces ("which view resolves which DOF"), and the Woodbury cap saturates the high-info directions
// toward Σc isotropically, which INVERTS the face ranking (verified via the self_test NBV check). The
// "don't be overconfident about a single dense view" concern is instead handled where it belongs: the
// planner scores gain against the REPORTED covariance (Σ with the yaw mode-entropy folded in), so an
// unresolved DOF dominates Σ and the NBV attacks it regardless of the raw predicted magnitude.
template <int N, class Model>
Eigen::Matrix<float, N, N> predicted_information(const Model& m, const typename Model::State& state,
                                                 const std::vector<Eigen::Vector3f>& pts, float R)
{
    Eigen::Matrix<float, N, N> I = Eigen::Matrix<float, N, N>::Zero();
    const float invR = 1.0f / std::max(1e-9f, R);
    const int   P = m.n_prims();
    for (const auto& p : pts)
    {
        int best = 0; float bd = std::abs(m.sdf_prim(p, state, 0));
        for (int prim = 1; prim < P; ++prim)
        {
            const float d = std::abs(m.sdf_prim(p, state, prim));
            if (d < bd) { bd = d; best = prim; }
        }
        const Eigen::Matrix<float, N, 1> J = m.sdf_jacobian(p, state, best);
        I.noalias() += invR * (J * J.transpose());
    }
    return I;
}

}  // namespace rc::ai
