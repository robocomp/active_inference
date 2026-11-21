/*
 * common/ai_belief/persistent_update.h  —  persistent shared-error latents (header-only, pure)
 *
 * Spec docs/superpowers/specs/2026-11-15-lidar-measures-design.md §7c.2. The per-frame Woodbury `saturate` of
 * recursive_laplace.h draws each frame's shared error afresh; the world is static, so a parked robot sees the same
 * surface through the same localisation, deprojection and mask bias frame after frame. Here the shared errors are
 * LATENTS b ∈ ℝᴷ carried beside the object state θ ∈ ℝᴺ, state x = (θ, b), covariance P ((N+K)×(N+K), double).
 *
 *   Dynamics, per latent DOF j:  b_t = ρ_j D_j b_{t−1} + w_t,  D_j = σ_now/σ_prev,
 *                                w_t ~ N(0, (1 − ρ_j²) σ_now²)        ⇒ marginal exactly σ_now², lag-1 corr. ρ_j.
 *   σ_prev = 0 means there was no error source before this step (a zero-variance latent: chain cov disabled, a channel
 *   never measured, a fresh birth). The source is then NEW, uncorrelated with the (non-existent) old one: ρ = 0 for that
 *   transition, so the latent is drawn afresh with variance σ_now² — not (1 − ρ²)σ_now², which at ρ = 1 would leave a new
 *   error source at variance 0 for ever (ruling T4-1c).
 *   Channel c measures θ + B_c b; its linearised data enters as information (I_c, g_c) on that quantity, with
 *   g_c the log-likelihood gradient at the linearisation point (the engine's bd = −Σ w J d). H_c = [𝟙 B_c].
 *   A θ-only channel (I_θ, g_θ), H_θ = [𝟙 0], carries what the biases must not absorb (one-sided constraints, anchors).
 *   Joint step (inverse-free; no P̄⁻¹, so zero-variance latents are valid):
 *       x_{k+1} = x̃_k + (𝟙 + P̄W)⁻¹ [P̄G − (x̃_k − x̄)],   W = Σ H_cᵀ I_c H_c + H_θᵀ I_θ H_θ,  G = Σ H_cᵀ g_c + H_θᵀ g_θ
 *       P       = (𝟙 + P̄W)⁻¹ P̄   (symmetrised)
 *   The caller restarts the biases at their prediction each GN iterate: x̃_k = (θ_k, b̄), keeps θ_{k+1}, and after
 *   the last iterate keeps the whole x_{k+1} (its b is the posterior b). At ρ = 0, eliminating b reproduces today's
 *   saturated step at every iterate; carrying the iterate's b belief instead re-counts the frame on b.
 *
 * All solves in double; fixed-size Eigen throughout (N, K known at compile time). No thresholds: the only case split is
 * σ_prev = 0 (a new error source: ρ = 0 for that transition, no 0/0).
 */
#pragma once

#include <span>
#include <Eigen/Dense>

namespace rc::ai::persistent
{

// Information form of one channel's linearised data on the quantity it measures.
template <int N>
struct FreshInfo
{
    Eigen::Matrix<double, N, N> I = Eigen::Matrix<double, N, N>::Zero();
    Eigen::Matrix<double, N, 1> g = Eigen::Matrix<double, N, 1>::Zero();
};

// Marginalise the channel's per-frame FRESH covariance Σf (ego-motion: drawn anew each frame, outside every latent),
// inverse-free: I = (𝟙 + Id Σf)⁻¹ Id, g = (𝟙 + Id Σf)⁻¹ bd. Exactly the identity at Σf = 0 (parked) — unlike
// `saturate`, which needs Σc⁻¹.
template <int N>
FreshInfo<N> marginalise_fresh(const Eigen::Matrix<float, N, N>& Id,
                               const Eigen::Matrix<float, N, 1>& bd,
                               const Eigen::Matrix<float, N, N>& Sigma_f)
{
    using MatN = Eigen::Matrix<double, N, N>;
    const MatN Idd = Id.template cast<double>();
    const MatN A   = MatN::Identity() + Idd * Sigma_f.template cast<double>();
    const Eigen::PartialPivLU<MatN> lu(A);
    FreshInfo<N> out;
    out.I = lu.solve(Idd);
    out.I = 0.5 * (out.I + out.I.transpose());   // symmetric in exact arithmetic
    out.g = lu.solve(bd.template cast<double>());
    return out;
}

// A channel measuring θ + B b.
template <int N, int K>
struct Channel
{
    FreshInfo<N> info;
    Eigen::Matrix<double, N, K> B = Eigen::Matrix<double, N, K>::Zero();
};

// Predict one latent block (x offset N + off, size n; spec 7c.7 a). The latent evolves b' = R F b + w, R = diag(ρ),
// Var w = S Σ_now S, S = diag(√(1 − ρ²)): a transport F, a correlation ρ with the past, and fresh variance that tops the
// marginal up to Σ_now. NO rescale: a change of our own σ is not an observation. Rows and columns of the block (θ–b and
// b–b cross blocks included) go by R F. new_source ⇒ R = 0: x_b = 0, cross blocks 0, P_bb = Σ_now.
template <int N, int K>
void predict_block(Eigen::Matrix<double, N + K, N + K>& P,
                   Eigen::Matrix<double, N + K, 1>& x,
                   int off, int n,
                   const Eigen::MatrixXd& F,
                   const Eigen::VectorXd& rho,
                   const Eigen::MatrixXd& Sigma_now,
                   bool new_source)
{
    const int r = N + off;
    const Eigen::VectorXd rv = new_source ? Eigen::VectorXd::Zero(n) : Eigen::VectorXd(rho);
    const Eigen::MatrixXd T = rv.asDiagonal() * F;                       // R F
    const Eigen::VectorXd S = (1.0 - rv.array().square()).sqrt().matrix();
    x.segment(r, n) = T * x.segment(r, n).eval();
    P.middleRows(r, n) = (T * P.middleRows(r, n)).eval();
    P.middleCols(r, n) = (P.middleCols(r, n) * T.transpose()).eval();
    P.block(r, r, n, n) += S.asDiagonal() * Sigma_now * S.asDiagonal();
}

namespace detail
{
// W = Σ H_cᵀ I_c H_c + H_θᵀ I_θ H_θ and G = Σ H_cᵀ g_c + H_θᵀ g_θ, H_c = [𝟙 B_c], H_θ = [𝟙 0].
template <int N, int K>
void information(std::span<const Channel<N, K>> channels, const FreshInfo<N>& theta_only,
                 Eigen::Matrix<double, N + K, N + K>& W, Eigen::Matrix<double, N + K, 1>& G)
{
    W.setZero(); G.setZero();
    W.template topLeftCorner<N, N>() = theta_only.I;
    G.template head<N>()             = theta_only.g;
    for (const auto& c : channels)
    {
        const Eigen::Matrix<double, N, K> IB = c.info.I * c.B;
        W.template topLeftCorner<N, N>()     += c.info.I;
        W.template topRightCorner<N, K>()    += IB;
        W.template bottomLeftCorner<K, N>()  += IB.transpose();
        W.template bottomRightCorner<K, K>() += c.B.transpose() * IB;
        G.template head<N>()                 += c.info.g;
        G.template tail<K>()                 += c.B.transpose() * c.info.g;
    }
}
}  // namespace detail

// One damped inverse-free Gauss-Newton step (spec 7c.7 d): (𝟙 + P̄(W + λM)) δ = P̄G − (x̃ − x̄), M = blockdiag(diag W_θθ, 0).
// y = G − (W + λM)δ is the prior's pull in information form (P̄⁻¹(x_new − x̄) when P̄ is invertible).
// damp_latents (the smooth beam model, spec 7c.8; Task 19): M = diag W over θ AND b (Levenberg–Marquardt on the whole joint
// state). The countable belief uses it with the CONSISTENT linearisation (x̃_k = (θ_k, b_k), each channel linearised at
// θ_k + B b_k; spec 7c.7 d's caveat): with M on θ only, λ → ∞ leaves the latents' undamped step, which on a non-quadratic
// channel (a ray refuted by a foreign return behind the model, whose curvature the beam's W does not carry) overshoots at
// every λ, so no proposal is ever accepted. The fixed point is unchanged (δ = 0 solves the system for every λ and M),
// x_new − x̄ = P̄y still holds (prior_energy unchanged), a zero-variance latent keeps δ = 0 (its row of P̄ is zero), and
// λ = 0 is the same step either way. false: M = blockdiag(diag W_θθ, 0), the 7c.7 (d) step bit for bit.
template <int N, int K>
struct DampedStep { Eigen::Matrix<double, N + K, 1> delta, y; };

template <int N, int K>
DampedStep<N, K> joint_step_damped(const Eigen::Matrix<double, N + K, 1>& x_tilde,
                                   const Eigen::Matrix<double, N + K, 1>& x_bar,
                                   const Eigen::Matrix<double, N + K, N + K>& P_bar,
                                   std::span<const Channel<N, K>> channels,
                                   const FreshInfo<N>& theta_only, double lambda, bool damp_latents = false)
{
    using MatD = Eigen::Matrix<double, N + K, N + K>;
    using VecD = Eigen::Matrix<double, N + K, 1>;
    MatD W; VecD G;
    detail::information<N, K>(channels, theta_only, W, G);
    MatD Wl = W;
    const int nd = damp_latents ? N + K : N;
    for (int i = 0; i < nd; ++i) Wl(i, i) += lambda * W(i, i);
    const MatD A = MatD::Identity() + P_bar * Wl;
    DampedStep<N, K> s;
    s.delta = A.partialPivLu().solve(P_bar * G - (x_tilde - x_bar));
    s.y     = G - Wl * s.delta;
    return s;
}

template <int N, int K>
double prior_energy(const DampedStep<N, K>& s, const Eigen::Matrix<double, N + K, 1>& x_new,
                    const Eigen::Matrix<double, N + K, 1>& x_bar)
{
    return 0.5 * s.y.dot(x_new - x_bar);
}

// x̃ + the undamped step.
template <int N, int K>
Eigen::Matrix<double, N + K, 1> joint_step(const Eigen::Matrix<double, N + K, 1>& x_tilde,
                                           const Eigen::Matrix<double, N + K, 1>& x_bar,
                                           const Eigen::Matrix<double, N + K, N + K>& P_bar,
                                           std::span<const Channel<N, K>> channels,
                                           const FreshInfo<N>& theta_only)
{
    return x_tilde + joint_step_damped<N, K>(x_tilde, x_bar, P_bar, channels, theta_only, 0.0).delta;
}

// The joint posterior covariance (𝟙 + P̄W)⁻¹ P̄, symmetrised. Published Σ = its θθ block.
// Evaluated as P̄ − P̄ (𝟙 + W P̄)⁻¹ W P̄ (the same matrix, push-through identity) so that a ZERO-VARIANCE latent (its row and
// column of P̄ exactly zero: Σ_pose = 0, a null ψ) keeps an exactly zero row and column: both terms are products with that
// zero row / column. The direct solve gave the column exactly zero but the row only to roundoff, and the symmetrisation
// carried that roundoff into the next prediction, where it surfaced as a −1e-35 variance (spec 7c.7 b, a Σ_pose with a
// null coordinate).
template <int N, int K>
Eigen::Matrix<double, N + K, N + K> joint_cov(const Eigen::Matrix<double, N + K, N + K>& P_bar,
                                              std::span<const Channel<N, K>> channels,
                                              const FreshInfo<N>& theta_only)
{
    using MatD = Eigen::Matrix<double, N + K, N + K>;
    using VecD = Eigen::Matrix<double, N + K, 1>;
    MatD W; VecD G;
    detail::information<N, K>(channels, theta_only, W, G);
    const MatD WP = W * P_bar;
    const MatD X  = (MatD::Identity() + WP).partialPivLu().solve(WP);
    const MatD P  = P_bar - P_bar * X;
    return 0.5 * (P + P.transpose());
}

}  // namespace rc::ai::persistent
