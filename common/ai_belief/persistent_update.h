/*
 * common/ai_belief/persistent_update.h  —  persistent shared-error latents (header-only, pure)
 *
 * Spec docs/superpowers/specs/2026-11-15-lidar-measures-design.md §7c.2. The per-frame Woodbury `saturate` of
 * recursive_laplace.h draws each frame's shared error afresh; the world is static, so a parked robot sees the same
 * surface through the same localisation, deprojection and mask bias frame after frame. Here the shared errors are
 * LATENTS b ∈ ℝᴷ carried beside the object state θ ∈ ℝᴺ, state x = (θ, b), covariance P ((N+K)×(N+K), double).
 *
 *   Dynamics, per latent DOF j:  b_t = ρ_j D_j b_{t−1} + w_t,  D_j = σ_now/σ_prev (1 when σ_prev = 0),
 *                                w_t ~ N(0, (1 − ρ_j²) σ_now²)        ⇒ marginal exactly σ_now², lag-1 corr. ρ_j.
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
 * All solves in double; fixed-size Eigen throughout (N, K known at compile time). No thresholds: the only guard is
 * D = 1 when σ_prev = 0 (a zero-variance latent), as specified.
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

// Predict the latents: F = blockdiag(𝟙_N, diag(ρ_j D_j)) applied to x and as F P Fᵀ (θ–b and b–b cross blocks included),
// then P_bb += diag((1 − ρ²) σ_now²). θ's own prediction (and its clamp) is the caller's, on the θθ block.
template <int N, int K>
void predict_latents(Eigen::Matrix<double, N + K, N + K>& P,
                     Eigen::Matrix<double, N + K, 1>& x,
                     const Eigen::Matrix<double, K, 1>& rho,
                     const Eigen::Matrix<double, K, 1>& sigma_prev,
                     const Eigen::Matrix<double, K, 1>& sigma_now)
{
    for (int j = 0; j < K; ++j)
    {
        const double D = (sigma_prev(j) == 0.0) ? 1.0 : sigma_now(j) / sigma_prev(j);   // the specified guard
        const double f = rho(j) * D;
        const int    r = N + j;
        x(r) *= f;
        P.row(r) *= f;
        P.col(r) *= f;
        P(r, r) += (1.0 - rho(j) * rho(j)) * sigma_now(j) * sigma_now(j);
    }
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

// One inverse-free Gauss-Newton step on the joint state: x̃ + (𝟙 + P̄W)⁻¹[P̄G − (x̃ − x̄)].
template <int N, int K>
Eigen::Matrix<double, N + K, 1> joint_step(const Eigen::Matrix<double, N + K, 1>& x_tilde,
                                           const Eigen::Matrix<double, N + K, 1>& x_bar,
                                           const Eigen::Matrix<double, N + K, N + K>& P_bar,
                                           std::span<const Channel<N, K>> channels,
                                           const FreshInfo<N>& theta_only)
{
    using MatD = Eigen::Matrix<double, N + K, N + K>;
    using VecD = Eigen::Matrix<double, N + K, 1>;
    MatD W; VecD G;
    detail::information<N, K>(channels, theta_only, W, G);
    const MatD A = MatD::Identity() + P_bar * W;
    return x_tilde + A.partialPivLu().solve(P_bar * G - (x_tilde - x_bar));
}

// The joint posterior covariance (𝟙 + P̄W)⁻¹ P̄, symmetrised. Published Σ = its θθ block.
template <int N, int K>
Eigen::Matrix<double, N + K, N + K> joint_cov(const Eigen::Matrix<double, N + K, N + K>& P_bar,
                                              std::span<const Channel<N, K>> channels,
                                              const FreshInfo<N>& theta_only)
{
    using MatD = Eigen::Matrix<double, N + K, N + K>;
    using VecD = Eigen::Matrix<double, N + K, 1>;
    MatD W; VecD G;
    detail::information<N, K>(channels, theta_only, W, G);
    const MatD A = MatD::Identity() + P_bar * W;
    MatD P = A.partialPivLu().solve(P_bar);
    return 0.5 * (P + P.transpose());
}

}  // namespace rc::ai::persistent
