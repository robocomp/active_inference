// persistent_update_test — the persistent shared-error latents (spec 2026-11-15 lidar-measures §7c.2): fresh-covariance
// marginalisation, latent dynamics, the inverse-free joint step with the bias restart, and the joint covariance.
// Header-only under test (common/ai_belief/persistent_update.h); nothing in the agents uses it yet. Run: ./persistent_update_test
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <span>
#include <Eigen/Dense>
#include "ai_belief/persistent_update.h"
#include "ai_belief/recursive_laplace.h"

namespace pu = rc::ai::persistent;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

template <class A, class B>
static double rel_err(const A& a, const B& b) { return (a - b).norm() / std::max(b.norm(), 1e-300); }

// ─── 1-D: the restart reproduces the eliminated (today's) update at every iterate ──────────────────
static void restart_matches_saturated_1d()
{
    using V2 = Eigen::Matrix<double, 2, 1>;
    using M2 = Eigen::Matrix<double, 2, 2>;
    const double z = 0.1, Idat = 1e4;
    M2 Pbar = M2::Zero(); Pbar(0, 0) = 0.0036; Pbar(1, 1) = 4e-4;
    const V2 xbar = V2::Zero();
    // A linear datum on θ + b: the log-likelihood gradient at the linearisation point x̃ is I (z − θ̃ − b̃).
    const auto channel_at = [&](const V2& xt)
    {
        pu::Channel<1, 1> c;
        c.info.I(0, 0) = Idat;
        c.info.g(0)    = Idat * (z - xt(0) - xt(1));
        c.B(0, 0)      = 1.0;
        return c;
    };
    pu::FreshInfo<1> none; none.I.setZero(); none.g.setZero();

    // Restarting: x̃_k = (θ_k, b̄).
    V2 x = xbar;
    std::printf("restart_matches_saturated_1d  restarting :");
    for (int k = 0; k < 4; ++k)
    {
        const V2 xt(x(0), xbar(1));
        const std::array<pu::Channel<1, 1>, 1> ch{channel_at(xt)};
        x = pu::joint_step<1, 1>(xt, xbar, Pbar, std::span<const pu::Channel<1, 1>>(ch), none);
        std::printf(" %.6f", x(0));
        CHECK(std::abs(x(0) - 0.0878) < 1e-4);
    }
    std::printf("\n");

    // Not restarting the latent BELIEF: the latent's posterior (mean as linearisation point, variance as prior) is
    // carried into the next iterate, so the same frame's data is counted again on b each iterate; b's variance
    // shrinks like 1/k and θ drifts toward the b-free answer z·Pθ/(Pθ + 1/I) = 0.0973. Printed only.
    x = xbar;
    M2 Pk = Pbar;
    std::printf("restart_matches_saturated_1d  carrying b :");
    for (int k = 0; k < 400; ++k)
    {
        const std::array<pu::Channel<1, 1>, 1> ch{channel_at(x)};
        const V2 xn = pu::joint_step<1, 1>(x, xbar, Pk, std::span<const pu::Channel<1, 1>>(ch), none);
        const M2 C  = pu::joint_cov<1, 1>(Pk, std::span<const pu::Channel<1, 1>>(ch), none);
        Pk(1, 1) = C(1, 1);
        x = xn;
        if (k < 4 or k == 39 or k == 399) std::printf(" [%d]%.6f", k, x(0));
    }
    std::printf("   (b-free limit %.6f)\n", z * 0.0036 / (0.0036 + 1.0 / Idat));
}

// ─── ρ = 0, one channel, B = I: the joint update equals today's per-frame Woodbury update ──────────
using M6f = Eigen::Matrix<float, 6, 6>;
using V6f = Eigen::Matrix<float, 6, 1>;
using M6d = Eigen::Matrix<double, 6, 6>;
using V6d = Eigen::Matrix<double, 6, 1>;
using M12 = Eigen::Matrix<double, 12, 12>;
using V12 = Eigen::Matrix<double, 12, 1>;

struct Rho0Fixture { M6f Id; V6f bd; M6f Sc; M6f Spred; };
static M6f random_spd(std::mt19937& rng, float scale)
{
    std::normal_distribution<float> n(0.0f, 1.0f);
    M6f A; for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) A(i, j) = n(rng);
    M6f S = scale * (A * A.transpose() / 6.0f + 0.2f * M6f::Identity());
    return 0.5f * (S + S.transpose());
}
static Rho0Fixture rho0_fixture()
{
    std::mt19937 rng(7);
    Rho0Fixture f;
    f.Id    = random_spd(rng, 1e4f);     // camera-scale data information (/m², /rad²)
    f.Sc    = random_spd(rng, 4e-4f);    // common-mode covariance ~ (2 cm)²
    f.Spred = random_spd(rng, 1e-3f);    // predicted posterior ~ (3 cm)²
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (int i = 0; i < 6; ++i) f.bd(i) = 50.0f * n(rng);
    return f;
}
static M12 rho0_joint_cov_and_mean(const Rho0Fixture& f, V12& xnext)
{
    M12 Pbar = M12::Zero();
    Pbar.topLeftCorner<6, 6>()     = f.Spred.cast<double>();
    Pbar.bottomRightCorner<6, 6>() = f.Sc.cast<double>();
    pu::Channel<6, 6> c;
    c.info = pu::marginalise_fresh<6>(f.Id, f.bd, M6f::Zero());   // fresh covariance 0 ⇒ the identity
    c.B    = M6d::Identity();
    const std::array<pu::Channel<6, 6>, 1> ch{c};
    pu::FreshInfo<6> none; none.I.setZero(); none.g.setZero();
    const V12 zero = V12::Zero();
    xnext = pu::joint_step<6, 6>(zero, zero, Pbar, std::span<const pu::Channel<6, 6>>(ch), none);
    return pu::joint_cov<6, 6>(Pbar, std::span<const pu::Channel<6, 6>>(ch), none);
}
static void rho0_equals_saturate()
{
    const Rho0Fixture f = rho0_fixture();
    V12 x; const M12 P = rho0_joint_cov_and_mean(f, x);
    const V6d theta = x.head<6>();
    const M6d Stt   = P.topLeftCorner<6, 6>();

    // Today's update, through the engine's own float saturate (prior mean 0, θ₀ = 0, one GN step).
    const M6f P0f = f.Spred.inverse();
    const auto [Ieff, beff] = rc::ai::saturate<6>(f.Id, f.bd, f.Sc.inverse());
    const V6d th_f = (P0f + Ieff).ldlt().solve(beff).cast<double>();
    const M6d S_f  = (P0f + Ieff).inverse().cast<double>();

    // The same in double (saturate re-implemented), to separate float noise from a modelling difference.
    const M6d P0 = f.Spred.cast<double>().inverse(), Sci = f.Sc.cast<double>().inverse();
    const M6d M  = (f.Id.cast<double>() + Sci).inverse();
    M6d Ie = Sci - Sci * M * Sci; Ie = 0.5 * (Ie + Ie.transpose());
    const V6d be  = Sci * M * f.bd.cast<double>();
    const V6d th_d = (P0 + Ie).ldlt().solve(be);
    const M6d S_d  = (P0 + Ie).inverse();

    const double e_th_f = rel_err(theta, th_f), e_S_f = rel_err(Stt, S_f);
    const double e_th_d = rel_err(theta, th_d), e_S_d = rel_err(Stt, S_d);
    std::printf("rho0_equals_saturate  vs float saturate: theta rel %.3e  Sigma rel %.3e\n", e_th_f, e_S_f);
    std::printf("rho0_equals_saturate  vs double re-impl: theta rel %.3e  Sigma rel %.3e\n", e_th_d, e_S_d);
    std::printf("rho0_equals_saturate  theta = %.6e %.6e %.6e %.6e %.6e %.6e\n",
                theta(0), theta(1), theta(2), theta(3), theta(4), theta(5));
    CHECK(e_th_d < 1e-5);
    CHECK(e_S_d < 1e-5);
    CHECK(e_th_f < 1e-5);
    CHECK(e_S_f < 1e-5);
}

// ─── fresh marginalisation: identity at Σf = 0, bounded otherwise ──────────────────────────────────
static void fresh_zero_is_identity()
{
    const Rho0Fixture f = rho0_fixture();
    const auto a = pu::marginalise_fresh<6>(f.Id, f.bd, M6f::Zero());
    CHECK(a.I == f.Id.cast<double>());
    CHECK(a.g == f.bd.cast<double>());

    const M6f Id = 2.19e6f * M6f::Identity();
    const M6f Sf = 1e-6f * M6f::Identity();
    const auto b = pu::marginalise_fresh<6>(Id, V6f::Ones(), Sf);
    const Eigen::SelfAdjointEigenSolver<M6d> es(b.I), ed(Id.cast<double>() - b.I);
    std::printf("fresh_zero_is_identity  I(0,0) = %.6e (expected %.6e)  min eig %.3e  min eig(Id-I) %.3e\n",
                b.I(0, 0), 2.19e6 / (1.0 + 2.19e6 * 1e-6), es.eigenvalues().minCoeff(), ed.eigenvalues().minCoeff());
    CHECK(b.I.allFinite() and b.g.allFinite());
    CHECK(es.eigenvalues().minCoeff() > 0.0);
    CHECK(ed.eigenvalues().minCoeff() >= 0.0);
}

// ─── a zero-variance latent (chain cov disabled) is valid: no NaN, it stays 0, D = 1 ──────────────
static void zero_variance_latent_runs()
{
    using V4 = Eigen::Matrix<double, 4, 1>;
    using M4 = Eigen::Matrix<double, 4, 4>;
    using V2 = Eigen::Matrix<double, 2, 1>;
    M4 P = M4::Zero(); P(0, 0) = 1e-3; P(1, 1) = 1e-3; P(3, 3) = 4e-4;   // latent 0 (row 2): zero variance
    V4 x = V4::Zero();
    const V2 rho(0.5, 0.5), s_prev(0.0, 0.02), s_now(0.0, 0.02);
    pu::predict_latents<2, 2>(P, x, rho, s_prev, s_now);
    CHECK(x.allFinite() and P.allFinite());
    CHECK(P(2, 2) == 0.0 and P.row(2).isZero(0.0) and P.col(2).isZero(0.0));

    pu::Channel<2, 2> c;
    c.info.I = 1e4 * Eigen::Matrix2d::Identity(); c.info.g = V2(30.0, -12.0);
    c.B = Eigen::Matrix2d::Identity();
    const std::array<pu::Channel<2, 2>, 1> ch{c};
    pu::FreshInfo<2> none; none.I.setZero(); none.g.setZero();
    const V4 xn = pu::joint_step<2, 2>(x, x, P, std::span<const pu::Channel<2, 2>>(ch), none);
    const M4 Pn = pu::joint_cov<2, 2>(P, std::span<const pu::Channel<2, 2>>(ch), none);
    std::printf("zero_variance_latent_runs  x = %.6e %.6e %.6e %.6e   P(2,2) = %.3e\n", xn(0), xn(1), xn(2), xn(3), Pn(2, 2));
    CHECK(xn.allFinite() and Pn.allFinite());
    CHECK(xn(2) == 0.0 and Pn(2, 2) == 0.0);

    // D = 1 when σ_prev = 0: with ρ = 1 a carried latent mean is untouched (no 0/0).
    V4 x2 = V4::Zero(); x2(2) = 0.3;
    M4 P2 = P;
    pu::predict_latents<2, 2>(P2, x2, V2(1.0, 1.0), s_prev, s_now);
    CHECK(x2(2) == 0.3 and x2.allFinite() and P2.allFinite());
}

// ─── ρ = 1, D = 1: n identical updates of one look = one update with n·Id ─────────────────────────
static void rho1_n_sweeps_equal_one_with_n_id()
{
    using V9 = Eigen::Matrix<double, 9, 1>;
    using M9 = Eigen::Matrix<double, 9, 9>;
    using M63 = Eigen::Matrix<double, 6, 3>;
    using V3 = Eigen::Vector3d;
    const Rho0Fixture f = rho0_fixture();
    M63 B = M63::Zero(); B(0, 0) = 1.0; B(1, 1) = 1.0; B(5, 2) = 1.0;   // latents on cx, cy, yaw
    M9 Pbar = M9::Zero();
    Pbar.topLeftCorner<6, 6>() = f.Spred.cast<double>();
    Pbar(6, 6) = 4e-4; Pbar(7, 7) = 9e-4; Pbar(8, 8) = 1e-3;
    V9 xbar = V9::Zero();
    V6d z; z << 0.04, -0.02, 0.01, 0.03, -0.01, 0.05;     // the datum on θ + B b
    pu::FreshInfo<6> none; none.I.setZero(); none.g.setZero();
    // Linear channel: gradient of the log-likelihood at x̃ is Id (z − θ̃ − B b̃).
    const auto channel_at = [&](const V9& xt, double n)
    {
        pu::Channel<6, 3> c;
        c.info.I = n * f.Id.cast<double>();
        c.info.g = c.info.I * (z - xt.head<6>() - B * xt.tail<3>());
        c.B = B;
        return c;
    };

    const V3 rho = V3::Ones(), sig(0.02, 0.03, std::sqrt(1e-3));
    V9 x = xbar; M9 P = Pbar;
    for (int k = 0; k < 10; ++k)
    {
        pu::predict_latents<6, 3>(P, x, rho, sig, sig);
        const std::array<pu::Channel<6, 3>, 1> ch{channel_at(x, 1.0)};
        const V9 xn = pu::joint_step<6, 3>(x, x, P, std::span<const pu::Channel<6, 3>>(ch), none);
        P = pu::joint_cov<6, 3>(P, std::span<const pu::Channel<6, 3>>(ch), none);
        x = xn;
    }
    const std::array<pu::Channel<6, 3>, 1> ch1{channel_at(xbar, 10.0)};
    const V9 x1 = pu::joint_step<6, 3>(xbar, xbar, Pbar, std::span<const pu::Channel<6, 3>>(ch1), none);
    const M9 P1 = pu::joint_cov<6, 3>(Pbar, std::span<const pu::Channel<6, 3>>(ch1), none);
    const double e_th = rel_err(x.head<6>(), x1.head<6>());
    const double e_S  = rel_err(P.topLeftCorner<6, 6>(), P1.topLeftCorner<6, 6>());
    std::printf("rho1_n_sweeps_equal_one_with_n_id  theta rel %.3e  Sigma_tt rel %.3e  (theta_x %.6e vs %.6e)\n",
                e_th, e_S, x(0), x1(0));
    CHECK(e_th < 1e-6);
    CHECK(e_S < 1e-6);
}

// ─── the chain latent is shared by both channels; the sensor latents add in parallel ───────────────
static void shared_latent_position_information()
{
    using V4 = Eigen::Matrix<double, 4, 1>;
    using M4 = Eigen::Matrix<double, 4, 4>;
    using M13 = Eigen::Matrix<double, 1, 3>;
    const double s_ch = 4e-4, s_c = 1e-4, s_L = 9e-4, s_th = 1e-2;
    M4 Pbar = M4::Zero(); Pbar(0, 0) = s_th; Pbar(1, 1) = s_ch; Pbar(2, 2) = s_c; Pbar(3, 3) = s_L;
    pu::Channel<1, 3> cam, lid;
    cam.info.I(0, 0) = 1e12; cam.info.g(0) = 0.0; cam.B = M13(1.0, 1.0, 0.0);
    lid.info.I(0, 0) = 1e12; lid.info.g(0) = 0.0; lid.B = M13(1.0, 0.0, 1.0);
    const std::array<pu::Channel<1, 3>, 2> ch{cam, lid};
    pu::FreshInfo<1> none; none.I.setZero(); none.g.setZero();
    const M4 P = pu::joint_cov<1, 3>(Pbar, std::span<const pu::Channel<1, 3>>(ch), none);
    const double info     = 1.0 / P(0, 0) - 1.0 / s_th;
    const double expected = 1.0 / (s_ch + 1.0 / (1.0 / s_c + 1.0 / s_L));
    std::printf("shared_latent_position_information  data info %.6e  expected %.6e  rel %.3e\n",
                info, expected, std::abs(info - expected) / expected);
    CHECK(std::abs(info - expected) / expected < 1e-4);
}

static void cov_symmetric_psd()
{
    const Rho0Fixture f = rho0_fixture();
    V12 x; const M12 P = rho0_joint_cov_and_mean(f, x);
    const double asym = (P - P.transpose()).cwiseAbs().maxCoeff();
    const double mine = Eigen::SelfAdjointEigenSolver<M12>(P).eigenvalues().minCoeff();
    std::printf("cov_symmetric_psd  max|P-P^T| %.3e  min eig %.3e\n", asym, mine);
    CHECK(asym <= 1e-12);
    CHECK(mine >= 0.0);
}

int main()
{
    restart_matches_saturated_1d();
    rho0_equals_saturate();
    fresh_zero_is_identity();
    zero_variance_latent_runs();
    rho1_n_sweeps_equal_one_with_n_id();
    shared_latent_position_information();
    cov_symmetric_psd();
    std::printf(bad ? "persistent_update_test: %d FAILED\n" : "persistent_update_test: all passed\n", bad);
    return bad ? 1 : 0;
}
