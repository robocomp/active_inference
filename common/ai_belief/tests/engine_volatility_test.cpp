// engine_volatility_test — the inferred process noise (plan rev 2 F1.2; Fable order review §2.3) on the minimal toy model.
// The engine's statistic per θ component is the EM one, E[w² | data] = q + (q/Σ̄)²(δθ² − (Σ̄ − Σ⁺)), averaged conjugately with
// the configured Q₀ worth n₀ = 1/σ_ω² pseudo-observations. Checked here:
//   (1) the first update (the class prior correcting the seed) teaches nothing: ω stays log Q₀;
//   (2) ω after the second update equals the formula, recomputed here from Σ̄ (the engine's predicted θθ), Σ⁺ and δθ;
//   (3) a large correction of a WIDE prior (a seed collapsing onto data, engine1's bed) barely moves ω — the raw-step rule
//       it replaces would have raised Q by ~δθ²/(n₀+1);
//   (4) a shove of a CONVERGED object (Σ̄ a few q) is learned: ω rises by orders of magnitude;
//   (5) without volatility() (infer off) Q is Q₀ throughout;
//   (6) PERMANENCE (owner ruling 11-25, spike-and-slab statistic) with the PRODUCTION bed class values (bed.concept.toml:
//       0.0027 moves/day, 0.5 m, dt 0.1 s ⇒ π = 3.125e-9 per step, q₀ = π·0.25 = 7.8e-10 m²): a static box keeps Q in
//       [n₀/(n₀ + n)·q₀, q₀] (it can only lower the statistic below the prior mean, and the hyperprior bounds that), and a 0.10 m
//       shove raises Q to at least a quarter of the shove's share δ²/(n₀ + n) — the Gaussian statistic would leave it at q₀.
//   (7) F1 round 7 A1: a refuted-ray-shaped θ-only term the stayed fit can free inside its prior is NOT a move (log BF < 0: the
//       data do not favour it; the evidence read at the stayed posterior mode); a Gaussian term saying the object is 0.20 m off IS one.
// Run: ./engine_volatility_test
#include <cmath>
#include <cstdio>

#include "toy_model.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

struct VolBox : toy::ToyBox
{
    rc::ai::engine::VolatilityConfig volatility() const { rc::ai::engine::VolatilityConfig v; v.infer = true; return v; }
};
// the bed class's permanence prior (production values)
constexpr double kBedPi = 0.0027 * 0.1 / 86400.0;
struct BedBox : toy::ToyBox
{
    rc::ai::engine::VolatilityConfig volatility() const
    {
        rc::ai::engine::VolatilityConfig v; v.infer = true;
        v.jump_var = {0.25, 0.25, 0.01, 0.04, 0.04, 0.25};   // 0.5 m, the class size stds (0.10 H, 0.20 w/h), 0.5 rad
        v.jump_prob = kBedPi;
        return v;
    }
};
// (7) a θ-only term shaped like a REFUTED LiDAR ray (bed_5 cycles 1708/1709, p1_bed): n rays crossing the box `depth` deep at
// σ_s and ending behind it, E = −n·log Φ((a − cx)/s) with a = −depth, its Fisher curvature n(φ/s)²[1/Φ + 1/(1 − Φ)] (the
// beam's I_d); `quad` instead: a Gaussian term saying cx = 0.20 (a real 0.20 m move, n points at σ 1 cm).
struct RefutedBox : BedBox
{
    double a = -0.02, s_ray = 0.01, n = 5.0;
    bool quad = false;
    void accumulate_extra(const State& st, const Frame&, MatN& I, VecN& b, double* E) const
    {
        const double cx = double(st.cx);
        if (quad)
        {
            const double lam = n / (0.01 * 0.01), r = cx - 0.20;
            I(0, 0) += float(lam); b(0) += float(-lam * r);
            if (E) *E += 0.5 * lam * r * r;
            return;
        }
        const double z = (a - cx) / s_ray, phi = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
        const double Phi = 0.5 * std::erfc(-z / std::sqrt(2.0)), Phic = 0.5 * std::erfc(z / std::sqrt(2.0));
        I(0, 0) += float(n * (phi / s_ray) * (phi / s_ray) * (1.0 / Phi + 1.0 / Phic));
        b(0) += float(-n * phi / (Phi * s_ray));                       // b = −∂E/∂cx
        if (E) *E += -n * std::log(Phi);
    }
};
using Engine = rc::ai::engine::PersistentEngine<VolBox>;
using Plain  = rc::ai::engine::PersistentEngine<toy::ToyBox>;
constexpr int N = Engine::N;

static rc::ai::engine::FrameLatents<N> cam_latents()
{
    rc::ai::engine::FrameLatents<N> fl;
    fl.rho_cam.fill(1.0f); fl.sig2_cam.fill(1e-6f); fl.fresh_c.fill(1e-8f);
    fl.rho_chain = 1.0f;
    fl.sigma_pose = Eigen::Vector3d(1e-8, 1e-8, 1e-8).asDiagonal();
    fl.robot_xy = {0.2f, -0.1f};
    fl.camera_measured = true;
    return fl;
}
template <class M> static void seed(M& m, const toy::ToyState& s)
{
    m.s = s;
    m.prior_mean << 0.f, 0.f, 0.8f, 0.6f, 0.4f, 0.f;
    m.Sigma = m.prior_diag.asDiagonal();
    m.q_diag.setConstant(0.005f * 0.005f);                              // the countable configs' Q₀ (AI2ProcessStdM 0.005)
}

int main()
{
    std::printf("engine_volatility_test\n");
    const toy::ToyState truth{1.0f, 2.0f, 0.8f, 0.6f, 0.4f, 0.3f};
    const auto f = toy::make_frame(toy::surface_points(truth, 6), {});
    const double n0 = 1.0 / (2.0 * 2.0);
    VolBox m;
    seed(m, toy::ToyState{1.05f, 1.96f, 0.85f, 0.57f, 0.43f, 0.35f});
    const Eigen::Matrix<double, N, 1> om0 = m.process_noise_diag().cast<double>().array().log().matrix();
    Engine eng;
    // (1)
    eng.update(m, f, cam_latents());
    const auto om1 = eng.log_volatility(m);
    std::printf("  (1) first update: max |d omega| %.3e\n", (om1 - om0).cwiseAbs().maxCoeff());
    CHECK((om1 - om0).cwiseAbs().maxCoeff() == 0.0);
    // (2) the formula, recomputed from the engine's Σ̄ / Σ⁺ / δθ
    {
        const Eigen::Matrix<float, N, 1> th_pred = m.state().vec();
        eng.update(m, f, cam_latents());
        const auto& Pb = eng.dbg_Pbar_tt();
        const auto om2 = eng.log_volatility(m);
        double worst = 0.0;
        for (int i = 0; i < N; ++i)
        {
            const double q = double(m.process_noise_diag()(i)), sbar = double(Pb(i, i)), spost = eng.P()(i, i);
            const double d = double(m.state().vec()(i)) - double(th_pred(i));
            const double ew2 = q + (q / sbar) * (q / sbar) * (d * d - (sbar - spost));
            const double want = std::log((n0 * q + ew2) / (n0 + 1.0));
            worst = std::max(worst, std::abs(om2(i) - want));
        }
        std::printf("  (2) second update: max |omega - formula| %.3e\n", worst);
        CHECK(worst < 1e-5);   // Σ̄ is read back through the float mirror dbg_Pbar_tt
    }
    // (3) a 0.3 m correction of a wide prior: the predicted Σ̄ is set wide (as a long predict-only stretch leaves it)
    {
        VolBox a;
        seed(a, toy::ToyState{1.3f, 2.0f, 0.8f, 0.6f, 0.4f, 0.3f});
        Engine e;
        e.update(a, f, cam_latents());                                  // the first update (class prior)
        e.set_first_update_done(true);
        a.s.cx = 1.3f;                                                  // the mean left 0.3 m off ...
        e.P_for_test()(0, 0) = 0.09;                                    // ... under a 0.3 m prior
        a.set_covariance(e.Sigma_theta());
        const double before = e.log_volatility(a)(0);
        e.update(a, f, cam_latents());
        const double dcx = double(a.state().cx) - 1.3, after = e.log_volatility(a)(0);
        const double q0 = double(a.process_noise_diag()(0));
        const double raw = std::log((n0 * q0 + dcx * dcx) / (n0 + 1.0));   // the rule it replaces (one step, from Q₀)
        std::printf("  (3) wide prior, step %.3f m: omega_cx %.3f -> %.3f (raw-step rule would give %.3f)\n", dcx, before, after, raw);
        CHECK(std::abs(dcx) > 0.2);
        CHECK(after - before < 0.05);
        CHECK(raw - before > 5.0);
    }
    // (4) a converged object shoved by 0.10 m
    {
        VolBox a;
        seed(a, toy::ToyState{1.05f, 1.96f, 0.85f, 0.57f, 0.43f, 0.35f});
        Engine e;
        for (int k = 0; k < 30; ++k) e.update(a, f, cam_latents());
        const double before = e.log_volatility(a)(0);
        toy::ToyState moved = truth; moved.cx += 0.10f;
        const auto g = toy::make_frame(toy::surface_points(moved, 6), {});
        const double sbar = e.P()(0, 0) + double(a.process_noise_diag()(0));
        e.update(a, g, cam_latents());
        const double after = e.log_volatility(a)(0);
        std::printf("  (4) converged (sigma_cx %.4f m), shoved 0.10 m: omega_cx %.3f -> %.3f\n", std::sqrt(sbar), before, after);
        CHECK(after - before > 1.0);
    }
    // (5) infer off: Q is the configured Q₀
    {
        toy::ToyBox a;
        seed(a, toy::ToyState{1.05f, 1.96f, 0.85f, 0.57f, 0.43f, 0.35f});
        Plain e;
        for (int k = 0; k < 5; ++k) e.update(a, f, cam_latents());
        CHECK(e.process_noise(a) == a.process_noise_diag());
    }
    // (6) permanence, production bed values
    {
        BedBox a;
        a.s = toy::ToyState{1.05f, 1.96f, 0.85f, 0.57f, 0.43f, 0.35f};
        a.prior_mean << 0.f, 0.f, 0.8f, 0.6f, 0.4f, 0.f;
        a.Sigma = a.prior_diag.asDiagonal();
        const auto vc = a.volatility();
        for (int i = 0; i < N; ++i) a.q_diag(i) = float(vc.jump_prob * vc.jump_var[std::size_t(i)]);
        const double q0 = double(a.q_diag(0)), n0 = 0.25;
        rc::ai::engine::PersistentEngine<BedBox> e;
        constexpr int kStatic = 30;
        for (int k = 0; k < kStatic; ++k) e.update(a, f, cam_latents());
        const double Qs = std::exp(e.log_volatility(a)(0));
        double n = 0.0; for (int k = 0; k < kStatic - 1; ++k) n = 0.995 * n + 1.0;   // the first update teaches nothing
        std::printf("  (6) bed permanence: q0 %.3e ; after %d static updates Q_cx %.3e (bounds [%.3e, %.3e])\n", q0, kStatic, Qs,
                    n0 / (n0 + n) * q0, q0);
        CHECK(Qs <= q0 * (1.0 + 1e-6) and Qs >= n0 / (n0 + n) * q0 * (1.0 - 1e-6));
        toy::ToyState moved = truth; moved.cx += 0.10f;
        e.update(a, toy::make_frame(toy::surface_points(moved, 6), {}), cam_latents());
        const double Qm = std::exp(e.log_volatility(a)(0)), share = 0.10 * 0.10 / (n0 + 0.995 * n + 1.0);
        std::printf("      shoved 0.10 m: Q_cx %.3e (omega %.2f -> %.2f) ; the shove's share %.3e\n", Qm, std::log(Qs), std::log(Qm), share);
        CHECK(Qm >= 0.25 * share);
    }
    // (7) F1 round 7 A1: the move evidence of a non-quadratic term is read at the STAYED POSTERIOR MODE. Five refuted rays 2σ_s
    //     deep under a 2 cm prior on cx (the LiDAR's ego-motion reach on bed_5): the stayed fit frees them inside its prior
    //     (cx ≈ −3.4 cm), so P(moved) must stay at the prior level (the factor at the PREDICTED state put the data's mean
    //     ~0.2 m away and gave log BF 26.5, P(moved) 0.999). A Gaussian term saying cx = 0.20 (a real move, 10 σ of the stayed prior) is detected.
    for (const bool quad : {false, true})
    {
        RefutedBox a;
        a.s = toy::ToyState{0.f, 0.f, 0.8f, 0.6f, 0.4f, 0.f};
        a.prior_mean << 0.f, 0.f, 0.8f, 0.6f, 0.4f, 0.f;
        a.Sigma = a.prior_diag.asDiagonal();
        const auto vc = a.volatility();
        for (int i = 0; i < N; ++i) a.q_diag(i) = float(vc.jump_prob * vc.jump_var[std::size_t(i)]);
        auto pts = toy::surface_points(a.s, 2);
        toy::ToyFrame fr = toy::make_frame(pts, {});
        fr.R.assign(pts.size(), 1.0f);                                  // the camera barely informs (σ 1 m per point)
        rc::ai::engine::PersistentEngine<RefutedBox> e;
        a.n = 0.0;                                                      // the first update: no extra term
        e.update(a, fr, cam_latents());
        a.s = toy::ToyState{0.f, 0.f, 0.8f, 0.6f, 0.4f, 0.f};
        e.P_for_test().row(0).setZero(); e.P_for_test().col(0).setZero();
        e.P_for_test()(0, 0) = 0.02 * 0.02;                             // the stayed prior on cx: 2 cm
        a.set_covariance(e.Sigma_theta());
        a.n = 5.0; a.quad = quad;
        e.update(a, fr, cam_latents());
        std::printf("  (7) %s: cx %+.4f, log BF %.3f, P(moved) %.3e\n", quad ? "real 0.20 m move (Gaussian term)" :
                    "5 refuted rays 2 sigma_s deep", double(a.s.cx), e.last_jump_log_bf(), e.last_jump_p());
        if (quad) CHECK(e.last_jump_p() > 0.5);                         // moved is the more probable hypothesis
        else      CHECK(e.last_jump_log_bf() < 0.0);                     // the data do not favour a move (fix round 1, I2)
    }
    std::printf(bad ? "engine_volatility_test: %d FAILED\n" : "engine_volatility_test: OK\n", bad);
    return bad != 0;
}
