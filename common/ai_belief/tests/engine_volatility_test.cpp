// engine_volatility_test — the inferred process noise (plan rev 2 F1.2; Fable order review §2.3) on the minimal toy model.
// The engine's statistic per θ component is the EM one, E[w² | data] = q + (q/Σ̄)²(δθ² − (Σ̄ − Σ⁺)), averaged conjugately with
// the configured Q₀ worth n₀ = 1/σ_ω² pseudo-observations. Checked here:
//   (1) the first update (the class prior correcting the seed) teaches nothing: ω stays log Q₀;
//   (2) ω after the second update equals the formula, recomputed here from Σ̄ (the engine's predicted θθ), Σ⁺ and δθ;
//   (3) a large correction of a WIDE prior (a seed collapsing onto data, engine1's bed) barely moves ω — the raw-step rule
//       it replaces would have raised Q by ~δθ²/(n₀+1);
//   (4) a shove of a CONVERGED object (Σ̄ a few q) is learned: ω rises by orders of magnitude;
//   (5) without volatility() (infer off) Q is Q₀ throughout.
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
    std::printf(bad ? "engine_volatility_test: %d FAILED\n" : "engine_volatility_test: OK\n", bad);
    return bad != 0;
}
