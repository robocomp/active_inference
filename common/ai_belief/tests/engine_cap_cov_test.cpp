// engine_cap_cov_test — an update that ends without reaching a mode does not collapse Σ (F1 round 9 b).
// joint_cov's P is the inverse curvature of the energy at the point the GN reached; it is a posterior only at a mode. An update
// stopped at the iteration cap (here every proposed step is rejected, SolverConfig::reject_all_for_test, so the cap binds as on
// k2c_table's table_2, whose first update ran 25 steps unconverged and took σ_cx 0.30 → 0.033, σ_h 0.572 → 0.036) carries the
// PRIOR moment: the covariance it returns is the predicted P̄ (the engine's dbg_Pbar_tt mirror), to the float mirror's rounding.
// A converged update of the same state and frame does shrink it (the data are informative), so the check is not vacuous.
// Before the fix the capped update returned the Laplace Σ: σ_cx ~ 1 mm against a predicted 0.30 m.
// (2) Steps accepted, then the cap (one GN step per update), the mean moved: the moved x̂ is kept and Σ = P̄ (the flagged
// threshold at persistent_engine.h, F1 round 9 fix round 2).
// Run: ./engine_cap_cov_test
#include <cmath>
#include <cstdio>
#include <limits>

#include "toy_model.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

struct CappedBox : toy::ToyBox
{
    rc::ai::engine::SolverConfig solver() const { rc::ai::engine::SolverConfig c; c.reject_all_for_test = true; return c; }
};
// steps accepted, then the cap: one GN step per update
struct OneStepBox : toy::ToyBox
{
    rc::ai::engine::SolverConfig solver() const { rc::ai::engine::SolverConfig c; c.iters_max = 1; return c; }
};
constexpr int N = toy::ToyBox::N;

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

int main()
{
    std::printf("engine_cap_cov_test\n");
    const toy::ToyState truth{1.0f, 2.0f, 0.8f, 0.6f, 0.4f, 0.3f};
    const auto f0 = toy::make_frame(toy::surface_points(truth, 6), {});
    double sd_prior[2] = {0, 0}, sd_post[2] = {0, 0};
    bool conv[2] = {false, false};
    for (int capped = 0; capped < 2; ++capped)
    {
        toy::ToyBox base;
        base.s = toy::ToyState{1.25f, 2.0f, 0.8f, 0.6f, 0.4f, 0.3f};   // the seed 0.25 m off, under a wide prior
        base.prior_mean << 0.f, 0.f, 0.8f, 0.6f, 0.4f, 0.f;
        base.Sigma = base.prior_diag.asDiagonal();
        base.q_diag.setConstant(1e-10f);
        const auto run = [&](auto& m, auto& e)
        {
            e.update(m, f0, cam_latents());                            // the first update (class prior)
            e.set_first_update_done(true);
            m.s.cx = 1.25f;
            e.P_for_test().row(0).setZero(); e.P_for_test().col(0).setZero();
            e.P_for_test()(0, 0) = 0.30 * 0.30;                        // a wide predicted σ_cx, as table_2's first update
            m.set_covariance(e.Sigma_theta());
            e.update(m, f0, cam_latents());
            sd_prior[capped] = std::sqrt(double(e.dbg_Pbar_tt()(0, 0)));
            sd_post[capped] = std::sqrt(e.P()(0, 0));
            conv[capped] = e.last_gn_converged();
        };
        if (capped) { CappedBox m; static_cast<toy::ToyBox&>(m) = base; rc::ai::engine::PersistentEngine<CappedBox> e; run(m, e); }
        else        { toy::ToyBox m = base; rc::ai::engine::PersistentEngine<toy::ToyBox> e; run(m, e); }
    }
    std::printf("  converged update: sigma_cx %.4f -> %.4f (converged %d); capped update: %.4f -> %.4f (converged %d)\n",
                sd_prior[0], sd_post[0], int(conv[0]), sd_prior[1], sd_post[1], int(conv[1]));
    CHECK(conv[0] and not conv[1]);
    CHECK(sd_post[0] < 0.5 * sd_prior[0]);                             // the data are informative: a converged update shrinks Σ
    const double rel = 4.0 * std::numeric_limits<float>::epsilon();   // the predicted Σ is read back through the float mirror
    CHECK(std::abs(sd_post[1] - sd_prior[1]) <= rel * sd_prior[1]);
    {   // (2) steps accepted, then the cap (iters_max = 1), the mean moved: the moved x̂ is kept and Σ = P̄ (the flagged switch)
        toy::ToyBox base;
        base.s = toy::ToyState{1.25f, 2.0f, 0.8f, 0.6f, 0.4f, 0.3f};
        base.prior_mean << 0.f, 0.f, 0.8f, 0.6f, 0.4f, 0.f;
        base.Sigma = base.prior_diag.asDiagonal();
        base.q_diag.setConstant(1e-10f);
        OneStepBox m; static_cast<toy::ToyBox&>(m) = base;
        rc::ai::engine::PersistentEngine<OneStepBox> e;
        e.update(m, f0, cam_latents());
        e.set_first_update_done(true);
        m.s.cx = 1.25f;
        e.P_for_test().row(0).setZero(); e.P_for_test().col(0).setZero();
        e.P_for_test()(0, 0) = 0.30 * 0.30;
        m.set_covariance(e.Sigma_theta());
        e.update(m, f0, cam_latents());
        const double cx = double(m.state().cx), sd = std::sqrt(e.P()(0, 0)), sdp = std::sqrt(double(e.dbg_Pbar_tt()(0, 0)));
        std::printf("  one step then the cap: cx 1.2500 -> %.4f (gn %d, converged %d), sigma_cx %.4f -> %.4f\n", cx, e.last_gn_iters(),
                    int(e.last_gn_converged()), sdp, sd);
        CHECK(not e.last_gn_converged());
        CHECK(std::abs(cx - 1.25) > 0.0);                               // the GN's step is kept
        CHECK(std::abs(sd - sdp) <= 4.0 * std::numeric_limits<float>::epsilon() * sdp);   // Σ = P̄ (the float mirror's rounding)
    }
    std::printf(bad ? "engine_cap_cov_test: %d FAILED\n" : "engine_cap_cov_test: OK\n", bad);
    return bad != 0;
}
