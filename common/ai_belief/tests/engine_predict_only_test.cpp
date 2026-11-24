// engine_predict_only_test — an update that measures nothing (shared-engine Task 4 (b), Review Focus 2). A frame with zero
// camera points and zero rays leaves θ bit-identical, and the joint covariance changes only by the prediction of the
// latents its channel flags open: per latent DOF j, P_jj ← ρ_j² P_jj + (1 − ρ_j²) σ_j² (b_cam, and e with F = 𝟙 since the
// robot did not move), u_cam re-drawn at its fresh variance, and nothing else. The expectation is computed here from ρ and
// σ², independently of the engine's predict_block. Runs on the minimal toy model (toy_model.h). Run: ./engine_predict_only_test
#include <cmath>
#include <cstdio>

#include "toy_model.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

using Engine = rc::ai::engine::PersistentEngine<toy::ToyBox>;
constexpr int N = Engine::N, K = Engine::K;

static bool close_rel(double a, double b) { return std::abs(a - b) <= 1e-12 * std::max({1e-300, std::abs(a), std::abs(b)}); }

// A box that has seen one camera frame (the class prior has entered; b_cam, e and u_cam are live latents).
static void seeded(toy::ToyBox& m, Engine& eng, const toy::ToyState& truth)
{
    m.s = toy::ToyState{1.05f, 1.96f, 0.85f, 0.57f, 0.43f, 0.35f};
    m.prior_mean << 0.f, 0.f, 0.75f, 0.55f, 0.45f, 0.f;
    m.Sigma = m.prior_diag.asDiagonal();
    rc::ai::engine::FrameLatents<N> fl;
    fl.rho_cam.fill(1.0f); fl.sig2_cam.fill(4e-4f); fl.fresh_c.fill(1e-4f);
    fl.rho_chain = 1.0f;
    fl.sigma_pose = Eigen::Vector3d(2e-4, 3e-4, 1e-4).asDiagonal();
    fl.robot_xy = {0.2f, -0.1f};
    fl.camera_measured = true;
    eng.update(m, toy::make_frame(toy::surface_points(truth, 5), {}), fl);
}

int main()
{
    const toy::ToyState truth{1.0f, 2.0f, 0.8f, 0.6f, 0.4f, 0.3f};

    // (1) Both channels flagged as measuring, but the frame holds no points and no rays.
    {
        toy::ToyBox m;
        Engine eng;
        seeded(m, eng, truth);
        const Eigen::Matrix<float, N, 1> theta0 = m.s.vec();
        const Eigen::Matrix<double, N + K, N + K> P0 = eng.P();

        rc::ai::engine::FrameLatents<N> fl;
        const std::array<float, N> rho  = {0.9f, 0.7f, 0.5f, 0.3f, 0.95f, 0.6f};
        const std::array<float, N> sig2 = {4e-4f, 9e-4f, 1e-4f, 2.5e-3f, 6e-4f, 3e-3f};
        const std::array<float, N> fresh = {1e-4f, 2e-4f, 3e-4f, 4e-4f, 5e-4f, 6e-4f};
        fl.rho_cam = rho; fl.sig2_cam = sig2; fl.fresh_c = fresh;
        fl.rho_L.fill(0.5f); fl.sig2_L.fill(1e-3f); fl.fresh_L.fill(1e-3f);
        fl.rho_chain = 0.8f;
        fl.sigma_pose = Eigen::Vector3d(5e-4, 7e-4, 2e-4).asDiagonal();
        fl.robot_xy = {0.2f, -0.1f};                     // chain_dr = 0: e is not transported
        fl.camera_measured = true;
        fl.lidar_measured  = true;                       // an EMPTY ray-set does not measure (spec 7c.8 rev 3.3.11)
        eng.update(m, toy::make_frame({}, {}), fl);

        CHECK(m.s.vec() == theta0);                      // bit-identical
        const Eigen::Matrix<double, N + K, N + K>& P1 = eng.P();
        CHECK(P1.allFinite());
        std::array<double, N + K> expect{};
        for (int i = 0; i < N + K; ++i) expect[i] = P0(i, i);   // θθ (Q = 0), b_L, u_L, s_*: unchanged
        const auto pred = [](double p, float r, double s2) { const double rr = double(r) * double(r); return rr * p + (1.0 - rr) * s2; };
        for (int i = 0; i < 3; ++i)
            expect[N + Engine::kChain + i] = pred(P0(N + Engine::kChain + i, N + Engine::kChain + i), fl.rho_chain, fl.sigma_pose(i, i));
        for (int j = 0; j < N; ++j)
        {
            expect[N + Engine::kCam + j] = pred(P0(N + Engine::kCam + j, N + Engine::kCam + j), rho[j], double(sig2[j]));
            expect[N + Engine::kFc + j]  = double(fresh[j]);   // re-drawn: the last frame's u_cam is marginalised out
        }
        for (int i = 0; i < N + K; ++i)
        {
            const bool ok = close_rel(P1(i, i), expect[i]);
            if (not ok) std::printf("  P(%d,%d) = %.17g, expected %.17g (before %.17g)\n", i, i, P1(i, i), expect[i], P0(i, i));
            CHECK(ok);
        }
        std::printf("measured-flags, empty frame: theta identical, %d diagonal entries checked\n", N + K);
    }

    // (2) Neither channel measures: θ and the whole joint covariance are untouched (Q = 0).
    {
        toy::ToyBox m;
        Engine eng;
        seeded(m, eng, truth);
        const Eigen::Matrix<float, N, 1> theta0 = m.s.vec();
        const Eigen::Matrix<double, N + K, N + K> P0 = eng.P();
        rc::ai::engine::FrameLatents<N> fl;
        eng.update(m, toy::make_frame({}, {}), fl);
        CHECK(m.s.vec() == theta0);
        CHECK(eng.P() == P0);
        std::printf("no-flags, empty frame: theta and P identical\n");
    }

    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}
