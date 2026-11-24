// engine_shape_latent_test — the shape misfit latent (spec 2026-11-23 shared-engine G4; Task 5). A box cannot say which
// layer of a real object it means: here the camera sees a 1.57 m width (the mattress) and the LiDAR a 1.77 m width (the
// frame), a layer spread Δ_w = 0.20 m. Each channel carries its own shape latent s_cam / s_L, born at Δ²/12 (uniform over the
// layer range) and carried with ρ = 1, so no number of views at any number of standpoints averages it away.
//
//   (1) Birth: one predict of the latents puts Δ²/12 on s_cam and s_L for w and exactly 0 on every DOF with Δ = 0; a
//       second predict (ρ = 1) leaves it unchanged.
//   (2) 200 updates from 4 standpoints, camera + LiDAR. The posterior σ_w cannot fall below the floor the two shape latents
//       set — σ_floor² = (1/P0_w + 2·12/Δ²)⁻¹, computed here from Δ and the class prior, the information of two
//       independent errors of variance Δ²/12 (every other latent only adds variance) — and the mean sits between the
//       layers: |w − 1.67| ≤ 3σ_w.
//   (3) Contrast: the same run with Δ = 0 ends far below that floor, so (2) is the latent's doing, not the data's.
//
// The floor is not Δ/√12 itself: the camera's and the LiDAR's shape errors are independent (they see different layers, spec
// "Answers to the builder check" 3), so the two channels do average each other's shape error, once, to Δ/√24; what no
// repetition can do is go below that.
//
// Tolerance: the solver stops when the undamped step is below tol_sigma·σ_k on every θ component (SolverConfig::tol_sigma,
// the stop rule of spec 7c.8 rev 3.3.8). The returned θ, where the posterior covariance is evaluated, may therefore sit up
// to tol_sigma·σ from the GN fixed point; we grant σ_w the same relative slack, (1 − tol_sigma) = 0.9. Not fitted.
//
// Runs on the minimal toy model (toy_model.h) with a LiDAR channel that measures w directly (the engine's
// accumulate_lidar_override hook). Run: ./engine_shape_latent_test
#include <cmath>
#include <cstdio>
#include <numbers>

#include "toy_model.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

namespace {

constexpr float kWCam = 1.57f, kWLidar = 1.77f, kDelta = 0.20f;
constexpr float kLidarSigma = 0.01f;                 // the LiDAR width reading's own noise (m)
constexpr float kRange = 3.0f;                       // standpoint range to the box centre (m)
constexpr float kRingGap = 2.0f * std::numbers::pi_v<float> / 180.0f;   // LiDAR ring gap (rad), rho_lidar_lateral

// A frame that may also carry one LiDAR width reading.
struct LayeredFrame : toy::ToyFrame
{
    bool  lidar = false;
    float w_lidar = 0.0f;
};

// The toy box whose LiDAR channel reads the width directly: a Gaussian on w at the frame's reading, through the engine's
// accumulate_lidar_override hook. It handles the channel on every frame (true), and adds its term only when the frame
// carries a reading.
struct LayeredToy : toy::ToyBox
{
    using Frame = LayeredFrame;
    bool accumulate_lidar_override(const State& s, const Frame& f, MatN* Id, VecN* bd, double* E) const
    {
        if (not f.lidar) return true;
        const float lam = 1.0f / (kLidarSigma * kLidarSigma);
        const float r = s.w - f.w_lidar;
        if (Id) (*Id)(3, 3) += lam;
        if (bd) (*bd)(3) += -lam * r;
        if (E)  *E += 0.5 * double(lam) * double(r) * double(r);
        return true;
    }
};
static_assert(rc::ai::engine::PersistentModel<LayeredToy>);
using Engine = rc::ai::engine::PersistentEngine<LayeredToy>;
constexpr int N = Engine::N;

const toy::ToyState kCamBox{0.0f, 0.0f, 0.6f, kWCam, 2.0f, 0.0f};

LayeredToy make_model(float delta_w)
{
    LayeredToy m;
    m.s = toy::ToyState{0.02f, -0.03f, 0.62f, 1.60f, 1.98f, 0.02f};   // the seed
    m.prior_mean << 0.f, 0.f, 0.6f, 0.5f * (kWCam + kWLidar), 2.0f, 0.f;
    m.Sigma = m.prior_diag.asDiagonal();
    m.layer_spread.setZero();
    m.layer_spread(3) = delta_w;
    return m;
}

rc::ai::engine::FrameLatents<N> base_latents()
{
    rc::ai::engine::FrameLatents<N> fl;
    const float p2 = 0.03f * 0.03f, s2 = 0.02f * 0.02f, y2 = 0.03f * 0.03f;
    fl.sig2_cam = {p2, p2, s2, s2, s2, y2};
    fl.sig2_L   = fl.sig2_cam;
    fl.rho_cam.fill(1.0f); fl.rho_L.fill(1.0f);
    fl.rho_chain = 1.0f;                               // Σ_pose = 0: e stays a zero-variance latent
    fl.camera_measured = true;
    fl.lidar_measured  = true;
    return fl;
}

// 200 updates, 50 at each of 4 standpoints 45° apart round the box at kRange. Each move's ρ comes from its path.
struct Run { float w = 0.0f, sigma_w = 0.0f; };
Run run(float delta_w)
{
    LayeredToy m = make_model(delta_w);
    Engine eng;
    const Eigen::Vector2f centre(kCamBox.cx, kCamBox.cy);
    float bearing = -0.5f * std::numbers::pi_v<float>;
    for (int sp = 0; sp < 4; ++sp)
    {
        float rho_c = 1.0f, rho_l = 1.0f;
        if (sp > 0)
        {   // the arc from the last standpoint, in small chords
            rc::ai::view::ViewpointPath path;
            constexpr int steps = 90;
            const float da = 0.25f * std::numbers::pi_v<float> / steps;
            for (int k = 0; k < steps; ++k)
            {
                const float a0 = bearing + k * da, a1 = a0 + da;
                const Eigen::Vector2f p0 = centre + kRange * Eigen::Vector2f(std::cos(a0), std::sin(a0));
                const Eigen::Vector2f p1 = centre + kRange * Eigen::Vector2f(std::cos(a1), std::sin(a1));
                rc::ai::view::accumulate_path(path, p0, p1 - p0, centre);
            }
            bearing += 0.25f * std::numbers::pi_v<float>;
            rho_c = rc::ai::view::rho_camera(path, kRange, kWCam);
            rho_l = rc::ai::view::rho_lidar_lateral(path, kRingGap);
        }
        const Eigen::Vector2f robot = centre + kRange * Eigen::Vector2f(std::cos(bearing), std::sin(bearing));
        for (int t = 0; t < 50; ++t)
        {
            auto fl = base_latents();
            fl.rho_cam.fill(t == 0 ? rho_c : 1.0f);
            fl.rho_L.fill(t == 0 ? rho_l : 1.0f);
            fl.robot_xy = robot;
            LayeredFrame f;
            f.points = toy::surface_points(kCamBox, 6);
            f.lidar = true;
            f.w_lidar = kWLidar;
            eng.update(m, f, fl);
        }
    }
    return {m.s.w, float(std::sqrt(eng.P()(3, 3)))};
}

}  // namespace

int main()
{
    const double tol = rc::ai::engine::SolverConfig{}.tol_sigma;

    // (1) Birth at Δ²/12, carried with ρ = 1.
    {
        LayeredToy m = make_model(kDelta);
        Engine eng;
        eng.ensure_joint(m.covariance());
        auto fl = base_latents();
        eng.predict_latents(m, fl);
        const double want = double(kDelta) * double(kDelta) / 12.0;
        for (const int off : {Engine::kSc, Engine::kSL})
            for (int j = 0; j < N; ++j)
            {
                const double got = eng.P()(N + off + j, N + off + j);
                const bool ok = j == 3 ? std::abs(got - want) <= 1e-6 * want : got == 0.0;
                if (not ok) std::printf("  birth: P(s%d) = %.9g, expected %.9g\n", off + j, got, j == 3 ? want : 0.0);
                CHECK(ok);
            }
        const auto P1 = eng.P();
        eng.predict_latents(m, fl);
        for (const int off : {Engine::kSc, Engine::kSL})
            CHECK(eng.P()(N + off + 3, N + off + 3) == P1(N + off + 3, N + off + 3));
        std::printf("birth: s_cam_w = s_L_w = %.6g (Δ²/12 = %.6g), carried unchanged\n",
                    eng.P()(N + Engine::kSc + 3, N + Engine::kSc + 3), want);
    }

    // (2) 200 updates from 4 standpoints: σ_w held at the shape floor, the mean between the layers.
    const double s2 = double(kDelta) * double(kDelta) / 12.0;
    const double p0 = double(make_model(kDelta).prior_diag(3));
    const double floor_w = std::sqrt(1.0 / (1.0 / p0 + 2.0 / s2));
    const Run r = run(kDelta);
    const float mid = 0.5f * (kWCam + kWLidar);
    std::printf("Δ = %.2f: w = %.4f, σ_w = %.4f; shape floor %.4f (bound %.4f), Δ/√12 = %.4f\n", kDelta, r.w, r.sigma_w,
                floor_w, (1.0 - tol) * floor_w, std::sqrt(s2));
    CHECK(r.sigma_w >= (1.0 - tol) * floor_w);
    CHECK(std::abs(r.w - mid) <= 3.0f * r.sigma_w);

    // (3) Without the shape latent the same data collapses σ_w far below that floor.
    const Run r0 = run(0.0f);
    std::printf("Δ = 0:    w = %.4f, σ_w = %.4f\n", r0.w, r0.sigma_w);
    CHECK(r0.sigma_w < (1.0 - tol) * floor_w);

    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}
