// parked_synthetic_test — the spec's parked test, offline (spec 2026-11-23 shared-engine G1, Task 5). Within one standpoint
// σ must stop shrinking; a new standpoint adds information in proportion to how far the view has decorrelated.
//
// Toy box, camera only, Δ = 0 (no shape latent), Σ_pose = 0 (no chain), no fresh share. The camera's shared error b_cam has
// marginal Σ_cam and lag-1 correlation ρ between updates. On cx the problem is scalar (the toy's points give cx no
// cross-information at yaw = 0), and the camera's points are dense enough that each frame pins θ + b_cam (its information
// is ~10³ times 1/Σ_cam), so the values below are the plateau limits:
//
//   (i)  200 frames at standpoint A (ρ = 1): σ_cx(200) ≥ 0.95 σ_cx(20) — parking does not keep buying precision — and it
//        sits at the plateau σ_A² = (1/P0 + 1/Σ_cam)⁻¹: one shared error, however many frames.
//   (ii) 200 frames at B, 45° round the object at the same range: b_cam at B is ρ_AB b_A + a fresh draw, ρ_AB = rho_camera
//        of the A→B path. Two views whose errors correlate at ρ carry 1ᵀC⁻¹1 = 2/(Σ_cam(1 + ρ)) on θ, so
//        σ_B² = (1/P0 + 2/(Σ_cam(1 + ρ_AB)))⁻¹ and the drop factor σ_B/σ_A must match √(σ_B²/σ_A²).
//   (iii) Back at A: the θ-marginal entropy drop of the return visit is ≤ the B visit's. (For an AR(1) shared error the
//        third view adds the same information as the second, on a larger base: (3 − ρ)/2 ≤ 2/(1 + ρ) ⇔ (1 − ρ)² ≥ 0.)
//
// Tolerances, from the solver's stop rule (SolverConfig::tol_sigma = 0.1: the undamped step is below 0.1σ_k on every θ
// component): the covariance is evaluated at a θ that may sit up to tol_sigma·σ from the GN fixed point, a relative slack
// tol_sigma on the variance; the drop factor and the plateau are ratios of σ = √variance, which halves it: tol_sigma/2 = 5 %.
// The 0.95 of (i) is the same 5 %. Neither is fitted.
//
// Run: ./parked_synthetic_test
#include <cmath>
#include <cstdio>
#include <numbers>

#include "toy_model.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

namespace {

using Engine = rc::ai::engine::PersistentEngine<toy::ToyBox>;
constexpr int N = Engine::N;

const toy::ToyState kTruth{0.0f, 0.0f, 0.6f, 1.6f, 2.0f, 0.0f};
constexpr float kRange = 2.0f;                       // both standpoints, to the box centre (m)
constexpr float kPosStd = 0.03f, kSizeStd = 0.02f, kYawStd = 0.03f;   // Σ_cam (the countable's common-mode stds)

double theta_entropy(const Engine& eng)
{
    const Eigen::Matrix<double, N, N> S = eng.P().topLeftCorner<N, N>();
    return 0.5 * std::log(S.determinant());
}

}  // namespace

int main()
{
    const double half_tol = 0.5 * rc::ai::engine::SolverConfig{}.tol_sigma;

    toy::ToyBox m;
    m.s = kTruth;                                    // seeded on the truth: θ never moves, P is the whole story
    m.prior_mean << 0.f, 0.f, kTruth.H, kTruth.w, kTruth.h, 0.f;
    m.Sigma = m.prior_diag.asDiagonal();
    m.sigma_m = 0.005f;
    Engine eng;

    const Eigen::Vector2f centre(kTruth.cx, kTruth.cy);
    const float bearing_A = -0.5f * std::numbers::pi_v<float>, bearing_B = bearing_A + 0.25f * std::numbers::pi_v<float>;
    const auto at = [&](float b) { return Eigen::Vector2f(centre + kRange * Eigen::Vector2f(std::cos(b), std::sin(b))); };
    // ρ of the arc between two bearings at kRange, from the path in small chords.
    const auto rho_arc = [&](float b0, float b1)
    {
        rc::ai::view::ViewpointPath path;
        constexpr int steps = 90;
        const float da = (b1 - b0) / steps;
        for (int k = 0; k < steps; ++k)
            rc::ai::view::accumulate_path(path, at(b0 + k * da), at(b0 + (k + 1) * da) - at(b0 + k * da), centre);
        return rc::ai::view::rho_camera(path, kRange, kTruth.w);
    };
    const auto frame = [&] { return toy::make_frame(toy::surface_points(kTruth, 6), {}); };
    // n frames at one standpoint; the first carries ρ from the move there, the rest ρ = 1 (parked). σ_cx after each.
    const auto visit = [&](const Eigen::Vector2f& robot, float rho_first, int n, std::vector<double>& sig)
    {
        for (int t = 0; t < n; ++t)
        {
            rc::ai::engine::FrameLatents<N> fl;
            const float p2 = kPosStd * kPosStd, s2 = kSizeStd * kSizeStd, y2 = kYawStd * kYawStd;
            fl.sig2_cam = {p2, p2, s2, s2, s2, y2};
            fl.rho_cam.fill(t == 0 ? rho_first : 1.0f);
            fl.rho_chain = 1.0f;
            fl.robot_xy = robot;
            fl.camera_measured = true;
            eng.update(m, frame(), fl);
            sig.push_back(std::sqrt(eng.P()(0, 0)));
        }
    };

    const double sc2 = double(kPosStd) * double(kPosStd);
    const double p0  = double(m.prior_diag(0));

    // (i) Parked at A.
    std::vector<double> sA;
    visit(at(bearing_A), 1.0f, 200, sA);
    const double var_A = 1.0 / (1.0 / p0 + 1.0 / sc2);
    std::printf("(i)   A: σ_cx(20) = %.6f, σ_cx(200) = %.6f, plateau %.6f\n", sA[19], sA[199], std::sqrt(var_A));
    CHECK(sA[199] >= (1.0 - half_tol) * sA[19]);
    CHECK(std::abs(sA[199] / std::sqrt(var_A) - 1.0) <= half_tol);
    const double H_A = theta_entropy(eng);

    // (ii) 45° round to B.
    const double rho_AB = rho_arc(bearing_A, bearing_B);
    std::vector<double> sB;
    visit(at(bearing_B), float(rho_AB), 200, sB);
    const double var_B = 1.0 / (1.0 / p0 + 2.0 / (sc2 * (1.0 + rho_AB)));
    const double drop = sB.back() / sA.back(), drop_pred = std::sqrt(var_B / var_A);
    std::printf("(ii)  B: ρ_AB = %.4f, σ_cx = %.6f, drop %.4f, analytic %.4f\n", rho_AB, sB.back(), drop, drop_pred);
    CHECK(std::abs(drop / drop_pred - 1.0) <= half_tol);
    const double H_B = theta_entropy(eng);

    // (iii) Back at A.
    const double rho_BA = rho_arc(bearing_B, bearing_A);
    std::vector<double> sA2;
    visit(at(bearing_A), float(rho_BA), 200, sA2);
    const double H_A2 = theta_entropy(eng);
    std::printf("(iii) entropy drop at B %.4f nats, back at A %.4f nats (σ_cx %.6f)\n", H_A - H_B, H_B - H_A2, sA2.back());
    CHECK(H_B - H_A2 <= H_A - H_B);

    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}
