// offer_return_view_test — offers under the same cap (spec 2026-11-23 shared-engine G5, Task 7, Review Focus 4).
//
// The engine's offer gain (PersistentEngine::predicted_gain) is the θ-marginal entropy drop of a hypothetical update at a
// standpoint: the latents predicted with ρ along the straight path from the last update's camera position, the look's
// camera points (offer_view.h, each weighted by its solid angle) and rays, and the engine's own joint covariance update.
// So a return to a seen view promises little and a new view promises a lot.
//
// Toy box, Δ = 0, Σ_pose = 0, no fresh share (the parked test's setting). The camera's frames ARE the engine's own
// synthesised observation at A (offer_observation: point k at variance R/weight_k, so a frame's information is exactly the
// hypothetical's), so every comparison below is against the same Λ and the tolerance covers the solver's stop rule only.
//
//   (a) 100 frames at A. For B (90° round, the short face) and C (A moved 0.5 m sideways: an intermediate ρ, checked to lie
//       in (0.3, 0.7) so the path integration is tested between its endpoints, not only at ρ = 1 and ρ ≈ 0):
//         predicted_gain(A) ≤ r_X · predicted_gain(X) · 1.05,   r_X = g(ρ_AA = 1, Λ_A) / g(ρ_AX, Λ_X),
//       and predicted_gain(X) within 5 % of g(ρ_AX, Λ_X). g is the textbook linear-Gaussian θ-entropy drop on (θ, b_cam)
//       after those 100 frames — b_cam predicted with ρ (b' = ρb + w, Var w = (1 − ρ²)Σ_cam), then one update with
//       information HᵀΛH, H = [𝟙 𝟙], Λ = Σ_k weight_k J_k J_kᵀ / R with the engine's own weights — in a separate 12-state
//       implementation. ρ_AX = rho_camera over the same 10-chord path, z and w⊥ at X. 1.05 = 1 + toy::sigma_tol_single(),
//       the stop rule's bound on a reported variance (the engine's GN stops within tol_sigma·σ of the fixed point).
//   (b) One more update at A: the delivered θ-marginal entropy drop ≤ predicted_gain(A) · 1.05.
//   (c) Ranking (Step 2): seen only from its long face, the box ranks a short face first among its four face standpoints.
//   (d) LiDAR: with a BeamConfig built by make and an OfferLidar, the look also stages rays; the hits carry m = 1 − ε, and
//       the promise with them is not below the camera-only promise (more channels cannot lose information).
//
// Run: ./offer_return_view_test
#include <cmath>
#include <cstdio>
#include <numbers>
#include <numeric>
#include <utility>
#include <vector>

#include "toy_model.h"
#include "ai_belief/offer_view.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

namespace {

using Engine = rc::ai::engine::PersistentEngine<toy::ToyBox>;
constexpr int N = Engine::N;
using Mat12 = Eigen::Matrix<double, 2 * N, 2 * N>;
using MatN = Eigen::Matrix<double, N, N>;
namespace of = rc::ai::offer;

const toy::ToyState kTruth{0.0f, 0.0f, 0.6f, 1.6f, 0.6f, 0.0f};   // long along x (w = 1.6), short along y (h = 0.6)
constexpr float kStandoff = 2.0f;                                 // from each face (m)
constexpr float kPosStd = 0.03f, kSizeStd = 0.02f, kYawStd = 0.03f;   // Σ_cam (the countable's common-mode stds)

rc::nbv::Sensor camera()
{
    rc::nbv::Sensor s;                                            // a complete model: hfov 110°, vfov 70°, ZED height
    s.hfov_rad = 110.0f * std::numbers::pi_v<float> / 180.0f;
    s.vfov_rad = 70.0f * std::numbers::pi_v<float> / 180.0f;
    s.height_m = 0.945f;
    return s;
}

of::Standpoint looking_at_centre(const Eigen::Vector2f& p) { return {p, std::atan2(-p.y(), -p.x())}; }
// The standpoint in front of face i ([+x, −x, +y, −y]) at kStandoff, looking at the centre (plan_faces' convention).
of::Standpoint face_standpoint(int i)
{
    const std::array<Eigen::Vector2f, 4> nrm{Eigen::Vector2f(1, 0), Eigen::Vector2f(-1, 0), Eigen::Vector2f(0, 1), Eigen::Vector2f(0, -1)};
    const std::array<float, 4> half{0.5f * kTruth.w, 0.5f * kTruth.w, 0.5f * kTruth.h, 0.5f * kTruth.h};
    return looking_at_centre(nrm[i] * (half[i] + kStandoff));
}

double half_logdet(const MatN& S) { return 0.5 * std::log(S.determinant()); }

// Λ = Σ weight_k J_k J_kᵀ / R over the look, at the truth (double).
MatN info_of(const toy::ToyBox& m, const Engine::OfferObservation& o)
{
    MatN L = MatN::Zero();
    for (std::size_t k = 0; k < o.points.size(); ++k)
    {
        const Eigen::Matrix<double, N, 1> J = m.sdf_jacobian(o.points[k], kTruth, 0).cast<double>();
        L += double(o.weight[k]) * J * J.transpose() / double(o.R);
    }
    return L;
}

// One information update of the joint (θ, b) with H = [𝟙 𝟙]: P − P X, X = (𝟙 + W P)⁻¹ W P, W = HᵀΛH.
Mat12 info_update(const Mat12& P, const MatN& L)
{
    Mat12 W;
    W << L, L, L, L;
    const Mat12 WP = W * P;
    const Mat12 X = (Mat12::Identity() + WP).partialPivLu().solve(WP);
    const Mat12 Pn = P - P * X;
    return 0.5 * (Pn + Pn.transpose());
}

// g(ρ, Λ): the θ-marginal entropy drop of one look after predicting b_cam with ρ.
double g_analytic(const Mat12& P, double rho, const Eigen::Matrix<double, N, 1>& sig2, const MatN& L)
{
    Mat12 F = Mat12::Identity();
    F.bottomRightCorner<N, N>() *= rho;
    Mat12 Pp = F * P * F.transpose();
    Pp.bottomRightCorner<N, N>() += ((1.0 - rho * rho) * sig2).asDiagonal();
    const Mat12 Pa = info_update(Pp, L);
    return half_logdet(P.topLeftCorner<N, N>()) - half_logdet(Pa.topLeftCorner<N, N>());
}

}  // namespace

int main()
{
    const rc::nbv::Sensor sensor = camera();
    const std::vector<rc::nbv::Obstacle> no_obstacles;
    const of::OfferSensors cam{sensor, no_obstacles, std::nullopt};

    toy::ToyBox m;
    m.s = kTruth;                                                 // seeded on the truth: θ never moves, P is the whole story
    m.prior_mean << 0.f, 0.f, kTruth.H, kTruth.w, kTruth.h, 0.f;
    m.Sigma = m.prior_diag.asDiagonal();
    m.sigma_m = 0.05f;                                            // a mask point's σ, the live order (R ≈ 0.004 m²)
    Engine eng;

    const float p2 = kPosStd * kPosStd, s2 = kSizeStd * kSizeStd, y2 = kYawStd * kYawStd;
    const Eigen::Matrix<double, N, 1> sig2 = (Eigen::Matrix<double, N, 1>() << p2, p2, s2, s2, s2, y2).finished();
    // A frame IS the engine's own look: point k at variance R / weight_k.
    const auto frame_of = [&](const Engine::OfferObservation& o)
    {
        toy::ToyFrame f = toy::make_frame(o.points, {});
        f.R.resize(o.points.size());
        for (std::size_t k = 0; k < o.points.size(); ++k) f.R[k] = o.R / o.weight[k];
        return f;
    };
    const auto latents_at = [&](const of::Standpoint& s)
    {
        rc::ai::engine::FrameLatents<N> fl;
        fl.sig2_cam = {p2, p2, s2, s2, s2, y2};
        fl.sig2_L = fl.sig2_cam;
        fl.rho_cam.fill(1.0f);                                    // parked at A
        fl.rho_chain = 1.0f;
        fl.robot_xy = s.xy;
        fl.camera_xy = s.xy;
        fl.camera_measured = true;
        return fl;
    };

    const auto A = face_standpoint(3);                            // −y: the long face
    const auto B = face_standpoint(0);                            // +x: the short face, 90° round
    const auto C = looking_at_centre(A.xy + Eigen::Vector2f(0.5f, 0.0f));   // A, 0.5 m sideways
    const auto obsA0 = eng.offer_observation(m, A, cam);
    std::printf("view from A: %zu samples, R %.5f, Σ weight %.1f\n", obsA0.points.size(), obsA0.R,
                std::accumulate(obsA0.weight.begin(), obsA0.weight.end(), 0.0));
    CHECK(not obsA0.points.empty());
    for (int t = 0; t < 100; ++t) eng.update(m, frame_of(eng.offer_observation(m, A, cam)), latents_at(A));

    // ── (a) ──
    Mat12 P = Mat12::Zero();                                      // the analytic: the same 100 looks, a separate 12-state
    P.topLeftCorner<N, N>() = m.prior_diag.cast<double>().asDiagonal();
    P.bottomRightCorner<N, N>() = sig2.asDiagonal();
    const MatN LA = info_of(m, eng.offer_observation(m, A, cam));
    for (int t = 0; t < 100; ++t) P = info_update(P, LA);
    const double gA = eng.predicted_gain(m, A, cam), gA_an = g_analytic(P, 1.0, sig2, LA);
    const double tol = 1.0 + toy::sigma_tol_single();
    const Eigen::Vector2f c(kTruth.cx, kTruth.cy);
    std::printf("(a) A: predicted %.6g, analytic %.6g (ratio %.6f)\n", gA, gA_an, gA / gA_an);
    CHECK(std::abs(gA / gA_an - 1.0) <= tol - 1.0);
    for (const auto& [name, X] : {std::pair{"B", B}, std::pair{"C", C}})
    {
        rc::ai::view::ViewpointPath path;
        const Eigen::Vector2f step = (X.xy - A.xy) / 10.0f;
        for (int k = 0; k < 10; ++k) rc::ai::view::accumulate_path(path, A.xy + float(k) * step, step, c);
        const float wX = of::extent_across(m.beam_parts(kTruth), X.xy, c);
        const double rho = rc::ai::view::rho_camera(path, (X.xy - c).norm(), wX);
        const double gX = eng.predicted_gain(m, X, cam);
        const double gX_an = g_analytic(P, rho, sig2, info_of(m, eng.offer_observation(m, X, cam)));
        const double r = gA_an / gX_an;
        std::printf("    %s: ρ_A%s = %.4g; predicted %.6g, analytic %.6g (ratio %.6f); r = %.4g, gA/(r·g%s) = %.6f\n",
                    name, name, rho, gX, gX_an, gX / gX_an, r, name, gA / (r * gX));
        CHECK(gA <= r * gX * tol);
        CHECK(std::abs(gX / gX_an - 1.0) <= tol - 1.0);
        CHECK(gA < gX);
        if (X.xy == C.xy) CHECK(rho > 0.3 and rho < 0.7);
    }

    // ── (b) ──
    eng.update(m, frame_of(eng.offer_observation(m, A, cam)), latents_at(A));
    const double delivered = eng.last_delivered_nats();
    std::printf("(b) one more look at A: delivered %.6g nats, promised %.6g (ratio %.6f)\n", delivered, gA, delivered / gA);
    CHECK(std::isfinite(delivered));
    CHECK(delivered <= gA * tol);

    // ── (c) ──
    std::array<double, 4> g{};
    for (int i = 0; i < 4; ++i) g[i] = eng.predicted_gain(m, face_standpoint(i), cam);
    const int best = int(std::max_element(g.begin(), g.end()) - g.begin());
    std::printf("(c) faces +x %.4f  −x %.4f  +y %.4f  −y %.4f nats → best %d\n", g[0], g[1], g[2], g[3], best);
    CHECK(best == 0 or best == 1);

    // ── (d) ──
    {
        m.beam_cfg = rc::ai::beam::BeamConfig::make(*rc::ai::beam::StudentT::make(2500.0, 0.05));
        of::OfferLidar L;
        L.offset = {0.0f, -0.08f, 1.075f};                        // the Helios over the camera (the ROBOT_GEOMETRY mount)
        for (int k = 0; k < 32; ++k) L.ring_elev_rad.push_back((-15.5 + k * 70.0 / 31.0) * std::numbers::pi / 180.0);
        L.azimuth_step_rad = 0.2 * std::numbers::pi / 180.0;
        L.eps_px = 0.05;
        L.room.floor_sigma = 0.005;
        L.st = m.beam_cfg.st;
        const of::OfferSensors with_lidar{sensor, no_obstacles, L};
        const auto o = eng.offer_observation(m, B, with_lidar);
        int hits = 0;
        for (const double mi : o.rays.m) hits += (mi == 1.0 - L.eps_px) ? 1 : 0;
        const double g_cam = eng.predicted_gain(m, B, cam), g_both = eng.predicted_gain(m, B, with_lidar);
        std::printf("(d) rays %zu (hits %d), measured %d; B camera-only %.4f, camera + LiDAR %.4f nats\n",
                    o.rays.endpoints.size(), hits, int(o.lidar), g_cam, g_both);
        CHECK(o.lidar and hits > 0);
        CHECK(std::isfinite(g_both) and g_both >= g_cam);
    }

    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}
