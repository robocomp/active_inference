// offer_return_view_test — offers under the same cap (spec 2026-11-23 shared-engine G5, Task 7, Review Focus 4).
//
// The engine's offer gain (PersistentEngine::predicted_gain) is the θ-marginal entropy drop of a hypothetical update at a
// standpoint: the latents predicted with ρ along the straight path from the last update's camera position, the camera
// points the standpoint would see (offer_view.h), and the engine's own joint covariance update. So a return to a seen view
// promises little and a new view promises a lot.
//
// Toy box, camera only, Δ = 0, Σ_pose = 0, no fresh share (the parked test's setting), the camera's frames ARE the
// synthesised view at A (so the hypothetical's point set at A is the one the frames had).
//
//   (a) 100 frames at A. Then predicted_gain(A) ≤ r · predicted_gain(B) · 1.05, B at 90° round the object (the short face).
//       r is the analytic ratio g(ρ_AA = 1, Λ_A) / g(ρ_AB, Λ_B): g is the textbook linear-Gaussian entropy drop on the
//       joint (θ, b_cam) after those 100 frames — b_cam predicted with ρ (b' = ρb + w, Var w = (1 − ρ²)Σ_cam), then one
//       update with information HᵀΛH, H = [𝟙 𝟙], Λ = Σ_points J Jᵀ / R at the truth — computed here in a separate 12-state
//       implementation. ρ_AB = rho_camera over the same 10-chord path from A to B, z and w⊥ at B. 1.05 is the stop-rule
//       tolerance (toy::sigma_tol_single(): the reported posterior's variance error is ≤ tol_sigma, its σ ≤ tol_sigma/2;
//       the entropy drop is a difference of ½ log det, on which the stop rule's relative variance error tol_sigma/2 per
//       reported Σ is the bound used here, 1 + tol_sigma/2 = 1.05).
//   (b) One more update at A: the delivered θ-marginal entropy drop ≤ predicted_gain(A) · 1.05.
//   (c) Ranking (Step 2): seen only from its long face, the box ranks a short face first among its four face standpoints.
//
// Run: ./offer_return_view_test
#include <cmath>
#include <cstdio>
#include <numbers>
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

// The standpoint in front of face i ([+x, −x, +y, −y]) at kStandoff, looking at the centre (plan_faces' convention).
rc::ai::offer::Standpoint face_standpoint(int i)
{
    const std::array<Eigen::Vector2f, 4> nrm{Eigen::Vector2f(1, 0), Eigen::Vector2f(-1, 0), Eigen::Vector2f(0, 1), Eigen::Vector2f(0, -1)};
    const std::array<float, 4> half{0.5f * kTruth.w, 0.5f * kTruth.w, 0.5f * kTruth.h, 0.5f * kTruth.h};
    const Eigen::Vector2f p = nrm[i] * (half[i] + kStandoff);
    return {p, std::atan2(-p.y(), -p.x())};
}

double half_logdet(const MatN& S) { return 0.5 * std::log(S.determinant()); }

// Λ = Σ J Jᵀ / R over the points, at the truth (double).
MatN info_of(const toy::ToyBox& m, const std::vector<Eigen::Vector3f>& pts)
{
    MatN L = MatN::Zero();
    for (const auto& p : pts)
    {
        const Eigen::Matrix<double, N, 1> J = m.sdf_jacobian(p, kTruth, 0).cast<double>();
        L += J * J.transpose() / double(m.sigma2());
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
    const rc::ai::offer::OfferCamera cam{sensor, no_obstacles};

    toy::ToyBox m;
    m.s = kTruth;                                                 // seeded on the truth: θ never moves, P is the whole story
    m.prior_mean << 0.f, 0.f, kTruth.H, kTruth.w, kTruth.h, 0.f;
    m.Sigma = m.prior_diag.asDiagonal();
    m.sigma_m = 0.05f;                                            // a mask point's σ, the live order (R ≈ 0.004 m²)
    Engine eng;

    const float p2 = kPosStd * kPosStd, s2 = kSizeStd * kSizeStd, y2 = kYawStd * kYawStd;
    const Eigen::Matrix<double, N, 1> sig2 = (Eigen::Matrix<double, N, 1>() << p2, p2, s2, s2, s2, y2).finished();
    const auto view_at = [&](const rc::ai::offer::Standpoint& s)
    { return rc::ai::offer::visible_surface(m.beam_parts(kTruth), rc::ai::offer::camera_at(s, cam), s.yaw, cam).points; };
    const auto update_at = [&](const rc::ai::offer::Standpoint& s)
    {
        rc::ai::engine::FrameLatents<N> fl;
        fl.sig2_cam = {p2, p2, s2, s2, s2, y2};
        fl.rho_cam.fill(1.0f);                                    // parked at A
        fl.rho_chain = 1.0f;
        fl.robot_xy = s.xy;
        fl.camera_xy = s.xy;
        fl.camera_measured = true;
        eng.update(m, toy::make_frame(view_at(s), {}), fl);
    };

    const auto A = face_standpoint(3);                            // −y: the long face
    const auto B = face_standpoint(0);                            // +x: the short face, 90° round
    const std::vector<Eigen::Vector3f> ptsA = view_at(A), ptsB = view_at(B);
    std::printf("view: %zu points from A, %zu from B\n", ptsA.size(), ptsB.size());
    CHECK(not ptsA.empty() and not ptsB.empty());

    // ── (a) ──
    for (int t = 0; t < 100; ++t) update_at(A);
    const double gA = eng.predicted_gain(m, A, cam), gB = eng.predicted_gain(m, B, cam);

    // the analytic: the same 100 looks in a separate 12-state (θ, b_cam) implementation
    Mat12 P = Mat12::Zero();
    P.topLeftCorner<N, N>() = m.prior_diag.cast<double>().asDiagonal();
    P.bottomRightCorner<N, N>() = sig2.asDiagonal();
    const MatN LA = info_of(m, ptsA), LB = info_of(m, ptsB);
    for (int t = 0; t < 100; ++t) P = info_update(P, LA);
    rc::ai::view::ViewpointPath path;
    const Eigen::Vector2f c(kTruth.cx, kTruth.cy);
    const Eigen::Vector2f step = (B.xy - A.xy) / 10.0f;
    for (int k = 0; k < 10; ++k) rc::ai::view::accumulate_path(path, A.xy + float(k) * step, step, c);
    const float wB = rc::ai::offer::extent_across(m.beam_parts(kTruth), B.xy, c);
    const double rho_AB = rc::ai::view::rho_camera(path, (B.xy - c).norm(), wB);
    const double gA_an = g_analytic(P, 1.0, sig2, LA), gB_an = g_analytic(P, rho_AB, sig2, LB);
    const double r = gA_an / gB_an;
    const double tol = 1.0 + toy::sigma_tol_single();
    std::printf("(a) ρ_AB = %.4g (w⊥ %.3f m); predicted A %.6g, B %.6g nats; analytic A %.6g, B %.6g; r = %.4g\n",
                rho_AB, wB, gA, gB, gA_an, gB_an, r);
    CHECK(gA <= r * gB * tol);
    CHECK(gA < gB);

    // ── (b) ──
    update_at(A);
    const double delivered = eng.last_delivered_nats();
    std::printf("(b) one more look at A: delivered %.6g nats, promised %.6g\n", delivered, gA);
    CHECK(std::isfinite(delivered));
    CHECK(delivered <= gA * tol);

    // ── (c) ──
    std::array<double, 4> g{};
    for (int i = 0; i < 4; ++i) g[i] = eng.predicted_gain(m, face_standpoint(i), cam);
    const int best = int(std::max_element(g.begin(), g.end()) - g.begin());
    std::printf("(c) faces +x %.4f  −x %.4f  +y %.4f  −y %.4f nats → best %d\n", g[0], g[1], g[2], g[3], best);
    CHECK(best == 0 or best == 1);

    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}
