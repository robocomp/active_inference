// lidar_only_update_test — the LiDAR-only frame (spec 2026-11-15 lidar-measures §7b, owner ruling C1 = A): a frame with no
// camera points, only (mask-weighted) LiDAR rays. It carries what was measured on it plus the ONE-SIDED constraints that are
// zero unless violated; the Gaussian class anchors are NOT re-added, and ω is not updated. As memberships -> 0 the update
// must tend continuously to predict(). Fixture: the golden nightstand belief (lidar_golden_test.cpp).
// Run: ./lidar_only_update_test
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <Eigen/Eigenvalues>
#include "countable_belief.h"
#include "test_scene.h"

using namespace rc;
using Mat6 = Eigen::Matrix<float, 6, 6>;
using Vec6 = Eigen::Matrix<float, 6, 1>;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static constexpr float tx = 0.30f, ty = -0.255f, tH = 0.62f, tw = 0.51f;

static CountableBeliefParams params()
{
    CountableBeliefParams p;
    p.prior_footprint_m = 0.51f; p.prior_footprint_std = 0.06f; p.prior_height_m = 0.62f; p.prior_height_std = 0.10f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;
    p.wall_flush_prior = 0.90f;
    p.room_axis_std    = 3.0f * 0.017453293f;
    p.volatility_infer = true;
    return p;
}
static CountableBelief make_belief(float dcx = 0.05f, float dcy = -0.04f, float dH = 0.03f)
{
    CountableBelief b(CountableBeliefState{tx + dcx, ty + dcy, tH + dH, tw, tw, 0.0f}, params());
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    return b;
}
// A LiDAR-only frame: no points, wall + room axis staged, z_top_obs = -1.
static CountableFrame lidar_frame(const rc::ai::LidarRays& r, float wall_y = 0.0f)
{
    CountableFrame f;
    f.lidar = r; f.lidar_only = true; f.z_top_obs = -1.0f;
    f.wall.ok = true; f.wall.p = Eigen::Vector2f(0, wall_y); f.wall.n = Eigen::Vector2f(0, -1);
    f.wall.sigma_m = 0.02f; f.wall.length_m = 4.0f;
    f.room_axis_ok = true; f.room_axis = 0.0f;
    return f;
}
// ---- Physical fixtures (spec 7c.0): returns come from ray-casting the TRUE scene (box, wall y = 0, floor), first hit only. ----
static const auto kElev = rc::test::helios_ring_elev_deg();
static rc::test::SceneBox truth_box(float cy = ty) { return {Eigen::Vector2f(tx, cy), 0.0f, tw, tw, tH}; }
static const rc::test::SceneWall kWall{Eigen::Vector2f(-4.0f, 0.0f), Eigen::Vector2f(4.0f, 0.0f), 2.5f};
// Returns of a Helios at `origin` that the mask selects: those within `margin` of the truth box (floor + wall included).
static rc::ai::LidarRays scene_rays(const Eigen::Vector3f& origin, float cy, float margin, bool with_wall = true,
                                    std::vector<float>* on_wall = nullptr)
{
    const auto box = truth_box(cy);
    const std::vector<rc::test::SceneBox> boxes{box};
    const std::vector<rc::test::SceneWall> walls = with_wall ? std::vector<rc::test::SceneWall>{kWall} : std::vector<rc::test::SceneWall>{};
    const auto all = rc::test::raycast_scene(origin, boxes, walls, true, rc::test::kHeliosAzStepDeg, kElev, 12.0f);
    rc::ai::LidarRays r;
    r.origin = origin; r.endpoints = rc::test::select_near(all, box, margin);
    if (on_wall) for (const auto& p : r.endpoints) on_wall->push_back(std::abs(p.y() - kWall.a.y()) < 1e-3f ? 1.0f : 0.0f);
    r.precision = 2500.0f; r.robust_c_m = 0.05f;
    return r;
}
static rc::ai::LidarRays front_rays(float cy = ty)           // Helios 1.5 m in front of the front face
{ return scene_rays(Eigen::Vector3f(tx, cy - 0.5f * tw - 1.5f, 1.1075f), cy, 0.10f); }
static rc::ai::LidarRays two_face_rays()                      // diagonal origin: front + left + top faces
{   // (the top face is what the sensor, mounted above the box, sees of H: side faces alone leave H unobserved)
    return scene_rays(Eigen::Vector3f(tx - 1.5f, ty - 1.5f, 1.1075f), ty, 0.10f);
}
static float centre_err(const CountableBelief& b)
{ const auto& s = b.state(); return std::hypot(s.cx - tx, s.cy - ty); }

static void empty_frame_is_predict_only()
{
    CountableBelief a = make_belief(), c = make_belief();
    CountableFrame f; f.lidar_only = true; f.z_top_obs = -1.0f;
    const float fe = a.update(f);
    c.predict();
    CHECK(fe == 0.0f);
    CHECK((a.state().vec() - c.state().vec()).norm() == 0.0f);
    CHECK((a.covariance() - c.covariance()).norm() == 0.0f);
    CHECK(a.covariance().allFinite() and a.state().vec().allFinite());
    // and a point-free, non-lidar_only frame (the legacy meaning) is the same
    CountableBelief d = make_belief(); CountableFrame g; d.update(g);
    CHECK((d.covariance() - c.covariance()).norm() == 0.0f);
    // a point-free frame WITH rays but lidar_only=false is still ignored by the engine (predict-only)
    CountableBelief e = make_belief(); auto h = lidar_frame(two_face_rays()); h.lidar_only = false;
    CHECK(e.update(h) == 0.0f);
    CHECK((e.state().vec() - c.state().vec()).norm() == 0.0f);
    CHECK((e.covariance() - c.covariance()).norm() == 0.0f);
}

static void lidar_only_moves_toward_rays()
{
    CountableBelief a = make_belief(), p = make_belief();
    const float e0 = centre_err(a), h0 = std::abs(a.state().H - tH);
    a.update(lidar_frame(two_face_rays()));
    p.predict();
    std::printf("  moves: centre err %.4f -> %.4f ; H err %.4f -> %.4f ; Sxx %.3e/%.3e Syy %.3e/%.3e SHH %.3e/%.3e (post/predict)\n",
                e0, centre_err(a), h0, std::abs(a.state().H - tH), a.covariance()(0, 0), p.covariance()(0, 0),
                a.covariance()(1, 1), p.covariance()(1, 1), a.covariance()(2, 2), p.covariance()(2, 2));
    CHECK(centre_err(a) < e0);
    CHECK(std::abs(a.state().H - tH) < h0);
    CHECK(a.covariance()(0, 0) < p.covariance()(0, 0));
    CHECK(a.covariance()(1, 1) < p.covariance()(1, 1));
    CHECK(a.covariance()(2, 2) < p.covariance()(2, 2));
}

// Run 300 LiDAR-only frames whose rays all carry membership w; return {state move, worst Sigma-diag ratio-1, omega move}.
struct Vanish { float move, sig, omega; };
static Vanish run_vanishing(float w)
{
    auto sweep = two_face_rays(); sweep.weights.assign(sweep.endpoints.size(), w);
    CountableBelief a = make_belief(), p = make_belief();
    const Vec6 start = a.state().vec(), om0 = a.log_volatility();
    for (int k = 0; k < 300; ++k) { a.update(lidar_frame(sweep)); p.predict(); }
    float worst = 0.0f;
    for (int i = 0; i < 6; ++i) worst = std::max(worst, std::abs(a.covariance()(i, i) / p.covariance()(i, i) - 1.0f));
    return {(a.state().vec() - start).norm(), worst, (a.log_volatility() - om0).norm()};
}

static void lidar_only_vanishing_membership_is_predict()
{
    // The brief's w = 1e-6 does NOT reach predict() in 300 frames: the rays' information is linear in w (~0.3 per frame at
    // 200 rays x 2500, so ~90 over 300 frames against a prior of ~11). That is the model being continuous, not a leak.
    // So the printed ladder shows the move scaling ~linearly with w, and the claim is asserted where the accumulated
    // information is below the 1e-4 tolerance.
    const Vanish v6 = run_vanishing(1e-6f), v9 = run_vanishing(1e-9f), v12 = run_vanishing(1e-12f);
    std::printf("  vanishing (lidar_only): w=1e-6 move %.3e sig %.3e | w=1e-9 move %.3e sig %.3e | w=1e-12 move %.3e sig %.3e omega %.1e\n",
                v6.move, v6.sig, v9.move, v9.sig, v12.move, v12.sig, v12.omega);
    CHECK(v12.move < 1e-4f);
    CHECK(v12.sig < 1e-3f);
    CHECK(v12.omega == 0.0f);
    CHECK(v9.move < v6.move and v12.move < v9.move);          // continuous: more vanishing membership, closer to predict()
}

static Vec6 invdiag(const CountableBelief& b, const CountableFrame& f) { return b.common_mode_inv_diag(f); }

static void lidar_only_saturates()
{
    const auto one = two_face_rays();
    rc::ai::LidarRays ten = one;
    for (int k = 1; k < 10; ++k) ten.endpoints.insert(ten.endpoints.end(), one.endpoints.begin(), one.endpoints.end());
    CountableBelief a1 = make_belief(), a10 = make_belief(), pred = make_belief();
    auto f1 = lidar_frame(one), f10 = lidar_frame(ten);
    f1.chain_cov_xx = f1.chain_cov_yy = f10.chain_cov_xx = f10.chain_cov_yy = 0.01f;
    const Vec6 sc = invdiag(a10, f10);
    Mat6 Id = Mat6::Zero(); Vec6 bd = Vec6::Zero();
    a10.accumulate_extra(a10.state(), f10, Id, bd);
    std::printf("  saturates: raw Id(0,0)=%.3e  10*Sc^-1(0)=%.3e\n", Id(0, 0), 10.0f * sc(0));
    CHECK(Id(0, 0) > 10.0f * sc(0));                                  // non-vacuous
    pred.predict();
    a1.update(f1); a10.update(f10);
    const Mat6 B = (pred.covariance().inverse() + Mat6(sc.asDiagonal())).inverse();
    for (const CountableBelief* b : {&a1, &a10})
    {
        Eigen::SelfAdjointEigenSolver<Mat6> es(B);
        const Mat6 Bih = es.eigenvectors() * es.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
        const Mat6 W = Bih * (b->covariance() - B) * Bih;
        const float lmin = Eigen::SelfAdjointEigenSolver<Mat6>(0.5f * (W + W.transpose())).eigenvalues().minCoeff();
        std::printf("  saturates: whitened min eig of (Sigma_post - bound) = %.4f\n", lmin);
        CHECK(lmin > -0.01f);
    }
    // (ruling T3-a') A front view makes Id large on every diagonal but near-singular along coupled directions ((cy,h), (cx,w)),
    // so a per-DOF x10/x1 test cannot hold. Two statements instead: (2) the model, (3) the data-dominated whitened directions.
    // (2) posterior Sigma == (Sigma_pred^-1 + Ieff_k)^-1, Ieff_k = saturate(Id_k, bd_k, Sc^-1) at the converged state.
    for (int k : {1, 10})
    {
        const CountableBelief& b = (k == 1) ? a1 : a10;
        const CountableFrame& fk = (k == 1) ? f1 : f10;
        Mat6 Idk = Mat6::Zero(); Vec6 bdk = Vec6::Zero();
        CountableBelief tmp = b;
        tmp.accumulate_extra(b.state(), fk, Idk, bdk);
        const Vec6 sck = invdiag(b, fk);
        const Mat6 Ieff = rc::ai::saturate<6>(Idk, bdk, Mat6(sck.asDiagonal())).first;
        const Mat6 model = (pred.covariance().inverse() + Ieff).inverse();
        for (int i = 0; i < 6; ++i)
        {
            const float rel = std::abs(b.covariance()(i, i) / model(i, i) - 1.0f);
            std::printf("  saturates: x%-2d diag %d  Sigma %.4e  model (Sp^-1+Ieff)^-1 %.4e  rel %.2e\n", k, i, b.covariance()(i, i), model(i, i), rel);
            CHECK(rel < 0.01f);
        }
    }
    // per-DOF diagonal ratios (printed only)
    for (int i = 0; i < 6; ++i)
        std::printf("  saturates: diag %d  x10/x1 = %.4f\n", i, a10.covariance()(i, i) / a1.covariance()(i, i));
    // (3) whitened eigen-directions of M = Sc^1/2 Id_10 Sc^1/2 with eigenvalue > 10
    {
        const Vec6 half = sc.cwiseInverse().cwiseSqrt();                      // diag of Sc^1/2
        const Mat6 M = half.asDiagonal() * Id * half.asDiagonal();
        Eigen::SelfAdjointEigenSolver<Mat6> es(0.5f * (M + M.transpose()));
        int n_dir = 0;
        for (int j = 5; j >= 0; --j)
        {
            const float lam = es.eigenvalues()(j);
            Vec6 u = half.cwiseProduct(es.eigenvectors().col(j)); u.normalize();
            const float rr = (u.dot(a10.covariance() * u)) / (u.dot(a1.covariance() * u));
            const bool dom = lam > 10.0f;
            std::printf("  saturates: whitened eig %.3e  direction x10/x1 = %.4f  %s\n", lam, rr, dom ? "(asserted)" : "(weak, not asserted)");
            if (dom) { ++n_dir; CHECK(std::abs(rr - 1.0f) < 0.05f); }
        }
        CHECK(n_dir >= 1);
    }
}

static void lidar_only_repeated_sweeps()
{
    CountableBelief b = make_belief();
    const auto f = lidar_frame(two_face_rays());
    const auto Q = b.process_noise_diag();
    const Vec6 sc = invdiag(b, f);
    for (int k = 0; k < 300; ++k) b.update(f);
    {   // diagnostics only: the one-sided view leaves (cx, w) and (cy, h) degenerate (a face fixes cx - w/2, cy - h/2)
        const Mat6& S = b.covariance();
        std::printf("  repeated sweeps (diag): sigma w %.4f h %.4f ; corr(cx,w) %.3f corr(cy,h) %.3f\n", std::sqrt(S(3, 3)), std::sqrt(S(4, 4)),
                    S(0, 3) / std::sqrt(S(0, 0) * S(3, 3)), S(1, 4) / std::sqrt(S(1, 1) * S(4, 4)));
    }
    const int idx[3] = {0, 1, 2};
    const char* nm[3] = {"cx", "cy", "H"};
    for (int j = 0; j < 3; ++j)
    {
        const int i = idx[j];
        const float sg = std::sqrt(b.covariance()(i, i)), an = std::pow(Q(i) / sc(i), 0.25f);   // sigma at saturation: variance sqrt(Q*Sigma_c)
        std::printf("  repeated sweeps: %s settled sigma %.5f  analytic (Q*Sc)^(1/4) %.5f (variance sqrt(Q*Sc) = %.3e)  ratio %.3f\n",
                    nm[j], sg, an, std::sqrt(Q(i) / sc(i)), sg / an);
        CHECK(std::isfinite(sg) and sg > 0.0f);
        // LOWER bound only: the failure to guard against is COLLAPSE (overconfidence). With one face per axis the rays leave
        // (cx,w) and (cy,h) degenerate, so sigma stays ABOVE the per-frame cap's fixed point (Q*Sc)^(1/4), which assumes every
        // DOF observed. Overconfidence against the true error is checked two-sided in the live run.
        CHECK(sg >= 0.5f * an);
    }
}

static void lidar_only_one_face_keeps_depth()
{
    CountableBelief b = make_belief(0.0f, 0.0f, 0.0f);
    const float cy0 = b.state().cy, H0 = b.state().H;
    const auto f = lidar_frame(front_rays());
    for (int k = 0; k < 200; ++k) b.update(f);
    const float dh = std::abs(b.state().H - H0), dcy = std::abs(b.state().cy - cy0);
    std::printf("  one face: |dH| %.5f |dcy| %.5f (bounds 0.02)\n", dh, dcy);
    CHECK(dh < 0.02f);
    CHECK(dcy < 0.02f);
}

static void scene_first_hit_only()
{   // a box 5 cm in front of the wall, Helios 1.5 m away: no wall return lies behind the box's silhouette
    const float cy = -(0.5f * tw + 0.05f);
    const Eigen::Vector3f o(tx, cy - 0.5f * tw - 1.5f, 1.1075f);
    const auto box = truth_box(cy);
    const std::vector<rc::test::SceneBox> boxes{box};
    const std::vector<rc::test::SceneWall> walls{kWall};
    const auto pts = rc::test::raycast_scene(o, boxes, walls, true, rc::test::kHeliosAzStepDeg, kElev, 12.0f);
    int nbox = 0, nwall = 0, occluded = 0;
    for (const auto& p : pts)
    {
        if (std::abs(p.y()) < 1e-3f)                                       // a wall return: does its segment cross the box?
        {
            ++nwall;
            const std::vector<rc::test::SceneBox> bx{box};
            const Eigen::Vector3f d = (p - o).normalized();
            // sample the segment finely; any sample strictly inside the box (shrunk 1 mm) is an occlusion violation
            for (float t = 0.0f; t < (p - o).norm() - 0.01f; t += 0.002f)
            {
                const Eigen::Vector3f q = o + t * d;
                if (std::abs(q.x() - box.c.x()) < 0.5f * box.w - 1e-3f and std::abs(q.y() - box.c.y()) < 0.5f * box.h - 1e-3f
                    and q.z() > 1e-3f and q.z() < box.H - 1e-3f) { ++occluded; break; }
            }
        }
        else if (std::abs(p.y() - (box.c.y() - 0.5f * box.h)) < 1e-3f and p.z() <= box.H + 1e-3f) ++nbox;
    }
    std::printf("  first-hit scene: %zu returns, %d on the box front face, %d on the wall, %d wall returns through the box\n",
                pts.size(), nbox, nwall, occluded);
    CHECK(nbox > 0);
    CHECK(nwall > 0);
    CHECK(occluded == 0);
}

static void wall_behind_legacy_pull_recorded()
{
    // Legacy flag-off path: Sigma is unbounded along unseen directions by design (C1 = A), so wall returns at weight w DO move
    // the back face on this path. This records the numbers (belief = truth, back face 2 cm from the wall y = 0, physical
    // returns, 2000 LiDAR-only sweeps) and asserts only finiteness; the bounded-Sigma assertion lives in Task 6.
    const float cy = -(0.5f * tw + 0.02f);
    auto build = [&]() { CountableBelief b(CountableBeliefState{tx, cy, tH, tw, tw, 0.0f}, params());
                         b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f)); return b; };
    std::vector<float> is_wall;
    const rc::ai::LidarRays r = scene_rays(Eigen::Vector3f(tx, cy - 0.5f * tw - 1.5f, 1.1075f), cy, 0.50f, true, &is_wall);
    auto gap = [](const CountableBelief& b) { return 0.0f - (b.state().cy + 0.5f * b.state().h); };   // wall y = 0
    auto run = [&](float ww) {
        auto f = lidar_frame(r);
        f.lidar.weights.resize(r.endpoints.size());
        for (std::size_t i = 0; i < r.endpoints.size(); ++i) f.lidar.weights[i] = is_wall[i] != 0.0f ? ww : 1.0f;
        CountableBelief b = build();
        for (int k = 0; k < 2000; ++k) b.update(f);
        return gap(b);
    };
    rc::ai::LidarRays rb = r; rb.endpoints.clear();
    for (std::size_t i = 0; i < r.endpoints.size(); ++i) if (is_wall[i] == 0.0f) rb.endpoints.push_back(r.endpoints[i]);
    CountableBelief base = build();
    const auto fb = lidar_frame(rb);
    for (int k = 0; k < 2000; ++k) base.update(fb);
    const float g0 = gap(base);
    std::printf("  wall behind (legacy, start gap 0.020): baseline (box only) gap %.5f m\n", g0);
    CHECK(std::isfinite(g0));
    for (const float ww : {0.0f, 0.001f, 0.1f, 1.0f})
    {
        const float g = run(ww);
        std::printf("    wall weight %-6g gap %.5f m  diff vs baseline %+.5f m\n", ww, g, g - g0);
        CHECK(std::isfinite(g));
    }
}

static void sigma_h_after_2000_sweeps()
{   // printed for the spec (7c.0: 20.9 cm on the old fixture); diagnostic, only finiteness is asserted
    CountableBelief b = make_belief(0.0f, 0.0f, 0.0f);
    const auto f = lidar_frame(front_rays());
    for (int k = 0; k < 2000; ++k) b.update(f);
    const float sh = std::sqrt(b.covariance()(4, 4));
    std::printf("  sigma_h after 2000 LiDAR-only sweeps (front view, %zu returns): %.4f m\n", f.lidar.endpoints.size(), sh);
    CHECK(std::isfinite(sh));
}

int main()
{
    empty_frame_is_predict_only();
    lidar_only_moves_toward_rays();
    lidar_only_vanishing_membership_is_predict();
    lidar_only_saturates();
    lidar_only_repeated_sweeps();
    lidar_only_one_face_keeps_depth();
    scene_first_hit_only();
    wall_behind_legacy_pull_recorded();
    sigma_h_after_2000_sweeps();
    std::printf("lidar_only_update_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
