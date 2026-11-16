// persistent_belief_test — the countable belief with persistent shared errors and the class prior counted once
// (spec 2026-11-15 lidar-measures §7c.1–7c.2; CountableBeliefParams::persistent = true). Fixture: the golden nightstand
// belief (lidar_golden_test.cpp); every LiDAR fixture ray-casts the TRUE scene (test_scene.h, first hit only).
// Run: ./persistent_belief_test
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>
#include <Eigen/Eigenvalues>
#include "countable_belief.h"
#include "test_scene.h"

using namespace rc;
namespace pu = rc::ai::persistent;
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
    p.persistent       = true;
    return p;
}
static CountableBelief make_belief(float dcx = 0.05f, float dcy = -0.04f, float dH = 0.03f, CountableBeliefParams p = params())
{
    CountableBelief b(CountableBeliefState{tx + dcx, ty + dcy, tH + dH, tw, tw, 0.0f}, p);
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    return b;
}
// The latents' marginals at the config floors (common_mode_*_std), the chain at 1 cm; ρ as given; no fresh covariance.
static void set_latents(CountableFrame& f, float rho, float chain2 = 1e-4f)
{
    const CountableBeliefParams p = params();
    const float pp = p.common_mode_pos_std * p.common_mode_pos_std, ss = p.common_mode_size_std * p.common_mode_size_std,
                yy = p.common_mode_yaw_std * p.common_mode_yaw_std;
    f.sig2_cam = {pp, pp, ss, ss, ss, yy};
    f.sig2_L   = {pp, pp, ss, ss, ss, yy};
    f.sig2_chain = {chain2, chain2};
    f.rho_cam.fill(rho); f.rho_L.fill(rho); f.rho_chain.fill(rho);
}
static void stage_wall(CountableFrame& f)
{
    f.wall.ok = true; f.wall.p = Eigen::Vector2f(0, 0); f.wall.n = Eigen::Vector2f(0, -1);
    f.wall.sigma_m = 0.02f; f.wall.length_m = 4.0f;
    f.room_axis_ok = true; f.room_axis = 0.0f;
}
// Golden camera frame: 200 noise-free points on the front and left faces of the truth box.
static CountableFrame camera_frame(int copies = 1, bool top = true)
{
    CountableFrame f;
    for (int c = 0; c < copies; ++c)
    {
        for (int i = 0; i < 12; ++i) for (int j = 0; j < 10; ++j)
            f.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 12.0f, ty - 0.5f * tw, tH * (j + 0.5f) / 10.0f);
        for (int i = 0; i < 8; ++i) for (int j = 0; j < 10; ++j)
            f.points.emplace_back(tx - 0.5f * tw, ty - 0.5f * tw + tw * (i + 0.5f) / 8.0f, tH * (j + 0.5f) / 10.0f);
    }
    f.R.assign(f.points.size(), params().sigma_base_m * params().sigma_base_m);
    f.z_top_obs = top ? tH : -1.0f;
    stage_wall(f);
    return f;
}
// ---- Physical LiDAR fixtures (Task 3): truth box, wall y = 0, floor; first hit only. ----
static const auto kElev = rc::test::helios_ring_elev_deg();
static rc::test::SceneBox truth_box(float cy = ty) { return {Eigen::Vector2f(tx, cy), 0.0f, tw, tw, tH}; }
static const rc::test::SceneWall kWall{Eigen::Vector2f(-4.0f, 0.0f), Eigen::Vector2f(4.0f, 0.0f), 2.5f};
static rc::ai::LidarRays scene_rays(const Eigen::Vector3f& origin, float cy, float margin, std::vector<float>* on_wall = nullptr)
{
    const auto box = truth_box(cy);
    const std::vector<rc::test::SceneBox> boxes{box};
    const std::vector<rc::test::SceneWall> walls{kWall};
    const auto all = rc::test::raycast_scene(origin, boxes, walls, true, rc::test::kHeliosAzStepDeg, kElev, 12.0f);
    rc::ai::LidarRays r;
    r.origin = origin; r.endpoints = rc::test::select_near(all, box, margin);
    if (on_wall) for (const auto& p : r.endpoints) on_wall->push_back(std::abs(p.y() - kWall.a.y()) < 1e-3f ? 1.0f : 0.0f);
    r.precision = 2500.0f; r.robust_c_m = 0.05f;
    return r;
}
static rc::ai::LidarRays front_rays(float cy = ty) { return scene_rays(Eigen::Vector3f(tx, cy - 0.5f * tw - 1.5f, 1.1075f), cy, 0.10f); }
static rc::ai::LidarRays two_face_rays() { return scene_rays(Eigen::Vector3f(tx - 1.5f, ty - 1.5f, 1.1075f), ty, 0.10f); }
static CountableFrame lidar_frame(const rc::ai::LidarRays& r)
{
    CountableFrame f;
    f.lidar = r; f.lidar_only = true; f.z_top_obs = -1.0f;
    stage_wall(f);
    return f;
}

// ---- The engine on the same frame content: ai::update through an adapter whose ONLY information is one channel of
//      accumulate_channels (n_prims = 0, so the engine's own point loop adds nothing) and whose Σc⁻¹ is the summed latent
//      marginals. Process noise 0: the caller passes the persistent path's PREDICTED θθ block as Σ. ----
struct EngineAdapter
{
    using State = CountableBeliefState;
    using Frame = CountableFrame;
    const CountableBelief& b;
    int   iters;
    Vec6  scinv;
    bool  camera;
    mutable float th_norm = 0.0f;   // the θ-only channel must be empty for the comparison to be the same model
    int   gn_iters() const { return iters; }
    int   n_prims()  const { return 0; }
    float sigma2()   const { return b.sigma2(); }
    float sdf_prim(const Eigen::Vector3f& p, const State& s, int k) const { return b.sdf_prim(p, s, k); }
    Vec6  sdf_jacobian(const Eigen::Vector3f& p, const State& s, int k) const { return b.sdf_jacobian(p, s, k); }
    std::array<float, 3> responsibilities(const Eigen::Vector3f& p, const State& s, float R) const { return b.responsibilities(p, s, R); }
    void  apply_constraints(State& s) const { b.apply_constraints(s); }
    void  canonicalize(State&) const {}
    Mat6  transition() const { return Mat6::Identity(); }
    Vec6  process_noise_diag() const { return Vec6::Zero(); }
    Vec6  prior_cov_diag() const { return b.prior_cov_diag(); }
    Vec6  common_mode_inv_diag(const Frame&) const { return scinv; }
    bool  has_extra_data(const Frame& f) const { return b.has_extra_data(f); }
    void  accumulate_extra(const State& s, const Frame& f, Mat6& Id, Vec6& bd) const
    {
        Mat6 Ic, Il, It; Vec6 bc, bl, bt;
        b.accumulate_channels(s, s, s, f, Ic, bc, Il, bl, It, bt);
        th_norm = std::max(th_norm, It.norm() + bt.norm());
        Id += camera ? Ic : Il;
        bd += camera ? bc : bl;
    }
};
template <class A, class B>
static double rel(const A& a, const B& b) { return double((a - b).norm()) / std::max(double(b.norm()), 1e-30); }

static void rho0_equals_engine(bool camera)
{
    CountableBeliefParams p = params();
    p.depth_unobs_precision = 0.0f;
    // The θ-only channel off as well: the truth box is flush (back face on the wall), so the one-sided (2c) no-cross term is
    // legitimately live around the converged state, and it enters unsaturated here but saturated in the engine.
    p.wall_no_cross_precision = 0.0f;
    CountableBelief fx = make_belief(0.05f, -0.04f, 0.03f, p);
    fx.set_prior_added(true, true);
    CountableFrame f = camera ? camera_frame() : lidar_frame(two_face_rays());
    set_latents(f, 0.0f, 2.5e-4f);
    // the summed marginals of what the channel measures: chain (cx, cy) + the channel's own latent
    Vec6 sc;
    for (int i = 0; i < 6; ++i) sc(i) = (camera ? f.sig2_cam[i] : f.sig2_L[i]) + (i < 2 ? f.sig2_chain[i] : 0.0f);
    const Vec6 scinv = sc.cwiseInverse();

    CountableBelief a = fx;
    a.update(f);
    const auto& its = a.dbg_iterates();
    const char* nm = camera ? "rho0_camera_only_equals_engine" : "rho0_lidar_only_equals_engine";
    CHECK(int(its.size()) == p.gn_iters);
    double worst = 0.0;
    for (int k = 1; k <= int(its.size()); ++k)
    {
        CountableBelief e = fx;
        e.set_frame_wall_for_test(f.wall);
        EngineAdapter ad{e, k, scinv, camera};
        CountableBeliefState st = fx.state();
        Mat6 S = a.dbg_Pbar_tt();
        Vec6 pm = Vec6::Zero();
        rc::ai::update<6>(ad, st, S, pm, f);
        const double r_th = rel(its[k - 1], st.vec());
        const double r_step = rel(its[k - 1] - fx.state().vec(), st.vec() - fx.state().vec());
        worst = std::max(worst, r_th);
        std::printf("  %s  iterate %d: theta rel %.3e (step rel %.3e)  theta-only |I|+|g| %.1e\n", nm, k, r_th, r_step, ad.th_norm);
        CHECK(r_th < 1e-4);
        CHECK(ad.th_norm == 0.0f);
        if (k == int(its.size()))
        {
            const double r_S = rel(a.covariance(), S);
            std::printf("  %s  Sigma_tt rel %.3e  (diag persistent %.4e %.4e %.4e %.4e %.4e %.4e)\n", nm, r_S,
                        a.covariance()(0, 0), a.covariance()(1, 1), a.covariance()(2, 2), a.covariance()(3, 3),
                        a.covariance()(4, 4), a.covariance()(5, 5));
            std::printf("  %s                     (diag engine     %.4e %.4e %.4e %.4e %.4e %.4e)\n", nm,
                        S(0, 0), S(1, 1), S(2, 2), S(3, 3), S(4, 4), S(5, 5));
            CHECK(r_S < 1e-4);
            CHECK(rel(a.state().vec(), st.vec()) < 1e-4);
        }
    }
}

static void prior_added_once()
{
    // (2c) no-cross off so the θ-only channel holds exactly the relational anchors: the flush truth box keeps the one-sided
    // term live near convergence (it is every-frame by design, and is not what this test is about). Ruling T6-1: one flag
    // per anchor, each set only on an admissible camera update that stages its context.
    CountableBeliefParams p = params();
    p.wall_no_cross_precision = 0.0f;
    CountableBelief b = make_belief(0.05f, -0.04f, 0.03f, p);
    CountableFrame fl = lidar_frame(two_face_rays()); set_latents(fl, 1.0f);
    CountableFrame fc = camera_frame();               set_latents(fc, 1.0f);
    CountableFrame fn = fc; fn.wall = WallRef{};       // same camera frame, no wall staged (room axis still staged)
    b.update(fl);
    const float n0 = b.dbg_Id_th().norm();
    std::printf("  prior_added_once: after LiDAR-only wall=%d axis=%d  |Id_th| %.3e\n", int(b.wall_prior_added()), int(b.axis_prior_added()), n0);
    CHECK(not b.wall_prior_added() and not b.axis_prior_added());
    CHECK(n0 == 0.0f);
    b.update(fn);
    const Mat6 I0 = b.dbg_Id_th();
    std::printf("  prior_added_once: camera, no wall: wall=%d axis=%d  Id_th(1,1) %.3e (flush)  Id_th(5,5) %.3e (axis)  |Id_th| %.3e\n",
                int(b.wall_prior_added()), int(b.axis_prior_added()), I0(1, 1), I0(5, 5), I0.norm());
    CHECK(not b.wall_prior_added() and b.axis_prior_added());
    CHECK(I0(5, 5) > 0.0f);
    {   // only the axis term: (5,5) and nothing else
        Mat6 rest = I0; rest(5, 5) = 0.0f;
        CHECK(rest.norm() == 0.0f);
    }
    b.update(fc);
    const Mat6 I1 = b.dbg_Id_th();
    std::printf("  prior_added_once: camera, wall staged: wall=%d axis=%d  Id_th(1,1) %.3e (flush)  Id_th(5,5) %.3e (parallel)\n",
                int(b.wall_prior_added()), int(b.axis_prior_added()), I1(1, 1), I1(5, 5));
    CHECK(b.wall_prior_added() and b.axis_prior_added());
    CHECK(I1(1, 1) > 0.0f);
    b.update(fc);
    const float n2 = b.dbg_Id_th().norm();
    std::printf("  prior_added_once: camera, wall staged again: |Id_th| %.3e\n", n2);
    CHECK(n2 == 0.0f);
}

// Under persistent, (2c) wall no-cross is a θ-only term every frame: a belief whose back face is 3 cm THROUGH the wall is
// pushed back toward the legal side, by a camera update and by a LiDAR-only update. Isolated: the relational anchors are
// already in (flags set), the depth boost is off, and both views look at the LEFT face (x = const), which says nothing about
// the depth direction (cy, h); control = the same update with (2c) off.
static void no_cross_pushes_back_persistent()
{
    const float cy = 0.03f - 0.5f * tw;                                       // back face at y = +0.03 (wall at y = 0)
    auto gap = [](const CountableBelief& b) { return 0.0f - (b.state().cy + 0.5f * b.state().h); };
    CountableFrame fcam;                                                      // left face only, truth box
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 10; ++j)
        fcam.points.emplace_back(tx - 0.5f * tw, ty - 0.5f * tw + tw * (i + 0.5f) / 8.0f, tH * (j + 0.5f) / 10.0f);
    fcam.R.assign(fcam.points.size(), params().sigma_base_m * params().sigma_base_m);
    stage_wall(fcam);
    CountableFrame flid = lidar_frame(scene_rays(Eigen::Vector3f(tx - 0.5f * tw - 1.5f, ty, 1.1075f), ty, 0.10f));
    for (const bool camera : {true, false})
    {
        CountableFrame f = camera ? fcam : flid; set_latents(f, 1.0f);
        float g[2], th = 0.0f, last_step = 0.0f;
        float g_off = 0.0f;
        {   // diagnostic: the flag-off path on the same update
            CountableBeliefParams p = params(); p.persistent = false; p.depth_unobs_precision = 0.0f;
            CountableBelief b(CountableBeliefState{tx, cy, tH, tw, tw, 0.0f}, p);
            b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
            b.update(f);
            g_off = gap(b);
        }
        for (const bool on : {true, false})
        {
            CountableBeliefParams p = params();
            p.depth_unobs_precision = 0.0f;
            if (not on) p.wall_no_cross_precision = 0.0f;
            CountableBelief b(CountableBeliefState{tx, cy, tH, tw, tw, 0.0f}, p);
            b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
            b.set_prior_added(true, true);
            b.update(f);
            g[on ? 0 : 1] = gap(b);
            if (on)
            {
                th = b.dbg_Id_th().norm();
                const auto& its = b.dbg_iterates();
                std::printf("    iterates (gap):");
                for (const auto& v : its) std::printf(" %+.5f", 0.0f - (v(1) + 0.5f * v(4)));
                std::printf("\n");
                last_step = its.size() >= 2 ? (its.back() - its[its.size() - 2]).cwiseAbs().maxCoeff() : 0.0f;
            }
        }
        std::printf("  no_cross (%s, %zu data): start gap -0.03000 ; after one update gap %.5f with (2c), %.5f without ; |Id_th| %.3e ; "
                    "last GN step %.2e m ; flag-off path %.5f\n",
                    camera ? "camera" : "LiDAR-only", camera ? f.points.size() : f.lidar.endpoints.size(), g[0], g[1], th, last_step, g_off);
        // camera: the view says nothing about depth, so the whole move is (2c)'s and it is toward the legal side. LiDAR-only:
        // this sweep's rays (left face + top, belief 3 cm off) themselves push the box deeper; (2c) must counter that push.
        if (camera) CHECK(g[0] > -0.03f + 0.005f);
        CHECK(g[0] - g[1] > 0.005f);          // the (2c)-attributable move toward the legal side
        CHECK(th > 0.0f);
        // (spec 7c.6) fixed-precision (2c) + first-crossing rays: the iterates converge (no half-Hessian overshoot, no wall ray
        // switching on and off) and land at the wall
        CHECK(last_step < 0.001f);
        CHECK(g[0] > -0.005f);
    }
}

static void aged_belief_returns_to_class_P()
{
    CountableBeliefParams p = params();
    p.volatility_infer = false;   // the configured Q: this test is about the clamp, not about how ω learnt Q
    CountableBelief b = make_belief(0.05f, -0.04f, 0.03f, p);
    CountableFrame fc = camera_frame(); set_latents(fc, 1.0f);
    b.update(fc);
    const Vec6 th = b.state().vec();
    const Vec6 s1 = b.covariance().diagonal();
    for (int k = 0; k < 5000; ++k) b.predict();
    const Vec6 p0 = b.prior_cov_diag(), d = b.covariance().diagonal();
    float worst = 0.0f;
    for (int i = 0; i < 6; ++i) worst = std::max(worst, std::abs(d(i) / p0(i) - 1.0f));
    std::printf("  aged: |dtheta| %.3e ; diag after 1 update %.3e %.3e %.3e %.3e %.3e %.3e ; after 5000 predicts / class P worst rel %.3e\n",
                (b.state().vec() - th).cwiseAbs().maxCoeff(), s1(0), s1(1), s1(2), s1(3), s1(4), s1(5), worst);
    CHECK((b.state().vec() - th).cwiseAbs().maxCoeff() < 1e-6f);
    CHECK(worst < 0.02f);
}

static void wall_behind_does_not_pull()
{
    // Task 3's physical scene: belief = truth, back face 2 cm from the wall y = 0, returns within 0.5 m of the box (wall returns
    // included), 2000 LiDAR-only sweeps, parked (ρ = 1). Wall returns at weight 0.1 vs the box-returns-only baseline.
    const float cy = -(0.5f * tw + 0.02f);
    auto build = [&]() { CountableBelief b(CountableBeliefState{tx, cy, tH, tw, tw, 0.0f}, params());
                         b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f)); return b; };
    std::vector<float> is_wall;
    const rc::ai::LidarRays r = scene_rays(Eigen::Vector3f(tx, cy - 0.5f * tw - 1.5f, 1.1075f), cy, 0.50f, &is_wall);
    auto gap = [](const CountableBelief& b) { return 0.0f - (b.state().cy + 0.5f * b.state().h); };
    rc::ai::LidarRays rb = r; rb.endpoints.clear();
    for (std::size_t i = 0; i < r.endpoints.size(); ++i) if (is_wall[i] == 0.0f) rb.endpoints.push_back(r.endpoints[i]);
    CountableBelief base = build();
    CountableFrame fb = lidar_frame(rb); set_latents(fb, 1.0f);
    for (int k = 0; k < 2000; ++k) base.update(fb);
    const float g0 = gap(base);
    std::printf("  wall_behind (persistent, start gap 0.020, %zu returns, %zu on the box): baseline (box only) gap %.5f m  sigma_h %.4f\n",
                r.endpoints.size(), rb.endpoints.size(), g0, std::sqrt(base.covariance()(4, 4)));
    float g01 = 0.0f;
    for (const float ww : {0.0f, 0.001f, 0.1f, 1.0f})
    {
        CountableFrame f = lidar_frame(r); set_latents(f, 1.0f);
        f.lidar.weights.resize(r.endpoints.size());
        for (std::size_t i = 0; i < r.endpoints.size(); ++i) f.lidar.weights[i] = is_wall[i] != 0.0f ? ww : 1.0f;
        CountableBelief b = build();
        for (int k = 0; k < 2000; ++k) b.update(f);
        const float g = gap(b);
        if (ww == 0.1f) g01 = g;
        std::printf("    wall weight %-6g gap %.5f m  diff vs baseline %+.5f m  sigma_h %.4f\n", ww, g, g - g0, std::sqrt(b.covariance()(4, 4)));
    }
    CHECK(std::isfinite(g0) and std::isfinite(g01));
    CHECK(std::abs(g01 - g0) < 0.005f);
    {   // self-control: the SAME scene on the flag-off path at wall weight 1 moves the gap (Task 3 recorded +0.209 m), so the
        // test can see a pull when there is one
        CountableBeliefParams p = params(); p.persistent = false;
        CountableBelief b(CountableBeliefState{tx, cy, tH, tw, tw, 0.0f}, p);
        b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
        CountableFrame f = lidar_frame(r);
        f.lidar.weights.assign(r.endpoints.size(), 1.0f);
        for (int k = 0; k < 2000; ++k) b.update(f);
        const float g = gap(b);
        std::printf("    self-control (flag off, wall weight 1): gap %.5f m  diff vs persistent baseline %+.5f m\n", g, g - g0);
        CHECK(std::abs(g - g0) > 0.01f);
    }
}

static void physical_lidar_only_sigma_bounded()
{
    CountableBelief b = make_belief(0.0f, 0.0f, 0.0f);
    CountableFrame f = lidar_frame(front_rays()); set_latents(f, 1.0f);
    const float back0 = b.state().cy + 0.5f * b.state().h, w0 = b.state().w;
    for (int k = 0; k < 2000; ++k) b.update(f);
    const float sh = std::sqrt(b.covariance()(4, 4)), back = b.state().cy + 0.5f * b.state().h;
    std::printf("  sigma bounded: sigma_h after 2000 LiDAR-only sweeps (front, %zu returns) %.4f m (class std %.4f) ; back face moved %.5f m ; "
                "w moved %.5f m\n", f.lidar.endpoints.size(), sh, params().prior_footprint_std, back - back0, b.state().w - w0);
    CHECK(sh <= 1.01f * params().prior_footprint_std);
    CHECK(std::abs(back - back0) < 0.001f);
    CHECK(std::abs(b.state().w - w0) < 0.002f);
}

static void parked_camera_is_one_look()
{
    CountableBeliefParams p = params();
    p.process_std_m = 0.0f; p.process_std_yaw = 0.0f; p.volatility_infer = false;   // no Q: the statement is about b
    p.depth_unobs_precision = 0.0f;                                                   // points only in the camera channel
    CountableBelief seq = make_belief(0.0f, 0.0f, 0.0f, p), one = make_belief(0.0f, 0.0f, 0.0f, p);
    CountableFrame f1 = camera_frame(1, false);   set_latents(f1, 1.0f);
    CountableFrame f100 = camera_frame(100, false); set_latents(f100, 1.0f);
    for (int k = 0; k < 100; ++k) seq.update(f1);
    one.update(f100);
    const double r = rel(seq.covariance(), one.covariance());
    std::printf("  parked: Sigma_tt rel (100 frames vs one frame with 100 Id) %.3e ; diag ratios", r);
    for (int i = 0; i < 6; ++i) std::printf(" %.4f", seq.covariance()(i, i) / one.covariance()(i, i));
    std::printf("\n");
    CHECK(r < 0.01);
}

static void zero_chain_cov_runs()
{
    CountableBelief b = make_belief();
    const Vec6 s0 = b.state().vec();
    CountableFrame f = camera_frame(); set_latents(f, 1.0f, 0.0f); f.lidar = two_face_rays();
    for (int k = 0; k < 10; ++k) b.update(f);
    const auto sd = b.latent_std();
    std::printf("  zero chain: state moved %.4f ; latent std chain %.2e %.2e ; P finite %d\n", (b.state().vec() - s0).norm(),
                sd(0), sd(1), int(b.joint_covariance().allFinite()));
    CHECK(b.state().vec().allFinite() and b.covariance().allFinite() and b.joint_covariance().allFinite());
    CHECK(b.latent_mean().allFinite());
    CHECK((b.state().vec() - s0).norm() > 1e-3f);
    CHECK(sd(0) == 0.0 and sd(1) == 0.0);
}

static void mode_rotation_permutes_latents()
{
    CountableBelief b = make_belief();
    CountableFrame f = camera_frame(); set_latents(f, 1.0f); f.lidar = two_face_rays();
    b.update(f);
    CountableBelief::VecKd v;
    v << 0.011, 0.012, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036;
    b.set_latent_mean_for_test(v);
    const CountableBelief::Mat20d P0 = b.joint_covariance();
    b.apply_mode_rotation_for_test(1);
    const auto& m = b.latent_mean();
    const auto& P = b.joint_covariance();
    std::printf("  mode rotation: b_cam w,h %.3f %.3f (was %.3f %.3f) ; b_L w,h %.3f %.3f ; b_chain %.3f %.3f\n",
                m(5), m(6), v(5), v(6), m(11), m(12), m(0), m(1));
    CHECK(m(5) == v(6) and m(6) == v(5));                 // b_cam w ↔ h
    CHECK(m(11) == v(12) and m(12) == v(11));             // b_L w ↔ h
    CHECK(m(0) == v(0) and m(1) == v(1));                 // b_chain untouched
    CHECK(m(2) == v(2) and m(7) == v(7) and m(13) == v(13));
    CHECK(P(3, 3) == P0(4, 4) and P(4, 4) == P0(3, 3));
    CHECK(P(6 + 5, 6 + 5) == P0(6 + 6, 6 + 6) and P(6 + 11, 6 + 11) == P0(6 + 12, 6 + 12));
    CHECK(P(3, 6 + 5) == P0(4, 6 + 6));                   // θ_w–b_cam,w cross term follows both relabels
    CHECK(b.covariance()(3, 3) == float(P(3, 3)));
    const double asym = (P - P.transpose()).cwiseAbs().maxCoeff();
    const double mine = Eigen::SelfAdjointEigenSolver<CountableBelief::Mat20d>(P).eigenvalues().minCoeff();
    std::printf("  mode rotation: max|P-P^T| %.3e  min eig %.3e\n", asym, mine);
    CHECK(asym == 0.0);
    CHECK(mine >= -1e-12);
}

static void sanitize_reseeds()
{
    CountableBelief b = make_belief();
    CountableFrame f = camera_frame(); set_latents(f, 1.0f); f.lidar = two_face_rays();
    b.update(f);
    CHECK(b.wall_prior_added() and b.axis_prior_added());
    b.joint_covariance_for_test()(0, 0) = std::numeric_limits<double>::quiet_NaN();
    b.sanitize_covariance();
    const auto& P = b.joint_covariance();
    const Vec6 p0 = b.prior_cov_diag();
    bool th_ok = true, lat_ok = true, cross_ok = true;
    for (int i = 0; i < 6; ++i) th_ok = th_ok and P(i, i) == double(p0(i));
    const std::array<float, 14> m2 = {f.sig2_chain[0], f.sig2_chain[1], f.sig2_cam[0], f.sig2_cam[1], f.sig2_cam[2], f.sig2_cam[3],
                                      f.sig2_cam[4], f.sig2_cam[5], f.sig2_L[0], f.sig2_L[1], f.sig2_L[2], f.sig2_L[3], f.sig2_L[4], f.sig2_L[5]};
    for (int j = 0; j < 14; ++j) lat_ok = lat_ok and std::abs(P(6 + j, 6 + j) / double(m2[j]) - 1.0) < 1e-6;
    for (int r = 0; r < 20; ++r) for (int c = 0; c < 20; ++c) if (r != c) cross_ok = cross_ok and P(r, c) == 0.0;
    std::printf("  sanitize: thetatheta = P %d ; latents = marginals %d ; cross 0 %d ; wall/axis added %d/%d ; b = 0 %d\n",
                int(th_ok), int(lat_ok), int(cross_ok), int(b.wall_prior_added()), int(b.axis_prior_added()), int(b.latent_mean().isZero(0.0)));
    CHECK(th_ok and lat_ok and cross_ok);
    CHECK(not b.wall_prior_added() and not b.axis_prior_added());
    CHECK(b.latent_mean().isZero(0.0));
    CHECK(b.covariance().allFinite());
}

// ─── Ruling T4-1a: a CURVED measurement h(θ + b). The caller's restarting loop matches today's eliminated (saturate) path
//     at every iterate; a loop that carries b̃ between iterates does not. 1-D, double. ────────────────────────────────
static void nonlinear_restart_matches_eliminated()
{
    using V2 = Eigen::Vector2d;
    using M2 = Eigen::Matrix2d;
    const double Pth = 0.0036, Sb = 4e-4, R = 1e-4, z = 0.1, kappa = 5.0;   // h(u) = u + κu²
    const auto h  = [&](double u) { return u + kappa * u * u; };
    const auto dh = [&](double u) { return 1.0 + 2.0 * kappa * u; };
    M2 Pbar = M2::Zero(); Pbar(0, 0) = Pth; Pbar(1, 1) = Sb;
    const V2 xbar = V2::Zero();
    const auto channel = [&](double u)   // linearised at u: information and log-likelihood gradient on u
    {
        pu::Channel<1, 1> c;
        const double J = dh(u);
        c.info.I(0, 0) = J * J / R;
        c.info.g(0)    = J * (z - h(u)) / R;
        c.B(0, 0)      = 1.0;
        return c;
    };
    pu::FreshInfo<1> none;
    double th_e = 0.0;                      // eliminated: today's engine step (saturate with Σc = Σb, b at b̄ = 0)
    V2 x_r = xbar, x_c = xbar;              // restarting / carrying
    double worst_r = 0.0, last_c = 0.0;
    std::printf("  nonlinear restart (h = u + %.0f u^2):", kappa);
    for (int k = 0; k < 6; ++k)
    {
        {   // eliminated
            const double J = dh(th_e), Id = J * J / R, bd = J * (z - h(th_e)) / R;
            const double Ieff = Id / (1.0 + Id * Sb), beff = bd / (1.0 + Id * Sb);
            th_e += (1.0 / Pth * (0.0 - th_e) + beff) / (1.0 / Pth + Ieff);
        }
        {   // restarting: x̃ = (θ_k, b̄)
            const V2 xt(x_r(0), xbar(1));
            const std::array<pu::Channel<1, 1>, 1> ch{channel(xt(0) + xt(1))};
            x_r = pu::joint_step<1, 1>(xt, xbar, Pbar, std::span<const pu::Channel<1, 1>>(ch), none);
        }
        {   // carrying: x̃ = the previous iterate's (θ, b)
            const std::array<pu::Channel<1, 1>, 1> ch{channel(x_c(0) + x_c(1))};
            x_c = pu::joint_step<1, 1>(x_c, xbar, Pbar, std::span<const pu::Channel<1, 1>>(ch), none);
        }
        const double er = std::abs(x_r(0) - th_e) / std::abs(th_e);
        worst_r = std::max(worst_r, er);
        last_c = std::abs(x_c(0) - th_e);
        std::printf(" [%d] elim %.6f restart %.6f (rel %.1e) carry %.6f", k, th_e, x_r(0), er, x_c(0));
    }
    std::printf("\n  nonlinear restart: worst restart rel %.3e ; carrying differs by %.3e at the last iterate\n", worst_r, last_c);
    CHECK(worst_r < 1e-9);
    CHECK(last_c > 1e-4);
}

static void flag_off_golden()
{   // the golden trajectory, run (not copied): the flag-off path must be bit-identical
    const int rc = std::system(GOLDEN_BIN);
    std::printf("  flag_off_golden: %s exit %d\n", GOLDEN_BIN, rc);
    CHECK(rc == 0);
}

int main()
{
    flag_off_golden();
    nonlinear_restart_matches_eliminated();
    rho0_equals_engine(true);
    rho0_equals_engine(false);
    prior_added_once();
    no_cross_pushes_back_persistent();
    aged_belief_returns_to_class_P();
    wall_behind_does_not_pull();
    physical_lidar_only_sigma_bounded();
    parked_camera_is_one_look();
    zero_chain_cov_runs();
    mode_rotation_permutes_latents();
    sanitize_reseeds();
    std::printf("persistent_belief_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
