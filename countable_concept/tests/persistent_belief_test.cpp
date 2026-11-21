// persistent_belief_test — the countable belief with persistent shared errors and the class prior counted once
// (spec 2026-11-15 lidar-measures §7c.1–7c.2; CountableBeliefParams::persistent = true). Fixture: the golden nightstand
// belief (lidar_golden_test.cpp); every LiDAR fixture ray-casts the TRUE scene (test_scene.h, first hit only).
// Run: ./persistent_belief_test
#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <vector>
#include <Eigen/Eigenvalues>
#include "countable_belief.h"
#include "countable_viewpoint.h"
#include "test_scene.h"
#include "beam_fixture.h"

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
    p.beam             = rc::test::test_beam_config();
    return p;
}
static CountableBelief make_belief(float dcx = 0.05f, float dcy = -0.04f, float dH = 0.03f, CountableBeliefParams p = params())
{
    CountableBelief b(CountableBeliefState{tx + dcx, ty + dcy, tH + dH, tw, tw, 0.0f}, p);
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    return b;
}
// The robot, parked 1.5 m in front of the golden box (the lever of e when Σ_pose has a ψ part).
static const Eigen::Vector2f kRobot(tx, ty - 1.5f);
// The latents' marginals at the config floors (common_mode_*_std), the robot pose error e at 1 cm on (x, y) and none on ψ
// (the old (cx, cy) chain at 1 cm); ρ as given (all latents); no fresh covariance; the robot parked at kRobot.
static void set_latents(CountableFrame& f, float rho, float chain2 = 1e-4f)
{
    const CountableBeliefParams p = params();
    const float pp = p.common_mode_pos_std * p.common_mode_pos_std, ss = p.common_mode_size_std * p.common_mode_size_std,
                yy = p.common_mode_yaw_std * p.common_mode_yaw_std;
    f.sig2_cam = {pp, pp, ss, ss, ss, yy};
    f.sig2_L   = {pp, pp, ss, ss, ss, yy};
    f.sigma_pose = Eigen::Vector3d(chain2, chain2, 0.0).asDiagonal();
    f.rho_cam.fill(rho); f.rho_L.fill(rho); f.rho_chain = rho;
    f.chain_dr = Eigen::Vector2f::Zero();
    f.robot_xy = kRobot;
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
// The persistent path reads the BEAM set (spec 7c.8), not f.lidar: the same sweep (same origin, same scene) staged by RAY
// GEOMETRY through the select box (the truth box + 10 cm) with hand labels — 0.9 on the truth box, ε_px = 0.02 on the wall /
// floor — κ = 1, σ_s = model_sigma_m. Endpoint staging (f.lidar's select_near, which drops the floor beside the box) would
// leave the silhouettes without the returns that refute a wider or deeper box: measured on the creep set, +6.7 cm on w.
// beam_of keeps the endpoint-selected set for the one fixture that compares against it (wall_behind's printed baseline).
static rc::ai::beam::BeamRays beam_of(const rc::ai::LidarRays& r, const rc::test::SceneBox& box)
{
    rc::ai::beam::BeamRays b;
    b.origin = r.origin.cast<double>(); b.kappa = 1.0; b.rho_max = 6.0; b.sigma_s = params().model_sigma_m;
    const std::vector<rc::test::SceneBox> obj{box};
    for (const auto& e : r.endpoints)
    {
        b.endpoints.push_back(e.cast<double>());
        b.m.push_back(rc::test::on_object(obj, e) ? rc::test::kMOn : rc::test::kEpsPx);
    }
    return b;
}
// Stage a ray-set on a frame for both paths: f.lidar (flag off) and f.beam (flag on), labelled against the truth box at cy.
static rc::ai::beam::BeamRays ray_staged(const Eigen::Vector3f& origin, float cy)
{
    const std::vector<rc::test::SceneBox> obj{truth_box(cy)};
    const std::vector<rc::test::SceneWall> walls{kWall};
    return rc::test::helios_beam(origin, obj, walls, truth_box(cy), 0.10f, rc::test::hand_labels(obj), 1.0, params().model_sigma_m);
}
static void stage(CountableFrame& f, const rc::ai::LidarRays& r, float cy = ty) { f.lidar = r; f.beam = ray_staged(r.origin, cy); }
static rc::ai::LidarRays front_rays(float cy = ty) { return scene_rays(Eigen::Vector3f(tx, cy - 0.5f * tw - 1.5f, 1.1075f), cy, 0.10f); }
static rc::ai::LidarRays two_face_rays() { return scene_rays(Eigen::Vector3f(tx - 1.5f, ty - 1.5f, 1.1075f), ty, 0.10f); }
static CountableFrame lidar_frame(const rc::ai::LidarRays& r, float cy = ty)
{
    CountableFrame f;
    stage(f, r, cy); f.lidar_only = true; f.z_top_obs = -1.0f;
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

// spec 7c.8 Pass ("Earlier items this section changes"): the ρ = 0 regression pins the persistent path against the flag-off
// path on CAMERA-ONLY frames. On frames with rays the two differ BY DESIGN: the flag-off engine reads f.lidar through the
// sphere-traced factor, the persistent path reads f.beam through the smooth beam model (7c.8), so there is no LiDAR-only case.
// spec 7c.7 Pass: the ρ = 0 regression runs with Σ_pose = 0 (a zero-variance e; b_cam, b_L as before) and pins the persistent
// path against the flag-off engine WITHOUT chain covariance. Ruling T14-3: the damped GN (λ ≥ 10⁻³·4⁻ᵏ on diag W_θθ) does not
// follow the undamped engine iterate by iterate, but both converge to the same stationary point (δ = 0 solves the damped
// system for every λ): the CONVERGED results are compared (1e-6), the engine run for kEngineIters undamped iterates.
static void rho0_regression_zero_pose_cov(bool camera)
{
    constexpr int kEngineIters = 50;
    CountableBeliefParams p = params();
    p.depth_unobs_precision = 0.0f;
    // The θ-only channel off as well: the truth box is flush (back face on the wall), so the one-sided (2c) no-cross term is
    // legitimately live around the converged state, and it enters unsaturated here but saturated in the engine.
    p.wall_no_cross_precision = 0.0f;
    CountableBelief fx = make_belief(0.05f, -0.04f, 0.03f, p);
    fx.set_prior_added(true, true);
    fx.set_first_update_done(true);   // the recursion (x̄ = the current mean), not the class-mean first update (Task 6c)
    CountableFrame f = camera ? camera_frame() : lidar_frame(two_face_rays());
    set_latents(f, 0.0f, 0.0f);
    // the marginal of what the channel measures: the channel's own latent only (e has zero variance)
    Vec6 sc;
    for (int i = 0; i < 6; ++i) sc(i) = camera ? f.sig2_cam[i] : f.sig2_L[i];
    const Vec6 scinv = sc.cwiseInverse();

    CountableBelief a = fx;
    a.update(f);
    const char* nm = camera ? "rho0_zero_pose_cov_camera_only" : "rho0_zero_pose_cov_lidar_only";
    CountableBelief e = fx;
    e.set_frame_wall_for_test(f.wall);
    EngineAdapter ad{e, kEngineIters, scinv, camera};
    CountableBeliefState st = fx.state();
    Mat6 S = a.dbg_Pbar_tt();
    Vec6 pm = Vec6::Zero();
    rc::ai::update<6>(ad, st, S, pm, f);
    const double r_th = rel(a.state().vec(), st.vec());
    // Σ without the solver's δ_last δ_lastᵀ (spec 7c.7 d); the engine's Σ is float, hence the 1e-4 bound kept from Task 13.
    const Mat6 Sp = (a.joint_covariance().topLeftCorner<6, 6>() - a.last_gn_step() * a.last_gn_step().transpose()).cast<float>();
    const double r_S  = rel(Sp, S);
    std::printf("  %s: persistent gn_it %d conv %d (accepted %zu) vs engine %d iterates: theta rel %.3e ; Sigma_tt rel %.3e ; "
                "theta-only |I|+|g| %.1e\n", nm, a.last_gn_iters(), int(a.last_gn_converged()), a.dbg_iterates().size(), kEngineIters,
                r_th, r_S, ad.th_norm);
    std::printf("  %s  (diag persistent %.4e %.4e %.4e %.4e %.4e %.4e)\n", nm, a.covariance()(0, 0), a.covariance()(1, 1),
                a.covariance()(2, 2), a.covariance()(3, 3), a.covariance()(4, 4), a.covariance()(5, 5));
    std::printf("  %s  (diag engine     %.4e %.4e %.4e %.4e %.4e %.4e)\n", nm, S(0, 0), S(1, 1), S(2, 2), S(3, 3), S(4, 4), S(5, 5));
    CHECK(a.last_gn_converged());
    CHECK(ad.th_norm == 0.0f);
    CHECK(r_th <= 1e-6);
    CHECK(r_S <= 1e-4);
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
    // included), 2000 LiDAR-only sweeps, parked (ρ = 1).
    // spec 7c.8: the wall returns' camera LABEL m (the beam model's ownership) replaces 7c.2's weight; a label is never certain,
    // so a weight of 0 or 1 reads as ε_px or 1 − ε_px. The baseline is the wall returns at m = ε_px (what a correct mask says):
    // 7c.6's box-returns-only baseline is an endpoint-staging artefact on this path — with no return behind the box nothing
    // refutes a deeper box, and the claimed returns near the top's back edge pull it onto the wall (measured: gap 0.020 → 0.000,
    // stopped by (2c)); it is printed. CHECK: the wall at m = 0.1 moves the back face < 5 mm from the baseline. m = 1 − ε_px (an
    // over-segmented mask on the wall) is the stated limit 7c.3 row E and is printed.
    const float cy = -(0.5f * tw + 0.02f);
    auto build = [&]() { CountableBelief b(CountableBeliefState{tx, cy, tH, tw, tw, 0.0f}, params());
                         b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f)); return b; };
    std::vector<float> is_wall;
    const rc::ai::LidarRays r = scene_rays(Eigen::Vector3f(tx, cy - 0.5f * tw - 1.5f, 1.1075f), cy, 0.50f, &is_wall);
    auto gap = [](const CountableBelief& b) { return 0.0f - (b.state().cy + 0.5f * b.state().h); };
    const auto run = [&](const CountableFrame& f) { CountableBelief b = build(); for (int k = 0; k < 2000; ++k) b.update(f); return b; };
    std::size_t n_box = 0;
    for (const float w : is_wall) n_box += (w == 0.0f);
    {   // printed: 7c.6's box-only baseline (endpoint staging without the returns behind the box)
        rc::ai::LidarRays rb = r; rb.endpoints.clear();
        for (std::size_t i = 0; i < r.endpoints.size(); ++i) if (is_wall[i] == 0.0f) rb.endpoints.push_back(r.endpoints[i]);
        CountableFrame fb = lidar_frame(rb, cy); set_latents(fb, 1.0f);
        fb.beam = beam_of(rb, truth_box(cy));
        const CountableBelief b = run(fb);
        std::printf("  wall_behind (persistent, start gap 0.020, %zu returns, %zu on the box): box-only (7c.6 baseline, printed) gap %.5f m  "
                    "sigma_h %.4f\n", r.endpoints.size(), n_box, gap(b), std::sqrt(b.covariance()(4, 4)));
    }
    float g0 = 0.0f, g01 = 0.0f;
    for (const float ww : {0.0f, 0.1f, 1.0f})
    {
        CountableFrame f = lidar_frame(r, cy); set_latents(f, 1.0f);
        for (std::size_t i = 0; i < f.beam.endpoints.size(); ++i)   // the wall's returns (y = 0), by geometry
            if (std::abs(f.beam.endpoints[i].y() - double(kWall.a.y())) < 1e-3)
                f.beam.m[i] = std::clamp(double(ww), rc::test::kEpsPx, 1.0 - rc::test::kEpsPx);
        const CountableBelief b = run(f);
        const float g = gap(b);
        if (ww == 0.0f) g0 = g;
        if (ww == 0.1f) g01 = g;
        std::printf("    wall label m %-6.3f gap %.5f m  diff vs baseline (m = eps_px) %+.5f m  sigma_h %.4f\n",
                    std::clamp(double(ww), rc::test::kEpsPx, 1.0 - rc::test::kEpsPx), g, g - g0, std::sqrt(b.covariance()(4, 4)));
    }
    CHECK(std::isfinite(g0) and std::isfinite(g01));
    CHECK(std::abs(g01 - g0) < 0.005f);
    {   // self-control: the SAME scene on the flag-off path at wall weight 1 moves the gap (Task 3 recorded +0.209 m), so the
        // test can see a pull when there is one
        CountableBeliefParams p = params(); p.persistent = false;
        CountableBelief b(CountableBeliefState{tx, cy, tH, tw, tw, 0.0f}, p);
        b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
        CountableFrame f = lidar_frame(r, cy);
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

// A full Σ_pose with cross terms (x, y, ψ): the robot pose error the published edge carries.
static Eigen::Matrix3d full_sigma_pose()
{
    Eigen::Matrix3d S;
    S << 9.0e-4, 2.0e-4, 1.0e-4,
         2.0e-4, 1.2e-3, -5.0e-5,
         1.0e-4, -5.0e-5, 4.0e-4;
    return S;
}

static void zero_pose_cov_runs()
{
    CountableBelief b = make_belief();
    const Vec6 s0 = b.state().vec();
    CountableFrame f = camera_frame(); set_latents(f, 1.0f, 0.0f); stage(f, two_face_rays());
    for (int k = 0; k < 10; ++k) b.update(f);
    const auto sd = b.latent_std();
    std::printf("  zero pose cov: state moved %.4f ; e std %.2e %.2e %.2e ; e mean %.2e ; P finite %d\n",
                (b.state().vec() - s0).norm(), sd(0), sd(1), sd(2), b.latent_mean().head<3>().norm(),
                int(b.joint_covariance().allFinite()));
    CHECK(b.state().vec().allFinite() and b.covariance().allFinite() and b.joint_covariance().allFinite());
    CHECK(b.latent_mean().allFinite());
    CHECK((b.state().vec() - s0).norm() > 1e-3f);
    CHECK(sd(0) == 0.0 and sd(1) == 0.0 and sd(2) == 0.0);
    CHECK(b.latent_mean().head<3>().isZero(0.0));
}

// spec 7c.7 b: canonicalize / mode rotation leave e and ℓ untouched (a pose error is a room-frame quantity); b_cam and b_L
// relabel w ↔ h with θ.
static void mode_rotation_leaves_e()
{
    CountableBelief b = make_belief();
    CountableFrame f = camera_frame(); set_latents(f, 1.0f); stage(f, two_face_rays());
    f.sigma_pose = full_sigma_pose();
    b.update(f);
    CountableBelief::VecKd v;
    v << 0.011, 0.012, 0.013, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036;
    b.set_latent_mean_for_test(v);
    const CountableBelief::MatJd P0 = b.joint_covariance();
    const ChainLever L0 = b.chain_lever();
    b.apply_mode_rotation_for_test(1);
    const auto& m = b.latent_mean();
    const auto& P = b.joint_covariance();
    constexpr int c = CountableBelief::kCam, l = CountableBelief::kL, N = 6;
    std::printf("  mode rotation: b_cam w,h %.3f %.3f (was %.3f %.3f) ; b_L w,h %.3f %.3f ; e %.3f %.3f %.3f ; lever (%.4f %.4f) was (%.4f %.4f)\n",
                m(c + 3), m(c + 4), v(c + 3), v(c + 4), m(l + 3), m(l + 4), m(0), m(1), m(2),
                b.chain_lever().l.x(), b.chain_lever().l.y(), L0.l.x(), L0.l.y());
    CHECK(m(c + 3) == v(c + 4) and m(c + 4) == v(c + 3));         // b_cam w ↔ h
    CHECK(m(l + 3) == v(l + 4) and m(l + 4) == v(l + 3));         // b_L w ↔ h
    CHECK(m(0) == v(0) and m(1) == v(1) and m(2) == v(2));        // e untouched
    CHECK(b.chain_lever().l == L0.l and b.chain_lever().set == L0.set);
    CHECK(m(c) == v(c) and m(c + 5) == v(c + 5) and m(l + 5) == v(l + 5));
    CHECK(P(3, 3) == P0(4, 4) and P(4, 4) == P0(3, 3));
    CHECK(P(N + c + 3, N + c + 3) == P0(N + c + 4, N + c + 4) and P(N + l + 3, N + l + 3) == P0(N + l + 4, N + l + 4));
    CHECK(P(3, N + c + 3) == P0(4, N + c + 4));                   // θ_w–b_cam,w cross term follows both relabels
    CHECK((P.block<3, 3>(N, N) == P0.block<3, 3>(N, N)));           // e's own block untouched
    CHECK(P(0, N) == P0(0, N) and P(5, N + 2) == P0(5, N + 2));   // θ–e cross terms of the unrelabelled DOFs untouched
    CHECK(b.covariance()(3, 3) == float(P(3, 3)));
    const double asym = (P - P.transpose()).cwiseAbs().maxCoeff();
    const double mine = Eigen::SelfAdjointEigenSolver<CountableBelief::MatJd>(P).eigenvalues().minCoeff();
    std::printf("  mode rotation: max|P-P^T| %.3e  min eig %.3e\n", asym, mine);
    CHECK(asym == 0.0);
    CHECK(mine >= -1e-12);
}

// spec 7c.7 b (R4-M7): sanitize re-seeds e at the last Σ_pose (its own 3×3, zero cross terms to everything else), b_cam and
// b_L at their last marginals, θθ at the class P; ℓ is kept.
static void sanitize_reseeds_e()
{
    CountableBelief b = make_belief();
    CountableFrame f = camera_frame(); set_latents(f, 1.0f); stage(f, two_face_rays());
    f.sigma_pose = full_sigma_pose();
    b.update(f);
    CHECK(b.wall_prior_added() and b.axis_prior_added());
    const ChainLever L0 = b.chain_lever();
    b.joint_covariance_for_test()(0, 0) = std::numeric_limits<double>::quiet_NaN();
    b.sanitize_covariance();
    const auto& P = b.joint_covariance();
    constexpr int N = 6, K = CountableBelief::K;
    const Vec6 p0 = b.prior_cov_diag();
    bool th_ok = true, lat_ok = true, cross_ok = true;
    for (int i = 0; i < N; ++i) th_ok = th_ok and P(i, i) == double(p0(i));
    const bool e_ok = (P.block<3, 3>(N, N) - f.sigma_pose).cwiseAbs().maxCoeff() == 0.0;
    const std::array<float, 12> m2 = {f.sig2_cam[0], f.sig2_cam[1], f.sig2_cam[2], f.sig2_cam[3], f.sig2_cam[4], f.sig2_cam[5],
                                      f.sig2_L[0], f.sig2_L[1], f.sig2_L[2], f.sig2_L[3], f.sig2_L[4], f.sig2_L[5]};
    for (int j = 0; j < 12; ++j) lat_ok = lat_ok and std::abs(P(N + 3 + j, N + 3 + j) / double(m2[j]) - 1.0) < 1e-6;
    for (int r = 0; r < N + K; ++r)
        for (int c = 0; c < N + K; ++c)
        {
            const bool in_e = r >= N and r < N + 3 and c >= N and c < N + 3;
            if (r != c and not in_e) cross_ok = cross_ok and P(r, c) == 0.0;
        }
    std::printf("  sanitize: thetatheta = P %d ; e = Sigma_pose %d ; latents = marginals %d ; cross 0 %d ; wall/axis added %d/%d ; "
                "b = 0 %d ; lever kept %d\n", int(th_ok), int(e_ok), int(lat_ok), int(cross_ok), int(b.wall_prior_added()),
                int(b.axis_prior_added()), int(b.latent_mean().isZero(0.0)), int(b.chain_lever().l == L0.l));
    CHECK(th_ok and e_ok and lat_ok and cross_ok);
    CHECK(not b.wall_prior_added() and not b.axis_prior_added());
    CHECK(not b.first_update_done());
    CHECK(b.latent_mean().isZero(0.0));
    CHECK(b.chain_lever().l == L0.l and b.chain_lever().set);
    CHECK(b.covariance().allFinite());
}

// spec 7c.7 b: ρ_chain = 0 (a relocalisation) re-draws e: after the latent predict e's mean is 0, its covariance is this
// frame's Σ_pose with no cross terms, the lever is re-taken at (belief centre − robot), and θ has not moved.
static void relocalisation_redraws_e()
{
    CountableBelief b = make_belief();
    CountableFrame f = camera_frame(); set_latents(f, 1.0f); stage(f, two_face_rays());
    f.sigma_pose = full_sigma_pose();
    for (int k = 0; k < 3; ++k) b.update(f);
    constexpr int N = 6, K = CountableBelief::K;
    const double cross_before = b.joint_covariance().block<N, 3>(0, N).cwiseAbs().maxCoeff();
    CountableFrame g = f;
    g.rho_chain = 0.0f;
    g.sigma_pose = 2.0 * full_sigma_pose();
    g.robot_xy = kRobot + Eigen::Vector2f(0.4f, -0.3f);
    g.chain_dr = Eigen::Vector2f(0.1f, 0.05f);
    const Vec6 th0 = b.state().vec();
    b.predict_latents_for_test(g);
    const auto& P = b.joint_covariance();
    const Eigen::Vector2f expect_l = Eigen::Vector2f(b.state().cx, b.state().cy) - g.robot_xy;
    double cross = 0.0;
    for (int r = 0; r < N + K; ++r)
        for (int c = N; c < N + 3; ++c)
            if (r < N or r >= N + 3) cross = std::max(cross, std::abs(P(r, c)));
    std::printf("  relocalisation: e mean %.2e ; |P_ee - Sigma_pose| %.2e ; cross before %.2e after %.2e ; lever (%.4f %.4f) "
                "expect (%.4f %.4f) ; |dtheta| %.2e\n", b.latent_mean().head<3>().norm(),
                (P.block<3, 3>(N, N) - g.sigma_pose).cwiseAbs().maxCoeff(), cross_before, cross,
                b.chain_lever().l.x(), b.chain_lever().l.y(), expect_l.x(), expect_l.y(), (b.state().vec() - th0).norm());
    CHECK(cross_before > 0.0);                                   // the test can see cross terms when there are some
    CHECK(b.latent_mean().head<3>().isZero(0.0));
    CHECK(((P.block<3, 3>(N, N) - g.sigma_pose).cwiseAbs().maxCoeff() <= 1e-15));
    CHECK(cross == 0.0);
    CHECK(b.chain_lever().l == expect_l);
    CHECK(b.state().vec() == th0);
}

// spec 7c.7 c: at ρ = 1 a parked range jitter reaches nothing — the range share of b_cam's σ changes only through S Σ S = 0.
// 1000 identical frames; in one run the mask range (hence the range share of b_cam's marginal) is jittered by ±1e-4
// relative from the second frame on (the first frame draws b_cam at its nominal range in both runs).
static void parked_range_jitter_equality()
{
    const CountableBeliefParams p = params();
    const auto cam_marginal = [&](float range)
    {
        const float pp = p.common_mode_pos_std * p.common_mode_pos_std, ss = p.common_mode_size_std * p.common_mode_size_std,
                    yy = p.common_mode_yaw_std * p.common_mode_yaw_std;
        const float lat = 0.02f * range, sz = 0.08f * range, yw = 0.03f * range;
        return std::array<float, 6>{pp + lat * lat, pp + lat * lat, ss + sz * sz, ss + sz * sz, ss + sz * sz, yy + yw * yw};
    };
    CountableFrame f = camera_frame(); set_latents(f, 1.0f); stage(f, two_face_rays());
    f.sigma_pose = full_sigma_pose();
    const float range = 1.65f;
    f.sig2_cam = cam_marginal(range);
    CountableBelief a = make_belief(), j = make_belief();
    std::uint32_t lcg = 12345u;
    for (int k = 0; k < 1000; ++k)
    {
        a.update(f);
        CountableFrame fj = f;
        if (k > 0)
        {
            lcg = lcg * 1664525u + 1013904223u;
            const float u = 2.0f * float(lcg >> 8) / float(1u << 24) - 1.0f;   // [-1, 1)
            fj.sig2_cam = cam_marginal(range * (1.0f + 1e-4f * u));
        }
        j.update(fj);
    }
    const double dth = rel(j.state().vec(), a.state().vec()), db = (j.latent_mean() - a.latent_mean()).cwiseAbs().maxCoeff(),
                 dP = rel(j.joint_covariance(), a.joint_covariance());
    std::printf("  parked range jitter (1000 frames): theta rel %.3e ; max|db| %.3e ; P rel %.3e\n", dth, db, dP);
    CHECK(dth <= 1e-6);
    CHECK(db <= 1e-6 * std::max(1e-3, a.latent_mean().cwiseAbs().maxCoeff()));
    CHECK(dP <= 1e-6);
}

// spec 7c.7 b, "chain once for both" in its 3×3 form: a parked camera + LiDAR update; the data information the two channels
// add on (cx, cy, yaw) equals (B Σ_pose Bᵀ + (Σ_cam⁻¹ + Σ_L⁻¹)⁻¹)⁻¹, B the three non-zero rows of B_chain, Σ_ch the channel's
// data covariance plus its own latent (inverse-free: Σ_ch⁻¹ = (𝟙 + I_ch Σ_b)⁻¹ I_ch). θ-only channel empty (anchors in,
// (2c) and the depth boost off), no fresh covariance, first update done (P̄ = the recursion's prediction, no θ–b cross terms).
static void chain_once_for_both_3x3()
{
    using M6d = Eigen::Matrix<double, 6, 6>;
    CountableBeliefParams p = params();
    p.depth_unobs_precision = 0.0f;
    p.wall_no_cross_precision = 0.0f;
    CountableBelief b = make_belief(0.05f, -0.04f, 0.03f, p);
    b.set_prior_added(true, true);
    b.set_first_update_done(true);
    // Camera points on all four sides (a two-face view leaves cx + w/2 and cy + h/2 unobserved: a zero-information direction
    // makes the channels' combined covariance, and so the comparison, undefined).
    CountableFrame f = camera_frame(); set_latents(f, 1.0f); stage(f, two_face_rays());
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 10; ++j)
        f.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 12.0f, ty + 0.5f * tw, tH * (j + 0.5f) / 10.0f);
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 10; ++j)
        f.points.emplace_back(tx + 0.5f * tw, ty - 0.5f * tw + tw * (i + 0.5f) / 8.0f, tH * (j + 0.5f) / 10.0f);
    f.R.assign(f.points.size(), params().sigma_base_m * params().sigma_base_m);
    f.sigma_pose = full_sigma_pose();
    f.robot_xy = Eigen::Vector2f(tx - 1.5f, ty - 1.5f);
    b.update(f);
    CHECK(b.dbg_Id_th().norm() == 0.0f);
    const M6d Pbar = b.dbg_Pbar_tt().cast<double>();
    // The joint posterior's θθ block WITHOUT the solver's remaining-step term δ_last δ_lastᵀ (spec 7c.7 d): the identity is
    // about the model's covariance, the outer product is the solver's own uncertainty added on top.
    const M6d Ppost = b.joint_covariance().topLeftCorner<6, 6>() - b.last_gn_step() * b.last_gn_step().transpose();
    const M6d I_data = Ppost.inverse() - Pbar.inverse();
    Mat6 Sbc = Mat6::Zero(), SbL = Mat6::Zero();
    for (int i = 0; i < 6; ++i) { Sbc(i, i) = f.sig2_cam[i]; SbL(i, i) = f.sig2_L[i]; }
    const M6d Ac = pu::marginalise_fresh<6>(b.dbg_Id_cam(), Vec6::Zero(), Sbc).I;
    const M6d AL = pu::marginalise_fresh<6>(b.dbg_Id_L(),   Vec6::Zero(), SbL).I;
    const Eigen::Matrix<double, 6, 3> B = chain_B(b.chain_lever().l);
    const std::array<int, 3> ix{0, 1, 5};
    const M6d Cch = (Ac + AL).inverse();                          // (Σ_cam⁻¹ + Σ_L⁻¹)⁻¹
    const double min_eig = Eigen::SelfAdjointEigenSolver<M6d>(Ac + AL).eigenvalues().minCoeff();
    CHECK(min_eig > 0.0);                                         // well posed: every DOF informed
    const M6d Cdata = I_data.inverse();
    Eigen::Matrix3d B3, C3, D3;
    for (int r = 0; r < 3; ++r)
    {
        B3.row(r) = B.row(ix[r]);
        for (int c = 0; c < 3; ++c) { C3(r, c) = Cch(ix[r], ix[c]); D3(r, c) = Cdata(ix[r], ix[c]); }
    }
    const Eigen::Matrix3d expect = (B3 * f.sigma_pose * B3.transpose() + C3).inverse();
    const Eigen::Matrix3d got    = D3.inverse();
    const double r3 = rel(got, expect);
    const double r6 = rel(I_data, (B * f.sigma_pose * B.transpose() + Cch).inverse().eval());
    std::printf("  chain once for both (3x3 on cx, cy, yaw): rel %.3e (6x6 form rel %.3e) ; min eig of the channels' info %.3e ; "
                "lever (%.3f %.3f) ; diag got %.4e %.4e %.4e expect %.4e %.4e %.4e\n", r3, r6, min_eig, b.chain_lever().l.x(),
                b.chain_lever().l.y(), got(0, 0), got(1, 1), got(2, 2), expect(0, 0), expect(1, 1), expect(2, 2));
    CHECK(r3 <= 1e-6);
}

// spec 7c.7 Pass, transport invariance: a ray-cast nightstand, the robot driving a 90° arc at 1.5 m around it, its pose
// dead-reckoned with a TRUE heading error δψ = 0.02 rad that arose at the arc's start p0 (the estimated path is the true one
// rotated by δψ about p0, so every room-frame observation is the true scene under that one rigid map). ρ_chain = 1 (no
// correction), ρ_cam = ρ_L = 1 (the claims are about e). True e_t = (estimated − true) robot pose.
static void transport_invariance_arc()
{
    constexpr int N = 6;
    const double dpsi = 0.02;
    const Eigen::Vector2f c(tx, ty);
    const Eigen::Rotation2Df Rd(static_cast<float>(dpsi));
    const int steps = 10;
    std::vector<Eigen::Vector2f> r_true(steps), r_est(steps);
    for (int k = 0; k < steps; ++k)
    {
        const float a = static_cast<float>((-135.0 + 90.0 * k / (steps - 1)) * M_PI / 180.0);
        r_true[k] = c + 1.5f * Eigen::Vector2f(std::cos(a), std::sin(a));
    }
    const Eigen::Vector2f p0 = r_true[0];
    for (int k = 0; k < steps; ++k) r_est[k] = p0 + Rd * (r_true[k] - p0);
    const Eigen::Vector2f c_obs = p0 + Rd * (c - p0);              // the box as the room-frame data shows it
    const rc::test::SceneBox seen{c_obs, static_cast<float>(dpsi), tw, tw, tH};
    const std::vector<rc::test::SceneBox> boxes{seen};
    const std::vector<rc::test::SceneWall> walls;
    Eigen::Matrix3d Sp = Eigen::Vector3d(0.03 * 0.03, 0.03 * 0.03, 0.02 * 0.02).asDiagonal();

    CountableBelief b(CountableBeliefState{c_obs.x(), c_obs.y(), tH, tw, tw, static_cast<float>(dpsi)}, params());
    Eigen::Vector3d sd1 = Eigen::Vector3d::Zero();
    double worst_bf = 0.0, worst_sd = 0.0, nees_lo = 1e30, nees_hi = 0.0, worst_th = 0.0;
    for (int k = 0; k < steps; ++k)
    {
        CountableFrame f;
        {
            const Eigen::Vector3f o(r_est[k].x(), r_est[k].y(), 0.945f);
            std::vector<float> el; for (float e = -40.0f; e <= 5.0f; e += 0.4f) el.push_back(e);
            f.points = rc::test::select_near(rc::test::raycast_scene(o, boxes, walls, true, 0.4f, el, 6.0f), seen, 0.02f);
            f.R.assign(f.points.size(), params().sigma_base_m * params().sigma_base_m);
            f.z_top_obs = tH;
        }
        {
            const Eigen::Vector3f o(r_est[k].x(), r_est[k].y(), 1.1075f);
            f.lidar.origin = o;
            f.lidar.endpoints = rc::test::select_near(rc::test::raycast_scene(o, boxes, walls, true, rc::test::kHeliosAzStepDeg,
                                                                              kElev, 12.0f), seen, 0.02f);
            f.lidar.precision = 2500.0f; f.lidar.robust_c_m = 0.05f;
            {
                const std::vector<rc::test::SceneBox> obj{seen};
                f.beam = rc::test::helios_beam(o, obj, walls, seen, 0.10f, rc::test::hand_labels(obj), 1.0, params().model_sigma_m);
            }
        }
        set_latents(f, 1.0f);
        f.sigma_pose = Sp;
        f.robot_xy = r_est[k];
        f.chain_dr = k == 0 ? Eigen::Vector2f::Zero() : Eigen::Vector2f(r_est[k] - r_est[k - 1]);
        const ChainLever Lp = b.chain_lever();
        b.update(f);
        const ChainLever L = b.chain_lever();
        if (k > 0)
        {
            const Eigen::Matrix<double, 6, 3> BF = chain_B(L.l) * chain_transport(f.chain_dr), Bp = chain_B(Lp.l);
            worst_bf = std::max(worst_bf, (BF - Bp).cwiseAbs().maxCoeff());
        }
        const auto& P = b.joint_covariance();
        const Eigen::Matrix3d Se = P.block<3, 3>(N, N);
        const Eigen::Vector3d sd = Se.diagonal().cwiseSqrt();
        if (k == 0) sd1 = sd;
        else worst_sd = std::max(worst_sd, (sd1 - sd).cwiseQuotient(sd1).maxCoeff());
        const Eigen::Vector2f exy = r_est[k] - r_true[k];
        const Eigen::Vector3d e_true(exy.x(), exy.y(), dpsi);
        const Eigen::Vector3d d = b.latent_mean().head<3>() - e_true;
        const double nees = d.dot(Se.ldlt().solve(d));
        nees_lo = std::min(nees_lo, nees); nees_hi = std::max(nees_hi, nees);
        const Vec6 th_true = (Vec6() << tx, ty, tH, tw, tw, 0.0f).finished();
        const Vec6 err = b.state().vec() - th_true;
        double wz = 0.0;
        for (int i = 0; i < 6; ++i) wz = std::max(wz, double(std::abs(err(i)) / std::sqrt(b.covariance()(i, i))));
        worst_th = std::max(worst_th, wz);
        std::printf("    arc %d: e (%+.4f %+.4f %+.5f) true (%+.4f %+.4f %+.5f) sd (%.4f %.4f %.5f) NEES %.3f ; theta err cx %+.4f "
                    "cy %+.4f yaw %+.4f worst |err|/sigma %.2f\n", k, b.latent_mean()(0), b.latent_mean()(1), b.latent_mean()(2),
                    e_true(0), e_true(1), e_true(2), sd(0), sd(1), sd(2), nees, err(0), err(1), err(5), wz);
    }
    std::printf("  transport invariance: max|B_t F_t - B_t-1| %.3e ; worst sd drop below update 1 %.3e ; NEES [%.3f, %.3f] ; "
                "worst theta |err|/sigma %.2f\n", worst_bf, worst_sd, nees_lo, nees_hi, worst_th);
    // The lever is float state (Task 12's ChainLever, ruling T12-1): ℓ − Δr rounds at ε_float·|ℓ|, so the identity holds to
    // float precision of the lever, not to 1e-9.
    CHECK(worst_bf <= 1e-6);
    CHECK(worst_sd <= 1e-3);
    CHECK(nees_lo >= 0.1 and nees_hi <= 10.0);
    CHECK(worst_th <= 2.0);
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

// ─── Task 6c (spec 7c.1 amendment 11-17): the class prior at the class mean, counted once; the seed is only the GN start ───
// One-face camera frame of the TRUE flush nightstand (back face on the wall y = 0): 120 points on the front face only.
static CountableFrame one_face_frame(bool top = true)
{
    CountableFrame f;
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 10; ++j)
        f.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 12.0f, ty - 0.5f * tw, tH * (j + 0.5f) / 10.0f);
    f.R.assign(f.points.size(), params().sigma_base_m * params().sigma_base_m);
    f.z_top_obs = top ? tH : -1.0f;
    stage_wall(f);
    set_latents(f, 1.0f);
    return f;
}

static void bad_seed_recovers()
{
    // live8's nightstand_1: born with depth h = 2.14 (a mask over the bed side), back face on the wall (birth_on_priors).
    const float h0 = 2.14f;
    CountableBelief b(CountableBeliefState{tx, -0.5f * h0, tH, tw, h0, 0.0f}, params());
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    const CountableFrame f = one_face_frame();
    b.update(f);
    const float h1 = b.state().h, cy1 = b.state().cy;
    std::printf("  bad_seed_recovers: seed h %.3f ; after 1 update h %.4f cy %.4f (truth h %.3f cy %.3f) first_done %d\n",
                h0, h1, cy1, tw, ty, int(b.first_update_done()));
    for (int k = 1; k < 10; ++k) b.update(f);
    const auto s = b.state();
    std::printf("  bad_seed_recovers: after 10 updates h %.4f cy %.4f w %.4f H %.4f yaw %.4f ; sigma_h %.4f ; Q_h exp(omega) %.3e\n",
                s.h, s.cy, s.w, s.H, s.yaw, std::sqrt(b.covariance()(4, 4)), std::exp(b.log_volatility()(4)));
    CHECK(std::abs(h1 - tw) < 0.10f);
    CHECK(std::abs(s.h - tw) < 0.05f);
    {   // control: the same seed with the class prior centred on the seed (the recursion from the start) does not recover
        CountableBelief c(CountableBeliefState{tx, -0.5f * h0, tH, tw, h0, 0.0f}, params());
        c.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
        c.set_first_update_done(true);
        for (int k = 0; k < 10; ++k) c.update(f);
        std::printf("  bad_seed_recovers: control (prior at the seed) after 10 updates h %.4f\n", c.state().h);
        CHECK(std::abs(c.state().h - tw) > 0.05f);   // the control does NOT recover: the test can see the failure it guards
    }
}

// T6c-2: the first update does not teach ω (its step is seed → posterior, not posterior → posterior); the second does.
static void first_update_no_volatility()
{
    const float h0 = 2.14f;
    CountableBelief b(CountableBeliefState{tx, -0.5f * h0, tH, tw, h0, 0.0f}, params());
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    const Vec6 w0 = b.log_volatility();
    const CountableFrame f = one_face_frame();
    b.update(f);
    const Vec6 w1 = b.log_volatility();
    b.update(f);
    const Vec6 w2 = b.log_volatility();
    std::printf("  first_update_no_volatility: |d omega| after the first update %.3e ; after the second %.3e\n",
                (w1 - w0).cwiseAbs().maxCoeff(), (w2 - w1).cwiseAbs().maxCoeff());
    CHECK((w1 - w0).cwiseAbs().maxCoeff() == 0.0f);
    CHECK((w2 - w1).cwiseAbs().maxCoeff() > 0.0f);
}

static void good_seed_stays()
{
    // Without (2d): its target is the observed top + top_no_float_margin_m (2 cm), a likelihood term that lifts H by ~2 cm on
    // this noise-free fixture whatever the prior (printed below, prior at the class mean vs at the seed: the same move).
    const auto size_move = [](bool top, bool first_done)
    {
        CountableBelief b = make_belief(0.0f, 0.0f, 0.0f);
        b.set_first_update_done(first_done);
        const Vec6 s0 = b.state().vec();
        b.update(one_face_frame(top));
        const Vec6 d = b.state().vec() - s0;
        return Eigen::Vector3f(d(2), d(3), d(4));
    };
    const Eigen::Vector3f d = size_move(false, false), dt = size_move(true, false), dc = size_move(true, true);
    const float dsize = d.cwiseAbs().maxCoeff();
    std::printf("  good_seed_stays: size move after the first update dH %+.5f dw %+.5f dh %+.5f (max %.5f) ; with (2d) top: "
                "class-mean prior dH %+.5f dw %+.5f dh %+.5f, prior at the seed dH %+.5f dw %+.5f dh %+.5f\n",
                d(0), d(1), d(2), dsize, dt(0), dt(1), dt(2), dc(0), dc(1), dc(2));
    CHECK(dsize < 0.01f);
    CHECK((dt - dc).cwiseAbs().maxCoeff() < 1e-4f);   // the class-mean first update adds nothing at a seed on the class mean
}

static void prior_counted_once()
{
    // depth_unobs_precision off: it is the one class-mean term still counted per camera frame (§7c.1), not the prior.
    CountableBeliefParams p = params();
    p.depth_unobs_precision = 0.0f;
    CountableBelief a = make_belief(0.05f, -0.04f, 0.03f, p);
    const CountableFrame f = one_face_frame();
    a.update(f);
    CountableBelief b = a;
    CountableBeliefParams q = p;
    q.prior_height_m = p.prior_height_m + 0.30f; q.prior_footprint_m = p.prior_footprint_m + 0.20f; q.prior_depth_m = 0.90f;
    b.set_params(q);
    a.update(f);
    b.update(f);
    const float diff = (a.state().vec() - b.state().vec()).cwiseAbs().maxCoeff();
    const float dcov = (a.covariance() - b.covariance()).cwiseAbs().maxCoeff();
    std::printf("  prior_counted_once: second update, class mean changed by (+0.30, +0.20, ->0.90): max|dtheta| %.3e  max|dSigma| %.3e\n",
                diff, dcov);
    CHECK(diff == 0.0f);
    CHECK(dcov == 0.0f);
}

static void first_update_flag_life()
{
    CountableBelief b = make_belief();
    CHECK(not b.first_update_done());
    CountableFrame none; stage_wall(none); set_latents(none, 1.0f);   // no points, not LiDAR-only: predict-only
    b.update(none);
    const bool after_none = b.first_update_done();
    CHECK(not after_none);
    b.update(one_face_frame());
    const bool after_cam = b.first_update_done();
    CHECK(after_cam);
    const CountableBelief snap = b;                                   // the fitter's step-bound snapshot is a value copy
    CHECK(snap.first_update_done());
    b.joint_covariance_for_test()(0, 0) = std::numeric_limits<double>::quiet_NaN();
    b.sanitize_covariance();
    std::printf("  first_update_flag_life: after predict-only %d ; after camera %d ; snapshot %d ; after sanitize re-seed %d\n",
                int(after_none), int(after_cam), int(snap.first_update_done()), int(b.first_update_done()));
    CHECK(not b.first_update_done());
    CountableBelief l = make_belief();                                // a LiDAR-only first update counts as the first
    CountableFrame fl = lidar_frame(two_face_rays()); set_latents(fl, 1.0f);
    l.update(fl);
    CHECK(l.first_update_done());
}

// Ruling T13-5: on the persistent path the published (cx, cy) covariance and the tracker's gate covariance are the joint's θθ
// marginal, which already integrates e; the fitter's per-frame chain cov is NOT added again. Parked camera frame, live9 Σ_pose.
// covariance() is the float copy of P's θθ block, so the published float equals float(P_ii) exactly. Flag off keeps the
// chain cov (self-control).
static void published_cov_no_chain_double_count()
{
    CountableBelief b = make_belief();
    CountableFrame f = camera_frame(); set_latents(f, 1.0f);
    f.sigma_pose = Eigen::Vector3d(0.034 * 0.034, 0.0435 * 0.0435, 0.024 * 0.024).asDiagonal();
    for (int k = 0; k < 3; ++k) b.update(f);
    const float chain_xx = 2.1e-3f, chain_yy = 2.9e-3f;            // what compute_chain_cov would stage
    const Eigen::Vector2f extra = b.published_chain_var(chain_xx, chain_yy);
    const auto& P = b.joint_covariance();
    double worst = 0.0;
    for (int i = 0; i < 2; ++i)
    {
        const double pub = double(b.covariance()(i, i) + extra(i)), marg = double(static_cast<float>(P(i, i)));
        worst = std::max(worst, std::abs(pub - marg) / marg);
    }
    CountableBeliefParams q = params(); q.persistent = false;
    CountableBelief off(CountableBeliefState{tx, ty, tH, tw, tw, 0.0f}, q);
    const Eigen::Vector2f extra_off = off.published_chain_var(chain_xx, chain_yy);
    std::printf("  published cov: (cx, cy) published %.6e %.6e ; joint marginal %.6e %.6e ; rel %.3e ; e share in it (B Sigma_e B^T)_xx "
                "%.3e ; flag-off extra %.1e %.1e\n", b.covariance()(0, 0) + extra(0), b.covariance()(1, 1) + extra(1), P(0, 0), P(1, 1),
                worst, (chain_B(b.chain_lever().l) * P.block<3, 3>(6, 6) * chain_B(b.chain_lever().l).transpose())(0, 0),
                extra_off(0), extra_off(1));
    CHECK(worst <= 1e-9);
    CHECK(extra_off(0) == chain_xx and extra_off(1) == chain_yy);
}

// ─── Task 14 (spec 7c.7 d): the update converges before P is taken ────────────────────────────────────────────────────
// Fixture: the golden nightstand (flush on the wall y = 0), camera on the front and left faces + Helios rays from the front,
// the wall and the room axis staged, parked (ρ = 1), the full Σ_pose (x, y, ψ). Class mean 6 cm off: the footprint class
// mean is tw + 6 cm (live8's shifted class, 0.57 against a true 0.51). Birth on priors: size at the class mean, back face on
// the wall, centre on the cloud (cx = tx), yaw 0.
static CountableBeliefParams shifted_params()
{
    CountableBeliefParams p = params();
    p.prior_footprint_m = tw + 0.06f;
    return p;
}
static CountableFrame converge_frame()
{
    CountableFrame f = camera_frame();
    set_latents(f, 1.0f);
    f.sigma_pose = full_sigma_pose();
    stage(f, front_rays());
    return f;
}
static CountableBelief born_on_priors(const CountableBeliefParams& p)
{
    const float h = p.depth_mean();
    CountableBelief b(CountableBeliefState{tx, -0.5f * h, p.prior_height_m, p.prior_footprint_m, h, 0.0f}, p);
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    return b;
}
static double max_theta_step(const Vec6& a, const Vec6& b, double& dyaw)
{
    double dm = 0.0;
    for (int i = 0; i < 5; ++i) dm = std::max(dm, double(std::abs(a(i) - b(i))));
    dyaw = std::abs(std::remainder(double(a(5)) - double(b(5)), 2.0 * M_PI));
    return dm;
}

// The central difference of persistent_data_energy in each θ DOF equals −(bd_cam + bd_L + bd_th), within 1 % (or 1e-6
// absolute). Three linearisation points: off the truth inside the room, the back face 2 cm INTO the wall (the (2c) hinge
// active), and short (H below plaus_height_min: the (2b) hinge active); each channel at its own state (θ, θ + b̄_cam, θ + b̄_L).
// spec 7c.8: the LiDAR channel is the smooth beam model, whose gradient is the EXACT gradient of its (discretised) energy
// (beam_ray.h; checked per ray in beam_ray_test), so the 7c.6 workarounds are gone: no active-ray-set ladder (the energy no
// longer jumps when a ray stops crossing), no seam distance (no finite-difference SDF Jacobian on this path), and σ_s =
// model_sigma_m is CHECKED (the first-crossing factor held n̂·u fixed in its variance and was printed only). Stated kink (iv)
// of the discretised energy (tied breakpoints, a one-sided derivative) can sit under a stencil: a DOF that misses the central
// difference is re-checked against the one-sided differences on each side and passes when it equals either.
static void energy_gradient_consistent()
{
    const CountableFrame f = converge_frame();
    // The fourth and fifth states are SQUARE (w == h) at yaw 0 exactly — the truth and a birth on the class mean (Task 19 review
    // D1: at hx == hy two pieces below the envelope cross at the envelope's vertex, and the vertex pair chosen there gave
    // g_w, g_h off by ±765 nats/m on the creep fixture).
    const std::array<Vec6, 5> thetas{(Vec6() << tx + 0.03f, ty - 0.02f, tH + 0.02f, tw + 0.03f, tw - 0.02f, 0.02f).finished(),
                                     (Vec6() << tx - 0.01f, ty + 0.02f, tH - 0.01f, tw, tw + 0.01f, -0.015f).finished(),
                                     (Vec6() << tx + 0.01f, ty - 0.01f, 0.35f, tw, tw, 0.01f).finished(),
                                     (Vec6() << tx, ty, tH, tw, tw, 0.0f).finished(),
                                     (Vec6() << tx + 0.004f, ty - 0.003f, tH, tw + 0.01f, tw + 0.01f, 0.0f).finished()};
    const Vec6 ocam_g = (Vec6() << 0.004f, -0.003f, 0.002f, 0.001f, -0.002f, 0.003f).finished();
    const Vec6 oL_g   = (Vec6() << -0.002f, 0.003f, -0.001f, 0.002f, 0.001f, -0.002f).finished();
    const float h = 1e-4f;
    double worst = 0.0;
    int one_sided = 0;
    // the square states keep each channel's state square at yaw 0 (equal w, h offsets, no yaw offset)
    const Vec6 ocam_sq = (Vec6() << 0.004f, -0.003f, 0.002f, 0.0015f, 0.0015f, 0.0f).finished();
    const Vec6 oL_sq   = (Vec6() << -0.002f, 0.003f, -0.001f, 0.0015f, 0.0015f, 0.0f).finished();
    for (std::size_t k = 0; k < thetas.size(); ++k)
    {
        const Vec6& th = thetas[k];
        const bool sq = k >= 3;
        const Vec6& ocam = sq ? ocam_sq : ocam_g;
        const Vec6& oL   = sq ? oL_sq : oL_g;
        CountableBelief b = make_belief(0.0f, 0.0f, 0.0f);
        b.set_frame_wall_for_test(f.wall);
        const auto S = [](const Vec6& v) { return CountableBeliefState::from_vec(v); };
        const auto E = [&](const Vec6& t) { return b.persistent_data_energy(S(t + ocam), S(t + oL), S(t), f); };
        Mat6 Ic, Il, It; Vec6 bc, bl, bt;
        b.accumulate_channels(S(th), S(th + ocam), S(th + oL), f, Ic, bc, Il, bl, It, bt);
        const Vec6 g = -(bc + bl + bt);
        std::printf("  energy_gradient theta (%.3f %.3f %.3f %.3f %.3f %+.3f): E %.6e ; %zu beam rays\n", th(0), th(1), th(2), th(3),
                    th(4), th(5), E(th), f.beam.endpoints.size());
        for (int i = 0; i < 6; ++i)
        {
            Vec6 tp = th, tm = th;
            tp(i) += h; tm(i) -= h;
            const double hp = double(tp(i)) - double(th(i)), hm = double(th(i)) - double(tm(i));   // the float steps taken
            const double E0 = E(th), Ep = E(tp), Em = E(tm);
            const double fd = (Ep - Em) / (hp + hm), fr = (Ep - E0) / hp, fl = (E0 - Em) / hm;
            const double tol = std::max(0.01 * std::abs(double(g(i))), 1e-6);
            double err = std::abs(fd - double(g(i)));
            bool os = false;
            if (err > tol)
            {   // stated kink (iv): one-sided on either side
                const double e1 = std::min(std::abs(fr - double(g(i))), std::abs(fl - double(g(i))));
                if (e1 <= std::max(0.01 * std::abs(double(g(i))), 1e-6) + 0.01 * std::abs(fr - fl)) { err = 0.0; os = true; ++one_sided; }
            }
            worst = std::max(worst, err / tol);
            std::printf("    DOF %d: dE/dtheta (central) %+.6e  (right %+.6e left %+.6e)  -(bd_cam+bd_L+bd_th) %+.6e  |diff| %.3e (tol %.3e)%s\n",
                        i, fd, fr, fl, double(g(i)), std::abs(fd - double(g(i))), tol, os ? "  one-sided (kink iv)" : "");
            CHECK(err <= tol);
        }
    }
    std::printf("  energy_gradient_consistent: worst |diff|/tol %.3f (beam rays at sigma_s = %.3f, checked) ; one-sided %d\n", worst,
                params().model_sigma_m, one_sided);
}

// The first update from the birth on priors: J is non-increasing over accepted steps, and θ and √diag P equal the same solver
// run for 200 steps (tolerance 0: it never stops early). Two fixtures, camera only and camera + rays; both must report
// gn_conv = 1 within gn_iters_max (spec 7c.8: the smooth beam energy has no ray edge, so T14-7's exemption is withdrawn).
static void first_update_converges_on(const char* nm, const CountableFrame& f, bool rays)
{
    CountableBelief b = born_on_priors(shifted_params());
    const Vec6 th0 = b.state().vec();
    b.update(f);
    const auto& J = b.dbg_gn_energy();
    bool mono = not J.empty();
    for (std::size_t k = 1; k < J.size(); ++k) mono = mono and J[k] <= J[k - 1];
    std::printf("  first_update_converges (%s): gn_it %d conv %d lambda %.3e ; accepted %zu ; J:", nm, b.last_gn_iters(),
                int(b.last_gn_converged()), b.last_lambda(), b.dbg_iterates().size());
    for (const double j : J) std::printf(" %.6f", j);
    std::printf("\n");
    for (const auto& v : b.dbg_iterates())
        std::printf("    iterate cx %.5f cy %.5f H %.5f w %.5f h %.5f yaw %+.5f\n", v(0), v(1), v(2), v(3), v(4), v(5));
    CountableBeliefParams q = shifted_params();
    q.gn_iters_max = 200; q.gn_tol_m = 0.0; q.gn_tol_rad = 0.0;
    CountableBelief r = born_on_priors(q);
    r.update(f);
    double dyaw = 0.0;
    const double dpos = max_theta_step(b.state().vec(), r.state().vec(), dyaw);
    double worst_sd = 0.0;
    std::printf("    sqrt diag P (12 / 200):");
    for (int i = 0; i < 6; ++i)
    {
        const double a = std::sqrt(double(b.covariance()(i, i))), c = std::sqrt(double(r.covariance()(i, i)));
        worst_sd = std::max(worst_sd, std::abs(a - c) / c);
        std::printf(" %.5f/%.5f", a, c);
    }
    std::printf("\n  first_update_converges (%s): theta (12-cap) vs 200 steps: max|d pos| %.3e m, |d yaw| %.3e rad ; worst sd rel %.3e ; "
                "200-step run: gn_it %d conv %d ; theta0 -> theta moved %.4f\n", nm, dpos, dyaw, worst_sd, r.last_gn_iters(),
                int(r.last_gn_converged()), (b.state().vec() - th0).norm());
    // spec 7c.8 Pass: the deferred item comes back as a CHECK — with the smooth beam energy the camera + ray fixture's first
    // update must return gn_conv = 1 too (T14-7's piecewise-edge reading no longer applies).
    (void)rays;
    CHECK(b.last_gn_converged());
    CHECK(dpos <= 1e-3 and dyaw <= 1e-3);
    CHECK(worst_sd <= 0.05);
    CHECK(b.last_gn_iters() <= b.params().gn_iters_max);   // 25 (spec 7c.8 rev 3.3.7)
    CHECK(mono);
}
static void first_update_converges()
{
    CountableFrame cam = converge_frame();
    cam.lidar = {}; cam.beam = {};
    first_update_converges_on("camera only", cam, false);
    first_update_converges_on("camera + rays", converge_frame(), true);
}

// Rulings T14-2, T14-5: repeated parked frames carry independent per-frame noise, so θ legitimately moves away from the
// class-prior pull on the frames after the first (the posterior with 2× the data is not the posterior with 1×).
// Camera only: over 10 identical parked frames the θ step sizes are non-increasing (1e-9 slack) and the step at frame 10 is
// below the tolerance. spec 7c.8 Pass: with the rays the same rule now holds (the camera-only rule of T14-5 for rays too).
static void parked_updates_contract()
{
    const auto run = [](const char* nm, const CountableFrame& f, std::vector<double>& pos, std::vector<double>& yaw, Vec6& drift)
    {
        CountableBelief b = born_on_priors(shifted_params());
        b.update(f);
        const Vec6 t1 = b.state().vec();
        std::printf("  parked_updates_contract (%s): frame steps (max |d pos| mm / |d yaw| mrad, gn_it, conv):", nm);
        for (int k = 2; k <= 10; ++k)
        {
            const Vec6 tp = b.state().vec();
            b.update(f);
            double dy = 0.0;
            pos.push_back(max_theta_step(b.state().vec(), tp, dy));
            yaw.push_back(dy);
            std::printf(" [%d] %.4f/%.4f %d %d", k, 1e3 * pos.back(), 1e3 * dy, b.last_gn_iters(), int(b.last_gn_converged()));
        }
        drift = (b.state().vec() - t1).cwiseAbs();
        std::printf("\n    drift after frame 1 (mm): %.3f %.3f %.3f %.3f %.3f ; yaw %.3f mrad\n", 1e3 * drift(0), 1e3 * drift(1),
                    1e3 * drift(2), 1e3 * drift(3), 1e3 * drift(4), 1e3 * drift(5));
    };
    {
        CountableFrame f = converge_frame();
        f.lidar = {}; f.beam = {};
        std::vector<double> pos, yaw; Vec6 drift;
        run("camera only", f, pos, yaw, drift);
        bool mono = true;
        for (std::size_t k = 1; k < pos.size(); ++k) mono = mono and pos[k] <= pos[k - 1] + 1e-9 and yaw[k] <= yaw[k - 1] + 1e-9;
        CHECK(mono);
        CHECK(pos.back() < 1e-3 and yaw.back() < 1e-3);
    }
    {
        std::vector<double> pos, yaw; Vec6 drift;
        run("camera + rays", converge_frame(), pos, yaw, drift);
        bool mono = true;
        for (std::size_t k = 1; k < pos.size(); ++k) mono = mono and pos[k] <= pos[k - 1] + 1e-9 and yaw[k] <= yaw[k - 1] + 1e-9;
        CHECK(mono);
        CHECK(pos.back() < 1e-3 and yaw.back() < 1e-3);
    }
}

// Every step rejected (test hook): θ returns at θ₀, P is still taken (finite, PSD), gn_conv = 0, no NaN.
static void all_steps_rejected_is_finite()
{
    const CountableFrame f = converge_frame();
    CountableBeliefParams p = shifted_params();
    p.gn_reject_all_for_test = true;
    CountableBelief b = born_on_priors(p);
    const Vec6 th0 = b.state().vec();
    b.update(f);
    const auto& P = b.joint_covariance();
    const Eigen::SelfAdjointEigenSolver<CountableBelief::MatJd> es(P);
    const double min_eig = es.eigenvalues().minCoeff(), max_eig = es.eigenvalues().maxCoeff();
    const double asym = (P - P.transpose()).cwiseAbs().maxCoeff();
    std::printf("  all_steps_rejected_is_finite: |theta - theta0| %.3e ; gn_it %d conv %d lambda %.3e ; P finite %d ; eig [%.3e, %.3e] ; "
                "asym %.1e ; accepted %zu\n", (b.state().vec() - th0).cwiseAbs().maxCoeff(), b.last_gn_iters(),
                int(b.last_gn_converged()), b.last_lambda(), int(P.allFinite()), min_eig, max_eig, asym, b.dbg_iterates().size());
    CHECK(b.state().vec() == th0);
    CHECK(P.allFinite() and b.latent_mean().allFinite());
    CHECK(min_eig >= -1e-12 * max_eig);
    CHECK(b.dbg_iterates().empty());                       // nothing accepted
    // Ruling T14-4: a growing λ cannot fake convergence — the undamped proposal from θ₀ is the seed's cm-scale correction.
    CHECK(not b.last_gn_converged());
    CHECK(b.last_gn_iters() == p.gn_iters_max);
}

// The cycle cost on the nightstand fixture (no assertion; the number goes into the report).
static void cycle_cost()
{
    const CountableFrame f = converge_frame();
    const CountableBelief born = born_on_priors(shifted_params());
    {
        CountableBelief b = born;
        int its = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < 100; ++k) { b.update(f); its += b.last_gn_iters(); }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 100.0;
        std::printf("  cycle_cost: 100 sequential (parked) updates: mean %.3f ms (%.1f Hz) ; mean gn_it %.2f\n", ms, 1000.0 / ms,
                    its / 100.0);
    }
    {
        int its = 0;
        double ms = 0.0;
        for (int k = 0; k < 100; ++k)
        {
            CountableBelief b = born;
            const auto t0 = std::chrono::steady_clock::now();
            b.update(f);
            ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            its += b.last_gn_iters();
        }
        ms /= 100.0;
        std::printf("  cycle_cost: 100 first updates (birth on priors): mean %.3f ms (%.1f Hz) ; mean gn_it %.2f\n", ms, 1000.0 / ms,
                    its / 100.0);
    }
}

// Final review I1: an object whose yaw sits at ±π. The golden box is square (w = h) and, with no wall staged, its energy is
// symmetric under yaw → yaw + π, so the SAME data labelled at yaw ε and at π + ε is one problem up to the label. Data box at
// yaw +10 mrad (camera points on two faces, rotated about the centre); room axis 0, so the class-mean yaw is the multiple
// of 90° nearest the seed. Run A seeds at −5 mrad (no wrap anywhere); run B seeds at π − 5 mrad, so x̄'s yaw is π and the
// iterates cross ±π on their way to π + 10 mrad. 4 parked frames with the full Σ_pose. θ (yaw compared wrapped, minus π),
// b and P must be the same in both runs: a 2π artefact in the prior pull would land in b_cam's yaw and stay there.
// (db, b's relative difference, is printed only: b is mm-scale, so its float floor reads ~1e-5 relative.)
static void wrap_invariance_near_pi()
{
    const float yd = 0.010f;
    CountableFrame f;
    {
        const CountableFrame g = camera_frame();
        const float c = std::cos(yd), s = std::sin(yd);
        for (const auto& p : g.points)
        {
            const float dx = p.x() - tx, dy = p.y() - ty;
            f.points.emplace_back(tx + c * dx - s * dy, ty + s * dx + c * dy, p.z());
        }
        f.R = g.R;
        f.z_top_obs = g.z_top_obs;
    }
    f.room_axis_ok = true; f.room_axis = 0.0f;                     // no wall: the wall-flush factor is not π-symmetric
    set_latents(f, 1.0f);
    f.sigma_pose = full_sigma_pose();
    const auto run = [&](float seed_yaw)
    {
        CountableBelief b(CountableBeliefState{tx, ty, tH, tw, tw, seed_yaw}, params());
        b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
        for (int k = 0; k < 4; ++k) b.update(f);
        return b;
    };
    const CountableBelief a = run(-0.005f), c = run(static_cast<float>(M_PI) - 0.005f);
    const Vec6 ta = a.state().vec(), tc = c.state().vec();
    const double dpos = (ta.head<5>() - tc.head<5>()).cwiseAbs().maxCoeff();
    const double dyaw = std::abs(std::remainder(double(tc(5)) - double(ta(5)) - M_PI, 2.0 * M_PI));
    const double db = rel(c.latent_mean(), a.latent_mean());
    const double db_abs = (c.latent_mean() - a.latent_mean()).cwiseAbs().maxCoeff();
    const double dP = rel(c.joint_covariance(), a.joint_covariance());
    std::printf("  wrap_invariance_near_pi: A yaw %+.6f, B yaw %+.6f ; max|d pos| %.3e m ; |d yaw| (mod pi) %.3e rad ; "
                "b rel %.3e max|d b| %.3e (b_cam yaw A %+.3e B %+.3e) ; P rel %.3e\n", ta(5), tc(5), dpos, dyaw, db, db_abs,
                a.latent_mean()(3 + 5), c.latent_mean()(3 + 5), dP);
    // Tolerances at the float floor: θ is stored in float, whose resolution at yaw ≈ π is 2.4e-7 rad (at 7 mrad, 5e-10), so the
    // B run's sdf and Jacobians differ from A's by float rounding; b is mm-scale, so it is compared absolutely. Before the fix
    // (unwrapped prior pull): |d pos| 5.3e-3 m, |d yaw| 1.5e-2 rad, b_cam yaw −8.1e-3 vs +2.2e-3, P rel 4.2e-3.
    CHECK(dpos <= 1e-6 and dyaw <= 1e-6);
    CHECK(db_abs <= 1e-6);
    CHECK(dP <= 1e-5);
}

int main()
{
    flag_off_golden();
    nonlinear_restart_matches_eliminated();
    rho0_regression_zero_pose_cov(true);   // camera-only frames only (spec 7c.8: frames with rays differ by design)
    prior_added_once();
    no_cross_pushes_back_persistent();
    aged_belief_returns_to_class_P();
    wall_behind_does_not_pull();
    physical_lidar_only_sigma_bounded();
    parked_camera_is_one_look();
    zero_pose_cov_runs();
    mode_rotation_leaves_e();
    sanitize_reseeds_e();
    relocalisation_redraws_e();
    parked_range_jitter_equality();
    chain_once_for_both_3x3();
    transport_invariance_arc();
    published_cov_no_chain_double_count();
    bad_seed_recovers();
    first_update_no_volatility();
    good_seed_stays();
    prior_counted_once();
    first_update_flag_life();
    energy_gradient_consistent();
    first_update_converges();
    parked_updates_contract();
    all_steps_rejected_is_finite();
    wrap_invariance_near_pi();
    cycle_cost();
    std::printf("persistent_belief_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
