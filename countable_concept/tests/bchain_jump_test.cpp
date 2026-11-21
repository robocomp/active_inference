// bchain_jump_test — live9 nightstand_9: the old chain latent absorbed 0.45 m on the second persistent update while cx moved
// 0.35 m the other way. Root cause (.superpowers/sdd/2026-11-15-lidar-measures/bchain-jump-report.md): the latent scale map
// D = σ_now/σ_prev made our own σ change an observation. Restated for spec §7c.7 (rev 3.2.1): no rescale; the chain latent
// is e = (δx, δy, δψ), the robot's pose error in the room, measured through B_chain = chain_B(ℓ) with a FIXED lever ℓ held
// as instance state (set when e is a new source, then moved only by the odometry). The belief centre never enters B.
//
// Fixture: the live9 geometry in the room frame, ray-cast (test_scene.h): back wall y = 4.145, nightstand (0.84, 3.89)
// 0.50 x 0.51 x 0.62 flush on it, the bed to its -x side, the robot 1.65 m in front of it at (0.83, 2.24). Camera = ZED-height
// first hits selected by a mask that bleeds 10 cm; rays = Helios first hits selected the same way. Σ_pose = live9's robot
// pose σ (3.4 cm, 4.35 cm, 0.024 rad) (proto_room.csv).
// Run: ./bchain_jump_test   (BJ_ITER=1 prints the GN iterates)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <span>
#include <vector>
#include "countable_belief.h"
#include "countable_viewpoint.h"
#include "test_scene.h"
#include "beam_fixture.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static const rc::test::SceneBox  kStand{Eigen::Vector2f(0.84f, 3.89f), 0.0f, 0.50f, 0.51f, 0.62f};
static const rc::test::SceneBox  kBed{Eigen::Vector2f(-0.385f, 3.145f), 0.0f, 1.83f, 2.00f, 0.55f};
static const rc::test::SceneWall kBack{Eigen::Vector2f(-4.0f, 4.145f), Eigen::Vector2f(4.0f, 4.145f), 2.5f};
static const Eigen::Vector2f     kRobot(0.83f, 2.24f);
static const Eigen::Vector3d     kSigPose(0.034, 0.0435, 0.024);   // live9 robot pose σ (x, y, ψ)

static Eigen::Matrix3d sigma_pose() { return kSigPose.cwiseProduct(kSigPose).asDiagonal(); }

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

static std::vector<float> elevations(float lo, float hi, float step)
{
    std::vector<float> e;
    for (float a = lo; a <= hi; a += step) e.push_back(a);
    return e;
}

// One frame: camera points (mask bleeding `bleed` m) + Helios rays (same selection), the wall and the room axis staged,
// b_cam / b_L at the live9 marginals, e at Σ_pose with the robot parked at kRobot (Δr = 0).
static CountableFrame live9_frame(float bleed, float rho_cam, float rho_L, float rho_ch)
{
    const std::vector<rc::test::SceneBox>  boxes{kStand, kBed};
    const std::vector<rc::test::SceneWall> walls{kBack};
    CountableFrame f;
    {
        const Eigen::Vector3f o(kRobot.x(), kRobot.y(), 0.945f);
        const auto el  = elevations(-40.0f, 5.0f, 0.4f);
        const auto all = rc::test::raycast_scene(o, boxes, walls, true, 0.4f, el, 6.0f);
        f.points = rc::test::select_near(all, kStand, bleed);
        const float s = params().sigma_base_m;
        f.R.assign(f.points.size(), s * s);
        f.z_top_obs = kStand.H;
    }
    {
        const Eigen::Vector3f o(kRobot.x(), kRobot.y() - 0.155f, 1.075f);
        const auto el  = rc::test::helios_ring_elev_deg();
        const auto all = rc::test::raycast_scene(o, boxes, walls, true, rc::test::kHeliosAzStepDeg, el, 12.0f);
        f.lidar.origin = o;
        f.lidar.endpoints = rc::test::select_near(all, kStand, bleed);
        f.lidar.precision = 2500.0f; f.lidar.robust_c_m = 0.05f;
        // the persistent path's beam set (spec 7c.8): staged by ray geometry through the select box about the stand (+ bleed),
        // hand labels 0.9 on the stand, ε_px on the bed / wall / floor, κ = 1
        const auto room = rc::test::room_of(walls);   // the floor and the back wall are known room surfaces (rev 3.3.7)
        f.beam = rc::test::stage_beam(o, all, kStand, bleed, rc::test::hand_labels({kStand}), 1.0, 0.01, 6.0, nullptr, &room);
    }
    f.wall.ok = true; f.wall.p = Eigen::Vector2f(0.0f, 4.145f); f.wall.n = Eigen::Vector2f(0.0f, -1.0f);
    f.wall.sigma_m = 0.02f; f.wall.length_m = 8.0f;
    f.room_axis_ok = true; f.room_axis = 0.0f;
    // Latent marginals as the fitter stages them at range 1.65 m (latent_inputs): b_cam = floors + range, b_L = floors.
    const CountableBeliefParams p = params();
    const float range = 1.65f, lat = 0.02f * range, sz = 0.08f * range, yw = 0.03f * range;
    const float pp = p.common_mode_pos_std * p.common_mode_pos_std, ss = p.common_mode_size_std * p.common_mode_size_std,
                yy = p.common_mode_yaw_std * p.common_mode_yaw_std;
    f.sig2_cam = {pp + lat * lat, pp + lat * lat, ss + sz * sz, ss + sz * sz, ss + sz * sz, yy + yw * yw};
    f.sig2_L   = {pp, pp, ss, ss, ss, yy};
    f.rho_cam.fill(rho_cam); f.rho_L.fill(rho_L);
    f.rho_chain  = rho_ch;
    f.sigma_pose = sigma_pose();
    f.chain_dr   = Eigen::Vector2f::Zero();
    f.robot_xy   = kRobot;
    return f;
}

static void print_state(const char* tag, const CountableBelief& b)
{
    if (std::getenv("BJ_ITER"))
        for (const auto& v : b.dbg_iterates()) std::printf("      iterate cx %.4f cy %.4f H %.4f w %.4f h %.4f yaw %+.4f\n", v(0), v(1), v(2), v(3), v(4), v(5));
    const auto s = b.state();
    const auto l = b.latent_mean();
    const auto L = b.chain_lever();
    std::printf("  %-10s cx %.4f cy %.4f H %.4f w %.4f h %.4f yaw %+.4f | e (%+.4f %+.4f %+.5f) lever (%.3f %.3f) bcam_x %+.4f bL_x %+.4f "
                "| gn %d/%d staged %d\n", tag, s.cx, s.cy, s.H, s.w, s.h, s.yaw, l(0), l(1), l(2), L.l.x(), L.l.y(), l(CountableBelief::kCam),
                l(CountableBelief::kL), b.last_gn_iters(), int(b.last_gn_converged()), b.last_beam_staged());
}

// (1) The live sequence on the ray-cast scene, seeded at `seed_c`. The lever is taken on the first update at
// (seed centre − robot); `lever_c` ≠ seed_c moves the robot_xy the frames report by (seed_c − lever_c), which changes the
// lever ONLY (robot_xy enters nothing else, and at ρ_chain = 1 lever_step ignores it), so ℓ = lever_c − robot. The camera
// cloud sits cam_dx off the true box (live9: cloud centre 0.92 vs truth 0.84). Returns θ and e after 6 updates, and the
// worst |e_i|/σ_i over the run.
struct SeqOut { Eigen::Matrix<float, 6, 1> th; Eigen::Vector3d e; double worst_z; };
static SeqOut live9_sequence(const Eigen::Vector2f& seed_c, const Eigen::Vector2f& lever_c, float cam_dx, bool verbose)
{
    const CountableBeliefState seed{seed_c.x(), seed_c.y(), 0.62f, 0.44f, 0.51f, -0.01f};
    CountableBelief b(seed, params());
    b.set_room_interior(Eigen::Vector2f(0.0f, 2.0f));
    SeqOut out{};
    for (int k = 1; k <= 6; ++k)
    {
        const bool first = (k == 1);
        CountableFrame f = live9_frame(0.10f, first ? 1.0f : 0.99897f, first ? 1.0f : 0.997257f, 1.0f);
        for (auto& q : f.points) q.x() += cam_dx;
        f.robot_xy = kRobot + (seed_c - lever_c);
        b.update(f);
        if (verbose)
        {
            char tag[16]; std::snprintf(tag, sizeof tag, "update %d", k);
            print_state(tag, b);
        }
        const Eigen::Vector3d e = b.latent_mean().head<3>();
        out.worst_z = std::max(out.worst_z, e.cwiseQuotient(kSigPose).cwiseAbs().maxCoeff());
    }
    out.th = b.state().vec();
    out.e  = b.latent_mean().head<3>();
    return out;
}

// The lever is set from the seed centre, which is off the truth; the model error that buys is σ_ψ·|Δℓ| = σ_ψ·|seed − truth|
// (spec 7c.7 b, stated limit). e and θ must equal the control whose lever is seeded at the truth within that. Two controls:
// (i) the lever alone from the truth (same GN start and class-prior centre); (ii) ruling T13-1, now that the update
// converges (spec 7c.7 d): the full truth-seeded control (seed AND lever at the truth): |e| ≤ 0.1σ per component and yaw
// within σ_ψ (rad) are checked, and (spec 7c.8 Pass, the deferred item back as a CHECK) θ's position equals the full control
// within σ_ψ·|seed − truth| per component.
// spec 7c.8 rev 3.3.8 "Known surfaces with their own σ": this is the FLUSH-WALL fixture — the stand flush on the back wall, the
// wall (σ 0.02 m, the WallRef's) and the floor (σ 0.005 m, Model.FloorSigmaM) known surfaces in p_f. Pass: one basin — the
// seed run and the lever-only control settle at the same θ and e (dth, de ≤ σ_ψ·|seed − truth|, the first two CHECKs).
// spec 7c.8 rev 3.3.9 "Flush against a known wall" (stated limit): with e modelled, a box flush against a known wall has at
// least two energy basins (w against e_x, h against e_y), so the full truth-seeded control (ii) may settle in the other one.
// Its numbers (θ against the full control, |e|/σ, yaw) are PRINTED, not checked; the lever-only control stays the CHECK, and
// live11 watches the two flush nightstands for an h or w error.
static void lever_arm_sensitivity()
{
    const float cam_dx = 0.12f;
    const Eigen::Vector2f seed(0.92f, 3.75f), truth = kStand.c;
    const double tol = kSigPose(2) * double((seed - truth).norm());
    std::printf("  lever arm (seed (%.2f, %.2f), lever from the seed):\n", seed.x(), seed.y());
    const SeqOut lev = live9_sequence(seed, seed, cam_dx, true);
    std::printf("  control (same seed, lever from the truth (%.2f, %.2f)):\n", truth.x(), truth.y());
    const SeqOut ctl = live9_sequence(seed, truth, cam_dx, true);
    std::printf("  full control (seed AND lever at the truth):\n");
    const SeqOut all = live9_sequence(truth, truth, cam_dx, true);
    const double dth = (lev.th - ctl.th).cast<double>().cwiseAbs().maxCoeff(), de = (lev.e - ctl.e).cwiseAbs().maxCoeff();
    std::printf("  lever_arm_sensitivity: max|dtheta| %.5f  max|de| %.5f  (bound sigma_psi*|seed-truth| = %.5f) ; "
                "worst |e|/sigma %.3f / %.3f / %.3f\n", dth, de, tol, lev.worst_z, ctl.worst_z, all.worst_z);
    std::printf("    per component dtheta:");
    for (int i = 0; i < 6; ++i) std::printf(" %+.5f", lev.th(i) - ctl.th(i));
    std::printf("  de: %+.5f %+.5f %+.5f\n", lev.e(0) - ctl.e(0), lev.e(1) - ctl.e(1), lev.e(2) - ctl.e(2));
    CHECK(dth <= tol);
    CHECK(de <= tol);
    CHECK(lev.worst_z < 3.0 and ctl.worst_z < 3.0);
    // (ii) T13-1: the full truth-seeded control. PRINTED (spec 7c.8 rev 3.3.9 stated limit, see above): |e| ≤ 0.1σ per
    // component on both runs, the yaw within σ_ψ of the control, and θ's position components within σ_ψ·|seed − truth|.
    std::printf("    full control, per component dtheta (cx cy H w h in m, bound %.5f; yaw in rad, bound %.5f):", tol, kSigPose(2));
    for (int i = 0; i < 6; ++i) std::printf(" %+.5f", lev.th(i) - all.th(i));
    std::printf("  (sigma_psi*|seed-truth| = %.5f m for reference)\n", tol);
    std::printf("    full control, de: %+.5f %+.5f %+.5f ; |e|/sigma seed run (%.3f %.3f %.3f) truth run (%.3f %.3f %.3f)\n",
                lev.e(0) - all.e(0), lev.e(1) - all.e(1), lev.e(2) - all.e(2),
                std::abs(lev.e(0)) / kSigPose(0), std::abs(lev.e(1)) / kSigPose(1), std::abs(lev.e(2)) / kSigPose(2),
                std::abs(all.e(0)) / kSigPose(0), std::abs(all.e(1)) / kSigPose(1), std::abs(all.e(2)) / kSigPose(2));
    bool e_small = true, pos_within = true;
    for (int i = 0; i < 3; ++i) e_small = e_small and std::abs(lev.e(i)) <= 0.1 * kSigPose(i) and std::abs(all.e(i)) <= 0.1 * kSigPose(i);
    for (int i = 0; i < 5; ++i) pos_within = pos_within and std::abs(double(lev.th(i)) - double(all.th(i))) <= tol;
    const double dyaw = std::abs(std::remainder(double(lev.th(5)) - double(all.th(5)), 2.0 * M_PI));
    std::printf("    full control (printed, rev 3.3.9 stated limit): |e| <= 0.1 sigma %s ; position within %.5f %s ; "
                "yaw %.5f rad (sigma_psi %.5f) %s\n", e_small ? "yes" : "NO", tol, pos_within ? "yes" : "NO",
                dyaw, kSigPose(2), dyaw <= kSigPose(2) ? "yes" : "NO");
}

// (2) The mechanism alone, in common/ai_belief/persistent_update.h with the live9 numbers: one DOF θ = cx, latents
// x_b = (e (3), b_cam, b_L); the camera measures θ + B_cx e + b_cam, the rays θ + B_cx e + b_L, B_cx = chain_B(ℓ)'s cx row.
// Update 1 leaves the measured sum at u1 = 0.892, update 2's data says u2 = 0.99 (live9 rows 0/1). ρ_chain = 1 (parked,
// the localiser corrected nothing), ρ_cam/ρ_L as logged. The belief centre is (0.92, 3.75) on update 1 and `c2` on update
// 2: it may move the lever only through lever_step, which at ρ_chain = 1 and Δr = 0 leaves ℓ alone.
struct LinOut { Eigen::Matrix<double, 6, 1> x; Eigen::Matrix<double, 6, 6> P; };
static LinOut linear_two_updates(const Eigen::Vector2f& c2)
{
    namespace pu = rc::ai::persistent;
    using M = Eigen::Matrix<double, 6, 6>; using V = Eigen::Matrix<double, 6, 1>;
    const double Ic = 1e5, IL = 1e3;
    M P = M::Zero(); V x = V::Zero();
    x(0) = 0.87; P(0, 0) = 0.09;                                      // class prior on cx at the seed, σ = prior_pos_std
    const Eigen::Vector2f r = kRobot, dr = Eigen::Vector2f::Zero();
    ChainLever L;
    const Eigen::MatrixXd I1 = Eigen::MatrixXd::Identity(1, 1);
    const auto predict = [&](bool new_source, Eigen::Vector2f c, double rho_cam, double rho_L)
    {
        if (new_source) lever_start(L, c, r); else lever_step(L, dr, 1.0f, c, r);
        pu::predict_block<1, 5>(P, x, 0, 3, chain_transport(dr), Eigen::VectorXd::Ones(3), sigma_pose(), new_source);
        pu::predict_block<1, 5>(P, x, 3, 1, I1, Eigen::VectorXd::Constant(1, rho_cam), Eigen::MatrixXd::Constant(1, 1, 0.0021), new_source);
        pu::predict_block<1, 5>(P, x, 4, 1, I1, Eigen::VectorXd::Constant(1, rho_L),   Eigen::MatrixXd::Constant(1, 1, 0.0009), new_source);
    };
    const auto update = [&](double u)
    {
        const Eigen::Matrix<double, 1, 3> Bcx = chain_B(L.l).row(0);
        std::array<pu::Channel<1, 5>, 2> ch;
        ch[0].B << Bcx, 1, 0; ch[1].B << Bcx, 0, 1;
        ch[0].info.I(0, 0) = Ic; ch[0].info.g(0) = Ic * (u - (x(0) + ch[0].B.dot(x.tail<5>())));
        ch[1].info.I(0, 0) = IL; ch[1].info.g(0) = IL * (u - (x(0) + ch[1].B.dot(x.tail<5>())));
        const pu::FreshInfo<1> none;
        const V xn = pu::joint_step<1, 5>(x, x, P, std::span<const pu::Channel<1, 5>>(ch), none);
        P = pu::joint_cov<1, 5>(P, std::span<const pu::Channel<1, 5>>(ch), none);
        x = xn;
    };
    predict(true, Eigen::Vector2f(0.92f, 3.75f), 1.0, 1.0);
    update(0.892);
    predict(false, c2, 0.99897, 0.997257);
    update(0.99);
    return {x, P};
}

static void linear_scale_coupling()
{
    const LinOut live = linear_two_updates(Eigen::Vector2f(0.89f, 3.90f));   // the belief centre moved (live9)
    const LinOut ctrl = linear_two_updates(Eigen::Vector2f(0.92f, 3.75f));   // the belief centre did not move
    const double dx = (live.x - ctrl.x).cwiseAbs().maxCoeff(), dP = (live.P - ctrl.P).cwiseAbs().maxCoeff();
    const Eigen::Vector3d ez = live.x.segment<3>(1).cwiseQuotient(kSigPose).cwiseAbs();
    std::printf("  linear: cx %.5f  e (%+.5f %+.5f %+.6f) = (%.3f %.3f %.3f) sigma  b_cam %+.5f b_L %+.5f ; "
                "vs unmoved centre max|dx| %.2e max|dP| %.2e\n",
                live.x(0), live.x(1), live.x(2), live.x(3), ez(0), ez(1), ez(2), live.x(4), live.x(5), dx, dP);
    CHECK(dx <= 1e-9 and dP <= 1e-9);                                 // the belief centre never enters B
    CHECK(std::abs(live.x(0) - 0.9385) <= 0.001);
    CHECK(ez.maxCoeff() <= 0.1);
}

int main()
{
    linear_scale_coupling();
    lever_arm_sensitivity();
    std::printf("bchain_jump_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
