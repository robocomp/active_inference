// creep_set_test — first-crossing LiDAR hits on the persistent path (spec 2026-11-15 lidar-measures §7c.6).
//  1. Unit, the shared factor (common/ai_belief/lidar_ray_factor.h):
//     - first_crossing_ignores_near_miss: a ray passing 1 mm outside a box edge contributes nothing under first_crossing;
//       the legacy loop (surf_eps stop) scores it as a hit.
//     - first_crossing_hit_matches_legacy_on_face: a ray square on a face gives the same residual and Jacobian.
//  2. The 16-case creep set: the parked flush scene of persistent_belief_test's physical_lidar_only_sigma_bounded (belief =
//     truth, front LiDAR returns ray-cast from the TRUE scene, wall y = 0 behind the box), belief offset by
//     cx ∈ {0, 0.5, 1, 2} mm × w ∈ {0, 1} mm, with and without the wall returns, N LiDAR-only sweeps on the persistent path.
//     Pass: |Δback face| < 1 mm, |Δw| < 2 mm, |Δfront face| < 2 mm at the end AND at every sweep (the maxima), no NaN
//     (Δ from the start state).
//     spec 7c.8: the persistent path now reads the BEAM set (smooth beam model), so the creep set runs on it, staged as the
//     spec stages it — by RAY GEOMETRY (every return whose ray crosses the select box: the truth box + 10 cm), hand labels 0.9
//     on the box and ε_px = 0.02 elsewhere, κ = 1 — "with and without the wall returns" being the scene with and without the
//     wall. Endpoint staging (f.lidar's select_near, which drops the floor beside the box) leaves the silhouettes with
//     claimed rays and nothing to refute a wider box: measured +6.7 cm on w in 20 sweeps (wall off), the a0 artefact the
//     spec's ray staging exists to remove. f.lidar is still staged for the record (unused on this path).
// Run: ./creep_set_test [N=200]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "countable_belief.h"
#include "test_scene.h"
#include "beam_fixture.h"

using namespace rc;
using Mat6 = Eigen::Matrix<float, 6, 6>;
using Vec6 = Eigen::Matrix<float, 6, 1>;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static constexpr float tx = 0.30f, ty = -0.255f, tH = 0.62f, tw = 0.51f;

// persistent_belief_test's params() (the golden nightstand belief, persistent on).
static CountableBeliefParams params()
{
    CountableBeliefParams p;
    p.prior_footprint_m = 0.51f; p.prior_footprint_std = 0.06f; p.prior_height_m = 0.62f; p.prior_height_std = 0.10f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;
    p.wall_flush_prior = 0.90f;
    p.room_axis_std    = 3.0f * 0.017453293f;
    p.volatility_infer = true;
    // F1 round 6: the nightstand class's permanence prior (nightstand.concept.toml [prior.permanence]: 0.011 moves/day, 0.3 m,
    // 0.5 rad; size jumps at the class size stds), so Q is the class's q0 (h: π·0.06² = 4.6e-11 m² per 0.1 s step), not 0.
    // PREDICTED before the run: the creep scaled from round 4's measured 1.25 mm per 200 sweeps at Q = 2.5e-5 (linear in q and
    // in the sweep count) is 1.25e-3 · (4.6e-11/2.5e-5) · (2000/200) ≈ 2e-8 m over 2000 sweeps — negligible against 1 mm.
    p.move_rate_per_day = 0.011f; p.move_scale_m = 0.3f; p.move_scale_rad = 0.5f;
    p.persistent       = true;
    p.beam             = rc::test::test_beam_config();
    return p;
}

// ---- 1. Unit: the factor on one ray ----

static void accumulate_one(const CountableBelief& b, const CountableBeliefState& s, const rc::ai::LidarRays& r, Mat6& Id, Vec6& bd)
{
    Id.setZero(); bd.setZero();
    rc::ai::accumulate_lidar_rays<6>(b, s, r, Id, bd);
}

static void first_crossing_ignores_near_miss()
{
    // Box at the origin, 0.5 × 0.5 × 0.6. A descending ray from the front passes 1 mm ABOVE the back-top edge (the wall return
    // over the top edge of the investigation): it never crosses the box. The legacy sphere-trace stops where SDF < surf_eps on
    // the top face and scores it.
    const CountableBeliefState s{0.0f, 0.0f, 0.60f, 0.50f, 0.50f, 0.0f};
    CountableBelief b(s, params());
    const Eigen::Vector3f o(0.05f, -2.0f, 1.10f), pass(0.05f, 0.25f, 0.601f);
    rc::ai::LidarRays r;
    r.origin = o; r.endpoints = {o + 1.15f * (pass - o)}; r.precision = 2500.0f; r.robust_c_m = 0.05f;
    Mat6 IdL, IdX; Vec6 bdL, bdX;
    accumulate_one(b, s, r, IdL, bdL);
    rc::ai::LidarRays rx = r; rx.first_crossing = true;
    accumulate_one(b, s, rx, IdX, bdX);
    rc::ai::LidarRays rs = rx; rs.surface_sigma_m = 0.01f;
    Mat6 IdS; Vec6 bdS;
    accumulate_one(b, s, rs, IdS, bdS);
    // control: the same ray 1 mm BELOW the edge crosses the top face, so first_crossing scores it
    rc::ai::LidarRays rc_ = rx; const Eigen::Vector3f pass_in(0.05f, 0.25f, 0.599f);
    rc_.endpoints = {o + 1.15f * (pass_in - o)};
    Mat6 IdC; Vec6 bdC;
    accumulate_one(b, s, rc_, IdC, bdC);
    std::printf("  near miss 1 mm over the back-top edge: legacy |Id| %.4e |bd| %.4e ; first_crossing |Id| %.4e |bd| %.4e (sigma_s 0.01: %.4e) ; "
                "1 mm under the edge (crosses): first_crossing |Id| %.4e\n",
                IdL.norm(), bdL.norm(), IdX.norm(), bdX.norm(), IdS.norm(), IdC.norm());
    CHECK(IdL.norm() > 0.0f);                       // the legacy loop scores the near miss
    CHECK(IdX.norm() == 0.0f and bdX.norm() == 0.0f);
    CHECK(IdS.norm() == 0.0f and bdS.norm() == 0.0f);
    CHECK(IdC.norm() > 0.0f);                       // a real crossing is a hit
}

static void first_crossing_hit_matches_legacy_on_face()
{
    // A ray square on the front face (y = −0.25), observed 1 cm behind the model surface: e = −0.01 on both loops.
    const CountableBeliefState s{0.0f, 0.0f, 0.60f, 0.50f, 0.50f, 0.0f};
    CountableBelief b(s, params());
    const Eigen::Vector3f o(0.05f, -2.0f, 0.30f);
    rc::ai::LidarRays r;
    r.origin = o; r.endpoints = {Eigen::Vector3f(0.05f, -0.24f, 0.30f)}; r.precision = 2500.0f; r.robust_c_m = 1e3f;
    Mat6 IdL, IdX; Vec6 bdL, bdX;
    accumulate_one(b, s, r, IdL, bdL);
    rc::ai::LidarRays rx = r; rx.first_crossing = true;
    accumulate_one(b, s, rx, IdX, bdX);
    // Id = w JJᵀ, bd = −w J e with w = precision (robust_c_m = 1e3 ⇒ the Cauchy factor is 1 to 1e-10) and J_cy = +1 > 0 on
    // the front face: √w J = Id.col(cy)/√Id(cy,cy), e = −bd(cy)/(w J_cy).
    const float w = r.precision;
    const auto recover = [&](const Mat6& Id, const Vec6& bd, float& e, Vec6& J)
    {
        J = Id.col(1) / std::sqrt(Id(1, 1) * w);
        e = -bd(1) / (w * J(1));
    };
    float eL, eX; Vec6 JL, JX;
    recover(IdL, bdL, eL, JL);
    recover(IdX, bdX, eX, JX);
    std::printf("  square on the front face: residual legacy %+.6f first_crossing %+.6f ; |J_legacy − J_first| %.2e ; J %s\n",
                eL, eX, (JL - JX).cwiseAbs().maxCoeff(), "(cx cy H w h yaw)");
    std::printf("    J legacy %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n    J first  %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n",
                JL(0), JL(1), JL(2), JL(3), JL(4), JL(5), JX(0), JX(1), JX(2), JX(3), JX(4), JX(5));
    CHECK(IdL.norm() > 0.0f and IdX.norm() > 0.0f);
    CHECK(std::abs(eL - eX) < 1e-5f);
    CHECK(std::abs(eX - (-0.01f)) < 1e-4f);
    CHECK((JL - JX).cwiseAbs().maxCoeff() < 1e-4f);
}

// ---- 2. The creep set ----

static const auto kElev = rc::test::helios_ring_elev_deg();
static const rc::test::SceneWall kWall{Eigen::Vector2f(-4.0f, 0.0f), Eigen::Vector2f(4.0f, 0.0f), 2.5f};

static rc::ai::LidarRays front_rays(bool wall_returns)
{
    const rc::test::SceneBox box{Eigen::Vector2f(tx, ty), 0.0f, tw, tw, tH};
    const std::vector<rc::test::SceneBox> boxes{box};
    const std::vector<rc::test::SceneWall> walls{kWall};
    const Eigen::Vector3f origin(tx, ty - 0.5f * tw - 1.5f, 1.1075f);
    const auto all = rc::test::raycast_scene(origin, boxes, walls, true, rc::test::kHeliosAzStepDeg, kElev, 12.0f);
    rc::ai::LidarRays r;
    r.origin = origin; r.precision = 2500.0f; r.robust_c_m = 0.05f;
    for (const auto& e : rc::test::select_near(all, box, 0.10f))
        if (wall_returns or std::abs(e.y() - kWall.a.y()) >= 1e-3f) r.endpoints.push_back(e);
    return r;
}

static CountableFrame lidar_frame(const rc::ai::LidarRays& r, bool wall_in_scene)
{
    const CountableBeliefParams p = params();
    CountableFrame f;
    f.lidar = r; f.lidar_only = true; f.z_top_obs = -1.0f;
    const rc::test::SceneBox box{Eigen::Vector2f(tx, ty), 0.0f, tw, tw, tH};
    const std::vector<rc::test::SceneBox> obj{box};
    const std::vector<rc::test::SceneWall> walls = wall_in_scene ? std::vector<rc::test::SceneWall>{kWall} : std::vector<rc::test::SceneWall>{};
    f.beam = rc::test::helios_beam(r.origin, obj, walls, box, 0.10f, rc::test::hand_labels(obj), 1.0, p.model_sigma_m);
    f.wall.ok = true; f.wall.p = Eigen::Vector2f(0, 0); f.wall.n = Eigen::Vector2f(0, -1);
    f.wall.sigma_m = 0.02f; f.wall.length_m = 4.0f;
    f.room_axis_ok = true; f.room_axis = 0.0f;
    const float pp = p.common_mode_pos_std * p.common_mode_pos_std, ss = p.common_mode_size_std * p.common_mode_size_std,
                yy = p.common_mode_yaw_std * p.common_mode_yaw_std;
    f.sig2_cam = {pp, pp, ss, ss, ss, yy};
    f.sig2_L   = {pp, pp, ss, ss, ss, yy};
    f.sigma_pose = Eigen::Vector3d(1e-4, 1e-4, 0.0).asDiagonal();   // the robot pose error e: 1 cm on (x, y), none on ψ
    f.rho_cam.fill(1.0f); f.rho_L.fill(1.0f); f.rho_chain = 1.0f;     // parked
    return f;
}

static void creep_set(int N)
{
    const float dcxs[] = {0.0f, 0.0005f, 0.001f, 0.002f};
    const float dws[]  = {0.0f, 0.001f};
    // spec 7c.8 rev 3.3.8 "Bounds against ray sampling", corrected by rev 3.3.9: w and h are distances between two
    // independently sampled edges, so Δw and Δh are bounded by the FULL Helios ray spacing s at the object (one centre, cx or
    // cy, would take ½s) — along w the azimuth step × range, along h the ring spacing × range (the front face is 1.5 m from the
    // sensor). The 7c.6 face bounds (back 1 mm, front 2 mm) stay.
    constexpr float d2r = 0.017453293f, range = 1.5f;
    const float bw = range * rc::test::kHeliosAzStepDeg * d2r;
    const float bh = range * (110.0f / 49.0f) * d2r;
    std::printf("  creep bounds: |dw| <= %.4f m (the azimuth spacing at %.1f m), |dh| <= %.4f m (the ring spacing)\n", bw, range, bh);
    int fails = 0;
    for (const bool wall : {true, false})
    {
        const CountableFrame f = lidar_frame(front_rays(wall), wall);
        for (const float dw : dws)
            for (const float dcx : dcxs)
            {
                const CountableBeliefState s0{tx + dcx, ty, tH, tw + dw, tw, 0.0f};
                CountableBelief b(s0, params());
                b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
                const float back0 = s0.cy + 0.5f * s0.h, front0 = s0.cy - 0.5f * s0.h;
                float maxb = 0.0f, maxw = 0.0f, maxf = 0.0f, maxh = 0.0f;
                bool finite = true;
                for (int k = 0; k < N; ++k)
                {
                    b.update(f);
                    const auto& s = b.state();
                    finite = finite and s.vec().allFinite() and b.covariance().allFinite();
                    maxb = std::max(maxb, std::abs(s.cy + 0.5f * s.h - back0));
                    maxf = std::max(maxf, std::abs(s.cy - 0.5f * s.h - front0));
                    maxw = std::max(maxw, std::abs(s.w - s0.w));
                    maxh = std::max(maxh, std::abs(s.h - s0.h));
                }
                const auto& s = b.state();
                const float db = s.cy + 0.5f * s.h - back0, dfr = s.cy - 0.5f * s.h - front0, dww = s.w - s0.w;
                const float dhh = s.h - s0.h;
                const bool ok = finite and std::abs(db) < 0.001f and std::abs(dww) <= bw and std::abs(dfr) < 0.002f and std::abs(dhh) <= bh
                                and maxb < 0.001f and maxw <= bw and maxf < 0.002f and maxh <= bh;
                fails += not ok;
                std::printf("  wall %-3s dcx %.4f dw %.4f | Δback %+.5f (max %.5f) Δw %+.5f (max %.5f, bound %.4f) Δh %+.5f (bound %.4f) "
                            "Δfront %+.5f (max %.5f) | cx %.5f H %.5f sig_h %.4f sig_cx %.4f | %zu rays %s\n",
                            wall ? "on" : "off", dcx, dw, db, maxb, dww, maxw, bw, dhh, bh, dfr, maxf, s.cx, s.H,
                            std::sqrt(b.covariance()(4, 4)), std::sqrt(b.covariance()(0, 0)), f.beam.endpoints.size(),
                            ok ? "ok" : "FAIL");
            }
    }
    std::printf("  creep set, %d sweeps: %d / 16 fail\n", N, fails);
    CHECK(fails == 0);   // spec 7c.8 rev 3.3.9: 0/16 with Δw, Δh within the ray spacing s
}

int main(int argc, char** argv)
{
    // Task 19: 200 by default (was 4000). On the beam path a parked sweep of the ~1000 ray-staged returns costs ~50 ms (the
    // first sweeps up to gn_iters_max), so 16 × 4000 sweeps is about an hour; θ stops moving within the first ~10 sweeps of
    // every case (the step falls below the tolerance and the update stops), so 200 sees the same end state. ./creep_set_test 4000
    // still runs the long set.
    const int N = argc > 1 ? std::atoi(argv[1]) : 200;
    first_crossing_ignores_near_miss();
    first_crossing_hit_matches_legacy_on_face();
    creep_set(N);
    std::printf("creep_set_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
