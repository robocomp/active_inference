// beam_live_staging_test — the persistent path's rays as the fitter stages them (spec 2026-11-15 lidar-measures §7c.8 a0 and
// Pass "live staging"): a ray-cast TRUE scene (test_scene.h, first hit only), a mask RENDERED by a pinhole ZED from the same
// scene, the label built from it by the agent's own make_persistent_label (σ_px and ε_px as live), and the rays staged by
// the agent's own stage_beam_cycle (countable_beam_staging.h: select_rays_beam by ray geometry on the live select box, κ as
// computed, σ_s = model σ, the floor and the room walls in p_f). Nothing is set by hand.
//
// Live values (nightstand_concept/etc/config.toml unless said): ε_px = MaskPixelErrorRate 0.094; the beam config at
// LidarPrecision 2500, LidarRobustCM 0.05 and the Helios σ_sensor 0.02 m (rc::kHeliosRangeSigmaM); σ_px from
// projection_sigma_px with ExistenceSensorSigmaM 0.03 m, σ_chain 0.01 m (the fixtures' Σ_pose of 1 cm on x, y), no ego-motion;
// π_mask = 0.9 (a slice confidence); the select box from BirthWidthM / BirthDepthM / BirthHeightM 0.5 / 0.5 / 0.45 and
// LidarSelectMarginM 0.10, anchored on the mask cloud; LidarCoverageN0 60, LidarCoverageAngPower 1.
// Fixture object: the golden nightstand (0.51 × 0.51 × 0.62 at (0.30, −0.255)); the robot 1.5 m in front of its front face;
// the ZED at the mount of ROBOT_GEOMETRY.md (0.945 m, 7.5 cm behind the body centre), 640 × 480 at f = 350 px (fixture choice:
// the whole nightstand in the image at 1.5 m). The silhouette fixtures put the Helios at 0.40 m like beam_belief_test's
// hand-label silhouettes (the rays then reach the background instead of the floor); the gap fixture at the Helios height.
// "Over 10 frames" bounds are the largest |Δ| over the 10 updates. Run: ./beam_live_staging_test
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <span>
#include <vector>
#include "countable_belief.h"
#include "countable_beam_staging.h"
#include "countable_mask_membership.h"
#include "beam_fixture.h"

using namespace rc;
namespace bm = rc::ai::beam;
namespace ts = rc::test;
using Vec6 = Eigen::Matrix<float, 6, 1>;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static constexpr float tx = 0.30f, ty = -0.255f, tH = 0.62f, tw = 0.51f;
static constexpr float kHeliosZ = 1.1075f, kSilZ = 0.40f;
static const Eigen::Vector2f kRobot(tx, ty - 0.5f * tw - 1.5f);
// Live values (see the header).
static constexpr double kEpsLive = 0.094;
static constexpr float  kPiMask = 0.9f, kSigmaRange = 0.03f, kSigmaChain = 0.01f, kMotVar = 0.0f;
static constexpr float  kSigmaS = 0.01f;            // ModelSigmaM
static constexpr float  kHalfSpacing = 0.005f;      // σ_s / 2: the live-staging and camera-coverage bound (spec Pass)

static bm::BeamConfig live_beam_config() { return *rc::make_beam_config(kEpsLive, 2500.0, 0.05, rc::kHeliosRangeSigmaM); }

static CountableBeliefParams params()
{
    CountableBeliefParams p;
    p.prior_footprint_m = 0.51f; p.prior_footprint_std = 0.06f; p.prior_height_m = 0.62f; p.prior_height_std = 0.10f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;
    p.wall_flush_prior = 0.90f;
    p.room_axis_std    = 3.0f * 0.017453293f;
    p.volatility_infer = true;
    p.persistent       = true;
    p.model_sigma_m    = kSigmaS;
    p.beam             = live_beam_config();
    return p;
}
static CountableBelief belief(const CountableBeliefState& s)
{
    CountableBelief b(s, params());
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    b.set_first_update_done(true);
    return b;
}
static ts::SceneBox truth_box() { return {Eigen::Vector2f(tx, ty), 0.0f, tw, tw, tH}; }
static ts::SceneWall back_wall(float gap)
{
    const float y = ty + 0.5f * tw + gap;
    return {Eigen::Vector2f(-4.0f, y), Eigen::Vector2f(4.0f, y), 2.5f};
}
static ts::Camera front_camera(float dx = 0.0f)
{
    ts::Camera c;
    c.pos = Eigen::Vector3f(kRobot.x() + dx, kRobot.y() - 0.075f, 0.945f);
    return c;
}

// The latents at the config floors, e at 1 cm on (x, y), parked, robot at kRobot.
static void set_latents(CountableFrame& f)
{
    const CountableBeliefParams p = params();
    const float pp = p.common_mode_pos_std * p.common_mode_pos_std, ss = p.common_mode_size_std * p.common_mode_size_std,
                yy = p.common_mode_yaw_std * p.common_mode_yaw_std;
    f.sig2_cam = {pp, pp, ss, ss, ss, yy};
    f.sig2_L   = {pp, pp, ss, ss, ss, yy};
    f.sigma_pose = Eigen::Vector3d(1e-4, 1e-4, 0.0).asDiagonal();
    f.rho_cam.fill(1.0f); f.rho_L.fill(1.0f); f.rho_chain = 1.0f;
    f.robot_xy = kRobot;
}

// ── The camera side: rendered mask → MaskMembership → the agent's persistent label (σ_px, ε_px live). ──
struct Live
{
    ts::Camera cam;
    ts::RenderedMask mask;
    MaskMembership mm;
    BeamLabel label;
    double eps = kEpsLive;
    // σ_px at depth z, in px and in metres at the object
    float sigma_px(float z) const { return projection_sigma_px(cam.f, z, kSigmaRange, kSigmaChain, kMotVar, mm.stride_px()); }
    float sigma_m(float z) const { return sigma_px(z) * z / cam.f; }
};
static Live make_live(const ts::Camera& cam, const ts::SceneBox& target, std::span<const ts::SceneBox> others = {},
                      float erode_m = 0.0f, double eps = kEpsLive)
{
    Live L;
    L.cam = cam;
    L.eps = eps;
    L.mask = ts::render_mask(cam, target, others, erode_m);
    L.mm = MaskMembership::build(L.mask.pixels);
    const ts::Camera c = cam;
    Projector proj = [c](const Eigen::Vector3f& p) { return c.project(p); };
    L.label = beam_label(make_persistent_label(proj, L.mm, cam.f, kSigmaRange, kSigmaChain, kMotVar, kPiMask, eps));
    return L;
}
// The mask cloud: points on the visible faces of `b` (front y = cy − h/2; left x = cx − w/2 when `left`) whose pixel lies
// inside the rendered mask — what the slice's deprojected depth would hand the fitter.
static std::vector<Eigen::Vector3f> mask_cloud(const Live& L, const ts::SceneBox& b, bool left = false)
{
    std::vector<Eigen::Vector3f> pts;
    const auto keep = [&](const Eigen::Vector3f& p)
    {
        const auto pr = L.cam.project(p);
        if (pr and L.mm.signed_distance_px(pr->x(), pr->y()) > 0.0f) pts.push_back(p);
    };
    for (int i = 0; i < 24; ++i) for (int j = 0; j < 12; ++j)
        keep({b.c.x() - 0.5f * b.w + b.w * (i + 0.5f) / 24.0f, b.c.y() - 0.5f * b.h, b.z0 + (b.H - b.z0) * (j + 0.5f) / 12.0f});
    if (left)
        for (int i = 0; i < 24; ++i) for (int j = 0; j < 12; ++j)
            keep({b.c.x() - 0.5f * b.w, b.c.y() - 0.5f * b.h + b.h * (i + 0.5f) / 24.0f, b.z0 + (b.H - b.z0) * (j + 0.5f) / 12.0f});
    return pts;
}

// The agent's staging (stage_beam_cycle) on the live select box and weights, the room = floor + `room_walls` (known surfaces).
static BeamStaging live_staging(const std::vector<ts::SceneWall>& room_walls)
{
    BeamStaging s;
    s.box = countable_select_box(0.5f, 0.5f, 0.45f, 0.10f);
    s.w = rc::lidar_select::Weights{60.0f, 1.0f, 0.05f};
    s.rho_max = 6.0;
    s.sigma_s = kSigmaS;
    s.room = ts::room_of(room_walls);
    s.st = live_beam_config().st;
    return s;
}
static bm::BeamRays stage(const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin,
                          std::span<const Eigen::Vector3f> anchor, const BeamLabel& label,
                          const std::vector<ts::SceneWall>& room_walls, BeamLog* log_out = nullptr)
{
    bm::BeamRays r;
    BeamLog log;
    stage_beam_cycle(&sweep, origin, anchor, live_staging(room_walls), label, r, log);
    if (log_out) *log_out = log;
    return r;
}
static std::vector<Eigen::Vector3f> helios_sweep(const Eigen::Vector3f& o, std::span<const ts::SceneBox> boxes,
                                                 std::span<const ts::SceneWall> walls)
{
    const auto rings = ts::helios_ring_elev_deg();
    return ts::raycast_scene(o, boxes, walls, true, ts::kHeliosAzStepDeg, rings, 12.0f);
}

static CountableFrame lidar_frame(const bm::BeamRays& r)
{
    CountableFrame f;
    f.beam = r; f.lidar_only = true; f.z_top_obs = -1.0f;
    f.room_axis_ok = true; f.room_axis = 0.0f;
    set_latents(f);
    return f;
}
static CountableFrame camera_frame(const std::vector<Eigen::Vector3f>& pts, const bm::BeamRays* r)
{
    CountableFrame f;
    f.points = pts;
    f.R.assign(f.points.size(), params().sigma_base_m * params().sigma_base_m);
    float zt = 0.0f;
    for (const auto& p : pts) zt = std::max(zt, p.z());
    f.z_top_obs = zt;
    f.room_axis_ok = true; f.room_axis = 0.0f;
    if (r) f.beam = *r;
    set_latents(f);
    return f;
}

struct Track { float max_dw = 0, max_dcx = 0, max_dcy = 0; };
static void track(Track& t, const CountableBelief& b)
{
    t.max_dw  = std::max(t.max_dw,  std::abs(b.state().w - tw));
    t.max_dcx = std::max(t.max_dcx, std::abs(b.state().cx - tx));
    t.max_dcy = std::max(t.max_dcy, std::abs(b.state().cy - ty));
}

// The "Near background" prediction of spec 7c.8 (a): the per-edge outward shift at σ_px = 1 / 2 / 3 / 4 cm at the object
// (π_mask 0.9, ε_px 2 %; the re-review's 1-D integration), linear between, the last slope beyond.
static double predicted_shift_m(double sigma_m)
{
    const double s[] = {0.01, 0.02, 0.03, 0.04}, v[] = {0.019, 0.037, 0.053, 0.069};
    if (sigma_m <= s[0]) return v[0] * sigma_m / s[0];
    for (int k = 0; k < 3; ++k)
        if (sigma_m <= s[k + 1]) return v[k] + (v[k + 1] - v[k]) * (sigma_m - s[k]) / (s[k + 1] - s[k]);
    return v[3] + (v[3] - v[2]) * (sigma_m - s[3]) / (s[3] - s[2]);
}

// ─── Silhouette, live staging (spec Pass): seeds w − 3 cm, correct, w + 3 cm; backgrounds 20 cm, 1 m (CHECK) and 5 cm (PRINT,
//     one bound: the shift per edge < 2σ_px at the object). LiDAR-only frames (the G1 gated cycle), 10 frames. ───
static void silhouette_live()
{
    const ts::Camera cam = front_camera();
    const std::vector<ts::SceneBox> boxes{truth_box()};
    const Live L = make_live(cam, truth_box());
    const auto anchor = mask_cloud(L, truth_box());
    const float zf = (ty - 0.5f * tw) - cam.pos.y();     // depth of the front face (its silhouette edges)
    const float sig_m = L.sigma_m(zf);
    std::printf("  silhouette live: mask %zu px, mask cloud %zu pts ; sigma_px at the front face %.2f px = %.4f m (z %.3f m)\n",
                L.mask.pixels.size(), anchor.size(), L.sigma_px(zf), sig_m, zf);
    const float prior_sw = params().prior_footprint_std;
    // The Helios ray spacing along w at the object: the HELIOS's horizontal range to the front face's silhouette edge × its
    // azimuth step (spec 7c.8 rev 3.3.9/3.3.10; the range is the LiDAR's, not the camera depth zf).
    const float r_edge = std::hypot(tx + 0.5f * tw - kRobot.x(), (ty - 0.5f * tw) - kRobot.y());
    const float s_helios = r_edge * ts::kHeliosAzStepDeg * std::numbers::pi_v<float> / 180.0f;
    for (const float gap : {0.20f, 1.0f, 0.05f})
    {
        const std::vector<ts::SceneWall> walls{back_wall(gap)};
        const Eigen::Vector3f o(kRobot.x(), kRobot.y(), kSilZ);
        BeamLog lg;
        const bm::BeamRays r = stage(helios_sweep(o, boxes, walls), o, anchor, L.label, walls, &lg);
        for (const float dw : {0.0f, -0.03f, 0.03f})
        {
            CountableBelief b = belief({tx, ty, tH, tw + dw, tw, 0.0f});
            Track t;
            for (int k = 0; k < 10; ++k) { b.update(lidar_frame(r)); track(t, b); }
            const float ew = b.state().w - tw, ecx = b.state().cx - tx, sw = std::sqrt(b.covariance()(3, 3));
            std::printf("  silhouette live bg %.2f m seed w %+.2f: staged %d nolabel %d kappa %.3f ; after 10: w - truth %+.5f (sigma_w %.5f, "
                        "prior %.3f) cx - truth %+.5f ; over 10 max |dw| %.5f |dcx| %.5f ; gn %d/%d\n", gap, dw, lg.staged, lg.nolabel,
                        lg.kappa, ew, sw, prior_sw, ecx, t.max_dw, t.max_dcx, b.last_gn_iters(), int(b.last_gn_converged()));
            if (gap > 0.1f)
            {
                // spec 7c.8 rev 3.3.10 (1): the correctly sized seed against the Helios spacing at the object, |Δw| ≤ s, |Δcx| ≤ ½s
                if (dw == 0.0f)
                {
                    std::printf("  silhouette live bg %.2f m correct seed: bound s = %.5f m (Helios horizontal range to the front edge "
                                "%.4f m x azimuth step %.2f deg)\n", gap, s_helios, r_edge, ts::kHeliosAzStepDeg);
                    CHECK(t.max_dw <= s_helios);
                    CHECK(t.max_dcx <= 0.5f * s_helios);
                }
                else CHECK(std::abs(ew) <= sw);
            }
            else if (dw == 0.0f)
            {
                const double right = ecx + 0.5 * ew, left = -ecx + 0.5 * ew;   // outward shift of each edge
                std::printf("  silhouette live bg 0.05 m (PRINTED, stated limit 4): dw %+.4f dcx %+.4f ; per-edge shift left %+.4f right "
                            "%+.4f against predicted %+.4f at sigma_px %.4f m (spec at eps_px 2 %%; live eps_px %.3f) ; bound 2 sigma_px "
                            "%.4f\n", ew, ecx, left, right, predicted_shift_m(sig_m), sig_m, kEpsLive, 2.0 * sig_m);
                CHECK(std::abs(left) < 2.0 * sig_m and std::abs(right) < 2.0 * sig_m);
            }
        }
    }
    // PRINTED (review I2): the 5 cm background with the back wall NOT a known room surface (floor only in p_f), the setting the
    // spec's "Near background" prediction was made in (rev 3.3.3, before rev 3.3.7's known surfaces): the wall's returns are then
    // foreign at p_pass, and a model that covers them is paid. Correct seed, 10 LiDAR-only frames. No bound.
    {
        const std::vector<ts::SceneWall> walls{back_wall(0.05f)};
        const Eigen::Vector3f o(kRobot.x(), kRobot.y(), kSilZ);
        BeamLog lg;
        const bm::BeamRays r = stage(helios_sweep(o, boxes, walls), o, anchor, L.label, {}, &lg);
        CountableBelief b = belief({tx, ty, tH, tw, tw, 0.0f});
        for (int k = 0; k < 10; ++k) b.update(lidar_frame(r));
        const double ew = b.state().w - tw, ecx = b.state().cx - tx;
        const double right = ecx + 0.5 * ew, left = -ecx + 0.5 * ew;
        std::printf("  silhouette live bg 0.05 m, wall NOT a known surface (PRINTED): staged %d kappa %.3f ; dw %+.4f dcx %+.4f ; per-edge "
                    "shift left %+.4f right %+.4f against predicted +0.019/+0.037/+0.053/+0.069 at sigma_px 1/2/3/4 cm (interpolated "
                    "%+.4f at this fixture's %.4f m; spec at eps_px 2 %%, live eps_px %.3f)\n", lg.staged, lg.kappa, ew, ecx, left, right,
                    predicted_shift_m(sig_m), sig_m, kEpsLive);
    }
    // PRINTED (review I2, the prediction's own geometry): the spec's b is the depth behind the FACE PLANE the silhouette ray
    // would enter if the model widened — for this front sweep the FRONT face. back_wall(gap) sits `gap` behind the BACK face,
    // i.e. tw + gap behind the front face plane, so the runs above never have b = 5 cm. Here a background surface 5 cm behind
    // the front face plane on both sides of the box (two 1.5 m segments starting at its edges; not a known surface), the back
    // wall 1 m (known), correct seed, 10 LiDAR-only frames. No bound.
    {
        const float yb = ty - 0.5f * tw + 0.05f;
        const std::vector<ts::SceneWall> room{back_wall(1.0f)};
        const std::vector<ts::SceneWall> scene{back_wall(1.0f),
                                               {Eigen::Vector2f(tx - 0.5f * tw - 1.5f, yb), Eigen::Vector2f(tx - 0.5f * tw, yb), 2.5f},
                                               {Eigen::Vector2f(tx + 0.5f * tw, yb), Eigen::Vector2f(tx + 0.5f * tw + 1.5f, yb), 2.5f}};
        const Eigen::Vector3f o(kRobot.x(), kRobot.y(), kSilZ);
        const auto sweep5 = helios_sweep(o, boxes, scene);
        // at the live ε_px, and at the spec prediction's ε_px = 2 % (labels only; the beam config is the live one)
        const Live L2 = make_live(cam, truth_box(), {}, 0.0f, 0.02);
        for (const Live* Lx : {&L, &L2})
        {
            BeamLog lg;
            const bm::BeamRays r = stage(sweep5, o, anchor, Lx->label, room, &lg);
            CountableBelief b = belief({tx, ty, tH, tw, tw, 0.0f});
            for (int k = 0; k < 10; ++k) b.update(lidar_frame(r));
            const double ew = b.state().w - tw, ecx = b.state().cx - tx;
            const double right = ecx + 0.5 * ew, left = -ecx + 0.5 * ew;
            std::printf("  silhouette live, background 5 cm behind the FRONT face plane, eps_px %.3f (PRINTED): staged %d kappa %.3f ; dw "
                        "%+.4f dcx %+.4f ; per-edge shift left %+.4f right %+.4f against predicted %+.4f at sigma_px %.4f m (spec at "
                        "eps_px 2 %%)\n", Lx->eps, lg.staged, lg.kappa, ew, ecx, left, right, predicted_shift_m(sig_m), sig_m);
        }
    }
    // PRINTED beside the CHECK: the correct seed's w at other Helios sampling phases (azimuth offset 0.1 / 0.2 / 0.3° of the
    // 0.4° step) and at a dense 0.05° sweep, background 1 m — what the 0.4° grid's phase does to w (the full ray spacing at the
    // front face is s = z·0.4°; spec rev 3.3.9 states the w bounds of creep and of the w optimum in s).
    {
        const std::vector<ts::SceneWall> walls{back_wall(1.0f)};
        const Eigen::Vector3f o(kRobot.x(), kRobot.y(), kSilZ);
        const auto rings = ts::helios_ring_elev_deg();
        for (const auto [step, off] : {std::pair{0.4f, 0.1f}, std::pair{0.4f, 0.2f}, std::pair{0.4f, 0.3f}, std::pair{0.05f, 0.0f}})
        {
            std::vector<Eigen::Vector3f> sweep;
            const int n = static_cast<int>(std::lround(360.0f / step));
            for (const float el : rings)
                for (int k = 0; k < n; ++k)
                {
                    const Eigen::Vector3f d = ts::beam_dir(el, off + static_cast<float>(k) * step);
                    const float t = ts::first_hit(o, d, boxes, walls, true);
                    if (t < 12.0f) sweep.push_back(o + t * d);
                }
            const bm::BeamRays r = stage(sweep, o, anchor, L.label, walls);
            CountableBelief b = belief({tx, ty, tH, tw, tw, 0.0f});
            for (int k = 0; k < 10; ++k) b.update(lidar_frame(r));
            std::printf("  silhouette live bg 1.00 m correct seed, azimuth step %.2f deg offset %.1f deg (PRINTED): %zu rays ; w - truth %+.5f "
                        "cx - truth %+.5f ; full spacing at the front face s = %.4f m\n", step, off, r.endpoints.size(),
                        b.state().w - tw, b.state().cx - tx, zf * 0.4f * std::numbers::pi_v<float> / 180.0f);
        }
    }
}

// ─── Flush neighbour, live staging, PRINTED (spec Pass, owner 11-18): a surface 2 cm behind the face plane extending 30 cm
//     beside the box (not a known room surface, not in the mask), background wall 1 m; 10 camera + ray frames against the
//     camera-only run from the same seed. ───
static void flush_neighbour_printed()
{
    const ts::Camera cam = front_camera();
    const float yf = ty - 0.5f * tw + 0.02f;
    const ts::SceneWall nb{Eigen::Vector2f(tx + 0.5f * tw, yf), Eigen::Vector2f(tx + 0.5f * tw + 0.30f, yf), tH};
    const std::vector<ts::SceneWall> room{back_wall(1.0f)}, scene{back_wall(1.0f), nb};
    const std::vector<ts::SceneBox> boxes{truth_box()};
    const Live L = make_live(cam, truth_box());
    const auto pts = mask_cloud(L, truth_box());
    const Eigen::Vector3f o(kRobot.x(), kRobot.y(), kSilZ);
    const bm::BeamRays r = stage(helios_sweep(o, boxes, scene), o, pts, L.label, room);
    CountableBelief br = belief({tx, ty, tH, tw, tw, 0.0f}), bc = br;
    for (int k = 0; k < 10; ++k) { br.update(camera_frame(pts, &r)); bc.update(camera_frame(pts, nullptr)); }
    std::printf("  flush neighbour (PRINTED): %zu rays ; after 10 frames w - truth with rays %+.5f (cx %+.5f), camera-only %+.5f (cx %+.5f) "
                "; dw rays - camera %+.5f (live11 bound 1 cm)\n", r.endpoints.size(), br.state().w - tw, br.state().cx - tx,
                bc.state().w - tw, bc.state().cx - tx, br.state().w - bc.state().w);
    CHECK(br.state().vec().allFinite());
}

// ─── Camera coverage, live staging (spec Pass, R6-C1). (i) the ZED displaced so the image's right border cuts the box: no staged
//     ray lies outside the image, and from the correct seed |Δw|, |Δcx|, |Δcy| < σ_s/2 over 10 camera + ray frames. (ii) the mask
//     eroded 3 cm on the right: the seed is the eroded cloud's (w − 3 cm, cx − 1.5 cm, what a cold start from that cloud takes),
//     and after 10 frames the bias of w and of cx with rays is not above the camera-only run's. ───
static void camera_coverage()
{
    const std::vector<ts::SceneWall> walls{back_wall(1.0f)};
    const std::vector<ts::SceneBox> boxes{truth_box()};
    const Eigen::Vector3f o(kRobot.x(), kRobot.y(), kSilZ);
    const auto sweep = helios_sweep(o, boxes, walls);
    {
        ts::Camera cam = front_camera();
        const float Y = (ty - 0.5f * tw) - cam.pos.y();
        cam.pos.x() = tx - Y * 0.5f * cam.cols / cam.f;   // the right border through the box centre at the front face
        const Live L = make_live(cam, truth_box());
        const auto pts = mask_cloud(L, truth_box(), true);
        BeamLog lg;
        const bm::BeamRays r = stage(sweep, o, pts, L.label, walls, &lg);
        int outside = 0;
        for (const auto& e : r.endpoints) if (not cam.project(e.cast<float>())) ++outside;
        CountableBelief b = belief({tx, ty, tH, tw, tw, 0.0f});
        Track t;
        for (int k = 0; k < 10; ++k) { b.update(camera_frame(pts, &r)); track(t, b); }
        std::printf("  camera coverage (i): mask %zu px, cloud %zu ; staged %d, left out for no label %d, staged outside the image %d ; over "
                    "10 max |dw| %.5f |dcx| %.5f |dcy| %.5f\n", L.mask.pixels.size(), pts.size(), lg.staged, lg.nolabel, outside,
                    t.max_dw, t.max_dcx, t.max_dcy);
        CHECK(lg.nolabel > 0);
        CHECK(outside == 0);
        CHECK(t.max_dw < kHalfSpacing and t.max_dcx < kHalfSpacing and t.max_dcy < kHalfSpacing);
    }
    {
        const ts::Camera cam = front_camera();
        ts::SceneBox eroded = truth_box();
        eroded.w -= 0.03f; eroded.c.x() -= 0.015f;
        const Live L = make_live(cam, eroded);
        const auto pts = mask_cloud(L, truth_box());
        const bm::BeamRays r = stage(sweep, o, pts, L.label, walls);
        CountableBelief br = belief({tx - 0.015f, ty, tH, tw - 0.03f, tw, 0.0f}), bc = br;
        for (int k = 0; k < 10; ++k) { br.update(camera_frame(pts, &r)); bc.update(camera_frame(pts, nullptr)); }
        const float bw_r = br.state().w - tw, bcx_r = br.state().cx - tx, bw_c = bc.state().w - tw, bcx_c = bc.state().cx - tx;
        std::printf("  camera coverage (ii) mask eroded 3 cm on the right (%zu rays): bias w with rays %+.5f camera-only %+.5f ; bias cx "
                    "with rays %+.5f camera-only %+.5f\n", r.endpoints.size(), bw_r, bw_c, bcx_r, bcx_c);
        CHECK(std::abs(bw_r) <= std::abs(bw_c));
        CHECK(std::abs(bcx_r) <= std::abs(bcx_c));
    }
}

// ─── Gap under furniture, live staging (spec Pass, R6-I3, stated limit 1): the TRUE body raised on a 15 cm gap (legs not
//     modelled), the model floor-anchored, background 1 m, Helios at its height. Against the same run without the gap rays
//     (staged rays whose segment from the sensor to the return enters the gap volume under the box; κ unchanged), cy, h and H
//     each move by less than 1 cm. LiDAR-only frames, 10, from the truth seed. A failure is reported, not tuned. ───
static void gap_under_furniture()
{
    ts::SceneBox raised = truth_box();
    raised.z0 = 0.15f;
    const std::vector<ts::SceneWall> walls{back_wall(1.0f)};
    const std::vector<ts::SceneBox> boxes{raised};
    const ts::Camera cam = front_camera();
    const Live L = make_live(cam, raised);
    const auto anchor = mask_cloud(L, raised);
    const Eigen::Vector3f o(kRobot.x(), kRobot.y(), kHeliosZ);
    const bm::BeamRays r = stage(helios_sweep(o, boxes, walls), o, anchor, L.label, walls);
    std::vector<std::size_t> gap_idx, rest_idx;
    for (std::size_t i = 0; i < r.endpoints.size(); ++i)
    {
        const Eigen::Vector3f e = r.endpoints[i].cast<float>(), u = (e - o).normalized();
        const float t = ts::detail::box_entry(o, u, 0.0f, raised.c, Eigen::Vector3f(-0.5f * tw, -0.5f * tw, 0.0f),
                                              Eigen::Vector3f(0.5f * tw, 0.5f * tw, raised.z0));
        ((t < (e - o).norm()) ? gap_idx : rest_idx).push_back(i);
    }
    const bm::BeamRays rg = ts::subset(r, gap_idx, kEpsLive), rn = ts::subset(r, rest_idx, kEpsLive);
    CountableBelief ba = belief({tx, ty, tH, tw, tw, 0.0f}), bb = ba;
    for (int k = 0; k < 10; ++k) { ba.update(lidar_frame(r)); bb.update(lidar_frame(rn)); }
    const auto gap_energy = [&](const CountableBelief& b)
    {
        const auto parts = b.beam_parts_for_test(b.state());
        Eigen::Matrix<double, 6, 6> W = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
        double E = 0.0;
        bm::accumulate_beam_rays<6>(parts, rg, b.params().beam, W, g, &E);
        return E;
    };
    const float dcy = ba.state().cy - bb.state().cy, dh = ba.state().h - bb.state().h, dH = ba.state().H - bb.state().H;
    std::printf("  gap under furniture: %zu staged, %zu gap rays (kappa %.3f) ; gap-ray energy at theta with them %.3f nats, without them "
                "%.3f nats ; belief gap counter (pi>0.99, e<-5c) %d ; with - without: cy %+.5f h %+.5f H %+.5f (bound 0.01) ; with: cy %+.5f "
                "h %+.5f H %+.5f\n", r.endpoints.size(), gap_idx.size(), r.kappa, gap_energy(ba), gap_energy(bb), ba.last_gap_rays(),
                dcy, dh, dH, ba.state().cy - ty, ba.state().h - tw, ba.state().H - tH);
    CHECK(not gap_idx.empty());
    CHECK(std::abs(dcy) < 0.01f and std::abs(dh) < 0.01f and std::abs(dH) < 0.01f);
}

// ─── Calibration (spec Pass): the silhouette fixture through live staging and the rendered mask, 200 noise draws (per ray a
//     surface offset δ ~ N(0, σ_s): the true box dilated by δ for that ray; range noise N(0, σ_sensor = 0.02 m)), one LiDAR-only
//     update from the truth seed per draw. The latents are zero-variance (sig2_L = 0, Σ_pose = 0) so P is what the beam's
//     curvature and the prior say, with no shared error the draws do not contain. NEES of (cx, w) / 2: mean in [0.5, 2] on the
//     20 cm and the 1 m backgrounds (CHECK), the 5 cm background printed; cx and w alone printed. ───
static void calibration_nees()
{
    const ts::Camera cam = front_camera();
    const Live L = make_live(cam, truth_box());
    const auto anchor = mask_cloud(L, truth_box());
    const Eigen::Vector3f o(kRobot.x(), kRobot.y(), kSilZ);
    const std::vector<ts::SceneBox> boxes{truth_box()};
    const auto rings = ts::helios_ring_elev_deg();
    const int n_az = static_cast<int>(std::lround(360.0f / ts::kHeliosAzStepDeg));
    for (const float gap : {0.20f, 1.0f, 0.05f})
    {
        const std::vector<ts::SceneWall> walls{back_wall(gap)};
        std::mt19937 rng(20261121u + static_cast<unsigned>(gap * 100.0f));
        std::normal_distribution<float> n01(0.0f, 1.0f);
        double s2 = 0.0, scx = 0.0, sw = 0.0;
        int n = 0, conv = 0;
        double nrays = 0.0;
        for (int draw = 0; draw < 200; ++draw)
        {
            std::vector<Eigen::Vector3f> sweep;
            for (const float el : rings)
                for (int k = 0; k < n_az; ++k)
                {
                    const Eigen::Vector3f d = ts::beam_dir(el, static_cast<float>(k) * ts::kHeliosAzStepDeg);
                    const float t = ts::first_hit(o, d, boxes, walls, true, kSigmaS * n01(rng));
                    if (not (t < 12.0f)) continue;
                    sweep.push_back(o + (t + static_cast<float>(rc::kHeliosRangeSigmaM) * n01(rng)) * d);
                }
            const bm::BeamRays r = stage(sweep, o, anchor, L.label, walls);
            nrays += double(r.endpoints.size());
            CountableBelief b = belief({tx, ty, tH, tw, tw, 0.0f});
            CountableFrame f = lidar_frame(r);
            f.sig2_L.fill(0.0f);   // spec 7c.8 rev 3.3.10 (2): shared latents (sig2_L, pose) held at zero variance in the draws
            f.sigma_pose.setZero();
            b.update(f);
            conv += int(b.last_gn_converged());
            const auto& P = b.covariance();
            Eigen::Matrix2d S;
            S << P(0, 0), P(0, 3), P(3, 0), P(3, 3);
            const Eigen::Vector2d e(b.state().cx - tx, b.state().w - tw);
            s2 += e.dot(S.ldlt().solve(e));
            scx += e(0) * e(0) / S(0, 0);
            sw  += e(1) * e(1) / S(1, 1);
            ++n;
        }
        const double nees2 = s2 / n / 2.0;
        std::printf("  calibration bg %.2f m (%d draws, %.0f rays/draw, gn_conv %d/%d): mean NEES(cx, w)/2 %.3f ; cx alone %.3f ; w alone %.3f%s\n",
                    gap, n, nrays / n, conv, n, nees2, scx / n, sw / n, gap < 0.1f ? "  (PRINTED: near background)" : "");
        if (gap > 0.1f) CHECK(nees2 >= 0.5 and nees2 <= 2.0);
    }
}

// ─── A cycle without membership (spec 7c.8 a0, owner 11-18): no label ⇒ nothing staged, the log counter counts it, and the
//     camera update of that cycle is exactly the camera-only update (P untouched by rays). No sweep is logged apart (code 2,
//     not counted). With a label the same sweep stages rays and changes P (the control). ───
static void no_membership_cycle_skips_rays()
{
    const std::vector<ts::SceneWall> walls{back_wall(1.0f)};
    const std::vector<ts::SceneBox> boxes{truth_box()};
    const Eigen::Vector3f o(kRobot.x(), kRobot.y(), kSilZ);
    const auto sweep = helios_sweep(o, boxes, walls);
    const Live L = make_live(front_camera(), truth_box());
    const auto pts = mask_cloud(L, truth_box());
    const BeamStaging cfg = live_staging(walls);
    BeamLog log;
    bm::BeamRays r0, r1, rl;
    log.begin_cycle(); stage_beam_cycle(&sweep, o, pts, cfg, BeamLabel{}, r0, log);
    const int code1 = log.nomem_cycle; const long c1 = log.nomem_cycles;
    log.begin_cycle(); stage_beam_cycle(&sweep, o, pts, cfg, BeamLabel{}, r1, log);
    const long c2 = log.nomem_cycles;
    log.begin_cycle(); stage_beam_cycle(nullptr, o, pts, cfg, L.label, r1, log);
    const int code_nosweep = log.nomem_cycle; const long c3 = log.nomem_cycles;
    // An empty slice: no membership AND no mask cloud — still a no-membership cycle, counted (review I1).
    log.begin_cycle(); stage_beam_cycle(&sweep, o, std::span<const Eigen::Vector3f>{}, cfg, BeamLabel{}, r1, log);
    const int code_empty = log.nomem_cycle; const long c4 = log.nomem_cycles;
    log.begin_cycle(); stage_beam_cycle(&sweep, o, pts, cfg, L.label, rl, log);
    const int code_ok = log.nomem_cycle;
    CountableBelief a = belief({tx, ty, tH, tw + 0.01f, tw, 0.0f}), c = a, d = a;
    a.update(camera_frame(pts, &r0));
    c.update(camera_frame(pts, nullptr));
    d.update(camera_frame(pts, &rl));
    const bool same = a.state().vec() == c.state().vec() and a.covariance() == c.covariance();
    const float dP = (d.covariance() - c.covariance()).norm();
    std::printf("  no membership: staged %zu (code %d, counter %ld -> %ld) ; no sweep code %d counter %ld ; with a label staged %zu (code %d) ; "
                "camera update identical to camera-only %d ; |P_rays - P_cam| %.3e ; empty slice code %d counter %ld\n", r0.endpoints.size(), code1,
                c1, c2, code_nosweep, c3, rl.endpoints.size(), code_ok, int(same), dP, code_empty, c4);
    CHECK(r0.endpoints.empty() and r0.q_sum == 0.0);
    CHECK(code1 == BeamLog::kNoMembership and c1 == 1 and c2 == 2);
    CHECK(code_nosweep == BeamLog::kNoSweep and c3 == 2 and r1.endpoints.empty());
    CHECK(code_empty == BeamLog::kNoMembership and c4 == 3 and r1.endpoints.empty());
    CHECK(code_ok == BeamLog::kMembership and not rl.endpoints.empty());
    CHECK(not a.lidar_measured(camera_frame(pts, &r0)));
    CHECK(same);
    CHECK(dP > 0.0f);
}

// ─── Staging cost (spec Pass "Cost", printed): one full Helios sweep through stage_beam_cycle (select_rays_beam by ray
//     geometry, one label per candidate, κ, p_f), median of 21 runs; the mask raster (MaskMembership::build) printed beside. ───
static void staging_cost_printed()
{
    const std::vector<ts::SceneWall> walls{back_wall(1.0f)};
    const std::vector<ts::SceneBox> boxes{truth_box()};
    const Eigen::Vector3f o(kRobot.x(), kRobot.y(), kHeliosZ);
    const auto sweep = helios_sweep(o, boxes, walls);
    const ts::Camera cam = front_camera();
    const Live L = make_live(cam, truth_box());
    const auto pts = mask_cloud(L, truth_box());
    const BeamStaging cfg = live_staging(walls);
    std::vector<double> ms, ms_mm;
    BeamLog log;
    bm::BeamRays r;
    for (int k = 0; k < 21; ++k)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const MaskMembership mm = MaskMembership::build(L.mask.pixels);
        const auto t1 = std::chrono::steady_clock::now();
        log.begin_cycle();
        stage_beam_cycle(&sweep, o, pts, cfg, L.label, r, log);
        const auto t2 = std::chrono::steady_clock::now();
        ms_mm.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() + 0.0 * mm.stride_px());
        ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    std::ranges::sort(ms); std::ranges::sort(ms_mm);
    std::printf("  staging cost (PRINTED): sweep %zu returns -> staged %d, nolabel %d, kappa %.3f ; stage_beam_cycle median %.3f ms ; mask "
                "raster (%zu px) median %.3f ms\n", sweep.size(), log.staged, log.nolabel, log.kappa, ms[ms.size() / 2],
                L.mask.pixels.size(), ms_mm[ms_mm.size() / 2]);
    CHECK(log.staged > 0);
}

// ─── One-sided view (spec 7c.8 rev 3.3.11 Pass, bounds per rev 3.3.12; κ ≡ 1): the box seen from ONE face only — the robot on
//     its +x side, 1.5 m from the +x face, looking −x (ZED heading π), the Helios at 0.40 m (below the 0.62 m top, so no ray sees
//     the top face), a known wall 1 m behind the far (−x) face. Live staging, live ε_px. LiDAR-only frames. The far face
//     x = cx − w/2 is never hit; the near face cx + w/2 is.
//     (i) 10 LiDAR-only frames from the truth seed: σ of the far face given the near face (= σ_w) ≥ ½ the prior σ_w (CHECK,
//         rev 3.3.12 (1)); σ per DOF, the prior σ, the far-face position σ and its σ conditional on the near face printed.
//     (ii) Calibration (rev 3.3.10 (2) protocol: 200 draws, per-ray surface offset δ ~ N(0, σ_s) and range noise
//         N(0, σ_sensor), one LiDAR-only update per draw, shared latents sig2_L and Σ_pose at zero variance). Each draw's prior
//         mean for w is drawn from the w prior, w₀ = tw + σ_w,prior·n (rev 3.3.12 (3)); the other seed components at the truth.
//         CHECK NEES over the constrained directions (near face cx + w/2, cy, H, h, yaw) / 5 ∈ [0.5, 2] (rev 3.3.12 (2)) and
//         NEES(w) ∈ [0.5, 2] (rev 3.3.12 (3)); each component's NEES, (cy, h)/2 and (cx, w)/2 printed. A failure is reported,
//         not tuned. ───
static void one_sided_view()
{
    const float x_near = tx + 0.5f * tw;
    const Eigen::Vector2f robot(x_near + 1.5f, ty);
    ts::Camera cam;
    cam.pos = Eigen::Vector3f(robot.x() + 0.075f, robot.y(), 0.945f);
    cam.heading = std::numbers::pi_v<float>;              // looks along −x; image right = +y
    const Live L = make_live(cam, truth_box());
    std::vector<Eigen::Vector3f> anchor;                  // the +x face points inside the rendered mask
    for (int i = 0; i < 24; ++i) for (int j = 0; j < 12; ++j)
    {
        const Eigen::Vector3f q(x_near, ty - 0.5f * tw + tw * (i + 0.5f) / 24.0f, tH * (j + 0.5f) / 12.0f);
        const auto pr = cam.project(q);
        if (pr and L.mm.signed_distance_px(pr->x(), pr->y()) > 0.0f) anchor.push_back(q);
    }
    const float xw = tx - 0.5f * tw - 1.0f;
    const std::vector<ts::SceneWall> walls{{Eigen::Vector2f(xw, -4.0f), Eigen::Vector2f(xw, 4.0f), 2.5f}};
    const std::vector<ts::SceneBox> boxes{truth_box()};
    const Eigen::Vector3f o(robot.x(), robot.y(), kSilZ);
    const auto side_frame = [&](const bm::BeamRays& r) { CountableFrame f = lidar_frame(r); f.robot_xy = robot; return f; };
    const CountableBeliefParams p = params();
    const Vec6 prior_sd = (Vec6() << p.prior_pos_std, p.prior_pos_std, p.prior_height_std, p.prior_footprint_std, p.prior_footprint_std,
                           p.prior_yaw_std).finished();
    // (i)
    {
        BeamLog lg;
        const bm::BeamRays r = stage(helios_sweep(o, boxes, walls), o, anchor, L.label, walls, &lg);
        int far_hits = 0;                                  // staged returns on the far face (the fixture's premise: 0)
        for (const auto& e : r.endpoints) if (std::abs(e.x() - (tx - 0.5 * tw)) < 1e-3 and std::abs(e.y() - ty) < 0.5 * tw) ++far_hits;
        CountableBelief b = belief({tx, ty, tH, tw, tw, 0.0f});
        for (int k = 0; k < 10; ++k) b.update(side_frame(r));
        const auto& P = b.covariance();
        const Vec6 sd = P.diagonal().cwiseSqrt();
        const double s_far = std::sqrt(double(P(0, 0)) + 0.25 * double(P(3, 3)) - double(P(0, 3)));
        const double s_far0 = std::sqrt(double(prior_sd(0) * prior_sd(0)) + 0.25 * double(prior_sd(3) * prior_sd(3)));
        // the far face given the near face: var(far) − cov(far, near)² / var(near)
        const double v_near = double(P(0, 0)) + 0.25 * double(P(3, 3)) + double(P(0, 3));
        const double c_fn = double(P(0, 0)) - 0.25 * double(P(3, 3));
        const double s_far_given_near = std::sqrt(std::max(0.0, s_far * s_far - c_fn * c_fn / v_near));
        std::printf("  one-sided view: mask %zu px, anchor %zu pts ; staged %d nolabel %d (old kappa %.4f) q_sum %.2f ; far-face hits %d ; "
                    "gn %d/%d after 10 frames\n", L.mask.pixels.size(), anchor.size(), lg.staged, lg.nolabel, lg.kappa, lg.qsum, far_hits,
                    b.last_gn_iters(), int(b.last_gn_converged()));
        std::printf("  one-sided view sigma per DOF (cx, cy, H, w, h, yaw) after 10 frames: %.5f %.5f %.5f %.5f %.5f %.5f ; prior %.3f %.3f "
                    "%.3f %.3f %.3f %.3f ; far face given near (= sigma_w) %.5f, / prior sigma_w %.3f (bound >= 0.5) ; conditional "
                    "sigma(far | near) %.5f ; far-face position cx - w/2 sigma %.5f (its prior %.4f, ratio %.3f) ; "
                    "theta - truth cx %+.5f cy %+.5f H %+.5f w %+.5f h %+.5f yaw %+.5f\n", sd(0), sd(1), sd(2), sd(3), sd(4), sd(5),
                    prior_sd(0), prior_sd(1), prior_sd(2), prior_sd(3), prior_sd(4), prior_sd(5), sd(3), sd(3) / prior_sd(3),
                    s_far_given_near, s_far, s_far0, s_far / s_far0, b.state().cx - tx, b.state().cy - ty, b.state().H - tH, b.state().w - tw, b.state().h - tw,
                    b.state().yaw);
        CHECK(lg.staged > 0 and far_hits == 0);
        CHECK(sd(3) >= 0.5f * prior_sd(3));
    }
    // (ii)
    {
        const auto rings = ts::helios_ring_elev_deg();
        const int n_az = static_cast<int>(std::lround(360.0f / ts::kHeliosAzStepDeg));
        std::mt19937 rng(20261122u), rng_w(20261123u);
        std::normal_distribution<float> n01(0.0f, 1.0f), nw(0.0f, 1.0f);
        double s_cyh = 0.0, s_cxw = 0.0, s_con = 0.0, nrays = 0.0;
        Eigen::Matrix<double, 6, 1> alone = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 5, 1> alone_con = Eigen::Matrix<double, 5, 1>::Zero();
        int n = 0, conv = 0;
        const auto nees2 = [](const Eigen::Matrix<float, 6, 6>& P, int i, int j, double ei, double ej)
        {
            Eigen::Matrix2d S;
            S << P(i, i), P(i, j), P(j, i), P(j, j);
            const Eigen::Vector2d e(ei, ej);
            return e.dot(S.ldlt().solve(e));
        };
        // the constrained directions: near face cx + w/2, cy, H, h, yaw (rows of A over θ = (cx, cy, H, w, h, yaw))
        Eigen::Matrix<double, 5, 6> A = Eigen::Matrix<double, 5, 6>::Zero();
        A(0, 0) = 1.0; A(0, 3) = 0.5; A(1, 1) = 1.0; A(2, 2) = 1.0; A(3, 4) = 1.0; A(4, 5) = 1.0;
        double w0_sum = 0.0, w0_sq = 0.0;
        for (int draw = 0; draw < 200; ++draw)
        {
            std::vector<Eigen::Vector3f> sweep;
            for (const float el : rings)
                for (int k = 0; k < n_az; ++k)
                {
                    const Eigen::Vector3f d = ts::beam_dir(el, static_cast<float>(k) * ts::kHeliosAzStepDeg);
                    const float t = ts::first_hit(o, d, boxes, walls, true, kSigmaS * n01(rng));
                    if (not (t < 12.0f)) continue;
                    sweep.push_back(o + (t + static_cast<float>(rc::kHeliosRangeSigmaM) * n01(rng)) * d);
                }
            const bm::BeamRays r = stage(sweep, o, anchor, L.label, walls);
            nrays += double(r.endpoints.size());
            const float w0 = tw + prior_sd(3) * nw(rng_w);   // rev 3.3.12 (3): this draw's prior mean for w, from the w prior
            w0_sum += double(w0 - tw); w0_sq += double(w0 - tw) * double(w0 - tw);
            CountableBelief b = belief({tx, ty, tH, w0, tw, 0.0f});
            CountableFrame f = side_frame(r);
            f.sig2_L.fill(0.0f);   // rev 3.3.10 (2): shared latents (sig2_L, pose) at zero variance in the draws
            f.sigma_pose.setZero();
            b.update(f);
            conv += int(b.last_gn_converged());
            const Eigen::Matrix<double, 6, 6> P = b.covariance().cast<double>();
            const CountableBeliefState& st = b.state();
            const Eigen::Matrix<double, 6, 1> e = (st.vec() - CountableBeliefState{tx, ty, tH, tw, tw, 0.0f}.vec()).cast<double>();
            s_cyh += nees2(b.covariance(), 1, 4, e(1), e(4));
            s_cxw += nees2(b.covariance(), 0, 3, e(0), e(3));
            const Eigen::Matrix<double, 5, 1> ec = A * e;
            const Eigen::Matrix<double, 5, 5> Pc = A * P * A.transpose();
            s_con += ec.dot(Pc.ldlt().solve(ec));
            for (int i = 0; i < 5; ++i) alone_con(i) += ec(i) * ec(i) / Pc(i, i);
            for (int i = 0; i < 6; ++i) alone(i) += e(i) * e(i) / P(i, i);
            ++n;
        }
        const double nees_con = s_con / n / 5.0;
        alone /= double(n);
        alone_con /= double(n);
        std::printf("  one-sided calibration (%d draws, %.0f rays/draw, gn_conv %d/%d ; seed w0 - truth mean %+.4f sd %.4f, prior sigma_w %.3f): "
                    "mean NEES(near, cy, H, h, yaw)/5 %.3f (bound [0.5, 2]) ; NEES(w) %.3f (bound [0.5, 2])\n", n, nrays / n, conv, n,
                    w0_sum / n, std::sqrt(std::max(0.0, w0_sq / n - (w0_sum / n) * (w0_sum / n))), prior_sd(3), nees_con, alone(3));
        std::printf("  one-sided calibration components (PRINTED): near face cx + w/2 %.3f cy %.3f H %.3f h %.3f yaw %.3f ; alone cx %.3f "
                    "cy %.3f H %.3f w %.3f h %.3f yaw %.3f ; NEES(cy, h)/2 %.3f ; NEES(cx, w)/2 %.3f\n", alone_con(0), alone_con(1),
                    alone_con(2), alone_con(3), alone_con(4), alone(0), alone(1), alone(2), alone(3), alone(4), alone(5), s_cyh / n / 2.0,
                    s_cxw / n / 2.0);
        CHECK(nees_con >= 0.5 and nees_con <= 2.0);
        CHECK(alone(3) >= 0.5 and alone(3) <= 2.0);
    }
}


// Model.GnTrace (diagnostic instrument): one camera + beam update from an offset seed with the trace on records one row per
// GN step and 2 linearisations x (camera, LiDAR, θ-only, all-along-θ) x 6 components of the finite-difference check, every
// value finite. The table is PRINTED (not bounded): it is the instrument's offline reading of G / W against the energy.
static void gn_trace_instrument(bool with_fresh = false)
{
    const ts::Camera cam = front_camera();
    const std::vector<ts::SceneBox> boxes{truth_box()};
    const Live L = make_live(cam, truth_box());
    const auto anchor = mask_cloud(L, truth_box());
    const std::vector<ts::SceneWall> walls{back_wall(0.20f)};
    const Eigen::Vector3f o(kRobot.x(), kRobot.y(), kSilZ);
    const bm::BeamRays r = stage(helios_sweep(o, boxes, walls), o, anchor, L.label, walls);
    CountableBeliefParams p = params();
    p.gn_trace = 2;
    CountableBelief b({tx + 0.03f, ty, tH, tw + 0.05f, tw, 0.0f}, p);
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    b.set_first_update_done(true);
    CountableFrame fr = camera_frame(anchor, &r);
    if (with_fresh)
    {   // the ego-motion share of a moving robot (live13trace motion_var 1e-4 … 5e-4 m²)
        fr.fresh_cam = {2e-4f, 2e-4f, 2e-4f, 2e-4f, 2e-4f, 1e-4f};
        fr.fresh_L   = fr.fresh_cam;
    }
    b.update(fr);
    const auto& st = b.dbg_gn_steps();
    const auto& fd = b.dbg_gn_fd();
    std::printf("  gn trace (fresh %s): gn %d/%d, %zu step rows, %zu fd rows\n", with_fresh ? "on" : "0", b.last_gn_iters(), int(b.last_gn_converged()), st.size(), fd.size());
    CHECK(st.size() >= std::size_t(b.last_gn_iters()) and st.size() <= std::size_t(b.last_gn_iters()) + 1);
    CHECK(fd.size() == 2u * 4u * 6u);
    for (const auto& x : st) CHECK(std::isfinite(x.J_before) and x.d.allFinite());
    const char* ch[] = {"cam", "L", "th", "all"};
    for (const auto& x : fd)
    {
        CHECK(std::isfinite(x.grad_an) and std::isfinite(x.grad_fd3) and std::isfinite(x.grad_fd4) and std::isfinite(x.curv_fd));
        std::printf("    at %d %-3s k%d grad an %+.4e marg %+.4e fd3 %+.4e fd4 %+.4e | curv an %.4e fd %.4e\n", x.at, ch[x.channel],
                    x.k, x.grad_an, x.grad_marg, x.grad_fd3, x.grad_fd4, x.curv_an, x.curv_fd);
    }
    // spec 7c.8 rev 3.3.13 Pass (1): the solver's joint G_θ is the gradient of J — ratio G / FD in [0.9, 1.1] on every
    // component whose |FD| is at least 1 % of the largest (per linearisation point).
    for (int at = 0; at < 2; ++at)
    {
        double big = 0.0;
        for (const auto& x : fd) if (x.channel == 3 and x.at == at) big = std::max(big, std::abs(x.grad_fd3));
        for (const auto& x : fd)
            if (x.channel == 3 and x.at == at and std::abs(x.grad_fd3) >= 0.01 * big)
            {
                const double ratio = x.grad_an / x.grad_fd3;
                if (not (ratio >= 0.9 and ratio <= 1.1)) std::printf("    G/FD out of band: at %d k%d ratio %.4f\n", at, x.k, ratio);
                CHECK(ratio >= 0.9 and ratio <= 1.1);
            }
    }
}


// spec 7c.8 rev 3.3.14: the birth cloud's excess position covariance enters the first update's class prior on (cx, cy),
// once: P̄_θθ(0:2, 0:2) = σ_pos² 𝟙 + excess; the second update no longer carries it.
static void birth_excess_enters_first_update()
{
    const ts::Camera cam = front_camera();
    const Live L = make_live(cam, truth_box());
    const auto anchor = mask_cloud(L, truth_box());
    CountableBelief b({tx, ty, tH, tw, tw, 0.0f}, params());
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    Eigen::Matrix2f ex; ex << 0.01f, 0.004f, 0.004f, 0.12f;
    b.set_birth_position_cov(ex);
    b.update(camera_frame(anchor, nullptr));
    const float pp = params().prior_pos_std * params().prior_pos_std;
    const Eigen::Matrix2f got = b.dbg_Pbar_tt().topLeftCorner<2, 2>();
    const Eigen::Matrix2f want = pp * Eigen::Matrix2f::Identity() + ex;
    std::printf("  birth excess: Pbar(cx,cy) %.5f %.5f / %.5f %.5f (want %.5f %.5f / %.5f %.5f) ; after: excess %.3e\n",
                got(0, 0), got(0, 1), got(1, 0), got(1, 1), want(0, 0), want(0, 1), want(1, 0), want(1, 1),
                b.birth_position_excess().norm());
    CHECK((got - want).cwiseAbs().maxCoeff() < 1e-6f);
    CHECK(b.birth_position_excess().isZero(0.0f));
}

int main()
{
    std::printf("beam_live_staging_test (spec 7c.8 Pass, live staging)\n");
    no_membership_cycle_skips_rays();
    staging_cost_printed();
    silhouette_live();
    flush_neighbour_printed();
    camera_coverage();
    gap_under_furniture();
    calibration_nees();
    one_sided_view();
    gn_trace_instrument(false);
    gn_trace_instrument(true);
    birth_excess_enters_first_update();
    std::printf(bad ? "beam_live_staging_test: %d FAILED\n" : "beam_live_staging_test: OK\n", bad);
    return bad != 0;
}
