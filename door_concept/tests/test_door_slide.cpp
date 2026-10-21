// try96 (2026-10-19): door_1 slid 1.1 m ALONG ITS WALL while room_1's localisation was wrong (heading 10-20°, position
// 0.3-0.6 m for ~30 s, gt_error.csv t≈274-303 s), then read phi 0 / P(open) 0 for a door standing open. Offline harness for
// the along-wall channel of DoorFitter::estimate_phi — rc::door::register_sweep, measure_openings, along_wall_update
// (door_opening_edges.h) and DoorBelief::fuse_along_wall / update_sharing_offset — on a synthetic ray-cast scene with
// try96's geometry: room_1 seen from the side the leaf opens into, the leaf open 90° on the far (x = W) jamb, the bpearl
// dome at 0.67 m, room_1's inside corner 0.85 m beyond the hinge jamb, the robot on try96's leg-3 path 3.3-5 m away.
// The room-frame cloud is produced through an ERRONEOUS pose (rotation about the sensor + translation), as live.
//
//   (a) try96 replay: the OLD channel (Task 15: the field's slide marginal fused into s every frame) slides the aperture
//       (> 0.3 m); the NEW channel holds it (< 0.10 m) through the bad stretch and reads the door OPEN afterwards.
//   (b) a door BORN at a wrong offset (0.8 m, σ_s at its prior 0.6 m), the pose good: converges onto the measured opening.
//   (c) a door CONFIDENTLY wrong (0.35 m, σ_s 3 cm), the pose good: still converges onto the measured opening.
//   (d) register_to_wall recovers an injected rotation / across offset of the scan.
//   (e) a SHUT door with a second opening 1.3 m along the same wall: the aperture is not dragged onto it.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tests/test_door_slide.cpp src/door_belief.cpp -o /tmp/test_door_slide && /tmp/test_door_slide
// SLIDE_TRACE=1 prints every frame.
#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>
#include <numbers>
#include <print>
#include <random>
#include <string>
#include <vector>

#include "../src/door_belief.h"
#include "../src/door_leaf_channel.h"   // try102: the leaf channel DoorFitter::estimate_phi runs
#include "../src/door_phi_posterior.h"
#include "../src/door_opening_edges.h"
#include "../src/door_passability.h"

using V2 = Eigen::Vector2f;
using V3 = Eigen::Vector3f;

static int g_fail = 0;
static void check(bool ok, const std::string& what)
{
    std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++g_fail;
}

// ─── the scene (the harness frame of test_door_phi_stability): wall face y = 0, robot side y > 0, doorway x ∈ [0, W] ───
// try96's room_1 maps onto it by x = −X − 0.05, y = 4.62 − Y (a 180° turn, handedness kept).
struct Box { V2 c; V2 ex; float hx, hy, z0, z1; };
static Box aabb(float x0, float x1, float y0, float y1, float z0, float z1)
{ return {V2(0.5f * (x0 + x1), 0.5f * (y0 + y1)), V2(1, 0), 0.5f * (x1 - x0), 0.5f * (y1 - y0), z0, z1}; }
static float ray_box(const V3& o, const V3& u, const Box& b)
{
    const V2 ey(-b.ex.y(), b.ex.x());
    const V3 c(b.c.x(), b.c.y(), 0.5f * (b.z0 + b.z1));
    const V3 ax[3] = {V3(b.ex.x(), b.ex.y(), 0), V3(ey.x(), ey.y(), 0), V3(0, 0, 1)};
    const float h[3] = {b.hx, b.hy, 0.5f * (b.z1 - b.z0)};
    const V3 d = c - o;
    float tmin = 0, tmax = 1e9f;
    for (int i = 0; i < 3; ++i)
    {
        const float e = ax[i].dot(d), f = ax[i].dot(u);
        if (std::abs(f) > 1e-7f)
        {
            float t1 = (e - h[i]) / f, t2 = (e + h[i]) / f;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1); tmax = std::min(tmax, t2);
            if (tmin > tmax) return -1;
        }
        else if (-e - h[i] > 0 or -e + h[i] < 0) return -1;
    }
    return tmin > 0 ? tmin : -1;
}

struct Scene
{
    float T = 0.11f, W = 1.0f, H = 2.1f;
    float phi = 1.5708f;         // the TRUE leaf angle
    float swing = +1.0f;         // +1: into the robot's room (try96: DOOR_1 opens into room_1)
    bool  hinge_far = true;      // hinged on the x = W jamb (try96: the west jamb, the belief's "far")
    float corner_x = 1.85f;      // room_1's inside corner (the notch) beyond the hinge jamb; <= 0: none
    bool  second_hole = false;   // (e): a second opening x ∈ [2.3, 3.2] in the same wall
    std::vector<Box> boxes() const
    {
        std::vector<Box> b;
        if (second_hole)
        {
            b.push_back(aabb(-6, 0, -T, 0, 0, 3));
            b.push_back(aabb(W, 2.3f, -T, 0, 0, 3));
            b.push_back(aabb(3.2f, 6, -T, 0, 0, 3));
            b.push_back(aabb(2.3f, 3.2f, -T, 0, H, 3));
        }
        else
        {
            b.push_back(aabb(-6, 0, -T, 0, 0, 3));
            b.push_back(aabb(W, 6, -T, 0, 0, 3));
        }
        b.push_back(aabb(0, W, -T, 0, H, 3));                 // lintel
        b.push_back(aabb(-6, 6, -4.2f, -4.0f, 0, 3));         // the next room's far wall + side walls
        b.push_back(aabb(-3.2f, -3.0f, -4.2f, -T, 0, 3));
        b.push_back(aabb(4.0f, 4.2f, -4.2f, -T, 0, 3));
        if (corner_x > 0.f) b.push_back(aabb(corner_x, 6.0f, 0.0f, 1.69f, 0, 3));   // room_1's notch
        const float y_h = swing < 0 ? -T : 0.0f;
        const V2 dir(hinge_far ? -std::cos(phi) : std::cos(phi), swing * std::sin(phi));
        const float lw = W - 0.01f;
        b.push_back({V2(hinge_far ? W : 0.0f, y_h) + 0.5f * lw * dir, dir, 0.5f * lw, 0.02f, 0.01f, H - 0.01f});
        return b;
    }
};

// The bpearl-like dome (720 azimuths × 32 rings, horizontal to straight down, 1 cm noise, a no-return beyond 8 m),
// half the rays kept; then the cloud is put in the ROOM frame through an ERRONEOUS pose: rotated about the sensor by
// `eps` and translated by `t` — what a mislocalised robot publishes. Returns the cloud and the published sensor origin.
struct Cloud { std::vector<V3> pts; V3 origin; };
static Cloud scan(const Scene& s, const V3& o, std::mt19937& rng, float eps = 0.f, V2 t = V2::Zero())
{
    std::normal_distribution<float> n(0, 0.01f);
    std::uniform_real_distribution<float> U(0, 1);
    const auto bx = s.boxes();
    Cloud c;
    c.origin = V3(o.x() + t.x(), o.y() + t.y(), o.z());
    const float ce = std::cos(eps), se = std::sin(eps);
    for (int a = 0; a < 720; ++a)
        for (int e = 0; e < 32; ++e)
        {
            if (U(rng) > 0.5f) continue;
            const float az = a * 0.5f * std::numbers::pi_v<float> / 180.f;
            const float el = -e * (90.f / 31.f) * std::numbers::pi_v<float> / 180.f;
            const V3 u(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
            float tt = 50;
            if (u.z() < -1e-6f) tt = std::min(tt, -o.z() / u.z());
            for (const auto& b : bx) { const float tb = ray_box(o, u, b); if (tb > 0) tt = std::min(tt, tb); }
            if (tt > 8.0f) continue;
            const V3 p = o + (tt + n(rng)) * u;
            const V2 d(p.x() - o.x(), p.y() - o.y());
            c.pts.emplace_back(c.origin.x() + ce * d.x() - se * d.y(), c.origin.y() + se * d.x() + ce * d.y(), p.z());
        }
    return c;
}

// ─── the model door: wall frame O = (−2, 0), u = +x ⇒ the true doorway is s ∈ [2, 3] ───────────────────────────────────
constexpr float S_TRUE = 2.0f;
static rc::DoorBelief make_belief(float s0, float sd_s, float phi0, rc::door::HingeSide hinge = rc::door::HingeSide::Far,
                                  float swing = +1.0f)
{
    rc::DoorBeliefParams p;
    p.wall_O = {-2, 0}; p.wall_u = {1, 0}; p.thickness = 0.05f; p.floor_z = 0;
    p.leaf = {phi0, hinge, swing};
    p.prior_s_std = sd_s;
    p.process_std_s = 0.005f;   // the agent's AI2ProcessStdM: what each mask frame injects
    return rc::DoorBelief({s0, 1.0f, 2.1f}, p);
}

constexpr float PHI_MIN = 0.0f, PHI_MAX = 2.0944f, PHI_STEP = 0.0873f, DT = 0.1f;
constexpr int   NSTEP = 25;

struct Tracker
{
    float hinge_llr = 5.f, swing_llr = 5.f, phi_est = 1.5f, phi_sigma = 0.1f;
    float p = 0.f;
    rc::door::AlongWallState along;
    rc::door::WallRegistration reg;
    rc::door::AlongWallResult aw;
    rc::door::LeafChannel leaf;   // try102: as DoorInstance::leaf_channel
};

// One cycle of DoorFitter::estimate_phi's LiDAR path. `old_channel`: Task 15 (no registration, the field's slide marginal
// fused into s every frame) — the code that ran in try96. Otherwise: register → select → field → posterior → the opening's
// span with the persistent registration offset (σ0 = 0.02/m · range, the agent's DoorModel.AI2RangeNoiseLatPerM).
static void step(Tracker& t, rc::DoorBelief& b, const Cloud& c, bool old_channel)
{
    auto nz = rc::door::phi_field_nuisance(b.aperture());
    nz.lat_sigma = std::sqrt(std::max(1e-8f, b.covariance()(0, 0)));
    nz.other_hinge = nz.other_swing = true;
    rc::door::RegisteredSweep rs;
    if (old_channel) { rs.pts = c.pts; rs.origin = c.origin; }
    else rs = rc::door::register_sweep(b.aperture(), c.origin, c.pts);
    t.reg = rs.reg;
    const auto sel = rc::door::select_doorway_rays(b.aperture(), rs.origin, rs.pts, std::max(-nz.lat_lo, nz.lat_hi), 1.0f);
    rc::door::LeafChannelInput in;
    if (not sel.rays.endpoints.empty())
    {
        auto fr = b.phi_field_loglik(sel.rays, PHI_MIN, PHI_MAX, NSTEP, nz);
        if (old_channel)
            if (const auto so = rc::door::slide_observation(fr.lat_loglik); so.var > 0.f)
                b.fuse_slide(so.mean, so.var);
        in.bb = std::move(fr.loglik); in.ob = std::move(fr.loglik_other); in.bs = std::move(fr.loglik_swing); in.os = std::move(fr.loglik_both);
    }
    in.have_sensor = true; in.sensor = c.origin.head<2>(); in.dt = DT;
    in.phi_min = PHI_MIN; in.phi_max = PHI_MAX; in.nstep = NSTEP; in.view_rad = PHI_STEP;
    const auto post = rc::door::leaf_channel_step(t.leaf, b, t.hinge_llr, t.swing_llr, t.phi_est, t.phi_sigma, in).post;
    if (not old_channel)
    {
        const auto ap = b.aperture();
        const float range = (ap.centre_xy() - rs.origin.head<2>()).norm();
        t.aw = rc::door::along_wall_update(b, t.along, rc::door::measure_openings(ap, rs.origin, rs.pts), rs.reg,
                                           rc::door::clear_fraction(post.curve), (0.02f * range) * (0.02f * range), range, DT);
    }
    b.set_leaf_phi(t.phi_est);
    t.p = rc::door::passable_prob(post.curve, b.aperture().w, 0.70f).value_or(-1.0f);
}

static void trace(int f, const V3& o, const Tracker& t, const rc::DoorBelief& b)
{
    if (not std::getenv("SLIDE_TRACE")) return;
    std::print("    f{:3d} o ({:.2f},{:.2f}) s {:.3f} sd {:.3f} phi {:.2f} P {:.2f} | reg {} eps {:+.3f} dep {:+.3f} | runs {} z {:.3f} "
               "beta {:.2f} b {:+.3f} sd_b {:.3f}\n", f, o.x(), o.y(), b.state().s, std::sqrt(b.covariance()(0, 0)), t.phi_est,
               t.p, t.reg.ok, t.reg.eps, t.reg.depth, t.aw.n_runs, t.aw.z, t.aw.beta, t.along.b, std::sqrt(t.along.Pb));
}

// try96's leg-3 path in room_1 (phi trace origins t = 262 … 303 s), mapped into the harness frame.
static std::vector<V2> try96_path()
{
    const std::vector<V2> room1 = {{-2.10f, -0.43f}, {-1.88f, -0.51f}, {-1.69f, -0.20f}, {-1.39f, -0.18f}, {-1.26f, -0.16f},
                                   {-1.32f, -0.03f}, {-1.17f, 0.23f},  {-1.07f, 0.45f},  {-1.13f, 0.61f},  {-1.20f, 0.92f},
                                   {-1.23f, 1.30f},  {-1.23f, 1.30f},  {-1.31f, 1.14f},  {-1.11f, 0.94f},  {-0.83f, 0.67f}};
    std::vector<V2> h;
    for (const auto& p : room1) h.emplace_back(-p.x() - 0.05f, 4.62f - p.y());
    return h;
}

struct Replay { float max_dev = 0.f, final_dev = 0.f, final_phi = 0.f, final_p = 0.f; };
static Replay replay_try96(bool old_channel, std::mt19937& rng)
{
    Scene sc;
    auto b = make_belief(S_TRUE, 0.006f, 1.5f);   // live: σ_s 5-6 mm before the slide
    Tracker t;
    const auto path = try96_path();
    constexpr int PER = 25;   // frames per waypoint: 350 frames, 35 s at 10 Hz
    Replay r;
    int f = 0;
    for (std::size_t k = 0; k + 1 < path.size(); ++k)
        for (int j = 0; j < PER; ++j, ++f)
        {
            const V2 P = path[k] + (path[k + 1] - path[k]) * (static_cast<float>(j) / PER);
            const V3 o(P.x(), P.y(), 0.67f);
            // The pose error (gt_error.csv, room_1 → harness axes): from frame 75 it ramps over 5 s to a 14° heading and
            // (0.53, −0.36) m, holds ~17 s, then the relocaliser fixes it (t ≈ 303 s) to ~1°, (−0.1, −0.29) m.
            float eps = 0.f; V2 tr = V2::Zero();
            if (f >= 75 and f < 300) { const float a = std::min(1.f, (f - 75) / 50.f); eps = -0.25f * a; tr = V2(0.53f, -0.36f) * a; }
            else if (f >= 300) { eps = -0.02f; tr = V2(-0.1f, -0.29f); }
            if (f % 30 == 0) b.predict();   // a mask frame's process noise (live: 3 masks in the stretch)
            step(t, b, scan(sc, o, rng, eps, tr), old_channel);
            trace(f, o, t, b);
            r.max_dev = std::max(r.max_dev, std::abs(b.state().s - S_TRUE));
        }
    r.final_dev = std::abs(b.state().s - S_TRUE);
    r.final_phi = t.phi_est;
    r.final_p = t.p;
    return r;
}

int main()
{
    std::setlocale(LC_ALL, "");
    std::mt19937 rng(96);

    std::print("(a) try96 replay: open door, robot 3.3-5 m away, room pose wrong by up to 14° / 0.64 m for 22 s\n");
    {
        const auto old_r = replay_try96(true, rng);
        std::print("  OLD (Task 15 slide fusion): max |s - truth| {:.3f} m, final {:.3f} m, phi {:.0f} deg, P(passable) {:.2f}\n",
                   old_r.max_dev, old_r.final_dev, old_r.final_phi * 57.3f, old_r.final_p);
        check(old_r.max_dev > 0.3f, std::format("the old channel reproduces the slide (max {:.2f} m > 0.3)", old_r.max_dev));
        const auto new_r = replay_try96(false, rng);
        std::print("  NEW (registered, opening edges, persistent offset): max |s - truth| {:.3f} m, final {:.3f} m, phi {:.0f} deg, "
                   "P(passable) {:.2f}\n", new_r.max_dev, new_r.final_dev, new_r.final_phi * 57.3f, new_r.final_p);
        check(new_r.max_dev < 0.10f, std::format("the aperture holds through the bad pose (max {:.3f} m < 0.10)", new_r.max_dev));
        check(new_r.final_dev < 0.05f, std::format("and ends on the doorway ({:.3f} m < 0.05)", new_r.final_dev));
        check(new_r.final_phi > 1.2f and new_r.final_p > 0.5f,
              std::format("the open door reads open afterwards (phi {:.0f} deg, P {:.2f})", new_r.final_phi * 57.3f, new_r.final_p));
    }

    // (b)/(c): the pose is good; the robot stands 2-3 m in front of the doorway, swaying.
    const auto converge = [&](float s0, float sd0, int frames, const char* what, float* dev_half = nullptr)
    {
        Scene sc;
        auto b = make_belief(s0, sd0, 1.5f);
        Tracker t;
        int first_ok = -1;
        for (int f = 0; f < frames; ++f)
        {
            if (dev_half and f == frames / 2) *dev_half = std::abs(b.state().s - S_TRUE);
            const V3 o(0.5f + 0.6f * std::sin(0.05f * f), 2.6f - 0.4f * std::sin(0.03f * f), 0.67f);
            if (f % 30 == 0) b.predict();
            step(t, b, scan(sc, o, rng), false);
            trace(f, o, t, b);
            if (first_ok < 0 and std::abs(b.state().s - S_TRUE) < 0.05f) first_ok = f;
            if (first_ok >= 0 and std::abs(b.state().s - S_TRUE) >= 0.05f) first_ok = -1;
        }
        std::print("  {}: s0 {:.2f} (σ {:.2f}) → s {:.3f} (truth {:.2f}), within 5 cm for good from frame {}\n", what, s0, sd0,
                   b.state().s, S_TRUE, first_ok);
        return std::pair{std::abs(b.state().s - S_TRUE), first_ok};
    };
    std::print("\n(b) a door BORN 0.8 m off its opening (σ_s 0.6 m prior), pose good\n");
    {
        const auto [dev, at] = converge(S_TRUE + 0.8f, 0.6f, 60, "born wrong");
        check(dev < 0.05f and at >= 0 and at <= 30, std::format("converges onto the measured opening ({:.3f} m, frame {})", dev, at));
        const auto [dev2, at2] = converge(S_TRUE - 0.8f, 0.6f, 60, "born wrong, other side");
        check(dev2 < 0.05f and at2 >= 0 and at2 <= 30, std::format("and from the other side ({:.3f} m, frame {})", dev2, at2));
    }
    // A belief 12σ wrong is an OVERCONFIDENT one; the evidence moving it is the opening seen through a pose whose along-wall
    // error persists for kRegistrationTauS, so it converges at that pace (a time constant of tens of seconds), not in a frame.
    std::print("\n(c) a door CONFIDENTLY 0.35 m off (σ_s 3 cm), pose good, 60 s\n");
    {
        float dev_30s = 1.f;
        const auto [dev, at] = converge(S_TRUE + 0.35f, 0.03f, 600, "confidently wrong", &dev_30s);
        std::print("  |s - truth| 0.350 → {:.3f} m at 30 s → {:.3f} m at 60 s\n", dev_30s, dev);
        check(dev_30s < 0.20f and dev < 0.10f and dev < dev_30s,
              std::format("still converges onto the measured opening (0.35 → {:.3f} → {:.3f} m)", dev_30s, dev));
    }

    std::print("\n(d) register_to_wall recovers an injected misregistration\n");
    for (const auto& [eps, ty] : {std::pair{0.0f, 0.0f}, std::pair{-0.25f, -0.36f}, std::pair{0.17f, 0.2f}})
    {
        Scene sc;
        const auto b = make_belief(S_TRUE, 0.01f, 1.5f);
        const V3 o(1.2f, 3.4f, 0.67f);
        const auto c = scan(sc, o, rng, eps, V2(0.2f, ty));
        const auto reg = rc::door::register_to_wall(b.aperture(), c.origin, c.pts);
        // The scan's wall, rotated by eps about the PUBLISHED origin and shifted by ty, sits across the model line by
        // ty + (rotation of the wall's foot about the sensor) at the aperture centre — compare after undoing it.
        auto pts = c.pts;
        const V3 o2 = rc::door::apply_registration(reg, b.aperture(), c.origin, pts);
        const auto reg2 = rc::door::register_to_wall(b.aperture(), o2, pts);
        std::print("  injected eps {:+.3f} rad, ty {:+.2f} m → measured eps {:+.4f} (sd {:.4f}) depth {:+.3f}, n {}; residual after "
                   "undoing it: eps {:+.4f} depth {:+.3f}\n", eps, ty, reg.eps, std::sqrt(reg.eps_var), reg.depth, reg.n_inliers,
                   reg2.eps, reg2.depth);
        check(reg.ok and std::abs(reg.eps - eps) < 0.0175f and reg2.ok and std::abs(reg2.eps) < 0.0087f and std::abs(reg2.depth) < 0.02f,
              std::format("rotation within 1° ({:+.4f} vs {:+.3f}), the registered wall on the model line ({:+.3f} m)",
                          reg.eps, eps, reg2.depth));
    }

    std::print("\n(d2) wall_in_view: out-of-view keeps a held registration, in-view-without-wall resets it\n");
    {
        Scene sc;
        const auto b = make_belief(S_TRUE, 0.01f, 1.5f);
        const V3 o(1.2f, 3.4f, 0.67f);
        const auto c = scan(sc, o, rng, 0.f, V2(0.f, 0.f));
        const auto ap = b.aperture();
        const auto good = rc::door::register_to_wall(ap, c.origin, c.pts);
        check(good.ok and good.wall_in_view, "a full sweep registers and reports the wall in view");
        const V2 cen = ap.centre_xy();
        const float bearing = std::atan2(cen.y() - c.origin.y(), cen.x() - c.origin.x());
        // (i) the robot turned away: only returns well off the window's bearing remain.
        std::vector<V3> away, shifted;
        for (const auto& p : c.pts)
        {
            const float az = std::atan2(p.y() - c.origin.y(), p.x() - c.origin.x());
            if (std::abs(std::remainder(az - bearing, 2.f * std::numbers::pi_v<float>)) > 2.0f) away.push_back(p);
            else { V3 q = p; q.head<2>() += 1.5f * ap.across_u(); shifted.push_back(q); }   // (ii) wall far off the search band
        }
        const auto r_away = rc::door::register_to_wall(ap, c.origin, away);
        check(not r_away.ok and not r_away.wall_in_view, "sweep not covering the window: not registered, NOT in view (held)");
        const auto r_none = rc::door::register_to_wall(ap, c.origin, shifted);
        check(not r_none.ok and r_none.wall_in_view, "sweep covering the window with no wall returns: not registered, in view (reset)");
    }

    std::print("\n(e) a SHUT door with a second opening 1.3 m along the same wall\n");
    {
        Scene sc; sc.phi = 0.0f; sc.second_hole = true; sc.corner_x = -1.f;
        auto b = make_belief(S_TRUE, 0.03f, 0.0f);
        Tracker t; t.phi_est = 0.0f; t.phi_sigma = 0.05f;
        float max_dev = 0.f;
        for (int f = 0; f < 200; ++f)
        {
            const V3 o(1.6f + 0.6f * std::sin(0.05f * f), 2.4f, 0.67f);
            if (f % 30 == 0) b.predict();
            step(t, b, scan(sc, o, rng), false);
            trace(f, o, t, b);
            max_dev = std::max(max_dev, std::abs(b.state().s - S_TRUE));
        }
        std::print("  max |s - truth| {:.3f} m, phi {:.0f} deg\n", max_dev, t.phi_est * 57.3f);
        check(max_dev < 0.05f, std::format("the shut door's aperture is not dragged onto the other opening ({:.3f} m)", max_dev));
    }

    std::print("\n{}\n", g_fail == 0 ? "ALL PASS" : std::format("{} FAILED", g_fail));
    return g_fail == 0 ? 0 : 1;
}
