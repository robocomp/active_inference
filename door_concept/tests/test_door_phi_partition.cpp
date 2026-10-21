// Task D-C (rtD1i door_5 = DOOR_0, a real SHUT door): the LiDAR leaf channel read it OPEN 89° (phi 0.18 ± 0.05 → 1.55 ± 0.18
// in 0.8 s) on 17-35 rays from an oblique ~3 m view, and the image channels then walked its existence +4 → −4. Replayed on the
// rtD1i tape (cycles 755-768, where the replay is still bit-equal to the live run for this door), the cause is NOT the recess
// or a jamb reveal: the robot stood on the far side of the PARTITION wall that meets DOOR_0's wall 7 cm from its hinge jamb
// (room_1 polygon edges (0.059,0.382)→(0.099,2.536) and (0.040,4.605)→(−0.060,0.396)); the doorway itself was hidden, and every
// return came from the partition's near face — 3-8 cm off the polygon's edge for it. The leaf hypothesis has two placement
// nuisances (the slide along the wall, the depth) and the room walls had none, so a leaf at 90° slid +0.20 m (3.3 σ of the
// placement, its hinge INSIDE the partition) fitted that face better than the face itself: +2…+12 nats a frame for 90°.
//
// The fixture is that scene, synthetic and exact: DOOR_0 shut in its wall; the partition built 7 cm off its polygon edge;
// rtD1i's own room_1 polygon; the bpearl dome (32 rings, 875 azimuths, at 0.70 m) driven along the live path behind the
// partition. The leaf channel runs as DoorFitter::estimate_phi does (register, select, phi_field_loglik, leaf_channel_step).
//   (a) CONTROL — rigid room walls, unbounded slide (the field before Task D-C): the shut leaf must be read OPEN (> 1.2 rad):
//       the fixture reproduces the defect.
//   (b) FIX — the room walls carry the polygon face's σ as a common-mode offset (FieldNuisance::room_wall_sigma) and the
//       aperture lies on its own wall (wall_seg_lo/hi): the leaf stays SHUT (< 0.5 rad) on every frame.
//   (c) the fix does not blind the field: the SAME door OPEN at 90° (leaf against that partition) seen from INSIDE room_1's
//       right half, with the doorway in view, still reads open (> 1.2 rad).
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tests/test_door_phi_partition.cpp src/door_belief.cpp -o /tmp/t && /tmp/t
#include <algorithm>
#include <array>
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
#include "../src/door_phi_posterior.h"
#include "../src/door_opening_edges.h"
#include "../src/door_leaf_channel.h"

using V2 = Eigen::Vector2f;
using V3 = Eigen::Vector3f;

static int g_fail = 0;
static void check(bool ok, const std::string& what)
{
    std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++g_fail;
}

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

// room_1's frame (rtD1i). DOOR_0: gap x ∈ [0.17, 1.17] in the wall y ∈ [2.538, 2.638]; hinge at the west jamb (x 0.17), the
// leaf opens INTO room_1 (−y). The partition stands 7 cm WEST of its polygon edges (the misregistration the tape measured).
static std::vector<Box> scene(float phi)
{
    std::vector<Box> b;
    b.push_back(aabb(-0.10f, 0.17f, 2.538f, 2.638f, 0, 3));    // DOOR_0's wall, west of the gap (meets the partition)
    b.push_back(aabb(1.17f, 4.03f, 2.538f, 2.638f, 0, 3));     // east of the gap
    b.push_back(aabb(0.17f, 1.17f, 2.538f, 2.638f, 2.1f, 3));  // lintel
    b.push_back(aabb(-0.13f, -0.01f, 0.39f, 4.60f, 0, 3));     // the PARTITION (polygon says x ∈ [-0.06, 0.06]..[0.04, 0.10])
    {   // the leaf, hinge (0.17, 2.56): shut along +x, swinging into room_1 (−y)
        const V2 d(std::cos(phi), -std::sin(phi));
        b.push_back({V2(0.17f, 2.56f) + 0.5f * d, d, 0.5f, 0.02f, 0.0f, 2.1f});
    }
    b.push_back(aabb(-4.25f, 4.25f, -4.65f, -4.55f, 0, 3));    // room_1's far walls (background returns)
    b.push_back(aabb(-4.35f, -4.25f, -4.6f, 2.6f, 0, 3));
    b.push_back(aabb(4.10f, 4.20f, -4.6f, 2.6f, 0, 3));
    b.push_back(aabb(-1.70f, 0.10f, 4.60f, 4.70f, 0, 3));
    return b;
}
// rtD1i's room_1 polygon, exactly as the tape carries it.
static std::vector<V2> room1_polygon()
{
    const float P[][2] = {{1.017f, -4.550f}, {1.021f, -3.836f}, {0.019f, -3.809f}, {0.057f, -2.123f}, {-0.063f, -2.130f},
                          {-0.065f, -3.813f}, {-0.953f, -3.806f}, {-0.931f, -4.560f}, {-4.028f, -4.449f}, {-4.253f, -4.167f},
                          {-4.252f, -4.018f}, {-3.856f, -4.018f}, {-3.856f, -3.418f}, {-4.241f, -3.418f}, {-4.139f, 2.462f},
                          {-1.679f, 2.474f}, {-1.667f, 2.535f}, {-2.082f, 2.541f}, {-2.076f, 3.398f}, {-1.630f, 3.379f},
                          {-1.617f, 4.628f}, {0.040f, 4.605f}, {-0.060f, 0.396f}, {0.059f, 0.382f}, {0.099f, 2.536f},
                          {4.031f, 2.538f}, {4.253f, 2.264f}, {4.076f, -3.434f}, {3.743f, -3.434f}, {3.743f, -4.034f},
                          {4.057f, -4.034f}, {4.038f, -4.628f}};
    std::vector<V2> p;
    for (const auto& q : P) p.emplace_back(q[0], q[1]);
    return p;
}
static std::vector<V3> scan(const std::vector<Box>& bx, const V3& o, std::mt19937& rng)
{
    std::normal_distribution<float> n(0, 0.01f);
    std::vector<V3> pts;
    for (int a = 0; a < 875; ++a)
        for (int e = 0; e < 32; ++e)
        {
            const float az = a * 2.f * std::numbers::pi_v<float> / 875.f;
            const float el = -e * (90.f / 31.f) * std::numbers::pi_v<float> / 180.f;
            const V3 u(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
            float t = 50;
            if (u.z() < -1e-6f) t = std::min(t, -o.z() / u.z());
            for (const auto& b : bx) { const float tb = ray_box(o, u, b); if (tb > 0) t = std::min(t, tb); }
            if (t > 8.0f) continue;
            pts.push_back(o + (t + n(rng)) * u);
        }
    return pts;
}

// The belief the tape carried at cycle 755: wall_O (4.031, 2.538), u = −x ⇒ across_u = −y (into room_1); s 2.89, w 1.035 ⇒
// the gap x ∈ [0.106, 1.141]; the hinge at the far end (x 0.106, HingeSide::Far), swing into room_1; σ_s 0.06 m.
static rc::DoorBelief make_belief(float phi0)
{
    rc::DoorBeliefParams p;
    p.thickness = 0.05f; p.floor_z = 0;
    p.wall_O = {4.031f, 2.538f}; p.wall_u = {-1, 0};
    p.leaf = {phi0, rc::door::HingeSide::Far, +1.f};
    rc::DoorBelief b({2.89f, 1.035f, 2.1f}, p);
    return b;
}

constexpr float PHI_MIN = 0.0f, PHI_MAX = 2.0944f, PHI_STEP = 0.0873f, DT = 0.1f;
constexpr int   NSTEP = 25;
constexpr float kFaceSigma = 0.10f;   // Existence.BoundarySigmaM (door_config.h default), the polygon face's σ

// Σ over the frames of this frame's evidence for an OPEN leaf against the believed shut one: max_{φ ≥ 75°} ll(φ) − ll(0.175),
// the tempered field log-likelihood exactly as the leaf channel receives it (FieldResult::loglik, the believed hinge/swing).
static double g_open_vs_shut = 0.0;
struct Tracker
{
    float hinge_llr = 25.f, swing_llr = 59.f, phi = 0.175f, sigma = 0.01f;   // the live state at cycle 755
    bool  first = false;
    rc::door::LeafChannel leaf;
};

static void step(Tracker& t, rc::DoorBelief& b, const V3& o, const std::vector<V3>& pts, const std::vector<V2>& poly, bool fixed)
{
    auto nz = rc::door::phi_field_nuisance(b.aperture());
    nz.lat_sigma = 0.06f;
    nz.other_hinge = nz.other_swing = true;
    nz.room_walls = rc::door::room_walls_beside(b.aperture(), poly, nz.depth_lo, nz.depth_hi);
    if (fixed)
    {
        nz.room_wall_sigma = kFaceSigma;
        std::tie(nz.wall_seg_lo, nz.wall_seg_hi) = rc::door::door_wall_segment(b.aperture(), poly, nz.depth_lo, nz.depth_hi);
    }
    const auto rs = rc::door::register_sweep(b.aperture(), o, pts, nz.room_walls);
    const auto sel = rc::door::select_doorway_rays(b.aperture(), rs.origin, rs.pts, std::max(-nz.lat_lo, nz.lat_hi), 1.0f);
    rc::door::LeafChannelInput in;
    in.dt = t.first ? std::numeric_limits<float>::infinity() : DT;
    t.first = false;
    if (not sel.rays.endpoints.empty())
    {
        auto fr = b.phi_field_loglik(sel.rays, PHI_MIN, PHI_MAX, NSTEP, nz);
        double l_shut = 0.0, l_open = -1e9;
        for (const auto& [ph, l] : fr.loglik)
        {
            if (std::abs(ph - 0.175f) < 0.5f * PHI_STEP) l_shut = l;
            if (ph >= 1.30f) l_open = std::max<double>(l_open, l);
        }
        g_open_vs_shut += l_open - l_shut;
        in.bb = std::move(fr.loglik); in.ob = std::move(fr.loglik_other); in.bs = std::move(fr.loglik_swing); in.os = std::move(fr.loglik_both);
    }
    in.have_sensor = true; in.sensor = o.head<2>();
    in.phi_min = PHI_MIN; in.phi_max = PHI_MAX; in.nstep = NSTEP; in.view_rad = PHI_STEP;
    (void)rc::door::leaf_channel_step(t.leaf, b, t.hinge_llr, t.swing_llr, t.phi, t.sigma, in);
    b.set_leaf_phi(t.phi);
}

// rtD1i cycles 752-775: the bpearl origin walked (−0.54, −0.80) → (−0.57, −0.12) at 0.70 m, behind the partition.
static float run(bool fixed, float phi_true, float phi0, const std::vector<V3>& path, const char* tag)
{
    g_open_vs_shut = 0.0;
    std::mt19937 rng(755);
    const auto bx = scene(phi_true);
    Tracker t; t.phi = phi0;
    auto b = make_belief(phi0);
    float lo = 9.f, hi = -9.f;
    for (std::size_t f = 0; f < path.size(); ++f)
    {
        step(t, b, path[f], scan(bx, path[f], rng), room1_polygon(), fixed);
        lo = std::min(lo, t.phi); hi = std::max(hi, t.phi);
        if (std::getenv("PHI_TRACE"))
            std::print("    {} f{:2d} o ({:+.2f},{:+.2f}) phi {:.3f} ± {:.3f}\n", tag, f, path[f].x(), path[f].y(), t.phi, t.sigma);
    }
    std::print("  {}: Σ frames [max ll(φ ≥ 75°) − ll(10°)] = {:+.1f} nats over {} frames; tracker phi range [{:.3f}, {:.3f}] rad\n",
               tag, g_open_vs_shut, path.size(), lo, hi);
    return hi;
}

int main()
{
    std::setlocale(LC_ALL, "");
    std::vector<V3> behind;
    for (int f = 0; f < 24; ++f)
    {
        const float s = static_cast<float>(f) / 23.f;
        behind.emplace_back(-0.54f - 0.03f * s, -0.80f + 0.68f * s, 0.70f);
    }
    // ★The checks are on the FRAME EVIDENCE the leaf channel receives, not on the tracker's flip: whether a given run of
    //  frames flips the tracker also depends on its viewpoint bank and move hazard (rtD1i: vp_n 22, p_ok 0.97 → 0.22 at the
    //  flip), which a fresh fixture does not carry. The tape's own frames 755-768 gave ≈ +60 nats for 90° (tempered, Σ).
    std::print("(a) CONTROL: rigid room walls, unbounded slide — the field before Task D-C\n");
    (void)run(false, 0.0f, 0.175f, behind, "control");
    const double ev_a = g_open_vs_shut;
    check(ev_a > 50.0, std::format("the fixture reproduces the defect: the shut door's frames argue OPEN by {:+.1f} nats (> 50)", ev_a));

    std::print("(b) FIX: room walls with the polygon face's σ, the aperture on its own wall\n");
    const float hi_b = run(true, 0.0f, 0.175f, behind, "fixed");
    const double ev_b = g_open_vs_shut;
    check(ev_b < 0.25 * ev_a, std::format("the same frames argue open by {:+.1f} nats, < 1/4 of the control's", ev_b));
    check(hi_b < 0.5f, std::format("the tracker keeps the SHUT door shut (max phi {:.2f} rad < 0.5)", hi_b));

    std::print("(c) the fix still sees an OPEN leaf (90°, against the partition) from inside, the doorway in view\n");
    std::vector<V3> inside;
    for (int f = 0; f < 24; ++f)
    {
        const float s = static_cast<float>(f) / 23.f;
        inside.emplace_back(1.6f + 0.6f * s, 0.2f + 0.4f * s, 0.70f);
    }
    std::mt19937 rng(90);
    const auto bx = scene(std::numbers::pi_v<float> / 2.f);
    Tracker t; t.phi = 0.175f; t.sigma = 0.6f;
    auto b = make_belief(0.175f);
    g_open_vs_shut = 0.0;
    for (const auto& o : inside) step(t, b, o, scan(bx, o, rng), room1_polygon(), true);
    std::print("  open, fixed field: Σ frames [max ll(φ ≥ 75°) − ll(10°)] = {:+.1f} nats\n", g_open_vs_shut);
    check(g_open_vs_shut > 50.0, std::format("the open leaf's frames still argue open ({:+.1f} nats > 50)", g_open_vs_shut));
    std::print("  open, fixed field: final phi {:.3f} ± {:.3f}\n", t.phi, t.sigma);
    check(t.phi > 1.2f, std::format("an open leaf still reads open with the fix (phi {:.2f} rad > 1.2)", t.phi));

    std::print("{}\n", g_fail == 0 ? "test_door_phi_partition: ALL PASS" : std::format("test_door_phi_partition: {} FAILURE(S)", g_fail));
    return g_fail == 0 ? 0 : 1;
}
