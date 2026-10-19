// try97 (2026-10-19): an OPEN door_1 (leaf at 90°, bridge pane: "swinging -1.57084 -> -1.5708" at the leg-2 OPEN, i.e. it
// already stood at 90.0°) read phi 73° → 69° from room_2 at 2.6 m and 49° from the parked pose 1.9 m out, clear span
// 0.72 → 0.65 m < 0.70 m of body ⇒ cross withdrawn. Offline harness for DoorFitter::estimate_phi's LiDAR path —
// register_sweep, select_doorway_rays, DoorBelief::phi_field_loglik, phi_leaf_marginal, phi_posterior_step — on the EXACT
// Webots DOOR_1 scene (webots-shadow/protos/Door.proto + worlds/piso/tworooms-piso.wbt + thicken_walls.py):
//   · room_2's south wall y ∈ [-0.11, 0] (door frame: x along the wall, gap centre 0; room_2 is y > 0), gap |x| < 0.53;
//   · room_1's mesh wall y ∈ [-0.24, -0.13] with its own 1.0 m opening, |x| < 0.50, and room_1's CORRIDOR behind it
//     (east side 7.5 cm past the east jamb, west side 0.6 m past the west jamb — the inner outline of thicken_walls.py);
//   · the frame's casings (5 cm, 0.20 m deep) and the leaf (0.95 × 2.075 × 0.05 m) hinged on the WEST jamb at
//     (-0.475, -0.20) — room_1's face of the frame — and opening INTO room_1;
//   · the bpearl dome at 0.67 m (32 rings 0..-90°, 875 azimuths: the live selected-ray counts, ~940 at 2.6 m).
//
//   (a) leg 1, room_1 side (the side the leaf opens into), approaching 3.7 → 1.3 m with try97's start-up belief (hinge right,
//       swing WRONG at +0.3 nats): the swing must end into room_1, phi 90 ± 10°, and no frame may register on the leaf
//       (live leg 1: eps +0.38..+0.41 rad, depth up to +0.59 m — the search's own bounds).
//   (b) leg 2, room_2 side, the try97 window (2.6 m centred, then the parked pose 0.77 m east / 1.57 m out) with the hinge,
//       swing and log-odds CARRIED from (a): phi 90 ± 10° and P(passable) > 0.8 on every frame after the first.
//   (c) a SHUT door from room_1's corridor (live try59: the corridor wall read as a leaf at 97°) and from room_2: P(passable)
//       stays below 0.05 — the reason the room's walls are modelled rather than every short return dropped.
//   (i) INFO, not a check: the live try97 carried belief (swing into room_2 at +2309 nats) — what an over-confident carried
//       log-odds still does from room_2, where the swing is weakly observable (see the report's concerns).
//   (j) try102: the swing carried WRONG at what the leaf channel itself accrues over (a)'s approach (≈ 3.5 nats per viewpoint,
//       ~+30 — not +2309): room_2 overturns it, and the open door then reads passable.
//
// Compiles against the code before the fix too (the room-wall API is reached through `requires`), so "before" can be
// measured: there (a) ends with the swing into room_2 at ~107° and (b) reads ~45-75°.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tests/test_door_phi_sides.cpp src/door_belief.cpp -o /tmp/test_door_phi_sides && /tmp/test_door_phi_sides
// PHI_TRACE=1 prints every frame.
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
#include "../src/door_passability.h"
#include "../src/door_leaf_channel.h"   // try102: the leaf channel DoorFitter::estimate_phi runs

using V2 = Eigen::Vector2f;
using V3 = Eigen::Vector3f;
using Seg = std::array<V2, 2>;

static int g_fail = 0;
static void check(bool ok, const std::string& what)
{
    std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++g_fail;
}

// ─── the scene ───────────────────────────────────────────────────────────────────────────────────────────────────────────
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

// World (x, y) → door frame: x + 4.77, y − 9.48 (room_2's wall face, the gap centre).
static V2 W(float x, float y) { return {x + 4.77f, y - 9.48f}; }

static std::vector<Box> scene(float phi)
{
    std::vector<Box> b;
    b.push_back(aabb(-3.06f, -0.53f, -0.11f, 0, 0, 3)); b.push_back(aabb(0.53f, 3.06f, -0.11f, 0, 0, 3));   // room_2 south wall
    b.push_back(aabb(-0.53f, 0.53f, -0.11f, 0, 2.1f, 3));                                                   // its lintel
    b.push_back(aabb(-1.21f, -0.501f, -0.24f, -0.13f, 0, 3)); b.push_back(aabb(0.499f, 0.67f, -0.24f, -0.13f, 0, 3));   // room_1 mesh wall
    b.push_back(aabb(-0.501f, 0.499f, -0.24f, -0.13f, 2.1f, 3));
    b.push_back(aabb(-0.5f, -0.45f, -0.20f, 0, 0, 2.1f));   // west casing (hinge side)
    b.push_back(aabb(0.45f, 0.5f, -0.20f, 0, 0, 2.1f));     // east casing
    b.push_back(aabb(-0.5f, 0.5f, -0.20f, 0, 2.05f, 2.1f)); // top frame
    {   // the leaf: hinge axis (-0.475, -0.20); shut it lies along +x, its 5 cm offset toward room_1; it opens INTO room_1
        const V2 d(std::cos(phi), -std::sin(phi)), n(-std::sin(phi), -std::cos(phi));
        b.push_back({V2(-0.475f, -0.20f) + 0.475f * d + 0.025f * n, d, 0.475f, 0.025f, 0.0f, 2.075f});
    }
    {   // room_1's corridor: east side (inner face) (0.574,-0.25) → (0.474,-4.46), 0.11 m thick outward
        const V2 a(0.574f, -0.25f), e(0.474f, -4.46f), dd = (e - a).normalized();
        b.push_back({0.5f * (a + e) + V2(0.055f, 0.f), dd, 0.5f * (e - a).norm(), 0.055f, 0, 3});
    }
    b.push_back(aabb(-1.22f, -1.11f, -1.47f, -0.24f, 0, 3));
    b.push_back(aabb(-1.67f, -1.11f, -1.58f, -1.47f, 0, 3));
    b.push_back(aabb(-1.67f, -1.56f, -2.31f, -1.47f, 0, 3));
    b.push_back(aabb(-6, -1.15f, -2.42f, -2.31f, 0, 3));
    b.push_back(aabb(-6, 6, -9.6f, -9.5f, 0, 3));
    b.push_back(aabb(-6, 6, -9.6f, -0.24f, 3.0f, 3.1f));      // room_1 ceiling
    b.push_back(aabb(-3.06f, 3.06f, 0, 4.11f, 3.0f, 3.1f));   // room_2 ceiling and its other walls
    b.push_back(aabb(-3.2f, 3.2f, 4.11f, 4.22f, 0, 3));
    b.push_back(aabb(-3.17f, -3.06f, 0, 4.11f, 0, 3)); b.push_back(aabb(3.06f, 3.17f, 0, 4.11f, 0, 3));
    return b;
}

// The room polygons the agent holds (room frame ≡ door frame here): room_2 a box; room_1 the inner outline of thicken_walls.py.
static std::vector<V2> room2_polygon() { return {V2(-3.06f, 0), V2(3.06f, 0), V2(3.06f, 4.11f), V2(-3.06f, 4.11f)}; }
static std::vector<V2> room1_polygon()
{
    const float L[][2] = {{-0.000f, 6.892f}, {-0.215f, 0.000f}, {-3.236f, 0.078f}, {-3.232f, 0.792f}, {-4.222f, 0.820f},
                          {-4.184f, 2.506f}, {-4.329f, 2.497f}, {-4.331f, 0.814f}, {-5.207f, 0.822f}, {-5.184f, 0.068f},
                          {-8.262f, 0.178f}, {-8.508f, 0.437f}, {-8.392f, 7.090f}, {-5.932f, 7.102f}, {-5.920f, 7.163f},
                          {-6.336f, 7.169f}, {-6.330f, 8.026f}, {-5.884f, 8.007f}, {-5.871f, 9.256f}, {-4.196f, 9.231f},
                          {-4.296f, 5.022f}, {-4.212f, 5.012f}, {-4.172f, 7.166f}, {-0.222f, 7.166f}};
    std::vector<V2> p;
    for (const auto& q : L) p.push_back(W(q[0], q[1]));
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

// ─── the model door (the belief the agent carries in each room's frame) ───────────────────────────────────────────────
// room_1: the model plane is room_1's wall face (y = -0.24), u = -x ⇒ across_u = -y points INTO room_1, Far = the west jamb.
// room_2: the model plane is room_2's face (y = 0), u = +x ⇒ across_u = +y points INTO room_2, Near = the west jamb.
// `into_room1`: the side the belief hangs the leaf on, as a physical side (the mirror carries it geometrically).
static rc::DoorBelief make_belief(bool room1, bool hinge_west, bool into_room1, float phi0)
{
    rc::DoorBeliefParams p;
    p.thickness = 0.05f; p.floor_z = 0;
    if (room1) { p.wall_O = {2.f, -0.24f}; p.wall_u = {-1, 0}; }
    else       { p.wall_O = {-2.f, 0.f};   p.wall_u = {1, 0}; }
    const auto west = room1 ? rc::door::HingeSide::Far : rc::door::HingeSide::Near;
    const auto east = room1 ? rc::door::HingeSide::Near : rc::door::HingeSide::Far;
    // across_u points into the room the belief is held in: "into room_1" is +1 in room_1's frame, -1 in room_2's.
    p.leaf = {phi0, hinge_west ? west : east, (into_room1 == room1) ? +1.f : -1.f};
    return rc::DoorBelief({1.5f, 1.0f, 2.1f}, p);
}
static bool leaf_west(const rc::DoorBelief& b, bool room1)
{ return (b.params().leaf.hinge == rc::door::HingeSide::Far) == room1; }
static bool leaf_into_room1(const rc::DoorBelief& b, bool room1) { return (b.params().leaf.swing > 0.f) == room1; }

constexpr float PHI_MIN = 0.0f, PHI_MAX = 2.0944f, PHI_STEP = 0.0873f, DT = 0.1f, NEEDED = 0.70f;
constexpr int   NSTEP = 25;

// The room's walls, through `requires` so this file also builds against the code before the fix (where they do not exist).
template <class NZ, class Ap>
static void set_room_walls(NZ& nz, const Ap& ap, const std::vector<V2>& poly)
{
    if constexpr (requires(NZ& n) { n.room_walls; })
    {
        // rc::door::room_walls_beside's rule, written here so the name stays dependent: the door's own wall is the edge
        // with both ends inside the field's depth support of the doorway plane.
        const V2 nrm = ap.across_u();
        const auto in_band = [&](const V2& q) { const float d = nrm.dot(q - ap.wall_O); return d >= nz.depth_lo and d <= nz.depth_hi; };
        for (std::size_t i = 0; i < poly.size(); ++i)
        {
            const V2& a = poly[i]; const V2& b = poly[(i + 1) % poly.size()];
            if (in_band(a) and in_band(b)) continue;
            nz.room_walls.push_back({a, b});
        }
    }
}
template <class NZ, class Ap>
static rc::door::RegisteredSweep do_register(const NZ& nz, const Ap& ap, const V3& o, const std::vector<V3>& pts)
{
    if constexpr (requires(const NZ& n) { rc::door::register_sweep(ap, o, pts, n.room_walls); })
        return rc::door::register_sweep(ap, o, pts, nz.room_walls);
    else
        return rc::door::register_sweep(ap, o, pts);
}

struct Tracker
{
    float hinge_llr = 0.f, swing_llr = 0.f, phi = 1.2f, sigma = PHI_MAX;
    bool  first = true;
    float p = -1.f;
    rc::door::WallRegistration reg;
    rc::door::LeafChannel leaf;   // try102: as DoorInstance::leaf_channel
};

// One cycle of DoorFitter::estimate_phi's LiDAR path (both jambs and both swings scored, as for a claimed door).
static void step(Tracker& t, rc::DoorBelief& b, const V3& o, const std::vector<V3>& pts, const std::vector<V2>& poly)
{
    auto nz = rc::door::phi_field_nuisance(b.aperture());
    nz.lat_sigma = std::sqrt(std::max(1e-8f, b.covariance()(0, 0)));
    nz.other_hinge = nz.other_swing = true;
    set_room_walls(nz, b.aperture(), poly);
    const auto rs = do_register(nz, b.aperture(), o, pts);
    t.reg = rs.reg;
    const auto sel = rc::door::select_doorway_rays(b.aperture(), rs.origin, rs.pts, std::max(-nz.lat_lo, nz.lat_hi), 1.0f);
    rc::door::LeafChannelInput in;
    in.dt = t.first ? std::numeric_limits<float>::infinity() : DT;
    t.first = false;
    if (not sel.rays.endpoints.empty())
    {
        auto fr = b.phi_field_loglik(sel.rays, PHI_MIN, PHI_MAX, NSTEP, nz);
        in.bb = std::move(fr.loglik); in.ob = std::move(fr.loglik_other); in.bs = std::move(fr.loglik_swing); in.os = std::move(fr.loglik_both);
    }
    in.have_sensor = true; in.sensor = o.head<2>();
    in.phi_min = PHI_MIN; in.phi_max = PHI_MAX; in.nstep = NSTEP; in.view_rad = PHI_STEP;
    const auto post = rc::door::leaf_channel_step(t.leaf, b, t.hinge_llr, t.swing_llr, t.phi, t.sigma, in).post;
    b.set_leaf_phi(t.phi);
    t.p = rc::door::passable_prob(post.curve, b.aperture().w, NEEDED).value_or(-1.f);
}

static void trace(const char* tag, int f, const V3& o, const Tracker& t, const rc::DoorBelief& b, bool room1)
{
    if (std::getenv("PHI_TRACE"))
        std::print("    {} f{:3d} o ({:+.2f},{:+.2f}) phi {:5.1f} ± {:4.1f} deg P {:.2f} | hinge {} swing into {} (llr {:+.0f} / {:+.0f}) reg {:+.3f} rad {:+.3f} m\n",
                   tag, f, o.x(), o.y(), t.phi * 57.2958f, t.sigma * 57.2958f, t.p, leaf_west(b, room1) ? "W" : "E",
                   leaf_into_room1(b, room1) ? "room_1" : "room_2", t.hinge_llr, t.swing_llr, t.reg.eps, t.reg.depth);
}

int main()
{
    std::setlocale(LC_ALL, "");   // CLAUDE.md: the agent's locale (nothing here parses text, but stay honest)
    std::mt19937 rng(97);
    const auto open = scene(std::numbers::pi_v<float> / 2.f);

    // ── (a) leg 1 from room_1 ───────────────────────────────────────────────────────────────────────────────────────
    std::print("(a) leg 1, room_1 side: approach 3.7 → 1.3 m, start-up belief hinge west (right), swing into room_2 (WRONG, +0.3)\n");
    Tracker ta; ta.hinge_llr = 0.0f; ta.swing_llr = 0.3f;
    auto ba = make_belief(true, true, false, 0.0f);
    float worst_reg = 0.f;
    const int NA = 60;
    for (int f = 0; f < NA; ++f)
    {
        const float s = static_cast<float>(f) / (NA - 1);
        const V3 o(0.10f - 0.14f * s + 0.25f * std::sin(0.3f * f), -3.74f + 2.46f * s, 0.67f);   // try97 leg-1 path, door frame
        step(ta, ba, o, scan(open, o, rng), room1_polygon());
        worst_reg = std::max(worst_reg, std::abs(ta.reg.eps));
        trace("a", f, o, ta, ba, true);
    }
    std::print("  end: phi {:.1f} deg σ {:.1f}, P(passable) {:.2f}, hinge {}, swing into {} (log-odds {:+.0f}), worst |reg eps| {:.3f} rad\n",
               ta.phi * 57.2958f, ta.sigma * 57.2958f, ta.p, leaf_west(ba, true) ? "west" : "east",
               leaf_into_room1(ba, true) ? "room_1" : "room_2", ta.swing_llr, worst_reg);
    check(leaf_into_room1(ba, true), "the swing ends on the true side (into room_1)");
    check(leaf_west(ba, true), "the hinge ends on the true jamb (west)");
    check(std::abs(ta.phi * 57.2958f - 90.f) < 10.f, std::format("phi {:.1f} deg within 90 ± 10", ta.phi * 57.2958f));
    check(worst_reg < 0.05f, std::format("no frame registers on the leaf (worst |eps| {:.3f} rad < 0.05)", worst_reg));

    // ── (b) leg 2 from room_2, the belief carried across ────────────────────────────────────────────────────────────
    std::print("\n(b) leg 2, room_2 side, try97 window: 2.6 m centred, then the parked pose (0.77 m east, 1.57 m out), belief from (a)\n");
    {
        Tracker tb; tb.hinge_llr = ta.hinge_llr; tb.swing_llr = ta.swing_llr; tb.phi = ta.phi; tb.sigma = ta.sigma; tb.first = false;
        auto bb = make_belief(false, leaf_west(ba, true), leaf_into_room1(ba, true), ta.phi);
        float pmin = 2.f, emax = 0.f, phi_park = 0.f;
        const int NB = 70;
        for (int f = 0; f < NB; ++f)
        {
            const float s = std::clamp((static_cast<float>(f) - 30.f) / 10.f, 0.f, 1.f);
            const V3 o(0.035f + 0.75f * s, 2.575f - 1.02f * s, 0.67f);
            step(tb, bb, o, scan(open, o, rng), room2_polygon());
            trace("b", f, o, tb, bb, false);
            if (f > 0) { pmin = std::min(pmin, tb.p); emax = std::max(emax, std::abs(tb.phi * 57.2958f - 90.f)); }
            if (f == NB - 1) phi_park = tb.phi * 57.2958f;
        }
        std::print("  worst |phi - 90| {:.1f} deg, min P(passable) {:.2f}, phi at the parked pose {:.1f} deg (live: 49)\n", emax, pmin, phi_park);
        check(emax < 10.f, std::format("phi within 90 ± 10 on every frame (worst {:.1f})", emax));
        check(pmin > 0.8f, std::format("P(passable) > 0.8 on every frame (min {:.2f})", pmin));
    }

    // ── (c) a SHUT door must stay shut — the room's walls are not a leaf ───────────────────────────────────────────────
    std::print("\n(c) SHUT door: from room_1's corridor (try59) and from room_2\n");
    {
        const auto shut = scene(0.0f);
        for (const bool room1 : {true, false})
        {
            Tracker tc;
            auto bc = make_belief(room1, true, true, 0.0f);
            float pmax = 0.f;
            const std::vector<V3> path = room1
                ? std::vector<V3>{V3(0.10f, -3.74f, 0.67f), V3(-0.60f, -3.50f, 0.67f), V3(-0.11f, -1.99f, 0.67f), V3(-0.04f, -1.28f, 0.67f), V3(-0.40f, -1.50f, 0.67f)}
                : std::vector<V3>{V3(0.035f, 2.575f, 0.67f), V3(0.785f, 1.555f, 0.67f), V3(-0.04f, 1.23f, 0.67f)};
            int f = 0;
            for (const V3& o : path)
                for (int k = 0; k < 15; ++k, ++f)
                {
                    step(tc, bc, o, scan(shut, o, rng), room1 ? room1_polygon() : room2_polygon());
                    trace("c", f, o, tc, bc, room1);
                    if (f > 2) pmax = std::max(pmax, tc.p);
                }
            check(pmax < 0.05f, std::format("from {}: P(passable) stays < 0.05 (max {:.3f}, final phi {:.1f} deg)",
                                            room1 ? "room_1" : "room_2", pmax, tc.phi * 57.2958f));
        }
    }

    // ── (i) the live carried belief, for the record ──────────────────────────────────────────────────────────────────
    std::print("\n(i) INFO: room_2 with try97's live carried belief (swing into room_2, +2309 nats) — not a check\n");
    {
        Tracker ti; ti.hinge_llr = 1019.f; ti.swing_llr = 2309.f; ti.phi = 1.92f; ti.sigma = 0.09f; ti.first = false;
        auto bi = make_belief(false, true, false, 1.92f);
        for (int f = 0; f < 40; ++f)
        {
            const float s = std::clamp((static_cast<float>(f) - 15.f) / 10.f, 0.f, 1.f);
            const V3 o(0.035f + 0.75f * s, 2.575f - 1.02f * s, 0.67f);
            step(ti, bi, o, scan(open, o, rng), room2_polygon());
            if (f == 14 or f == 39)
                std::print("  {}: phi {:.1f} deg, P(passable) {:.2f}, swing into {} (log-odds {:+.0f})\n",
                           f == 14 ? "2.6 m centred" : "parked pose  ", ti.phi * 57.2958f, ti.p,
                           leaf_into_room1(bi, false) ? "room_1" : "room_2", ti.swing_llr);
        }
    }

    // ── (j) try102: a WRONG side carried at what the channel itself accrues — overturned from room_2 ──────────────────────
    // Under the leaf channel (door_leaf_channel.h) a side is worth log((1−ε_v)/ε_v) ≈ 3.5 nats per VIEWPOINT, so leg 1's
    // whole approach (a) carries ~+30, not +2309. Carried wrong at that value into room_2, the try97 window must overturn it.
    const float carried = std::abs(ta.swing_llr);
    std::print("\n(j) room_2 with the swing carried WRONG at the channel's own leg-1 value (+{:.0f} nats into room_2)\n", carried);
    {
        Tracker tj; tj.hinge_llr = std::abs(ta.hinge_llr); tj.swing_llr = carried; tj.phi = 1.92f; tj.sigma = 0.09f; tj.first = false;
        auto bj = make_belief(false, true, false, 1.92f);
        int f_flip = -1;
        float p_tail = 2.f;
        const int NJ = 180;
        for (int f = 0; f < NJ; ++f)
        {
            // the try97 window, then a slow sideways wander across the doorway's front, 1.6 m (≈ 10 viewpoints at 1.5 m)
            const float s = std::clamp((static_cast<float>(f) - 15.f) / 10.f, 0.f, 1.f);
            const V3 o(0.035f + 0.75f * s - 1.6f * std::clamp((static_cast<float>(f) - 40.f) / 120.f, 0.f, 1.f),
                       2.575f - 1.02f * s, 0.67f);
            step(tj, bj, o, scan(open, o, rng), room2_polygon());
            trace("j", f, o, tj, bj, false);
            if (f_flip < 0 and leaf_into_room1(bj, false)) f_flip = f;
            if (f >= NJ - 20) p_tail = std::min(p_tail, tj.p);
        }
        std::print("  swing into {} after {} frames (flipped at frame {}), {} viewpoints; phi {:.1f} deg, P(passable) over the last 20 frames ≥ {:.2f}\n",
                   leaf_into_room1(bj, false) ? "room_1 (true)" : "room_2 (WRONG)", NJ, f_flip, tj.leaf.views.views.size(), tj.phi * 57.2958f, p_tail);
        check(leaf_into_room1(bj, false), std::format("a wrong side carried at +{:.0f} is overturned from room_2 (frame {})", carried, f_flip));
        check(p_tail > 0.8f, std::format("…and the open door reads passable once it is (min P {:.2f} over the last 20 frames)", p_tail));
    }

    std::print("\n{}\n", g_fail == 0 ? "ALL PASS" : std::format("{} FAILURE(S)", g_fail));
    return g_fail == 0 ? 0 : 1;
}
