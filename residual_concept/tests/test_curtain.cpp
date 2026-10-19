/*
 * test_curtain.cpp — the try96 "curtain", reproduced through the grid alone (no DSR, no Qt).
 *
 * What happened live (2026-10-19, try96 leg 3, see .superpowers/sdd/2026-10-18-interaction-anchor/
 * residual-curtain-report.md): room_concept's pose was 0.47 m / 13.5 deg wrong for ~20 s while the robot dwelt
 * 0.7 m from a wall, reporting sigma 2.4 cm the whole time. Every sweep registered that WALL half a metre into
 * the room, both lidars confirmed it every cycle, and the grid latched a 3 m "curtain" of full-height columns.
 * After the localiser recovered, the robot looked at the curtain from 2-4 m for half a minute, but with the
 * live pencil refutation (LidarModel.*BeamSpacingRad = 0) only the 2.5 cm bins that a ring line happened to
 * cross could ever be refuted; the bins between the ring lines are unobservable from a parked pose, so every
 * column kept its max positive and the curtain stayed for the rest of the run.
 *
 * Three properties, each its own scenario:
 *   (1) CURTAIN  — the phantom born through a wrong pose must be REMOVED once correctly-registered beams pass
 *                  through it and return nothing. Fails on the pre-fix code.
 *   (2) STANDING — a real obstacle seen repeatedly from a moving robot must NOT be removed.
 *   (3) OCCLUDED — a real obstacle the robot can no longer see (an occluder stands between) must HOLD.
 *
 * The sensors are the real mounts (ROBOT_GEOMETRY.md): helios INVERTED at 1.075 m (-15.5..+54.5 deg, 32 rings)
 * and bpearl a DOWNWARD dome at 0.670 m (0..-90 deg, 32 rings), each ray-traced from its OWN origin. The grid is
 * configured as the live agent runs it (etc/config.toml): 1.6 m bin span, pencil refutation, per-device floor
 * bands, pose precision at the sigma room_concept reported during the incident.
 *
 * Build + run: tests/run_tests.sh
 */
#include "../src/residual_occupancy_grid.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <set>
#include <utility>
#include <vector>

using Eigen::Vector3f;

namespace
{
int g_fail = 0;
void check(bool c, const char* m)
{
    if (not c) { ++g_fail; std::printf("  FAIL: %s\n", m); }
    else       std::printf("  ok:   %s\n", m);
}

// ── the world: a 6 x 6 m room, walls 2.4 m tall, and any number of axis-aligned boxes ──────────────────────
struct Box { float x0, x1, y0, y1, z0, z1; };
std::vector<Box> g_boxes;

float raycast(const Vector3f& o, const Vector3f& d, float tmax)
{
    float best = tmax;
    const auto slab = [](float lo, float hi, float p, float v, float& t0, float& t1)
    {
        if (std::abs(v) < 1e-9f) { if (p < lo or p > hi) { t0 = 1.f; t1 = -1.f; } return; }
        float a = (lo - p) / v, b = (hi - p) / v; if (a > b) std::swap(a, b);
        t0 = std::max(t0, a); t1 = std::min(t1, b);
    };
    if (d.z() < -1e-9f) { const float t = -o.z() / d.z(); if (t > 0.f and t < best) best = t; }       // floor
    if (d.z() >  1e-9f) { const float t = (2.4f - o.z()) / d.z(); if (t > 0.f and t < best) best = t; } // ceiling
    { float t0 = 0.f, t1 = best;                                                                       // walls
      slab(-3.f, 3.f, o.x(), d.x(), t0, t1); slab(-3.f, 3.f, o.y(), d.y(), t0, t1);
      if (t1 > 0.f and t1 < best) best = t1; }
    for (const auto& b : g_boxes)
    { float t0 = 0.f, t1 = best;
      slab(b.x0, b.x1, o.x(), d.x(), t0, t1); slab(b.y0, b.y1, o.y(), d.y(), t0, t1);
      slab(b.z0, b.z1, o.z(), d.z(), t0, t1);
      if (t1 >= t0 and t0 > 1e-3f and t0 < best) best = t0; }
    return best;
}

// A rigid SE(2) pose of the robot base (Shadow): position + heading.
struct Pose { float x, y, th; };
Vector3f apply(const Pose& p, const Vector3f& v)   // robot frame -> room frame
{ const float c = std::cos(p.th), s = std::sin(p.th);
  return {p.x + c * v.x() - s * v.y(), p.y + s * v.x() + c * v.y(), v.z()}; }
Vector3f unapply(const Pose& p, const Vector3f& w)  // room frame -> robot frame
{ const float c = std::cos(p.th), s = std::sin(p.th), dx = w.x() - p.x, dy = w.y() - p.y;
  return {c * dx + s * dy, -s * dx + c * dy, w.z()}; }

struct Device { Vector3f mount; float el_lo, el_hi; int rings; float min_r; float floor_z0; std::uint8_t id; };
// ROBOT_GEOMETRY.md: helios hangs inverted (fan UP at the walls), bpearl is a downward dome.
const Device kHelios{{0.f, -0.155f, 1.075f}, -15.5f, 54.5f, 32, 0.40f, 0.20f, 2};
const Device kBpearl{{0.f,  0.140f, 0.670f}, -90.0f,  0.0f, 32, 0.10f, 0.12f, 1};

// One device sweep. The rays are cast in the TRUE world from the TRUE pose; the returns are then REGISTERED
// through `believed` — exactly what a localiser error does to a lidar sweep. Returns the sensor origin as the
// agent would compute it (through the believed pose).
Vector3f sweep(const Device& dv, const Pose& truth, const Pose& believed, std::vector<Vector3f>& out,
               std::mt19937& rng)
{
    std::normal_distribution<float> noise(0.0f, 0.01f);
    // A spinning lidar's azimuth phase is not locked from one revolution to the next; without this a parked
    // 2-deg fan would leave the same cells permanently between two rays, which no real sensor does.
    std::uniform_real_distribution<float> phase(0.0f, 2.0f);
    const float az0 = phase(rng);
    out.clear();
    const Vector3f eye_true = apply(truth, dv.mount);
    for (int L = 0; L < dv.rings; ++L)
    {
        const float el = (dv.el_lo + (dv.el_hi - dv.el_lo) * L / (dv.rings - 1)) * std::numbers::pi_v<float> / 180.f;
        for (int A = 0; A < 180; ++A)                              // 2 deg azimuth (keeps the test fast)
        {
            const float az = (az0 + A * 2.0f) * std::numbers::pi_v<float> / 180.f;
            const Vector3f d(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
            float t = raycast(eye_true, d, 12.0f);
            if (t >= 12.0f or t < dv.min_r) continue;
            t += noise(rng);
            const Vector3f p_true = eye_true + t * d;
            out.push_back(apply(believed, unapply(truth, p_true)));   // registered through the believed pose
        }
    }
    return apply(believed, dv.mount);
}

rc::OccGridParams live_params()
{
    rc::OccGridParams P;               // compiled defaults, then what etc/config.toml overrides
    P.bin_span_m        = 1.6f;        // Grid.BinSpanM
    P.beam_spacing_rad  = 0.0f;        // LidarModel.*BeamSpacingRad = 0 (the live pencil)
    P.floor_z0 = 0.06f; P.floor_slope = 0.04f; P.ceil_z = 1.80f;
    P.lidar_clearance_m = 0.55f;
    P.inflate_radius_m  = 0.0f;
    return P;
}

struct Rig
{
    rc::OccupancyGrid g;
    std::mt19937 rng{1234};
    std::vector<Vector3f> pts;
    explicit Rig(const rc::OccGridParams& P)
    {
        g.reset(-3.4f, -3.4f, 3.4f, 3.4f, P);
        g.set_floor_plane(0.f, 0.f, 0.f, 0.01f);
        g.set_pose_sigma(0.024f, 0.0066f);     // what room_concept reported DURING the incident
    }
    // One agent cycle: bpearl first (begin_cycle), then helios, each from its own origin — as
    // SpecificWorker::integrate_lidar_per_device does.
    void cycle(const Pose& truth, const Pose& believed)
    {
        const Vector3f axis = apply(believed, {0.f, 0.f, 0.f});
        g.set_self_body(axis.x(), axis.y(), 0.55f);
        bool first = true;
        for (const Device* dv : {&kBpearl, &kHelios})
        {
            const Vector3f o = sweep(*dv, truth, believed, pts, rng);
            g.set_device_floor_z0(dv->floor_z0);
            g.set_sensor_min_range(dv->min_r);
            g.set_sensor_noise(0.f, 0.f);
            g.set_sensor_beam_spacing(0.0f);
            g.set_sensor_id(dv->id);
            g.integrate_sweep(o, pts, first, 1.0f);
            first = false;
        }
        g.commit_cycle(0.1f);
    }
};

using CellSet = std::set<std::pair<int,int>>;
float cx_of(const rc::OccupancyGrid& g, int ix) { return g.xmin() + (ix + 0.5f) * g.cell_size(); }
float cy_of(const rc::OccupancyGrid& g, int iy) { return g.ymin() + (iy + 0.5f) * g.cell_size(); }

// Occupied cells at least `m` inside every wall and outside every box (inflated by m): in an otherwise empty
// room these can only be phantoms.
CellSet phantoms(const rc::OccupancyGrid& g, float m = 0.30f)
{
    CellSet s;
    for (int iy = 0; iy < g.height(); ++iy) for (int ix = 0; ix < g.width(); ++ix)
    {
        if (not g.occupied(ix, iy)) continue;
        const float x = cx_of(g, ix), y = cy_of(g, iy);
        if (std::abs(x) > 3.f - m or std::abs(y) > 3.f - m) continue;
        bool in_box = false;
        for (const auto& b : g_boxes)
            in_box = in_box or (x > b.x0 - m and x < b.x1 + m and y > b.y0 - m and y < b.y1 + m);
        if (not in_box) s.insert({ix, iy});
    }
    return s;
}
CellSet box_cells(const rc::OccupancyGrid& g, const Box& b)
{
    CellSet s;
    for (int iy = 0; iy < g.height(); ++iy) for (int ix = 0; ix < g.width(); ++ix)
    {
        const float x = cx_of(g, ix), y = cy_of(g, iy);
        if (x > b.x0 and x < b.x1 and y > b.y0 and y < b.y1) s.insert({ix, iy});
    }
    return s;
}
long count_occ(const rc::OccupancyGrid& g, const CellSet& s)
{ long n = 0; for (const auto& [x, y] : s) n += g.occupied(x, y) ? 1 : 0; return n; }

Pose lerp(const Pose& a, const Pose& b, float t) { return {a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.th + t * (b.th - a.th)}; }

// ── (1) CURTAIN ───────────────────────────────────────────────────────────────────────────────────────────────
void test_curtain(const rc::OccGridParams& P)
{
    std::printf("\n[1] CURTAIN: a wall registered through a 0.47 m / 13.5 deg pose error, then looked through\n");
    g_boxes.clear();
    Rig r(P);
    // Robot dwells 0.75 m from the east wall (x = 3) facing +y. The localiser's belief is 0.47 m west and
    // rotated +0.235 rad, i.e. the try96 numbers (est - truth = (-0.47, +0.04), heading +0.235 rad).
    const Pose truth{2.25f, 0.0f, 0.0f};
    const Pose wrong{truth.x - 0.47f, truth.y + 0.04f, truth.th + 0.235f};
    for (int k = 0; k < 6; ++k) r.cycle(truth, truth);          // first, a correctly-registered look (clean map)
    for (int k = 0; k < 100; ++k) r.cycle(truth, wrong);        // ~10 s dwell through the wrong pose
    const CellSet curtain = phantoms(r.g);
    std::printf("  after the wrong-pose dwell: %zu phantom cells latched in open floor\n", curtain.size());
    check(curtain.size() >= 20, "the wrong-pose dwell must reproduce the curtain (>= 20 phantom cells)");

    // The localiser recovers (try96: 12:14:37). As live, the robot drives ~3 m away and parks for 30 s looking
    // back across the room: every curtain cell is now crossed by correctly-registered beams that return from
    // the true wall behind it.
    const Pose park{-0.8f, -0.6f, 0.3f};
    for (int k = 0; k <= 50; ++k) { const Pose p = lerp(truth, park, k / 50.0f); r.cycle(p, p); }
    for (int k = 0; k < 300; ++k) r.cycle(park, park);
    const long left_parked = count_occ(r.g, curtain);
    // ...and then carries on with its mission: a 26 s drive through the room. From a single parked pose a few
    // of the thinnest cells (a phantom struck at one 2.5 cm height) sit between two ring lines that never move,
    // which the model rightly reads as UNRESOLVED; any change of viewpoint resolves them.
    const Pose tour[] = {park, {0.5f, 1.5f, 1.2f}, {1.0f, -1.5f, -1.0f}, {1.6f, 1.0f, 0.5f}, {-1.5f, 0.5f, 2.5f}};
    for (int leg = 0; leg < 4; ++leg)
        for (int k = 0; k < 66; ++k) { const Pose p = lerp(tour[leg], tour[leg + 1], k / 66.0f); r.cycle(p, p); }
    const long left = count_occ(r.g, curtain);
    std::printf("  curtain cells still occupied: %ld after 30 s parked, %ld after a 26 s drive (of %zu)\n",
                left_parked, left, curtain.size());
    check(left_parked * 10 <= static_cast<long>(curtain.size()),
          "looking through the curtain from one parked pose must remove at least 90% of it");
    check(left == 0, "every curtain cell must be removed once the robot moves on");
}

// ── (2) STANDING ──────────────────────────────────────────────────────────────────────────────────────────────
void test_standing(const rc::OccGridParams& P)
{
    std::printf("\n[2] STANDING: a real 0.4 x 0.4 x 0.6 m box, the robot orbiting it for a minute\n");
    g_boxes = {{0.8f, 1.2f, -0.2f, 0.2f, 0.0f, 0.6f}};
    Rig r(P);
    const auto orbit = [](int k) { const float a = 0.01f * k;
        return Pose{-0.4f + 0.6f * std::cos(a), 0.6f * std::sin(a), a}; };
    for (int k = 0; k < 150; ++k) r.cycle(orbit(k), orbit(k));
    const CellSet box = box_cells(r.g, g_boxes[0]);
    const long before = count_occ(r.g, box);
    long worst = before, released_on_box = 0;
    for (int k = 150; k < 750; ++k)
    {
        r.cycle(orbit(k), orbit(k));
        for (const auto& e : r.g.last_releases())
        { int ix, iy; if (r.g.world_to_cell(e.x, e.y, ix, iy) and box.contains({ix, iy})) ++released_on_box; }
        worst = std::min(worst, count_occ(r.g, box));
    }
    const long after = count_occ(r.g, box);
    std::printf("  box cells occupied: %ld at 15 s -> %ld at 75 s (worst %ld) of %zu; releases on the box %ld\n",
                before, after, worst, box.size(), released_on_box);
    check(before >= 8, "the box must be on the map");
    check(after >= before, "a real obstacle seen repeatedly must not lose cells");
    check(worst * 10 >= before * 9, "...nor dip by more than 10% while it is being looked at");
}

// ── (3) OCCLUDED ──────────────────────────────────────────────────────────────────────────────────────────────
void test_occluded(const rc::OccGridParams& P)
{
    std::printf("\n[3] OCCLUDED: a real box, then the robot parks where a tall partition hides it\n");
    const Box obstacle{1.6f, 2.0f, 1.6f, 2.0f, 0.0f, 0.7f};
    const Box partition{0.6f, 0.7f, 0.4f, 2.9f, 0.0f, 2.4f};      // floor-to-ceiling, between box and parking spot
    g_boxes = {obstacle};
    Rig r(P);
    const Pose look{1.8f, 0.6f, 1.57f};                          // 1.2 m south of the box, looking at it
    for (int k = 0; k < 60; ++k) r.cycle(look, look);
    const CellSet box = box_cells(r.g, obstacle);
    const long seen = count_occ(r.g, box);
    g_boxes = {obstacle, partition};
    const Pose hide{-1.0f, 1.6f, 0.0f};                          // west of the partition: the box is behind it
    for (int k = 0; k <= 20; ++k) { const Pose p = lerp(look, hide, k / 20.0f); r.cycle(p, p); }
    // The drive itself passes in front of the box, so it may be re-struck; what is held is measured from here.
    const long at_hide = count_occ(r.g, box);
    for (int k = 0; k < 400; ++k) r.cycle(hide, hide);
    const long held = count_occ(r.g, box);
    std::printf("  box cells occupied: %ld seen, %ld when hidden, %ld after 40 s hidden (of %zu)\n",
                seen, at_hide, held, box.size());
    check(seen >= 4, "the box must be on the map");
    check(held == at_hide, "an occluded obstacle must HOLD: nothing can see it, so nothing may remove it");
}
}   // namespace

int main(int argc, char** argv)
{
    rc::OccGridParams P = live_params();
    // `--control`: run with the absence channel OFF — the pre-fix behaviour, for the A/B.
    if (argc > 1 and std::string_view(argv[1]) == "--control") P.absence_removes = false;
    std::printf("test_curtain (%s)\n", P.absence_removes ? "absence channel ON" : "CONTROL: absence channel OFF");
    test_curtain(P);
    test_standing(P);
    test_occluded(P);
    std::printf("\ntest_curtain %s (%d failure(s))\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
