/*
 * grid_dynamics_probe.cpp — the three questions a navigation map has to answer, measured.
 *
 *   (A) an object APPEARS in space the robot had already cleared: how many cycles until it is on the map?
 *   (B) an object is REMOVED: how many cycles until it is off the map?
 *   (C) a single spurious return lands on empty floor: how long does the phantom live?
 *
 * These are exactly the three failures reported live ("speckle on the floor", "a removed object takes minutes",
 * "a new object takes forever"). Nothing here touches DSR: the grid is pure Eigen/STL, so the sensor is a real
 * 32-layer ring fan raycast against an analytic room, at the real helios mount height.
 *
 * Build:  g++ -O2 -std=c++23 -I/usr/include/eigen3 tools/grid_dynamics_probe.cpp src/residual_occupancy_grid.cpp
 */
#include "../src/residual_occupancy_grid.h"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <vector>

using Eigen::Vector3f;

// ── the room: 6x6 m, walls at +-3, floor at z=0, and (optionally) one box obstacle ───────────────────────────
struct Box { float x0, x1, y0, y1, z1; };
static bool g_box_present = true;
static Box g_box{1.40f, 1.80f, -0.20f, 0.20f, 0.60f};
// Stray returns per cycle — dust, rain, a reflective surface, a mis-registered scan. A real sweep has
// them and the clean analytic room does not, so without this the speckle filter has nothing to filter
// and the probe would report it as free.
static int g_stray = 0;

// nearest surface along a ray, analytic. Returns range along the ray (<=0 ⇒ nothing).
static float raycast(const Vector3f& o, const Vector3f& d, float tmax)
{
    float best = tmax;
    const auto slab = [&](float lo, float hi, float p, float v, float& t0, float& t1)
    {
        if (std::abs(v) < 1e-9f) { if (p < lo or p > hi) { t0 = 1.f; t1 = -1.f; } return; }
        float a = (lo - p) / v, b = (hi - p) / v; if (a > b) std::swap(a, b);
        t0 = std::max(t0, a); t1 = std::min(t1, b);
    };
    // floor
    if (d.z() < -1e-9f) { const float t = -o.z() / d.z(); if (t > 0.f and t < best) best = t; }
    // the four walls (a slab from inside: the exit point)
    { float t0 = 0.f, t1 = best;
      slab(-3.f, 3.f, o.x(), d.x(), t0, t1); slab(-3.f, 3.f, o.y(), d.y(), t0, t1);
      if (t1 > 0.f and t1 < best) best = t1; }
    // the box
    if (g_box_present)
    { float t0 = 0.f, t1 = best;
      slab(g_box.x0, g_box.x1, o.x(), d.x(), t0, t1);
      slab(g_box.y0, g_box.y1, o.y(), d.y(), t0, t1);
      slab(0.f,      g_box.z1, o.z(), d.z(), t0, t1);
      if (t1 >= t0 and t0 > 1e-3f and t0 < best) best = t0; }
    return best;
}

// one helios sweep from `eye`: 32 layers over [-55,+15] deg, 1 deg azimuth.
static void sweep(const Vector3f& eye, std::vector<Vector3f>& out, std::mt19937& rng)
{
    std::normal_distribution<float> noise(0.0f, 0.015f);      // 1.5 cm range noise
    out.clear(); out.reserve(32 * 360);
    for (int L = 0; L < 32; ++L)
    {
        const float el = (-55.0f + 70.0f * L / 31.0f) * std::numbers::pi_v<float> / 180.0f;
        for (int A = 0; A < 360; ++A)
        {
            const float az = A * std::numbers::pi_v<float> / 180.0f;
            const Vector3f d(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
            float t = raycast(eye, d, 12.0f);
            if (t >= 12.0f) continue;
            t = std::max(0.05f, t + noise(rng));
            out.push_back(eye + t * d);
        }
    }
}

// cells whose centre lies inside the box footprint
static std::vector<std::pair<int,int>> box_cells(const rc::OccupancyGrid& g)
{
    std::vector<std::pair<int,int>> v;
    for (float x = g_box.x0 + 0.01f; x < g_box.x1; x += 0.02f)
        for (float y = g_box.y0 + 0.01f; y < g_box.y1; y += 0.02f)
        { int ix, iy; if (g.world_to_cell(x, y, ix, iy)) { std::pair p{ix, iy};
            if (std::find(v.begin(), v.end(), p) == v.end()) v.push_back(p); } }
    return v;
}
static long n_occupied(const rc::OccupancyGrid& g, const std::vector<std::pair<int,int>>& c)
{ long n = 0; for (const auto& [x, y] : c) n += g.occupied(x, y) ? 1 : 0; return n; }
// ...and what actually SHIPS: the read-out mask, i.e. after the speckle filter. Assert on what leaves the agent,
// not only on what it computes — a publish path can go quiet while every upstream number looks healthy.
static long n_shipped(const rc::OccupancyGrid& g, const std::vector<std::pair<int,int>>& c)
{
    const auto m = g.residual_mask_for_test();
    long n = 0; for (const auto& [x, y] : c) n += m[y * g.width() + x] ? 1 : 0; return n;
}

// occupied cells that are neither the box nor within 0.35 m of a wall = FLOOR SPECKLE.
// `shipped` = the same count taken through the read-out mask, i.e. after the speckle filter.
static long speckle(const rc::OccupancyGrid& g, bool shipped = false)
{
    const auto m = shipped ? g.residual_mask_for_test() : std::vector<std::uint8_t>{};
    long n = 0;
    for (int y = 0; y < g.height(); ++y) for (int x = 0; x < g.width(); ++x)
    {
        if (shipped ? not m[y * g.width() + x] : not g.occupied(x, y)) continue;
        const float wx = g.xmin() + (x + 0.5f) * g.cell_size(), wy = g.ymin() + (y + 0.5f) * g.cell_size();
        if (std::abs(wx) > 2.65f or std::abs(wy) > 2.65f) continue;                 // wall
        if (g_box_present and wx > g_box.x0 - 0.1f and wx < g_box.x1 + 0.1f
            and wy > g_box.y0 - 0.1f and wy < g_box.y1 + 0.1f) continue;            // the box
        ++n;
    }
    return n;
}

int main(int argc, char** argv)
{
    rc::OccGridParams P;                       // the LIVE defaults (config sets these to the same values)
    // allow single-parameter A/B from the command line: name=value
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i]; const auto eq = a.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = a.substr(0, eq); const float v = std::stof(a.substr(eq + 1));
        if      (k == "stable_gain")       P.stable_gain = v;
        else if (k == "l_clamp")           P.l_clamp = v;
        else if (k == "l_miss")            P.l_miss = v;
        else if (k == "occ_clear")         P.occ_clear = v;
        else if (k == "occ_set")           P.occ_set = v;
        else if (k == "lidar_clearance_m") P.lidar_clearance_m = v;
        else if (k == "clear_stop_max_m")  P.clear_stop_max_m = v;
        else if (k == "bin_span_m")        P.bin_span_m = v;
        else if (k == "forget_can_unlatch")P.forget_can_unlatch = v != 0.0f;
        else if (k == "forget_half_life_s")P.forget_half_life_s = v;
        else if (k == "beam_spacing_rad") P.beam_spacing_rad = v;
        else if (k == "speckle_min_neighbours") P.speckle_min_neighbours = static_cast<int>(v);
        else if (k == "speckle_grace_cycles")   P.speckle_grace_cycles   = static_cast<int>(v);
        else if (k == "stray")                  g_stray = static_cast<int>(v);
        else { std::printf("unknown knob '%s'\n", k.c_str()); return 2; }
        std::printf("  [knob] %s = %g\n", k.c_str(), v);
    }

    rc::OccupancyGrid g;
    g.reset(-3.2f, -3.2f, 3.2f, 3.2f, P);
    g.set_floor_plane(0.f, 0.f, 0.f);
    g.set_sensor_min_range(0.40f);             // helios
    g.set_sensor_noise(0.02f, 0.0f);           // lidar-grade: full clearing authority
    g.set_sensor_beam_spacing(-1.0f);          // ...and its vertical sampling interval, from the params

    std::mt19937 rng(1234);
    std::vector<Vector3f> pts;
    const auto cells = box_cells(g);
    const float dt = 0.1f;

    // The robot drives a slow loop around the room centre — a parked robot cannot show any of this.
    const auto pose = [&](int k) {
        const float a = 0.010f * k;
        return Vector3f(-0.9f + 0.55f * std::cos(a), 0.55f * std::sin(a), 1.075f);
    };
    std::uniform_real_distribution<float> ux(-2.6f, 2.6f), uz(0.20f, 1.40f);
    const auto cycle = [&](int k) {
        const Vector3f eye = pose(k);
        g.set_self_body(eye.x(), eye.y(), 0.30f);
        sweep(eye, pts, rng);
        for (int j = 0; j < g_stray; ++j) pts.push_back({ux(rng), ux(rng), uz(rng)});   // dust / mis-registration
        g.integrate_sweep(eye, pts);
        g.commit_cycle(dt);
    };

    // ── phase 0: the room is EMPTY and gets thoroughly cleared ───────────────────────────────────────────────
    g_box_present = false;
    for (int k = 0; k < 400; ++k) cycle(k);
    std::printf("\n[0] empty room, 400 cycles: occupied=%ld  box-cells occupied=%ld/%zu  floor speckle=%ld\n",
                g.occupied_count(), n_occupied(g, cells), cells.size(), speckle(g));

    // ── (A) the object APPEARS ───────────────────────────────────────────────────────────────────────────────
    // A box's FAR half is occluded and never latches, so "50% of the footprint" is not a criterion. What the
    // planner needs is the FIRST cell (does the map know something is there at all) and the plateau.
    g_box_present = true;
    long first_cell = -1;
    std::printf("[A] APPEAR trace (cells of %zu):", cells.size());
    for (int k = 400; k < 700; ++k)
    {
        cycle(k);
        const long n = n_occupied(g, cells);
        if (first_cell < 0 and n > 0) first_cell = k - 400;
        if ((k - 400) % 20 == 0) std::printf(" %ld", n);
    }
    std::printf("\n    first cell latched at cycle %ld (%.1f s)\n", first_cell, first_cell * dt);

    // let it stand, so it banks the evidence a real obstacle banks. A STANDING obstacle must not erode: count
    // every release and the worst dip in coverage — that is the failure the grazing guard exists to prevent, and
    // any change that speeds removal up must be shown NOT to have bought it here.
    long stand_rel = 0, dip = 1 << 30, zheld = 0;
    for (int k = 900; k < 1500; ++k)
    { cycle(k); stand_rel += g.last_sweep_diag().cells_released;
      zheld += g.last_sweep_diag().cells_zheld;
      dip = std::min<long>(dip, n_occupied(g, cells)); }
    const long ship_box = n_shipped(g, cells);      // ...evaluated FIRST: speckle_dropped() reports the LAST
    const long withheld  = g.speckle_dropped();     // read-out, and printf argument order is unspecified
    std::printf("    STANDING 600 cycles: %ld/%zu cells (worst dip %ld), releases(map-wide)=%ld, speckle=%ld,"
                " zheld=%ld/cycle | SHIPPED %ld of the box, filter withheld %ld map-wide\n",
                n_occupied(g, cells), cells.size(), dip, stand_rel, speckle(g), zheld / 600, ship_box, withheld);

    // ── (B) the object is REMOVED ────────────────────────────────────────────────────────────────────────────
    g_box_present = false;
    long gone_half = -1, gone_all = -1;
    const long n0 = n_occupied(g, cells);
    int k = 1500;
    std::printf("[B] REMOVE trace (from %ld cells):", n0);
    for (; k < 1500 + 400; ++k)
    {
        cycle(k);
        const long n = n_occupied(g, cells);
        if (gone_half < 0 and n * 2 <= n0) gone_half = k - 1500;
        if (gone_all < 0 and n == 0) { gone_all = k - 1500; }
        if ((k - 1500) % 40 == 0 and (k - 1500) <= 800) std::printf(" %ld", n);
        if ((k - 1500) % 20 == 0 and (k - 1500) <= 300)
        {   // WHY is it flat? Watch one cell's ledger and the guards that refuse its refutation.
            int px, py; g.world_to_cell(1.45f, 0.0f, px, py);
            const auto& d = g.last_sweep_diag();
            std::printf("\n      c%-4d cell(1.45,0) occ=%d lo=%+6.2f thick=%2d | bins_conf=%-5ld bins_refut=%-7ld"
                        " stopped=%-7ld blind=%-7ld",
                        k - 1500, g.occupied(px, py), g.logodds(px, py), g.column_thickness_bins(px, py),
                        d.bins_confirmed, d.bins_refuted, d.clear_stopped, d.clear_blind_shell);
        }
    }
    std::printf("\n    HALF gone at cycle %ld (%.1f s)   ALL gone at cycle %ld (%.1f s)   still %ld cells\n",
                gone_half, gone_half * dt, gone_all, gone_all * dt, n_occupied(g, cells));

    // what is left standing, and at what height — the orphan-voxel signature
    if (n_occupied(g, cells) > 0)
    {
        std::printf("    surviving columns (bin value at 3.125 cm resolution):\n");
        int shown = 0;
        for (const auto& [cx, cy] : cells)
        {
            if (not g.occupied(cx, cy) or shown++ >= 4) continue;
            const float wx = g.xmin() + (cx + 0.5f) * g.cell_size(), wy = g.ymin() + (cy + 0.5f) * g.cell_size();
            std::printf("      (%.2f,%.2f) lo=%.2f  bins:", wx, wy, g.logodds(cx, cy));
            for (int b = 0; b < 64; ++b)
                if (g.voxel_has_material(cx, cy, b)) std::printf(" %.2fm", (b + 0.5f) * P.bin_span_m / 64.0f);
            std::printf("\n");
        }
    }

    // ── (C) A TRANSIENT OBSTACLE — the honest source of floor speckle ────────────────────────────────────────
    // Not a stray point: a person who walks in, stands 3 s, and leaves. Every return is a true measurement; the
    // world simply changed. Whatever the map keeps afterwards is a phantom the sensor never asked for.
    for (int j = 0; j < 200; ++j) cycle(k + j);       // settle (k is now arm-independent)
    k += 200;
    const long base_speckle = speckle(g);
    g_box_present = true;
    const_cast<Box&>(g_box) = Box{-1.75f, -1.45f, 1.05f, 1.35f, 1.60f};   // a person, mid-room
    for (int j = 0; j < 30; ++j, ++k) cycle(k);
    const long stood = speckle(g);
    g_box_present = false;
    long ghost_life = -1;
    const long ghost0 = speckle(g);                  // cells the person left behind, the moment it left
    for (int j = 0; j < 900; ++j, ++k)
    { cycle(k); if (ghost_life < 0 and speckle(g) <= base_speckle) ghost_life = j; }
    std::printf("[C] TRANSIENT: a person stands 3 s and leaves %ld phantom cells; cleared after %ld cycles"
                " (%.1f s); after 90 s still %ld occupied / %ld SHIPPED  (baseline %ld, %ld while standing)\n",
                ghost0, ghost_life, ghost_life < 0 ? -1.0f : ghost_life * dt, speckle(g), speckle(g, true),
                base_speckle, stood);
    if (ghost_life < 0) std::printf("    ★ NEVER CLEARED in 300 s of driving — %ld phantom cells remain.\n",
                                    speckle(g) - base_speckle);
    return 0;
}
