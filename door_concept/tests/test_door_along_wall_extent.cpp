// Task D-C round 2 (tour3C): THE APERTURE IS IN ITS OWN WALL — rc::door::along_wall_update's own-wall term and
// rc::door::door_wall_extent. tour3C, leg 5: DOOR_0 was RESTORED in room_1 at (0.55, 2.54) yaw −3.14 with the birth seed's
// σ_s 0.6 m; the LiDAR's along-wall channel (the door remembered open, P(span) 0.93) associated the through-span BEYOND its
// wall's end — the open stretch past the partition corner at x 0.10, aw_z 4.1-4.4 m along against s 2.98 — and slid the
// aperture 0.9 m to x −0.39, off its wall; the wall-band prior then walked L from +2.6 to −3.47 and the REAL door was
// removed (with its identity row). The geometry below is tour3C's room_1 wall (polygon edge (0.099, 2.536)→(4.032, 2.538)).
//   (a) RED before the fix: with no extent the far span drags s by > 0.3 m; with the extent it moves s by < 2 cm.
//   (b) the TRUE opening (z 2.93, inside the wall) is still associated and fused (β > 0.5, s moves onto it).
//   (c) door_wall_extent: the run containing the centre; a doorway in a GAP between two runs gets both flanking runs.
// Build: g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_along_wall_extent.cpp src/door_belief.cpp -o /tmp/t
#include <cmath>
#include <cstdio>
#include <print>
#include <vector>

#include "../src/door_belief.h"
#include "../src/door_opening_edges.h"
#include "../src/door_phi_posterior.h"

namespace
{
int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("  FAIL line {}: {}\n", __LINE__, #c); } } while (0)

rc::DoorBelief restored(float sd_s)
{
    rc::DoorBeliefParams p;
    p.wall_O = {4.032f, 2.538f};
    p.wall_u = {-1.0f, 0.0f};
    p.leaf.phi = 1.5f;   // remembered open (the leaf memory)
    rc::DoorBeliefState s0;
    s0.s = 2.98f; s0.w = 1.01f; s0.h = 2.11f;
    rc::DoorBelief b(s0, p);
    b.set_placement_sd(sd_s);
    return b;
}

// A full-width through-span measuring s = z (OpeningRun edges in the wall frame: s ∈ [a_right − w, a_left]).
rc::door::OpeningRun span_at(float z, float w)
{
    rc::door::OpeningRun r;
    r.a_left = z; r.a_right = z + w;
    r.var_left = r.var_right = 0.0004f;
    r.n_through = 40;
    return r;
}
}  // namespace

int main()
{
    // tour3C's room_1 polygon around DOOR_0's wall: the wall y 2.536 from x 0.099 to 4.032, the partition sliver at x ≈ 0.
    const std::vector<Eigen::Vector2f> poly{{-4.139f, 2.462f}, {-1.679f, 2.474f}, {-1.667f, 2.535f}, {-2.082f, 2.541f},
                                            {-2.076f, 3.398f}, {-1.630f, 3.379f}, {-1.617f, 4.628f}, {0.040f, 4.605f},
                                            {-0.060f, 0.396f}, {0.059f, 0.382f}, {0.099f, 2.536f}, {4.032f, 2.538f},
                                            {4.253f, 2.264f}, {4.076f, -3.434f}, {-4.241f, -3.418f}};
    const float sigma_b = 0.10f;   // Existence.BoundarySigmaM
    const auto nz0 = rc::door::phi_field_nuisance(restored(0.6f).aperture());   // the field's depth band, as the agent passes it
    const float depth_lo = nz0.depth_lo, depth_hi = nz0.depth_hi;

    // (c) the extent of the wall the aperture is cut in
    {
        const auto b = restored(0.6f);
        const auto [lo, hi] = rc::door::door_wall_extent(b.aperture(), poly, depth_lo, depth_hi);
        std::print("door wall extent along u: [{:.3f}, {:.3f}] (expected [0, 3.933])\n", lo, hi);
        CHECK(std::abs(lo - 0.0f) < 0.01f and std::abs(hi - 3.933f) < 0.01f);
        // a doorway in a GAP: the wall line broken at the opening — both flanking runs
        const std::vector<Eigen::Vector2f> gap{{0.f, 0.f}, {4.f, 0.f}, {4.f, 2.5f}, {2.4f, 2.5f}, {2.4f, 2.9f}, {1.4f, 2.9f},
                                               {1.4f, 2.5f}, {0.f, 2.5f}};
        rc::DoorBeliefParams p; p.wall_O = {4.f, 2.5f}; p.wall_u = {-1.f, 0.f};
        rc::DoorBeliefState s0; s0.s = 1.6f; s0.w = 1.0f;
        const rc::DoorBelief g(s0, p);
        const auto [glo, ghi] = rc::door::door_wall_extent(g.aperture(), gap, -0.3f, 0.3f);
        std::print("doorway in a gap: extent [{:.2f}, {:.2f}] (expected [0, 4])\n", glo, ghi);
        CHECK(std::abs(glo) < 0.01f and std::abs(ghi - 4.0f) < 0.01f);
    }

    // (a) the span past the wall's end, measured with the live Pb (b's variance had grown to 0.4 m² by then)
    for (const bool extent : {false, true})
    {
        auto b = restored(0.6f);
        rc::door::AlongWallState st;
        st.init = true; st.b = 0.f; st.Pb = 0.4f;
        const auto [lo, hi] = rc::door::door_wall_extent(b.aperture(), poly, depth_lo, depth_hi);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float s0 = b.state().s;
        for (int k = 0; k < 6; ++k)
            (void)rc::door::along_wall_update(b, st, {span_at(4.30f, 1.01f)}, rc::door::WallRegistration{}, 0.93f,
                                              0.004f, 3.0f, 0.2f, extent ? lo : nan, extent ? hi : nan, sigma_b);
        const float ds = b.state().s - s0;
        std::print("span 1.3 m past the wall's end, {} extent: s {:.3f} → {:.3f} (Δ {:+.3f}), b {:+.3f}\n",
                   extent ? "WITH" : "without", s0, b.state().s, ds, st.b);
        if (extent) CHECK(std::abs(ds) < 0.02f and std::abs(st.b) < 0.02f);
        else        CHECK(std::abs(ds) > 0.3f);   // the defect, reproduced
    }

    // (b) the true opening still fuses
    {
        auto b = restored(std::hypot(0.04f, sigma_b));
        rc::door::AlongWallState st;
        st.init = true; st.b = 0.f; st.Pb = 0.004f;
        const auto [lo, hi] = rc::door::door_wall_extent(b.aperture(), poly, depth_lo, depth_hi);
        const auto r = rc::door::along_wall_update(b, st, {span_at(2.93f, 1.01f)}, rc::door::WallRegistration{}, 0.93f,
                                                   0.004f, 3.0f, 0.2f, lo, hi, sigma_b);
        std::print("true opening z 2.93: beta {:.3f}, s 2.980 → {:.3f}\n", r.beta, b.state().s);
        CHECK(r.updated and r.beta > 0.5f);
        CHECK(b.state().s < 2.975f);
    }

    if (fails == 0) std::print("test_door_along_wall_extent: ALL PASS\n");
    else            std::print("test_door_along_wall_extent: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
