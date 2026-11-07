// Offline harness for src/core/place_prior.h (the structural prior of a door place). Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_place_prior.cpp -o /tmp/test_place_prior && /tmp/test_place_prior
#include "../src/core/place_prior.h"
#include <cmath>
#include <cstdio>

using namespace rc::door::core;
static int g_fail = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// 6 x 4 room, room-side faces; the bottom wall is y = 0, so an aperture in its band sits at y in [-0.2, 0].
static const std::vector<Vec2f> kRoom{{0, 0}, {6, 0}, {6, 4}, {0, 4}};

static PlacePriorInput input(const std::vector<Vec2f>& poly, Vec2f ap, float w, int before = 0, bool proto = false)
{
    return PlacePriorInput{poly, proto, rc::door::RoomBand{0.20f, 0.10f}, ap, Vec2f(1, 0), w, 2.05f, 0.0f, 0.0f, true, before};
}

int main()
{
    const float first = std::log(0.4f / 0.6f);
    // typical_doors
    for (float w : {0.80f, 1.00f})
        CHECK(std::abs(place_prior(input(kRoom, {3, -0.1f}, w)).total() - first) < 1.0f);
    // wide_candidate_pays
    CHECK(place_prior(input(kRoom, {3, -0.1f}, 0.85f)).total() - place_prior(input(kRoom, {3, -0.1f}, 3.0f)).total() >= 10.0f);
    // off_wall_floor
    CHECK(place_prior(input(kRoom, {3, 1.5f}, 0.85f)).total() < -10.0f);
    // second_door
    CHECK(std::abs(place_prior(input(kRoom, {3, -0.1f}, 0.85f, 1)).base_rate - std::log(0.01f / 0.99f)) < 1e-5f);
    CHECK(std::abs(place_prior(input(kRoom, {3, -0.1f}, 0.85f, 0)).base_rate - first) < 1e-5f);
    // proto_frontier: one-sided, no cutoff. Which sign is outside: y = -0.5 is beyond the wall (the promoted room pays there).
    {
        const auto inside_proto = place_prior(input(kRoom, {3, 0.5f}, 0.85f, 0, true));
        const auto inside_prom  = place_prior(input(kRoom, {3, 0.5f}, 0.85f, 0, false));
        CHECK(inside_proto.band < -1.0f);                                       // inside the measured floor, off the band: pays
        CHECK(inside_proto.band == inside_prom.band);
        // at the frontier (on the band) a proto door pays ~0
        CHECK(std::abs(place_prior(input(kRoom, {3, -0.1f}, 0.85f, 0, true)).band) < 1e-6f);
        // 6.7 m beyond the measured floor pays as off-wall (the same as a promoted room), not 0
        const auto far_proto = place_prior(input(kRoom, {3, -6.7f}, 0.85f, 0, true));
        CHECK(far_proto.band < -100.0f and far_proto.band == place_prior(input(kRoom, {3, -6.7f}, 0.85f, 0, false)).band);
        // a published edge uncertainty widens sigma on the outside only: 0.4 m past the band pays less with sigma 0.5 than without
        auto in_u = input(kRoom, {3, -0.6f}, 0.85f, 0, true); in_u.edge_sigma_m = 0.5f;
        CHECK(place_prior(in_u).band > place_prior(input(kRoom, {3, -0.6f}, 0.85f, 0, true)).band);
        auto in_i = input(kRoom, {3, 0.5f}, 0.85f, 0, true); in_i.edge_sigma_m = 0.5f;
        CHECK(place_prior(in_i).band == inside_proto.band);                     // the inside is unchanged
    }
    // jogged_wall_is_one_group
    {
        const std::vector<Vec2f> jog{{0, 0}, {3, 0}, {3, 0.05f}, {6, 0.05f}, {6, 4}, {0, 4}};
        const auto g = wall_groups(jog, 0.10f);
        CHECK(g.size() == 6);
        CHECK(g[0] == g[1] and g[1] == g[2]);
        CHECK(g[2] != g[3]);
        const auto g2 = wall_groups(jog, 0.10f);
        CHECK(g2 == g);
        // edges: 0 (0,0)-(3,0), 1 jog, 2 (3,.05)-(6,.05), 3 right, 4 top, 5 left  -> 4 walls
        const std::vector<int> want{0, 0, 0, 1, 2, 3};
        CHECK(g == want);
        const auto r = wall_groups(kRoom, 0.10f);
        const std::vector<int> want_r{0, 1, 2, 3};
        CHECK(r == want_r);
    }
    // unresolved_size_pays_nothing
    {
        auto in = input(kRoom, {3, -0.1f}, 3.0f);
        in.size_resolved = false;
        CHECK(place_prior(in).size == 0.0f);
        in.size_resolved = true;
        CHECK(place_prior(in).size < 0.0f);
        CHECK(std::abs(place_prior(input(kRoom, {3, -0.1f}, 0.85f)).size) < 1e-6f);
    }
    std::printf(g_fail ? "FAILED\n" : "ALL PASS\n");
    return g_fail != 0;
}
