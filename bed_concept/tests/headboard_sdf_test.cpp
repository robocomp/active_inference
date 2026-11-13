// headboard_sdf_test — the bed's compound SDF (box ∪ headboard panel, 2026-11-14). Run: ./headboard_sdf_test
// Checks the geometry the owner asked for: the mattress box is the height, the headboard is the bed's but never the top.
#include <cmath>
#include <cstdio>
#include "bed_belief.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static bool near(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) < tol; }

int main()
{
    using namespace rc;
    BedBeliefParams p; p.headboard_h_m = 1.25f; p.headboard_t_m = 0.10f;
    const BedBeliefState s{0.0f, 0.0f, 0.65f, 1.75f, 2.10f, 0.0f};   // centred, yaw 0: back (+Y) at y = 1.05
    const BedBelief b(s, p);
    // a point ON the headboard top, at the back, 1.25 m up: explained by the panel, far from the mattress box
    const Eigen::Vector3f hb_top(0.0f, 1.0f, 1.25f);
    CHECK(near(b.sdf_headboard(hb_top, s), 0.0f));
    CHECK(b.sdf_box(hb_top, s) > 0.5f);
    CHECK(near(b.sdf_compound(hb_top, s), 0.0f));
    // the headboard's SDF does not depend on H: the panel never pulls the mattress height
    BedBeliefState s2 = s; s2.H = 0.40f;
    CHECK(near(b.sdf_headboard(hb_top, s2), b.sdf_headboard(hb_top, s)));
    // a point on the mattress top mid-bed is the box's
    const Eigen::Vector3f mat(0.0f, 0.0f, 0.65f);
    CHECK(near(b.sdf_box(mat, s), 0.0f) and b.sdf_headboard(mat, s) > 0.5f);
    // the same height at the FOOT end is NOT explained: the panel is on the back only (front/back evidence)
    const Eigen::Vector3f foot_high(0.0f, -1.0f, 1.25f);
    CHECK(b.sdf_compound(foot_high, s) > 0.5f);
    // yaw 180°: the headboard moves to −Y
    BedBeliefState s3 = s; s3.yaw = 3.14159265f;
    CHECK(near(b.sdf_headboard(foot_high, s3), 0.0f, 1e-3f));
    // off = plain box (the refrigerator parent's model)
    const BedBelief plain(s, BedBeliefParams{});
    CHECK(std::isinf(plain.sdf_headboard(hb_top, s)) and near(plain.sdf_compound(hb_top, s), plain.sdf_box(hb_top, s)));
    std::printf("headboard_sdf_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
