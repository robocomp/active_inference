// self_body_test — the SELF-BODY birth term (src/bed_self_body.h) against the real compiled Shadow hull.
//
// The live case it pins (objects_live3, 2026-11-14): bed_5 was born from a YOLOE "bed" 0.34 mask on the robot's
// own top plate — the strip the ZED's bottom rows graze, y ∈ [0.156, 0.230] at z ≈ 0.79 across the body width.
// The fix must (a) make that mask worth only a small fraction of an observation, (b) leave a real bed seen from
// ~1–2 m untouched, and (c) degrade gracefully (not veto) when a real mask bleeds a few pixels onto the plate.
#include "bed_self_body.h"
#include "robot_footprint/robot_footprint.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(cond, ...) do { if (not (cond)) { ++failures; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } \
                              else { std::printf("ok  : " __VA_ARGS__); std::printf("\n"); } } while (0)

int main()
{
    using rc::self_body::signed_distance;
    using rc::self_body::world_fraction;
    constexpr float sigma = 0.03f;   // AI2SigmaBaseM in bed_concept/etc/config.toml

    // 1. Signed distance on a unit CCW square.
    const std::vector<Eigen::Vector2f> sq{{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}, {-0.5f, 0.5f}};
    CHECK(std::abs(signed_distance(sq, {0.f, 0.f}) + 0.5f) < 1e-6f, "centre of unit square is 0.5 inside");
    CHECK(std::abs(signed_distance(sq, {1.f, 0.f}) - 0.5f) < 1e-6f, "point 0.5 right of the square is +0.5");
    CHECK(std::abs(signed_distance(sq, {1.f, 1.f}) - std::sqrt(0.5f)) < 1e-6f, "corner distance is Euclidean");
    CHECK(world_fraction({}, std::vector<Eigen::Vector3f>{{0, 0, 0}}, sigma) == 1.0f, "no polygon ⇒ asserts nothing (1)");
    CHECK(world_fraction(sq, std::vector<Eigen::Vector3f>{}, sigma) == 1.0f, "no points ⇒ asserts nothing (1)");

    const auto body = rc::RobotFootprint::shadow();
    const auto& poly = body.polygon();
    CHECK(std::abs(signed_distance(poly, {0.f, 0.23f})) < 0.01f, "Shadow hull front edge is at y = +0.23 (robot frame)");

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::normal_distribution<float> N(0.f, 0.005f);

    // 2. The bed_5 cloud: the ZED (body y = -0.075, z ≈ 0.98) sees its own top plate (z ≈ 0.79) from y = 0.156
    //    (the -39° bottom image row) to the front edge y = 0.230, across x ∈ [-0.20, 0.20]. Plus the ~5% of
    //    points that bled off the edge onto the far floor (the live mask's z-range was far larger than the
    //    plate's, inferred from its birth speed), which ARE world.
    std::vector<Eigen::Vector3f> plate;
    for (int i = 0; i < 13000; ++i)
        plate.emplace_back(-0.20f + 0.40f * U(rng) + N(rng), 0.156f + 0.074f * U(rng) + N(rng), 0.79f + N(rng));
    const float f_plate = world_fraction(poly, plate, sigma);
    std::vector<Eigen::Vector3f> plate_bleed = plate;
    for (int i = 0; i < 650; ++i)
        plate_bleed.emplace_back(-0.3f + 0.6f * U(rng), 1.1f + 0.3f * U(rng), 0.0f);
    const float f_plate_bleed = world_fraction(poly, plate_bleed, sigma);
    std::printf("      plate-only world fraction %.3f, with 5%% floor bleed %.3f\n", f_plate, f_plate_bleed);
    CHECK(f_plate < 0.25f, "the robot's own top plate is mostly NOT world (f=%.3f)", f_plate);
    CHECK(f_plate_bleed < 0.25f, "…and stays so with floor bleed (f=%.3f)", f_plate_bleed);

    // Birth arithmetic at the live numbers: conf 0.34, BirthFrames 8, every other factor ≈ 1.
    const float before = 8.0f / 0.34f, after = 8.0f / (0.34f * f_plate_bleed);
    std::printf("      observations to birth the plate: before %.0f (~%.1f s at 9.5 Hz), after %.0f (~%.1f s)\n",
                before, before / 9.5f, after, after / 9.5f);
    CHECK(after > 4.0f * before, "the plate needs >4x the observations it needed live");

    // 3. A real bed at bed_4's live geometry relative to the robot (its mask centroid 1.1 m ahead, 0.9 m to the
    //    side): a 1.75 x 2.10 x 0.65 box whose near edge is 0.6 m ahead. Every point is world.
    std::vector<Eigen::Vector3f> bed;
    for (int i = 0; i < 20000; ++i)
    {
        const float x = -0.9f - 0.875f + 1.75f * U(rng), y = 0.6f + 2.10f * U(rng);
        bed.emplace_back(x, y, 0.65f * U(rng));
    }
    const float f_bed = world_fraction(poly, bed, sigma);
    CHECK(f_bed > 0.999f, "a real bed 1-2 m away is entirely world (f=%.4f)", f_bed);

    // 4. A real bed mask standing right in front of the robot (the near face 0.35 m ahead, the closest a still
    //    robot comes) that bleeds 5% of its pixels onto the plate: degraded, not vetoed.
    std::vector<Eigen::Vector3f> bed_near;
    for (int i = 0; i < 19000; ++i)
        bed_near.emplace_back(-0.875f + 1.75f * U(rng), 0.35f + 2.10f * U(rng), 0.65f * U(rng));
    for (int i = 0; i < 1000; ++i) bed_near.push_back(plate[static_cast<std::size_t>(i)]);
    const float f_near = world_fraction(poly, bed_near, sigma);
    CHECK(f_near > 0.94f and f_near < 0.999f, "a near bed with 5%% plate bleed keeps its evidence (f=%.3f)", f_near);

    // 5. Continuity across the body edge: Φ(0) = 1/2 on the silhouette, monotone outward.
    CHECK(std::abs(rc::self_body::p_world(0.f, sigma) - 0.5f) < 1e-6f, "P_world = 1/2 on the body surface");
    CHECK(rc::self_body::p_world(-0.03f, sigma) < rc::self_body::p_world(0.03f, sigma), "P_world grows outward");

    std::printf("%s (%d failure%s)\n", failures == 0 ? "self_body_test PASS" : "self_body_test FAIL", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
