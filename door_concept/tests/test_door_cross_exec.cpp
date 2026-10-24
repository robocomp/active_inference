// test_door_cross_exec — the cross contract's `pre:` clause (door_cross_exec.h): P(the run robot -> through-pose is clear
// of the LEAF), marginalised over the phi / hinge / swing beliefs.
//   1. gold13 geometry: leaf opened 90 deg toward the robot's room, robot BESIDE it (0.4 m off the aperture axis, level with
//      the leaf) => low;
//   2. the same leaf, robot ON AXIS in front at the standpoint => high;
//   3. a SHUT leaf (phi 0) blocks the run => low (the passable factor is low then too);
//   4. leaf swung to the far side, robot on axis in front: clear;
//   5. the product (cross_exec_prob) is continuous: no step in the robot's lateral offset.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_cross_exec.cpp -o /tmp/test_door_cross_exec && /tmp/test_door_cross_exec
#include <cmath>
#include <print>

#include "../src/door_cross_exec.h"

namespace
{
int failures = 0;
void check(bool ok, int line, const char* what) { if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); } }
#define CHECK(x) check((x), __LINE__, #x)
using namespace rc::door;

// door in the wall y = 0, aperture x in [-0.5, +0.5] (s = 0, wall_O = (-0.5, 0)), across_u = (0, +1)... wall_u = (1,0).
CrossCorridorBelief belief(float phi, float swing, float swing_llr = 6.0f)
{
    CrossCorridorBelief b;
    b.ap.wall_O = {-0.5f, 0.0f}; b.ap.wall_u = {1.0f, 0.0f}; b.ap.s = 0.0f; b.ap.w = 1.0f;
    b.hinge = HingeSide::Near; b.hinge_llr = 6.0f;   // hinge at x = -0.5
    b.swing = swing; b.swing_llr = swing_llr;
    b.phi_curve_log = {{phi, 0.0}};                  // a collapsed posterior at phi
    return b;
}
constexpr float kBody = 0.40f, kSigma = 0.10f;
const Eigen::Vector2f through{0.0f, -1.8f};          // far side of the wall
}

int main()
{
    const float half_pi = 1.5708f;
    {   // 1. leaf open 90 deg into the robot's side (+y), robot beside it, level with the leaf at x = -0.45 -> run crosses it
        const auto b = belief(half_pi, +1.0f);
        const float p = prob_corridor_clear(b, {-0.20f, 0.60f}, through, kBody, kSigma);
        std::println("beside the open leaf: P(clear) = {:.3f}", p);
        CHECK(p < 0.30f);
    }
    {   // 2. on axis in front at the standpoint (1.75 m out): the leaf (x = -0.5, y 0..1) is 0.5 m from the axis > body 0.4
        const auto b = belief(half_pi, +1.0f);
        const float p = prob_corridor_clear(b, {0.0f, 1.75f}, through, kBody, kSigma);
        std::println("on axis in front: P(clear) = {:.3f}", p);
        CHECK(p > 0.80f);
    }
    {   // 3. a shut leaf fills the aperture: the run goes through it
        const auto b = belief(0.0f, +1.0f);
        CHECK(prob_corridor_clear(b, {0.0f, 1.75f}, through, kBody, kSigma) < 0.10f);
    }
    {   // 4. leaf swung to the FAR side (-y), robot on axis in front: the leaf is 0.5 m off the axis => clear
        CHECK(prob_corridor_clear(belief(half_pi, -1.0f, 8.0f), {0.0f, 1.75f}, through, kBody, kSigma) > 0.80f);
    }
    {   // 5. the product is continuous in the robot's lateral offset (largest step between 5 mm samples stays small)
        const auto b = belief(half_pi, +1.0f);
        float prev = -1.f, worst = 0.f;
        for (int i = 0; i <= 200; ++i)
        {
            const float x = -0.6f + 0.01f * float(i);
            const float p = cross_exec_prob(0.9f, prob_corridor_clear(b, {x, 1.2f}, through, kBody, kSigma), 0.95f);
            if (prev >= 0.f) worst = std::max(worst, std::abs(p - prev));
            prev = p;
        }
        std::println("largest step over 1 cm of lateral offset: {:.3f}", worst);
        CHECK(worst < 0.15f);
    }
    {   // 6. the product: low corridor => low pre even with P(at door) and P(passable) high; empty curve => 0 (not measured)
        CHECK(cross_exec_prob(0.67f, 0.1f, 1.0f) < 0.10f);
        CrossCorridorBelief empty = belief(half_pi, +1.0f); empty.phi_curve_log.clear();
        CHECK(prob_corridor_clear(empty, {0.f, 1.f}, through, kBody, kSigma) == 0.0f);
    }
    std::println("{}", failures == 0 ? "ALL PASS" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
