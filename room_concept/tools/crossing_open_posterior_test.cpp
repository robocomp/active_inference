// Task C: the crossing posterior CONDITIONS on the passage. A shut leaf cannot be passed without contact, so a
// pose reading "past the door" is evidence the door is OPEN (and, for a shut door, evidence the pose is off).
// Regression for crossing/gold3 leg 4: p_open 0.65-0.79, p_geom 1, p_cross = 0.79 < BirthProb 0.95 ⇒ no proto.
//   g++ -std=c++23 -I/usr/include/eigen3 -I../src crossing_open_posterior_test.cpp -o crossing_open_posterior_test
#include "door_crossing.h"
#include <cstdio>

using Eigen::Vector2f;
static int failures = 0;
static void check(bool ok, const char *what)
{ if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }

static float p_cross(float p_open, Vector2f robot, float sigma)
{
    const std::vector<Vector2f> rect = {{-3, -2}, {3, -2}, {3, 3}, {-3, 3}};
    rc::DoorAperture ap; ap.a = {-0.5f, 3}; ap.b = {0.5f, 3}; ap.p_open = p_open;
    return rc::crossing::evaluate(ap, rect, robot, Eigen::Matrix2f::Identity() * sigma * sigma, 0.25f).p_cross;
}

int main()
{
    constexpr float birth = 0.95f;
    std::printf("door P(open)=0.75, robot measured 0.3 m past, inside the span\n");
    check(p_cross(0.75f, {0.f, 3.3f}, 0.02f) > birth, "p_cross > 0.95 (the passage is evidence of the open leaf)");
    std::printf("door P(open)=0.65 (gold3 floor), robot 0.32 m past, sigma 0.05\n");
    check(p_cross(0.65f, {0.f, 3.32f}, 0.05f) > birth, "p_cross > 0.95");
    std::printf("door P(open)=0.75, robot in front\n");
    check(p_cross(0.75f, {0.f, 2.5f}, 0.02f) < 0.01f, "no birth in front of the door");
    check(p_cross(0.75f, {0.f, 3.0f}, 0.05f) < birth, "on the line: no birth");
    std::printf("shut door (P(open)=0), pose jitter across the plane\n");
    check(p_cross(0.f, {0.f, 3.05f}, 0.10f) < 0.05f, "jitter of one sigma past a shut door: no birth");
    check(p_cross(0.f, {0.f, 3.30f}, 0.10f) < birth, "3 sigma past a shut door: still no birth (shut outweighs)");
    std::printf("unmeasured door\n");
    check(p_cross(-1.f, {0.f, 3.3f}, 0.02f) > 0.9f, "unmeasured: passage still decides");
    std::printf("monotone in P(open)\n");
    check(p_cross(0.9f, {0.f, 3.2f}, 0.05f) >= p_cross(0.3f, {0.f, 3.2f}, 0.05f), "more open => more crossed");
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
