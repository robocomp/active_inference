// Task S: the proto-room frame comes from the APERTURE (x = a->b, y = its perpendicular), never from the
// nearest polygon edge. Regression for crossing/gold1 (2026-10-22): on a recalled many-vertex outline the
// edge nearest the aperture centre was a jamb running ACROSS the door, y came out ~parallel to x, det 0.019.
//   g++ -std=c++23 -I/usr/include/eigen3 -I../src proto_frame_test.cpp -o proto_frame_test && ./proto_frame_test
#include "door_crossing.h"
#include <cstdio>

using Eigen::Vector2f;
static int failures = 0;
static void check(bool ok, const char *what)
{ if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }
static bool near(float a, float b, float tol = 1e-4f) { return std::abs(a - b) <= tol; }

// Subdivide each edge into k pieces: a recalled outline is a many-vertex polygon.
static std::vector<Vector2f> densify(const std::vector<Vector2f> &p, int k)
{
    std::vector<Vector2f> o;
    for (std::size_t i = 0; i < p.size(); ++i)
        for (int j = 0; j < k; ++j)
            o.push_back(p[i] + (p[(i + 1) % p.size()] - p[i]) * (float(j) / float(k)));
    return o;
}

int main()
{
    // Room x[-3,3] y[-2,3], CCW, door in the y=3 wall as a NARROW recess: jambs at x=±0.1 (0.1 from the
    // aperture centre) are nearer than the back wall (0.5) — the gold1 configuration.
    const std::vector<Vector2f> notch = densify({{-3, -2}, {3, -2}, {3, 3}, {0.1f, 3}, {0.1f, 3.5f},
                                                 {-0.1f, 3.5f}, {-0.1f, 3}, {-3, 3}}, 4);
    const std::vector<Vector2f> rect = {{-3, -2}, {3, -2}, {3, 3}, {-3, 3}};
    const Eigen::Matrix2f cov = Eigen::Matrix2f::Identity() * 0.0004f;
    const Vector2f robot(0.3f, 3.8f);   // past the door, into the next room

    for (bool reversed : {false, true})
    {
        rc::DoorAperture ap;
        ap.a = reversed ? Vector2f(0.1f, 3) : Vector2f(-0.1f, 3);
        ap.b = reversed ? Vector2f(-0.1f, 3) : Vector2f(0.1f, 3);
        ap.p_open = 1.f;
        std::printf("jamb-nearest outline (%zu vertices), a/b %s\n", notch.size(), reversed ? "reversed" : "forward");
        const auto c = rc::crossing::evaluate(ap, notch, robot, cov, 0.25f);
        check(c.valid, "valid");
        check(near(c.n_out.dot(c.tangent), 0.f), "n_out perpendicular to a->b");
        check(near(c.n_out.x(), 0.f) and near(c.n_out.y(), 1.f), "n_out = +y (out of the room)");
        check(near(c.s, 0.8f) and near(std::abs(c.u), 0.3f), "s along the aperture normal, u along its tangent");
        const auto T = rc::crossing::proto_frame(c);
        check(near(T.linear().determinant(), 1.f), "det(T) = +1");
        const Vector2f in_proto = T.inverse() * robot;
        check(near(in_proto.y(), 0.8f) and near(std::abs(in_proto.x()), 0.3f), "robot pose in the proto = its pose relative to the aperture");
    }

    std::printf("plain rectangle\n");
    rc::DoorAperture ap; ap.a = {-0.5f, 3}; ap.b = {0.5f, 3}; ap.p_open = 1.f;
    const auto c = rc::crossing::evaluate(ap, rect, robot, cov, 0.25f);
    check(near(c.n_out.x(), 0.f) and near(c.n_out.y(), 1.f), "n_out = +y");
    const auto T = rc::crossing::proto_frame(c);
    check(near(T.linear().determinant(), 1.f) and near(T.linear()(0, 0), 1.f) and near(T.linear()(1, 1), 1.f), "frame = identity rotation");
    check(near(c.s, 0.8f) and near(c.u, 0.3f), "s, u");

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
