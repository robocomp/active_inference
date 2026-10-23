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

    // ── Task M: the published mouth pose is the proto frame's pose in the old room ──────────────────────────────
    std::printf("mouth pose (Task M)\n");
    for (const double yaw : {0.0, 0.7, -2.4})
        for (const bool dense : {false, true})
        {
            // A room rotated by `yaw`, door in its +y wall, 32-vertex (jamb-nearest) outline when dense.
            const Eigen::Rotation2Df R(static_cast<float>(yaw));
            auto rotv = [&](std::vector<Vector2f> v) { for (auto &p : v) p = R * p; return v; };
            const auto outline = rotv(dense ? notch : rect);
            rc::DoorAperture a2;
            a2.a = R * Vector2f(-0.1f, 3); a2.b = R * Vector2f(0.1f, 3); a2.p_open = 1.f; a2.sigma_along = 0.02f;
            const auto cc = rc::crossing::evaluate(a2, outline, R * Vector2f(0.3f, 3.8f), cov, 0.25f);
            const auto mo = rc::crossing::mouth_of(cc, a2.sigma_along, 0.06f);
            const auto Tp = rc::crossing::proto_frame(cc);
            std::printf("  yaw %.1f %s (%zu vertices)\n", yaw, dense ? "notch" : "rect", outline.size());
            check(cc.valid, "valid");
            check(near(mo.x, Tp.translation().x(), 1e-6f) and near(mo.y, Tp.translation().y(), 1e-6f), "mouth xy = proto origin in old room");
            // rebuild the rotation from the published yaw: must equal the proto frame's linear part, orthonormal
            const Eigen::Matrix2f Rm = Eigen::Rotation2Df(mo.yaw).toRotationMatrix();
            check((Rm - Tp.linear()).cwiseAbs().maxCoeff() < 1e-5f, "mouth yaw reproduces the proto frame rotation");
            check(near((Rm.transpose() * Rm - Eigen::Matrix2f::Identity()).cwiseAbs().maxCoeff(), 0.f, 1e-5f) and near(Rm.determinant(), 1.f), "orthonormal, det +1");
            {   // tangent sigma 0.02 (door_concept), normal sigma 0.06 (layout bar): C = 0.02^2 t t' + 0.06^2 n n' in the old frame
                const Eigen::Vector2f t = Tp.linear().col(0), n = Tp.linear().col(1);
                const Eigen::Matrix2f Ce = 4e-4f * t * t.transpose() + 3.6e-3f * n * n.transpose();
                check(mo.has_cov and near(mo.cov[0], Ce(0, 0), 1e-7f) and near(mo.cov[1], Ce(0, 1), 1e-7f) and near(mo.cov[7], Ce(1, 1), 1e-7f)
                          and near(mo.cov[5 * 6 + 5], 2.f * 3.6e-3f / (0.2f * 0.2f), 1e-4f),
                      "covariance: tangent from the aperture sigma, normal from the layout bar, yaw from both ends");
            }
            // the robot expressed through the published pose = through the proto frame
            const Vector2f robot_old = R * Vector2f(0.3f, 3.8f);
            const Vector2f via_pub = Rm.transpose() * (robot_old - Vector2f(mo.x, mo.y));
            check((via_pub - Tp.inverse() * robot_old).norm() < 1e-4f, "robot in the proto frame via the published pose");
        }
    check(not rc::crossing::mouth_of(c, 0.f, 0.f).has_cov, "mouth_of itself states nothing from zero sigmas (the caller supplies the layout bar)");

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
