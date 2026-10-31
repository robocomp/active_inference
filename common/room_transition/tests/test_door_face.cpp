// door_face.h: the doorway as the rooms' one attaching point. Two rectangular rooms sharing a wall with a 1 m door; every
// combination of outline winding, door-node yaw (wall direction) and room-b frame pose must place room b exactly where it is,
// and never over room a.
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -o /tmp/tdf test_door_face.cpp && /tmp/tdf
#include "../door_face.h"

#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, const char* what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL: {}", what); }
}
Eigen::Affine2f pose(float x, float y, float yaw)
{
    Eigen::Affine2f T = Eigen::Affine2f::Identity();
    T.translation() = Eigen::Vector2f(x, y);
    T.linear() = Eigen::Rotation2Df(yaw).toRotationMatrix();
    return T;
}
}  // namespace

int main()
{
    using namespace rc::door_face;
    // World: room a = [-4,4] x [-4.6,4.6]; room b = [-4,4] x [4.6,13]; DOOR at world (0, 4.6), wall along x.
    const Poly a_ccw{{-4, -4.6f}, {4, -4.6f}, {4, 4.6f}, {-4, 4.6f}};
    const Poly a_cw(a_ccw.rbegin(), a_ccw.rend());
    const Poly b_world_ccw{{-4, 4.6f}, {4, 4.6f}, {4, 13}, {-4, 13}};
    const Poly b_world_cw(b_world_ccw.rbegin(), b_world_ccw.rend());
    const Eigen::Affine2f W_a = pose(0, 0, 0);   // room a's frame = the world
    for (const float bx : {0.f, -0.6f, 2.f})
        for (const float byaw : {0.f, 0.3f, static_cast<float>(M_PI), -1.2f})
        {
            const Eigen::Affine2f W_b = pose(bx, 4.6f + 0.1f, byaw);   // room b's frame anywhere (here near the door)
            for (const auto* A : {&a_ccw, &a_cw})
                for (const auto* Bw : {&b_world_ccw, &b_world_cw})
                    for (const float door_yaw_a : {0.f, static_cast<float>(M_PI)})   // the wall tangent either way
                        for (const float door_yaw_b : {0.f, static_cast<float>(M_PI)})
                        {
                            // the door node's pose in each room (room <- door), and room b's outline in its own frame
                            const Eigen::Affine2f door_a = W_a.inverse() * pose(0, 4.6f, door_yaw_a);
                            const Eigen::Affine2f door_b = W_b.inverse() * pose(0, 4.6f, door_yaw_b);
                            Poly B;
                            for (const auto& v : *Bw) B.push_back(W_b.inverse() * v);
                            const Eigen::Affine2f got = attach(face(door_a, *A, 0.5f), face(door_b, B, 0.5f));
                            const Eigen::Affine2f want = W_a.inverse() * W_b;   // T(a <- b), truth
                            check((got.translation() - want.translation()).norm() < 1e-4f
                                      and std::abs(std::remainder(Eigen::Rotation2Df(got.linear()).angle()
                                                                  - Eigen::Rotation2Df(want.linear()).angle(), 2 * M_PI)) < 1e-4,
                                  "room b placed where it is");
                            Eigen::Vector2f c = Eigen::Vector2f::Zero();
                            for (const auto& v : B) c += v;
                            check(not inside(*A, got * Eigen::Vector2f(c / 4.f)), "never over room a");
                        }
        }
    std::println("test_door_face: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
