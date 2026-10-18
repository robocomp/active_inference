// The pure geometry of an object's `approach` affordances (object_approach_geometry.h): the standoff rule, the side
// directions, the room-axis face names and the per-side precondition. No graph.
// Build+run:
//   g++ -std=c++23 -O0 -g -I/usr/include/eigen3 -o /tmp/toag test_object_approach_geometry.cpp && /tmp/toag
#include "../object_approach_geometry.h"

#include <cmath>
#include <numbers>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)
bool near(double a, double b, double tol = 1e-4) { return std::abs(a - b) < tol; }
bool near(const Eigen::Vector2f& a, const Eigen::Vector2f& b) { return (a - b).norm() < 1e-4f; }

using namespace rc::pragmatic::approach;
using V = Eigen::Vector2f;
constexpr float pi = std::numbers::pi_v<float>;
V dir(float a) { return {std::cos(a), std::sin(a)}; }
const Dir* find(const std::vector<Dir>& ds, std::string_view kind)
{
    for (const auto& d : ds)
        if (d.kind == kind) return &d;
    return nullptr;
}
}   // namespace

int main()
{
    // ── h(d): the footprint's support function — centre to edge along d ──────────────────────────────────────────
    {
        const V half{0.6f, 0.4f};
        CHECK(near(support(half, 0.0f, V{1, 0}), 0.6));
        CHECK(near(support(half, 0.0f, V{-1, 0}), 0.6));
        CHECK(near(support(half, 0.0f, V{0, 1}), 0.4));
        CHECK(near(support(half, 0.0f, dir(pi / 4)), (0.6 + 0.4) / std::sqrt(2.0)));   // towards a corner
        CHECK(near(support(half, pi / 2, V{1, 0}), 0.4));                               // rotated box
        CHECK(near(support(V{0, 0}, 0.3f, V{1, 0}), 0.0));                              // a person: a point
    }

    // ── σ(d): the object's spread along the approach line plus the robot's own ───────────────────────────────────
    {
        Eigen::Matrix2f cov;
        cov << 0.04f, 0.0f, 0.0f, 0.01f;
        CHECK(near(sigma_along(cov, V{1, 0}, 0.0f), 0.2));
        CHECK(near(sigma_along(cov, V{0, 1}, 0.1f), std::sqrt(0.02)));
    }

    // ── the standpoint: centre + d·(h(d) + clearance + k·σ(d)), facing the centre ────────────────────────────────
    {
        const StandCfg cfg{.clearance_m = 0.398f, .sigma_k = 2.0f};
        const auto p = standpoint(V{2, 0}, 0.0f, V{0.6f, 0.4f}, V{-1, 0}, Eigen::Matrix2f::Zero(), 0.0f, cfg);
        CHECK(near(p.x, 2.0 - 0.6 - 0.398) and near(p.y, 0.0));
        CHECK(near(p.yaw, 0.0));                                    // facing +x, towards the object
        Eigen::Matrix2f cov = Eigen::Matrix2f::Identity() * 0.01f;  // σ = 0.1 m on both axes
        const auto q = standpoint(V{2, 0}, 0.0f, V{0.6f, 0.4f}, V{-1, 0}, cov, 0.0f, cfg);
        CHECK(near(q.x, 2.0 - 0.6 - 0.398 - 2 * 0.1));             // an uncertain object is approached less closely
        const auto r = standpoint(V{0, 0}, 0.0f, V{0.6f, 0.4f}, V{0, 1}, Eigen::Matrix2f::Zero(), 0.0f, cfg);
        CHECK(near(r.y, 0.4 + 0.398) and near(r.yaw, -pi / 2));    // from the narrow side: closer, facing −y
    }

    // ── a fronted object's sides, from its front direction; left/right are the object's OWN ──────────────────────
    {
        const auto s = fronted_sides(pi / 2);   // a chair facing +y
        CHECK(s.size() == 4);
        CHECK(find(s, "front") and near(find(s, "front")->dir, V{0, 1}));
        CHECK(find(s, "back") and near(find(s, "back")->dir, V{0, -1}));
        CHECK(find(s, "left") and near(find(s, "left")->dir, V{-1, 0}));    // sitting in it facing +y, left is −x
        CHECK(find(s, "right") and near(find(s, "right")->dir, V{1, 0}));
    }

    // ── a frontless object's faces, named by the room axis their normal is nearest ───────────────────────────────
    {
        FaceFrame f;
        auto s = f.update(0.1f, 0.02f);
        CHECK(s.size() == 4);
        CHECK(find(s, "xp") and near(find(s, "xp")->dir, dir(0.1f)));          // the true normal, not the axis
        CHECK(find(s, "yp") and near(find(s, "yp")->dir, dir(0.1f + pi / 2)));
        CHECK(find(s, "xn") and near(find(s, "xn")->dir, dir(0.1f + pi)));
        CHECK(find(s, "yn") and near(find(s, "yn")->dir, dir(0.1f - pi / 2)));
        s = f.update(0.1f + pi, 0.02f);   // a refit flipped the table's yaw by 180°: the names stay put
        CHECK(near(find(s, "xp")->dir, dir(0.1f)));
        CHECK(near(find(s, "yn")->dir, dir(0.1f - pi / 2)));
    }
    {
        // near 45° the names would swap on noise: they hold until the yaw is past 45° by 2σ of itself
        FaceFrame f;
        f.update(0.70f, 0.02f);
        auto s = f.update(0.80f, 0.02f);   // past π/4 = 0.785, inside 0.785 + 0.04
        CHECK(near(find(s, "xp")->dir, dir(0.80f)));
        s = f.update(0.86f, 0.02f);        // past the band: renamed
        CHECK(near(find(s, "xp")->dir, dir(0.86f - pi / 2)));
        CHECK(near(find(s, "yp")->dir, dir(0.86f)));
    }

    // ── the default side of a frontless object: the face towards the robot ───────────────────────────────────────
    {
        FaceFrame f;
        const auto s = f.update(0.0f, 0.01f);
        CHECK(facing(s, V{0, 0}, V{-3.0f, 0.5f}).kind == "xn");
        CHECK(facing(s, V{0, 0}, V{0.2f, 4.0f}).kind == "yp");
    }

    // ── the per-side precondition: the standpoint inside the room with room to turn ──────────────────────────────
    {
        const std::vector<V> room{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
        CHECK(fits_in_room(V{2, 2}, room, 0.4f));
        CHECK(not fits_in_room(V{0.3f, 2}, room, 0.4f));   // inside, but the body could not turn against the wall
        CHECK(not fits_in_room(V{5, 2}, room, 0.4f));      // outside
        const std::vector<V> ell{{0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 4}, {0, 4}};   // non-convex
        CHECK(not fits_in_room(V{3, 3}, ell, 0.1f));       // in the notch
        CHECK(fits_in_room(V{0.5f, 3}, ell, 0.4f));
    }

    // ── the value falls as the robot arrives (the door's ping-pong lesson) ───────────────────────────────────────
    {
        CHECK(arrival_value(0.3f, 0.0f, 0.25f, 0.05f) < 0.001f);   // standing on it: nothing left to gain
        CHECK(near(arrival_value(0.3f, 5.0f, 0.25f, 0.05f), 0.3));
        CHECK(near(arrival_value(0.3f, 0.25f, 0.25f, 0.05f), 0.15)); // at the arrival radius: half
    }

    std::println("test_object_approach_geometry: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
