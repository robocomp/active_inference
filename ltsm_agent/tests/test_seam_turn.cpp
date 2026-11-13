// Standalone (from ltsm_agent/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_seam_turn.cpp -o /tmp/tst && /tmp/tst
//
// Which of the two seams T(L <- P) through a crossed door is true — the faces as they are, or turned by pi — is decided
// by the robot (seam::turned_seam_fits), not by L's outline. clean33 (2026-11-10): the old rule compared the robot with
// L's outline VERTEX MEAN; a degenerate outline whose zigzag tail ran out past the door wall put that mean beyond the
// door, the correct seam was turned, the robot was mirrored into room_1 and the real crossing withdrawn.
#include "../src/seam_fusion.h"

#include <cstdio>
#include <vector>

using namespace ltsm::seam;

static int g_fail = 0;
#define CHECK(c) do { if (not (c)) { std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static Measured M(double x, double y, double th, double sxy = 0.05, double sth = 0.03)
{
    return {Vec3(x, y, th), Vec3(sxy * sxy, sxy * sxy, sth * sth).asDiagonal()};
}

// the world: door D in L at (2, 5) facing +y (L below, P above); the entry mirror M as P's estimator placed it
struct Case { Measured fL, fM, robot_in_P, robot_in_L; };
static Case make(bool faces_turned, const Vec3& robot_L_truth)
{
    Case c;
    c.fL = M(2.0, 5.0, 0.0);
    c.fM = M(1.0, -1.0, 0.7);
    // the true T(L <- P): the faces as given, or with M's frame turned half a turn
    const Measured inv_fM = inverse_cov(c.fM);
    const Measured truth = faces_turned ? compose_cov(compose_cov(c.fL, Measured{Vec3(0, 0, M_PI), Mat3::Zero()}), inv_fM)
                                        : compose_cov(c.fL, inv_fM);
    const Measured rL{robot_L_truth, Vec3(0.05 * 0.05, 0.05 * 0.05, 0.03 * 0.03).asDiagonal()};
    c.robot_in_P = compose_cov(inverse_cov(Measured{truth.pose, Mat3::Zero()}), rL);
    c.robot_in_P.cov = rL.cov;
    c.robot_in_L = rL;
    return c;
}

// the rule this replaces: flip when the seam puts the robot on the same side of D as L's outline vertex mean
static bool old_rule(const Case& c, const std::vector<Eigen::Vector2d>& outline_L)
{
    Eigen::Vector2d cL = Eigen::Vector2d::Zero();
    for (const auto& v : outline_L) cL += v;
    cL /= double(outline_L.size());
    const Measured seam = compose_cov(c.fL, inverse_cov(c.fM));
    const Eigen::Vector2d r = seam.pose.head<2>() + rot(seam.pose.z()) * c.robot_in_P.pose.head<2>();
    const Eigen::Vector2d o = c.fL.pose.head<2>(), n(-std::sin(c.fL.pose.z()), std::cos(c.fL.pose.z()));
    return n.dot(cL - o) * n.dot(r - o) > 0.0;
}

static void faces_as_given()
{
    const auto c = make(false, Vec3(2.3, 5.9, 1.2));   // the robot 0.9 m past the door, in P
    CHECK(not turned_seam_fits(c.fL, inverse_cov(c.fM), c.robot_in_P, c.robot_in_L));
}

static void faces_turned()
{
    const auto c = make(true, Vec3(2.3, 5.9, 1.2));
    CHECK(turned_seam_fits(c.fL, inverse_cov(c.fM), c.robot_in_P, c.robot_in_L));
}

static void on_the_threshold()
{   // the robot still straddles the door: position barely tells the sides apart, its heading does
    for (const bool turned : {false, true})
    {
        const auto c = make(turned, Vec3(2.1, 5.02, 1.4));
        CHECK(turned_seam_fits(c.fL, inverse_cov(c.fM), c.robot_in_P, c.robot_in_L) == turned);
    }
}

static void clean33_outline_does_not_matter()
{   // L's outline: a 3 x 4.5 room below the door wall (y < 5) plus a 0.1 m-wide zigzag tail running out to x = 7.9,
    // 15 vertices beyond the wall line — the vertex mean lands past the door. The faces are NOT turned.
    std::vector<Eigen::Vector2d> L = {{0.5, 0.5}, {3.5, 0.5}, {3.5, 5.0}, {2.5, 5.0}};
    for (int k = 0; k < 15; ++k) L.emplace_back(4.0 + 0.26 * k, (k % 2) ? 7.2 : 7.1);
    L.emplace_back(0.5, 5.0);
    const auto c = make(false, Vec3(2.3, 5.9, 1.2));
    CHECK(old_rule(c, L));    // the defect, reproduced: the vertex-mean rule turns a seam that is right
    CHECK(not turned_seam_fits(c.fL, inverse_cov(c.fM), c.robot_in_P, c.robot_in_L));   // the robot does not
}

int main()
{
    faces_as_given();
    faces_turned();
    on_the_threshold();
    clean33_outline_does_not_matter();
    std::printf(g_fail == 0 ? "test_seam_turn: ALL PASSED\n" : "test_seam_turn: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
