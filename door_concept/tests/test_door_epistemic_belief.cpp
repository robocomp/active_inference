// test_door_epistemic_belief — the live crash of 915b2e22 (rtD1g, 17:25:08): Eigen's `row < rows() && col < cols()`
// on a Matrix<float,3,3> inside rc::EpistemicPlanner::compute. The belief became [s, w, h, d] (Task 5b: d = the leaf
// plane's depth) while every consumer of its matrices kept a 3-DOF view; a 4-sized DOF table (kDoorDofs, sized by
// DoorBelief::N) then indexed the 3×3 reported Σ at (3,3). Built WITHOUT -DNDEBUG on purpose: the agent ships with
// Eigen's asserts live, and that is how it died.
//   1. EpistemicPlanner::compute on a 4-state belief, several cycles of the dashboard log (its (dbg % 30) branch), runs;
//   2. every reader of the belief's matrices agrees on the reported [s, w, h] dimension (the DOF table, Σ, information);
//   3. predicted_information is the information on [s, w, h] with d MARGINALISED (the Schur complement of the 4×4,
//      regularised by d's own posterior), never the raw top-left block, which overstates it: a view whose s/w lever is
//      collinear with d gains nothing on [s, w, h] it cannot tell from d.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O0 -g -I/usr/include/eigen3 tests/test_door_epistemic_belief.cpp src/epistemic_planner.cpp \
//       src/door_belief.cpp -o /tmp/test_door_epistemic_belief && /tmp/test_door_epistemic_belief
#include <cmath>
#include <print>
#include <random>

#include "../src/door_belief.h"
#include "../src/door_dof.h"
#include "../src/epistemic_planner.h"

namespace
{
int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what)
{
    ++g_checks;
    if (not ok) { ++g_fail; std::println("  FAIL: {}", what); }
}
}  // namespace

int main()
{
    rc::DoorBeliefParams p;
    p.wall_O = {0.f, 5.f}; p.wall_u = {1.f, 0.f};
    p.recess_prior_m = 0.f; p.recess_prior_std = 0.115f;
    p.tpl_w = 1.0f; p.tpl_h = 2.1f;
    rc::DoorBelief b({0.5f, 1.0f, 2.1f, 0.0f}, p);

    std::println("1. EpistemicPlanner::compute on a [s, w, h, d] belief");
    rc::EpistemicPlanner planner(1.8f);
    rc::nbv::Sensor sensor;
    sensor.hfov_rad = 1.2f; sensor.vfov_rad = 0.9f; sensor.height_m = 1.2f;
    std::vector<Eigen::Vector2f> room{{-3.f, -3.f}, {6.f, -3.f}, {6.f, 5.f}, {-3.f, 5.f}};
    int valid = 0;
    for (int k = 0; k < 61; ++k)   // > 30: the planner's periodic dashboard print reads the DOF table
    {
        const auto prop = planner.compute(b, 0.02f, 0.03f, sensor, {}, room);
        valid += prop.valid ? 1 : 0;
    }
    std::println("    61 cycles ran, {} valid proposals", valid);
    check(true, "unreachable");

    std::println("2. one reported dimension");
    check(static_cast<int>(rc::kDoorDofs.size()) == rc::DoorBelief::kReported, "the DOF table is not the reported [s, w, h]");
    check(b.covariance_reported().rows() == rc::DoorBelief::kReported, "covariance_reported is not [s, w, h]");

    std::println("3. predicted_information marginalises d");
    {
        // a face of points ON the leaf plane: every point's across residual has a d lever; the shared part with s/w/h
        // must not be booked as [s, w, h] information.
        std::mt19937 rng(3);
        std::uniform_real_distribution<float> U(0.f, 1.f);
        std::vector<Eigen::Vector3f> pts;
        for (int i = 0; i < 400; ++i) pts.emplace_back(0.5f + U(rng), 5.0f - 0.02f, 2.1f * U(rng));
        const Eigen::Matrix3f I3 = b.predicted_information(pts, 0.002f);
        const Eigen::Matrix4f I4 = rc::ai::predicted_information<4>(b, b.state(), pts, 0.002f);
        const Eigen::Matrix3f block = I4.topLeftCorner<3, 3>();
        std::println("    trace: marginal {:.1f} vs raw block {:.1f}", I3.trace(), block.trace());
        check(I3.allFinite(), "the marginal information is not finite");
        check(I3.trace() <= block.trace() + 1e-3f, "the marginal information exceeds the raw block");
        const Eigen::Matrix3f diff = block - I3;   // the Schur term is PSD
        check(diff.selfadjointView<Eigen::Lower>().eigenvalues().minCoeff() >= -1e-2f * std::max(1.0f, block.norm()),
              "block − marginal is not PSD");
    }
    std::println("{} / {} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
