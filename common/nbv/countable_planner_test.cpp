// countable_planner_test.cpp — the shared countable NBV core (common/nbv/countable_planner.h). Standalone:
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 countable_planner_test.cpp -o t && ./t
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "countable_planner.h"

static int fails = 0;
#define CHECK(cond, msg) do { const bool ok_ = (cond); std::printf("  %-60s %s\n", msg, ok_ ? "ok" : "FAIL"); if (not ok_) ++fails; } while (0)

namespace
{
rc::nbv::Target unit_box()
{
    rc::nbv::Target t;                 // 1×1 m box at the origin, 1 m tall
    t.w = 1.0f; t.h = 1.0f; t.z0 = 0.0f; t.z1 = 1.0f;
    return t;
}
rc::nbv::Sensor camera()
{
    rc::nbv::Sensor s;                 // a complete model: hfov 70°, vfov 55°, 1.2 m above the floor
    s.vfov_rad = 55.0f * std::numbers::pi_v<float> / 180.0f;
    s.height_m = 1.2f;
    return s;
}
std::vector<Eigen::Vector2f> rect(float x0, float x1, float y0, float y1)
{
    return {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
}
bool inside(const std::vector<Eigen::Vector2f>& poly, float x, float y)
{
    return rc::nbv::detail::point_in_polygon(Eigen::Vector2f(x, y), poly);
}
const auto kConst = [](int, float) { return 1.0f; };
const auto kOne   = [](float) { return 1.0f; };
}

int main()
{
    std::printf("countable_planner_test\n");
    const std::vector<rc::nbv::Obstacle> no_obstacles;

    {   // refuses_when_every_face_outside_room: the room is the box's own footprint — no viewpoint fits.
        const auto room = rect(-0.5f, 0.5f, -0.5f, 0.5f);
        const rc::nbv::CountableInputs in{camera(), 0.3f, no_obstacles, room};
        const auto r = rc::nbv::plan_countable(unit_box(), in, kConst, kOne);
        CHECK(r.plan.valid, "refuses_when_every_face_outside_room: plan itself is valid");
        CHECK(r.refused_no_usable_face, "refuses_when_every_face_outside_room: refused");
        CHECK(not r.proposal.valid, "refuses_when_every_face_outside_room: no proposal");
    }
    {   // wall_object_viewed_from_room_side: a 1.2 m wide corridor whose top wall the box touches ⇒ only -y is open.
        const auto room = rect(-0.6f, 0.6f, -8.0f, 0.5f);
        const rc::nbv::CountableInputs in{camera(), 0.3f, no_obstacles, room};
        const auto r = rc::nbv::plan_countable(unit_box(), in, kConst, kOne);
        int reachable = 0;
        for (const bool b : r.plan.face_reachable) reachable += b ? 1 : 0;
        CHECK(reachable == 1, "wall_object_viewed_from_room_side: 3 faces unreachable");
        CHECK(r.proposal.valid, "wall_object_viewed_from_room_side: valid proposal");
        CHECK(inside(room, r.proposal.epistemic_target_x_m, r.proposal.epistemic_target_y_m),
              "wall_object_viewed_from_room_side: viewpoint inside the room");
        CHECK(r.proposal.epistemic_target_y_m < -0.5f, "wall_object_viewed_from_room_side: on the open (-y) side");
    }
    {   // useful_fraction_unbounded_without_sigma_star
        const std::array<rc::DofSpec, 2> dofs{{{"cx", "m", -1.0f}, {"cy", "m", -1.0f}}};
        CHECK(rc::nbv::useful_fraction(dofs, [](std::size_t) { return 0.04f; }, 2.0f) == 1.0f,
              "useful_fraction_unbounded_without_sigma_star");
        // with a demand: var 0.04 (σ 0.2) vs σ* 0.1 ⇒ gap ½ln4 = 0.693 per DOF on one DOF ⇒ min(2, .693)/2
        const std::array<rc::DofSpec, 2> dem{{{"cx", "m", 0.1f}, {"cy", "m", -1.0f}}};
        const float f = rc::nbv::useful_fraction(dem, [](std::size_t) { return 0.04f; }, 2.0f);
        CHECK(std::abs(f - 0.5f * std::log(4.0f) / 2.0f) < 1e-5f, "useful_fraction with a sigma* = gap/raw");
    }
    {   // d_optimal_matches_formula: S = diag(1,4), dI = diag(3,0.5) ⇒ ½ ln((1+3)(1+2)) = ½ ln 12
        Eigen::Matrix2f S = Eigen::Matrix2f::Zero();  S(0, 0) = 1.0f; S(1, 1) = 4.0f;
        Eigen::Matrix2f dI = Eigen::Matrix2f::Zero(); dI(0, 0) = 3.0f; dI(1, 1) = 0.5f;
        CHECK(std::abs(rc::nbv::d_optimal_gain<2>(S, dI) - 0.5f * std::log(12.0f)) < 1e-5f, "d_optimal_matches_formula");
    }
    {   // non_finite_gain_refused
        const auto room = rect(-8.0f, 8.0f, -8.0f, 8.0f);
        const rc::nbv::CountableInputs in{camera(), 0.3f, no_obstacles, room};
        // A NaN from `useful` (e.g. a NaN variance in the adequacy gap) makes the gain non-finite ⇒ refused.
        const auto nan_useful = [](float) { return std::numeric_limits<float>::quiet_NaN(); };
        const auto r = rc::nbv::plan_countable(unit_box(), in, kConst, nan_useful);
        CHECK(not r.proposal.valid, "non_finite_gain_refused");
        // A NaN RAW gain never gets that far: plan_faces scores p_detect·max(0, raw) = 0 (pinned, not changed here).
        const auto nan_gain = [](int, float) { return std::numeric_limits<float>::quiet_NaN(); };
        const auto z = rc::nbv::plan_countable(unit_box(), in, nan_gain, kOne);
        CHECK(not z.proposal.valid or z.proposal.epistemic_gain == 0.0f, "NaN raw gain: never a positive bid");
        const auto ok = rc::nbv::plan_countable(unit_box(), in, kConst, kOne);
        CHECK(ok.proposal.valid and ok.proposal.epistemic_gain > 0.0f, "control: finite gain in an open room is valid");
    }

    std::printf(fails == 0 ? "countable_planner_test: PASS\n" : "countable_planner_test: %d FAIL\n", fails);
    return fails == 0 ? 0 : 1;
}
