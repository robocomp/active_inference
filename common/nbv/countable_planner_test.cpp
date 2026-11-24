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

    {   // standpoint_gain_is_scored_at_the_published_pose (shared-engine G5): raw_gain_from_standpoints evaluates the
        // agent's gain AT the pose plan_faces publishes for that face. A gain that rewards one spot (+y side, 3 m out)
        // must make that face win, and the winner's raw gain must be gain_at(best_pos, best_yaw) exactly.
        const auto room = rect(-8.0f, 8.0f, -8.0f, 8.0f);
        const rc::nbv::CountableInputs in{camera(), 0.3f, no_obstacles, room};
        const auto gain_at = [](const Eigen::Vector2f& p, float yaw)
        { return std::exp(-(p - Eigen::Vector2f(0.0f, 3.0f)).squaredNorm()) + 0.0f * yaw; };
        const auto r = rc::nbv::plan_countable(unit_box(), in, rc::nbv::raw_gain_from_standpoints(unit_box(), gain_at), kOne);
        CHECK(r.proposal.valid and r.plan.best_face == 2, "standpoint gain: the rewarded (+y) face wins");
        CHECK(r.plan.face_raw_gains[2] == static_cast<float>(gain_at(r.plan.best_pos, r.plan.best_yaw)),
              "standpoint gain: scored at the published pose");
        const auto fv = rc::nbv::face_viewpoint(unit_box(), 2, r.plan.best_standoff_m);
        CHECK(fv.pos == r.plan.best_pos and fv.yaw == r.plan.best_yaw, "face_viewpoint is plan_faces' pose");
    }

    {   // G6: an offer's standpoint the body cannot stand at loses the ranking; all blocked ⇒ the best face, p_stand 0; no world ⇒ nothing.
        const auto room = rect(-8.0f, 8.0f, -8.0f, 8.0f);
        const auto prefer_plus_y = [](int face, float) { return face == 2 ? 2.0f : 1.0f; };
        const rc::RobotFootprint shadow = rc::RobotFootprint::shadow();
        const rc::nbv::StandBody body = rc::nbv::StandBody::of(shadow, 0.05f);
        rc::nbv::CountableInputs in{camera(), 0.3f, no_obstacles, room};
        const auto base = rc::nbv::plan_countable(unit_box(), in, prefer_plus_y, kOne);
        CHECK(base.proposal.valid and base.plan.best_face == 2 and base.p_stand == 1.0f, "G6: without a world the +y face wins, p_stand 1");

        const auto block = [&](const std::vector<int>& faces) {   // latch a cell under each listed face's standpoint
            std::vector<float> f;
            for (const int face : faces)
            {
                const auto v = rc::nbv::face_viewpoint(unit_box(), face, face == 2 ? base.plan.best_standoff_m : base.plan.face_standoff_m[face]);
                f.insert(f.end(), {v.pos.x(), v.pos.y(), 0.02f});
            }
            return rc::nbv::OccupiedCells::from_xyz(f, 0.05f, rc::residual_risk::BeliefField{});   // no field: cells hard
        };
        const rc::nbv::OccupiedCells none;
        rc::nbv::StandWorld clear{&body, &none, room, {}};
        in.stand = &clear;
        const auto free = rc::nbv::plan_countable(unit_box(), in, prefer_plus_y, kOne);
        CHECK(free.proposal.epistemic_gain == base.proposal.epistemic_gain and free.p_stand == 1.0f, "G6: an empty world changes nothing");

        const auto cells = block({2});
        rc::nbv::StandWorld w{&body, &cells, room, {}};
        in.stand = &w;
        const auto blocked = rc::nbv::plan_countable(unit_box(), in, prefer_plus_y, kOne);
        CHECK(blocked.proposal.valid and blocked.plan.best_face != 2 and blocked.p_stand == 1.0f, "G6: a blocked standpoint loses to a free one");
        CHECK(blocked.plan.face_raw_gains[2] == 2.0f, "G6: face_raw_gains stays the belief's UNSCALED gain");

        const auto all = block({0, 1, 2, 3});
        rc::nbv::StandWorld wa{&body, &all, room, {}};
        in.stand = &wa;
        const auto none_free = rc::nbv::plan_countable(unit_box(), in, prefer_plus_y, kOne);
        CHECK(none_free.proposal.valid and not none_free.refused_no_usable_face, "G6: every standpoint blocked: the best face is still proposed");
        // the proposed face is the BELIEF's best (argmax of its raw gain), not the first usable one
        const auto best_raw = std::max_element(none_free.plan.face_raw_gains.begin(), none_free.plan.face_raw_gains.end());
        CHECK(none_free.plan.best_face == static_cast<int>(best_raw - none_free.plan.face_raw_gains.begin()) and none_free.plan.best_face == 2,
              "G6: ... and it is the belief's raw-best face (+y here)");
        CHECK(none_free.p_stand == 0.0f and none_free.proposal.epistemic_gain == 0.0f, "G6: ... at gain 0 with p_stand 0 (no offer value, not a skip)");
    }

    std::printf(fails == 0 ? "countable_planner_test: PASS\n" : "countable_planner_test: %d FAIL\n", fails);
    return fails == 0 ? 0 : 1;
}
