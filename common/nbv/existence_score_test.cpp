// existence_score_test.cpp — the EXISTENCE term of rc::nbv::score / plan_faces / plan_countable (viewpoint_score.h).
// It replaced the 5-nat verification floor (owner ruling 2026-11-23). Standalone:
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 existence_score_test.cpp -o t && ./t
#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

#include "countable_planner.h"

static int fails = 0;
#define CHECK(cond, msg) do { const bool ok_ = (cond); std::printf("  %-66s %s\n", msg, ok_ ? "ok" : "FAIL"); if (not ok_) ++fails; } while (0)

namespace
{
rc::nbv::Target unit_box()
{
    rc::nbv::Target t;                 // 1×1 m box at the origin, 1 m tall
    t.w = 1.0f; t.h = 1.0f; t.z0 = 0.0f; t.z1 = 1.0f;
    t.sigma_pos_m = 0.05f; t.sigma_extent_m = 0.05f;
    return t;
}
rc::nbv::Sensor camera()
{
    rc::nbv::Sensor s;                 // a complete model: default hfov, vfov 55°, 1.2 m above the floor
    s.vfov_rad = 55.0f * std::numbers::pi_v<float> / 180.0f;
    s.height_m = 1.2f;
    return s;
}
}

int main()
{
    std::printf("existence_score_test\n");
    const rc::nbv::Sensor cam = camera();
    const std::vector<rc::nbv::Obstacle> none;
    const float ph = 0.85f, pf = 0.05f;

    // A sweep of candidate poses at several ranges, all looking at the box.
    bool reg_ok = true, add_ok = true, any_pd = false;
    for (float d : {0.8f, 1.5f, 2.5f, 4.0f, 7.0f})
        for (float raw : {0.0f, 0.3f, 2.0f})
        {
            rc::nbv::Candidate c;
            c.pos = {d, 0.2f};
            c.yaw = std::atan2(-c.pos.y(), -c.pos.x());
            c.raw_gain = raw;

            const rc::nbv::Target t0 = unit_box();                 // p_exists unset (-1)
            const rc::nbv::Score s0 = rc::nbv::score(c, t0, cam, none, 0.3f);
            // REGRESSION: unset ⇒ exactly the old formula, bit for bit.
            reg_ok = reg_ok and s0.expected_gain == s0.p_detect * std::max(0.0f, raw) and s0.existence_gain == 0.0f;

            rc::nbv::Target t1 = unit_box();
            t1.set_existence({0.5f, ph, pf});
            const rc::nbv::Score s1 = rc::nbv::score(c, t1, cam, none, 0.3f);
            const float mi = rc::exist::existence_mutual_info(0.5f, ph * s1.p_detect, pf);
            add_ok = add_ok and s1.p_detect == s0.p_detect and s1.existence_gain == mi
                     and s1.expected_gain == s0.expected_gain + mi;
            any_pd = any_pd or s1.p_detect > 0.1f;
        }
    CHECK(any_pd, "sanity: some candidate has P(detect) > 0.1");
    CHECK(reg_ok, "score(): p_exists < 0 ⇒ expected_gain == p_detect·max(0,raw) exactly");
    CHECK(add_ok, "score(): p_exists = 0.5 adds exactly MI(0.5, p_hit·p_detect, p_false)");

    {   // existence settled ⇒ no term, even though it is priced
        rc::nbv::Candidate c; c.pos = {2.0f, 0.0f}; c.yaw = std::numbers::pi_v<float>; c.raw_gain = 1.0f;
        rc::nbv::Target t = unit_box(); t.set_existence({1.0f, ph, pf});
        const auto s = rc::nbv::score(c, t, cam, none, 0.3f);
        CHECK(std::abs(s.existence_gain) < 1e-6f, "score(): p_exists = 1 ⇒ existence_gain == 0");
    }

    {   // plan_faces: per-face split and selection include the term; an unusable face gets nothing
        const auto raw = [](int, float) { return 0.0f; };           // a fit that wants nothing more
        rc::nbv::Target t = unit_box(); t.set_existence({0.5f, ph, pf});
        const auto plan = rc::nbv::plan_faces(t, cam, 0.3f, none, raw);
        bool split = true;
        for (int i = 0; i < 4; ++i)
            split = split and plan.face_gains[i] == plan.face_geometric_gains[i] + plan.face_existence_gains[i]
                          and plan.face_existence_gains[i] > 0.0f;
        CHECK(plan.valid and plan.any_usable, "plan_faces: valid with raw gain 0 (existence open)");
        CHECK(split, "plan_faces: face_gains = geometric + existence, existence > 0 on usable faces");

        const auto room = std::vector<Eigen::Vector2f>{{-0.6f, -8.0f}, {0.6f, -8.0f}, {0.6f, 0.5f}, {-0.6f, 0.5f}};
        const auto p2 = rc::nbv::plan_faces(t, cam, 0.3f, none, raw, room);   // only -y reachable
        bool unusable_zero = true;
        for (int i = 0; i < 4; ++i)
            if (not p2.face_reachable[i])
                unusable_zero = unusable_zero and p2.face_gains[i] == 0.0f and p2.face_existence_gains[i] == 0.0f;
        CHECK(unusable_zero, "plan_faces: an unreachable face carries no existence gain");
    }

    {   // plan_countable: the adequacy factor scales ONLY the geometric part
        const rc::nbv::CountableInputs in{cam, 0.3f, none, {}};
        const auto raw = [](int, float) { return 2.0f; };
        const auto zero = [](float) { return 0.0f; };
        const auto one  = [](float) { return 1.0f; };

        const auto r_unset = rc::nbv::plan_countable(unit_box(), in, raw, one);
        const int b = r_unset.plan.best_face;
        CHECK(r_unset.proposal.valid and r_unset.proposal.epistemic_gain == r_unset.plan.face_gains[b] * 1.0f,
              "plan_countable: unset ⇒ gain == face_gains[best]·useful (unchanged)");

        rc::nbv::Target t = unit_box(); t.set_existence({0.5f, ph, pf});
        const auto r0 = rc::nbv::plan_countable(t, in, raw, zero);
        CHECK(r0.proposal.valid and r0.proposal.epistemic_gain == r0.plan.face_existence_gains[r0.plan.best_face]
                  and r0.proposal.epistemic_gain > 0.0f,
              "plan_countable: useful = 0 ⇒ gain == existence term (> 0)");
        const auto r1 = rc::nbv::plan_countable(t, in, raw, one);
        const int b1 = r1.plan.best_face;
        CHECK(r1.proposal.epistemic_gain == r1.plan.face_geometric_gains[b1] + r1.plan.face_existence_gains[b1],
              "plan_countable: useful = 1 ⇒ gain == geometric + existence");
        std::printf("    useful=0: gain %.4f nats (p_detect %.3f)\n", r0.proposal.epistemic_gain,
                    r0.plan.face_p_detect[r0.plan.best_face]);
    }

    std::printf(fails ? "existence_score_test: %d FAILED\n" : "existence_score_test: all passed\n", fails);
    return fails ? 1 : 0;
}
