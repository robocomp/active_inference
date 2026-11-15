// planner_ab_test — Phase 0 Task 11: chair's EpistemicPlanner on the shared core (rc::nbv::plan_countable) must
// propose what the pre-task planner proposed. tests/legacy_epistemic_planner.* is a VERBATIM copy of the planner
// before the move (namespace rc::legacy). For each of ≥ 20 belief states built through ChairBelief's own API, both
// are called with identical inputs; a difference FAILS unless its state is listed in kExpectedDifferences with a
// reason. Run: LANG=es_ES.UTF-8 ./planner_ab_test
#include <clocale>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

#include "chair_belief.h"
#include "chair_prior.h"
#include "epistemic_planner.h"
#include "legacy_epistemic_planner.h"

using namespace rc;

namespace
{
struct Case
{
    std::string                    name;
    ChairBelief                    belief;
    std::vector<nbv::Obstacle>     obstacles;
    std::vector<Eigen::Vector2f>   room;
};

// index → reason. Empty: the move is meant to be behaviour-identical for chair.
const std::vector<std::pair<int, const char*>> kExpectedDifferences = {};

std::vector<Eigen::Vector2f> rect(float x0, float x1, float y0, float y1) { return {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}; }

// Points on the chair's vertical seat face that faces `bearing` (room frame, chair→camera), plus the seat top.
std::vector<Eigen::Vector3f> visible_points(const ChairBelief& b, float bearing)
{
    const auto& s = b.state();
    const float rel = std::remainder(bearing - s.yaw, 2.0f * std::numbers::pi_v<float>);
    const float hw = 0.5f * b.seat_w(), hd = 0.5f * b.seat_d();
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    std::vector<Eigen::Vector3f> pts;
    for (int m = 0; m < 9; ++m)
    {
        const float t = -1.0f + 2.0f * m / 8.0f;
        float lx, ly;
        if (std::abs(rel) < std::numbers::pi_v<float> / 4)            { lx =  hw; ly = t * hd; }
        else if (std::abs(rel) > 3 * std::numbers::pi_v<float> / 4)   { lx = -hw; ly = t * hd; }
        else if (rel > 0)                                              { lx = t * hw; ly =  hd; }
        else                                                           { lx = t * hw; ly = -hd; }
        for (int k = 0; k < 5; ++k)
        {
            const float z = b.cz() + b.seat_h() * k / 4.0f;
            pts.emplace_back(s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, z);
        }
        pts.emplace_back(s.cx + c * (t * hw) - sn * (0.3f * t * hd), s.cy + sn * (t * hw) + c * (0.3f * t * hd), b.cz() + b.seat_h());
    }
    return pts;
}

void observe(ChairBelief& b, int frames, float bearing0, float spread)
{
    for (int f = 0; f < frames; ++f)
    {
        const float bearing = bearing0 + spread * std::sin(0.7f * f);
        ChairFrame fr;
        fr.points = visible_points(b, bearing);
        fr.R.assign(fr.points.size(), 0.03f * 0.03f);
        b.predict();
        b.update(fr);
        b.resolve_orientation(fr.points, 0.03f * 0.03f, 1.0f, bearing);
    }
}

ChairBelief fresh(const ChairBeliefParams& p, float x, float y, float yaw)
{
    ChairBeliefState s; s.cx = x; s.cy = y; s.yaw = yaw;
    return ChairBelief(s, p);
}
}  // namespace

int main()
{
    std::setlocale(LC_ALL, "");
    const auto pr = load_chair_prior(std::string(AGENT_DIR) + "/../common/concept_manifest/chair.concept.toml");
    if (not pr) { std::printf("prior: %s\n", pr.error().c_str()); return 1; }
    ChairBeliefParams p; p.seed_template(*pr);

    const auto open_room = rect(-6.0f, 6.0f, -6.0f, 6.0f);
    std::vector<Case> cases;
    const float yaws[4] = {0.0f, 0.7f, 1.5708f, -2.3f};
    for (int i = 0; i < 4; ++i)
        cases.push_back({"fresh prior yaw " + std::to_string(yaws[i]), fresh(p, 0.5f * i - 1.0f, 0.3f * i, yaws[i]), {}, open_room});
    for (const int n : {1, 5, 30})
        for (const float b0 : {0.4f, -2.0f})
        {
            auto b = fresh(p, 1.0f, -0.5f, 0.3f);
            observe(b, n, b0, 0.8f);
            cases.push_back({"after " + std::to_string(n) + " updates, bearing " + std::to_string(b0), b, {}, open_room});
        }
    {   // mode-ambiguous: seen only from one side, so the backrest direction is unresolved
        auto b = fresh(p, 0.0f, 0.0f, 0.0f); observe(b, 10, 1.5708f, 0.0f);
        cases.push_back({"mode-ambiguous: one side only", b, {}, open_room});
        auto b2 = fresh(p, 2.0f, 1.0f, 2.0f); observe(b2, 3, 2.0f + 1.5708f, 0.05f);
        cases.push_back({"mode-ambiguous: short side look", b2, {}, open_room});
        auto b3 = fresh(p, -1.0f, 2.0f, -1.0f); observe(b3, 40, -1.0f, 0.0f);
        cases.push_back({"one bearing spent (novelty exhausted)", b3, {}, open_room});
    }
    {   // bearing-only seeds (a peripheral 360 detection)
        auto b = fresh(p, 3.0f, 0.0f, 0.0f); b.seed_bearing({0, 0}, 0.0f, 1.5f, 0.05f, 1.5f);
        cases.push_back({"bearing-only seed east", b, {}, open_room});
        auto b2 = fresh(p, 0.0f, -3.0f, 1.0f); b2.seed_bearing({0, 0}, -1.5708f, 1.0f, 0.08f, 1.0f);
        cases.push_back({"bearing-only seed south", b2, {}, open_room});
    }
    {   // near a wall of the room polygon
        const auto room = rect(-3.0f, 3.0f, -3.0f, 1.0f);   // the chair's +y side is 0.6 m from the wall
        for (const float yaw : {0.0f, 1.5708f, 3.0f})
        {
            auto b = fresh(p, 0.0f, 0.4f, yaw); observe(b, 5, -1.5708f, 0.6f);
            cases.push_back({"near the +y wall, yaw " + std::to_string(yaw), b, {}, room});
        }
    }
    {   // one face blocked by an obstacle (a table standing off the chair's +x face)
        nbv::Obstacle table; table.cx = 1.4f; table.cy = 0.0f; table.w = 1.2f; table.h = 0.8f;
        auto b = fresh(p, 0.0f, 0.0f, 0.0f); observe(b, 5, 0.0f, 0.5f);
        cases.push_back({"+x face blocked by a table", b, {table}, open_room});
        nbv::Obstacle tall = table; tall.z0 = 0.0f; tall.z1 = 0.75f;
        cases.push_back({"+x face behind a 0.75 m table", b, {tall}, open_room});
        auto b2 = fresh(p, 0.0f, 0.4f, 0.0f); observe(b2, 5, -1.5708f, 0.6f);
        cases.push_back({"wall + obstacle", b2, {table}, rect(-3.0f, 3.0f, -3.0f, 1.0f)});
    }
    {   // no usable face (the room is the chair's own footprint) ⇒ both must refuse
        auto b = fresh(p, 0.0f, 0.0f, 0.0f);
        cases.push_back({"boxed in: no usable face", b, {}, rect(-0.4f, 0.4f, -0.4f, 0.4f)});
    }

    nbv::Sensor sensor;
    sensor.vfov_rad = 55.0f * std::numbers::pi_v<float> / 180.0f;
    sensor.height_m = 1.2f;
    const detect::DetectorEnvelope env{};
    legacy::EpistemicPlanner old_p(1.8f); old_p.set_detector_envelope(env); old_p.set_robot_radius(0.30f);
    EpistemicPlanner         new_p(1.8f); new_p.set_detector_envelope(env); new_p.set_robot_radius(0.30f);
    const float lat_rate = 0.01f, sigma_base = 0.03f;

    int fails = 0, valid = 0;
    for (int i = 0; i < static_cast<int>(cases.size()); ++i)
    {
        const auto& c = cases[i];
        const auto a = old_p.compute(c.belief, lat_rate, sigma_base, sensor, c.obstacles, c.room);
        const auto b = new_p.compute(c.belief, lat_rate, sigma_base, sensor, c.obstacles, c.room);
        const bool same = a.valid == b.valid
            and (not a.valid or (std::abs(a.epistemic_target_x_m - b.epistemic_target_x_m) < 1e-4f
                                 and std::abs(a.epistemic_target_y_m - b.epistemic_target_y_m) < 1e-4f
                                 and std::abs(a.epistemic_target_yaw_rad - b.epistemic_target_yaw_rad) < 1e-4f
                                 and std::abs(a.epistemic_gain - b.epistemic_gain) <= 1e-5f * std::abs(a.epistemic_gain)));
        valid += a.valid ? 1 : 0;
        const char* reason = nullptr;
        for (const auto& [k, why] : kExpectedDifferences) if (k == i) reason = why;
        const bool ok = same or reason != nullptr;
        std::printf("  [%2d] %-44s %s%s\n", i, c.name.c_str(), same ? "same" : (reason ? "EXPECTED: " : "DIFF"), reason ? reason : "");
        if (not same)
            std::printf("       legacy v=%d (%.4f,%.4f,%.4f) g=%.6f | new v=%d (%.4f,%.4f,%.4f) g=%.6f\n",
                        a.valid, a.epistemic_target_x_m, a.epistemic_target_y_m, a.epistemic_target_yaw_rad, a.epistemic_gain,
                        b.valid, b.epistemic_target_x_m, b.epistemic_target_y_m, b.epistemic_target_yaw_rad, b.epistemic_gain);
        if (not ok) ++fails;
    }
    std::printf("planner_ab_test: %zu states, %d valid proposals — %s\n", cases.size(), valid, fails == 0 ? "PASS" : "FAIL");
    if (cases.size() < 20 or valid < 15) { std::printf("planner_ab_test: harness too weak (need ≥20 states, ≥15 valid)\n"); return 1; }
    return fails == 0 ? 0 : 1;
}
