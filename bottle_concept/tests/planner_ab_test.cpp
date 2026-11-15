// planner_ab_test — Phase 0 Task 15: bottle's EpistemicPlanner after taking the shared proposal type and
// rc::nbv::d_optimal_gain, vs a VERBATIM pre-task copy (tests/legacy_epistemic_planner.*, namespace rc::legacy).
// Bottle does NOT go through plan_countable (one far-side viewpoint, not four faces — see the ledger). ≥ 20 states
// through BottleBelief's own API. A difference FAILS unless listed. Run: LANG=es_ES.UTF-8 ./planner_ab_test
#include <clocale>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

#include "bottle_belief.h"
#include "epistemic_planner.h"
#include "legacy_epistemic_planner.h"

using namespace rc;

namespace
{
struct Case { std::string name; BottleBelief belief; Eigen::Vector2f camera; std::vector<nbv::Obstacle> obstacles; std::vector<Eigen::Vector2f> room; };
const std::vector<std::pair<int, const char*>> kExpectedDifferences = {};
std::vector<Eigen::Vector2f> rect(float x0, float x1, float y0, float y1) { return {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}; }

// The visible half of the cylinder wall, as seen from `cam`.
std::vector<Eigen::Vector3f> visible_points(const BottleBeliefState& s, const Eigen::Vector2f& cam)
{
    const float az = std::atan2(cam.y() - s.cy, cam.x() - s.cx);
    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i < 9; ++i)
    {
        const float a = az + (-0.5f + i / 8.0f) * std::numbers::pi_v<float> * 0.9f;
        for (int k = 0; k < 6; ++k)
            pts.emplace_back(s.cx + s.radius * std::cos(a), s.cy + s.radius * std::sin(a),
                             s.cz - 0.5f * s.height + s.height * k / 5.0f);
    }
    return pts;
}
void observe(BottleBelief& b, int frames, const Eigen::Vector2f& cam0, float wobble)
{
    for (int f = 0; f < frames; ++f)
    {
        const Eigen::Vector2f cam = cam0 + Eigen::Vector2f(wobble * std::sin(0.7f * f), wobble * std::cos(0.9f * f));
        BottleFrame fr;
        fr.points = visible_points(b.state(), cam);
        fr.R.assign(fr.points.size(), 0.01f * 0.01f);
        b.predict();
        b.update(fr);
    }
}
BottleBelief fresh(float x, float y, float z = 0.85f)
{
    BottleBeliefState s; s.cx = x; s.cy = y; s.cz = z;
    return BottleBelief(s, BottleBeliefParams{});
}
}  // namespace

int main()
{
    std::setlocale(LC_ALL, "");
    const auto open_room = rect(-6.0f, 6.0f, -6.0f, 6.0f);
    std::vector<Case> cases;
    const Eigen::Vector2f cams[4] = {{-2.0f, 0.0f}, {0.0f, -2.5f}, {1.5f, 1.5f}, {-1.0f, 3.0f}};
    for (int i = 0; i < 4; ++i)
        cases.push_back({"fresh, camera " + std::to_string(i), fresh(0.2f * i, -0.1f * i), cams[i], {}, open_room});
    for (const int n : {1, 5, 30})
        for (int c = 0; c < 2; ++c)
        {
            auto b = fresh(0.5f, 0.3f); observe(b, n, cams[c], 0.2f);
            cases.push_back({"after " + std::to_string(n) + " updates, camera " + std::to_string(c), b, cams[c], {}, open_room});
        }
    {
        auto b = fresh(0.0f, 0.0f, 0.78f); observe(b, 12, {-1.2f, 0.0f}, 0.0f);
        cases.push_back({"close camera, one bearing", b, {-1.2f, 0.0f}, {}, open_room});
        auto b2 = fresh(1.0f, 1.0f, 0.95f); observe(b2, 8, {4.0f, 4.0f}, 0.3f);
        cases.push_back({"far camera", b2, {4.0f, 4.0f}, {}, open_room});
        auto b3 = fresh(-2.0f, 2.0f); observe(b3, 3, {-2.0f, -1.0f}, 0.05f);
        cases.push_back({"short look from below", b3, {-2.0f, -1.0f}, {}, open_room});
        auto b4 = fresh(0.0f, 0.0f); observe(b4, 40, {0.0f, 2.0f}, 0.0f);
        cases.push_back({"one bearing x40", b4, {0.0f, 2.0f}, {}, open_room});
    }
    {   // on a table, a chair off its far side (the far-side viewpoint is partly occluded)
        nbv::Obstacle chair; chair.cx = 1.4f; chair.cy = 0.0f; chair.w = 0.6f; chair.h = 0.6f;
        auto b = fresh(0.0f, 0.0f); observe(b, 5, {-2.0f, 0.0f}, 0.2f);
        cases.push_back({"far side behind a chair", b, {-2.0f, 0.0f}, {chair}, open_room});
        nbv::Obstacle low = chair; low.z0 = 0.0f; low.z1 = 0.9f;
        cases.push_back({"far side behind a 0.9 m chair", b, {-2.0f, 0.0f}, {low}, open_room});
        nbv::Obstacle table; table.cx = 0.0f; table.cy = 0.0f; table.w = 1.2f; table.h = 0.8f; table.z0 = 0.0f; table.z1 = 0.75f;
        cases.push_back({"standing on its table", b, {-2.0f, 0.0f}, {table}, open_room});
    }
    {   // walls: far side inside the room, then beyond the wall (refused)
        auto b = fresh(0.0f, 0.5f); observe(b, 5, {0.0f, -2.0f}, 0.2f);
        cases.push_back({"far side toward a wall 3 m away", b, {0.0f, -2.0f}, {}, rect(-3.0f, 3.0f, -3.0f, 3.5f)});
        cases.push_back({"far side beyond the wall", b, {0.0f, -2.0f}, {}, rect(-3.0f, 3.0f, -3.0f, 0.8f)});
        auto b2 = fresh(2.5f, 0.0f); observe(b2, 5, {0.0f, 0.0f}, 0.2f);
        cases.push_back({"bottle near the +x wall", b2, {0.0f, 0.0f}, {}, rect(-3.0f, 3.0f, -3.0f, 3.0f)});
    }
    cases.push_back({"degenerate: camera on the bottle", fresh(0.0f, 0.0f), {0.0f, 0.0f}, {}, open_room});

    nbv::Sensor sensor;
    sensor.vfov_rad = 55.0f * std::numbers::pi_v<float> / 180.0f;
    sensor.height_m = 1.2f;
    const detect::DetectorEnvelope env{};
    legacy::EpistemicPlanner old_p; old_p.set_detector_envelope(env); old_p.set_robot_radius(0.30f);
    EpistemicPlanner         new_p; new_p.set_detector_envelope(env); new_p.set_robot_radius(0.30f);
    const float sigma_base = 0.01f;

    const auto near = [](float a, float b, float tol) { return std::abs(a - b) <= tol; };
    int fails = 0, valid = 0;
    for (int i = 0; i < static_cast<int>(cases.size()); ++i)
    {
        const auto& c = cases[i];
        const auto a = old_p.compute(c.belief, c.camera, sigma_base, sensor, c.obstacles, c.room);
        const auto b = new_p.compute(c.belief, c.camera, sigma_base, sensor, c.obstacles, c.room);
        const bool same = a.valid == b.valid
            and (not a.valid or (near(a.epistemic_target_x_m, b.epistemic_target_x_m, 1e-4f) and near(a.epistemic_target_y_m, b.epistemic_target_y_m, 1e-4f)
                                 and near(a.epistemic_target_yaw_rad, b.epistemic_target_yaw_rad, 1e-4f)
                                 and near(a.epistemic_gain, b.epistemic_gain, 1e-5f * std::abs(a.epistemic_gain))));
        valid += a.valid ? 1 : 0;
        const char* reason = nullptr;
        for (const auto& [k, why] : kExpectedDifferences) if (k == i) reason = why;
        std::printf("  [%2d] %-36s %s%s\n", i, c.name.c_str(), same ? "same" : (reason ? "EXPECTED: " : "DIFF"), reason ? reason : "");
        if (not same)
            std::printf("       legacy v=%d (%.4f,%.4f,%.4f) g=%.6f | new v=%d (%.4f,%.4f,%.4f) g=%.6f\n",
                        a.valid, a.epistemic_target_x_m, a.epistemic_target_y_m, a.epistemic_target_yaw_rad, a.epistemic_gain,
                        b.valid, b.epistemic_target_x_m, b.epistemic_target_y_m, b.epistemic_target_yaw_rad, b.epistemic_gain);
        if (not same and reason == nullptr) ++fails;
    }
    std::printf("planner_ab_test: %zu states, %d valid proposals — %s\n", cases.size(), valid, fails == 0 ? "PASS" : "FAIL");
    if (cases.size() < 20 or valid < 15) { std::printf("planner_ab_test: harness too weak (need ≥20 states, ≥15 valid)\n"); return 1; }
    return fails == 0 ? 0 : 1;
}
