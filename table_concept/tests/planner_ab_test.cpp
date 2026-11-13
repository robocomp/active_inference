// planner_ab_test — Phase 0 Task 12 (+ the owner's gain-formula change, 2026-11-13): table's EpistemicPlanner on the shared core (rc::nbv::plan_countable) vs a
// VERBATIM pre-task copy (tests/legacy_epistemic_planner.*, namespace rc::legacy). ≥ 20 belief states through
// TableBelief's own API; every field the controller reads is compared. A difference FAILS unless listed in
// kExpectedDifferences with a reason. Run: LANG=es_ES.UTF-8 ./planner_ab_test
#include <clocale>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

#include "table_belief.h"
#include "epistemic_planner.h"
#include "legacy_epistemic_planner.h"

using namespace rc;

namespace
{
struct Case { std::string name; TableBelief belief; std::vector<nbv::Obstacle> obstacles; std::vector<Eigen::Vector2f> room; };

// Owner 2026-11-13: table's gain moves from min(p·raw, gap) to p·min(raw, gap). Only epistemic_gain may differ, and it
// can only FALL (p ≤ 1 ⇒ p·min(raw, gap) ≤ min(p·raw, gap)); equal where p = 1 or the gap does not bind. The check
// below enforces exactly that for listed states.
constexpr const char* kExpectation = "gain p*min(raw,gap) replaces min(p*raw,gap) (owner 2026-11-13): gain only, never higher";
const std::vector<std::pair<int, const char*>> kExpectedDifferences = {
    {0, kExpectation}, {1, kExpectation}, {2, kExpectation}, {3, kExpectation}, {4, kExpectation}, {5, kExpectation},
    {6, kExpectation}, {7, kExpectation}, {8, kExpectation}, {9, kExpectation}, {10, kExpectation}, {11, kExpectation},
    {12, kExpectation}, {13, kExpectation}, {15, kExpectation}, {16, kExpectation}, {17, kExpectation}, {18, kExpectation}};

std::vector<Eigen::Vector2f> rect(float x0, float x1, float y0, float y1) { return {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}; }

// Tabletop grid + the vertical face toward `bearing` (room frame, table→camera).
std::vector<Eigen::Vector3f> visible_points(const TableBeliefState& s, float bearing)
{
    const float rel = std::remainder(bearing - s.yaw, 2.0f * std::numbers::pi_v<float>);
    const float hw = 0.5f * s.w, hh = 0.5f * s.h, c = std::cos(s.yaw), sn = std::sin(s.yaw);
    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 5; ++j)
        {
            const float lx = hw * (-1.0f + 2.0f * i / 6.0f), ly = hh * (-1.0f + 2.0f * j / 4.0f);
            pts.emplace_back(s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, s.H);
        }
    for (int m = 0; m < 9; ++m)
    {
        const float t = -1.0f + 2.0f * m / 8.0f;
        float lx, ly;
        if (std::abs(rel) < std::numbers::pi_v<float> / 4)          { lx =  hw; ly = t * hh; }
        else if (std::abs(rel) > 3 * std::numbers::pi_v<float> / 4) { lx = -hw; ly = t * hh; }
        else if (rel > 0)                                            { lx = t * hw; ly =  hh; }
        else                                                         { lx = t * hw; ly = -hh; }
        for (int k = 0; k < 4; ++k)
            pts.emplace_back(s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, s.H * (0.4f + 0.2f * k));
    }
    return pts;
}

void observe(TableBelief& b, int frames, float bearing0, float spread)
{
    for (int f = 0; f < frames; ++f)
    {
        const float bearing = bearing0 + spread * std::sin(0.7f * f);
        TableFrame fr;
        fr.points = visible_points(b.state(), bearing);
        fr.R.assign(fr.points.size(), 0.02f * 0.02f);
        b.predict();
        b.update(fr);
        b.resolve_orientation(fr.points, 0.02f * 0.02f, 1.0f);
    }
}

TableBelief fresh(float x, float y, float yaw, float w = 1.2f, float h = 0.8f)
{
    TableBeliefState s; s.cx = x; s.cy = y; s.yaw = yaw; s.w = w; s.h = h;
    return TableBelief(s, TableBeliefParams{});
}
}  // namespace

int main()
{
    std::setlocale(LC_ALL, "");
    const auto open_room = rect(-6.0f, 6.0f, -6.0f, 6.0f);
    std::vector<Case> cases;
    const float yaws[4] = {0.0f, 0.7f, 1.5708f, -2.3f};
    for (int i = 0; i < 4; ++i)
        cases.push_back({"fresh prior yaw " + std::to_string(yaws[i]), fresh(0.5f * i - 1.0f, 0.3f * i, yaws[i]), {}, open_room});
    for (const int n : {1, 5, 30})
        for (const float b0 : {0.4f, -2.0f})
        {
            auto b = fresh(1.0f, -0.5f, 0.3f); observe(b, n, b0, 0.8f);
            cases.push_back({"after " + std::to_string(n) + " updates, bearing " + std::to_string(b0), b, {}, open_room});
        }
    {   // mode-ambiguous: a near-square table seen from one side
        auto b = fresh(0.0f, 0.0f, 0.0f, 0.9f, 0.85f); observe(b, 10, 1.5708f, 0.0f);
        cases.push_back({"near-square, one side only", b, {}, open_room});
        auto b2 = fresh(2.0f, 1.0f, 2.0f, 1.0f, 0.95f); observe(b2, 3, 2.0f, 0.05f);
        cases.push_back({"near-square, short look", b2, {}, open_room});
        auto b3 = fresh(-1.0f, 2.0f, -1.0f, 1.8f, 0.9f); observe(b3, 40, -1.0f, 0.0f);
        cases.push_back({"long table, one bearing x40", b3, {}, open_room});
        auto b4 = fresh(0.0f, 0.0f, 0.4f, 0.6f, 0.6f); observe(b4, 8, 0.0f, 1.5f);
        cases.push_back({"small square table, wide orbit", b4, {}, open_room});
    }
    {   // near a wall
        const auto room = rect(-3.0f, 3.0f, -3.0f, 1.0f);
        for (const float yaw : {0.0f, 1.5708f, 3.0f})
        {
            auto b = fresh(0.0f, 0.3f, yaw); observe(b, 5, -1.5708f, 0.6f);
            cases.push_back({"near the +y wall, yaw " + std::to_string(yaw), b, {}, room});
        }
    }
    {   // obstacles
        nbv::Obstacle chair; chair.cx = 1.5f; chair.cy = 0.0f; chair.w = 0.6f; chair.h = 0.6f;
        auto b = fresh(0.0f, 0.0f, 0.0f); observe(b, 5, 0.0f, 0.5f);
        cases.push_back({"+x face blocked by a chair", b, {chair}, open_room});
        nbv::Obstacle low = chair; low.z0 = 0.0f; low.z1 = 0.9f;
        cases.push_back({"+x face behind a 0.9 m chair", b, {low}, open_room});
        auto b2 = fresh(0.0f, 0.3f, 0.0f); observe(b2, 5, -1.5708f, 0.6f);
        cases.push_back({"wall + obstacle", b2, {chair}, rect(-3.0f, 3.0f, -3.0f, 1.0f)});
    }
    cases.push_back({"boxed in: no usable face", fresh(0.0f, 0.0f, 0.0f), {}, rect(-0.6f, 0.6f, -0.4f, 0.4f)});

    nbv::Sensor sensor;
    sensor.vfov_rad = 55.0f * std::numbers::pi_v<float> / 180.0f;
    sensor.height_m = 1.2f;
    const detect::DetectorEnvelope env{};
    legacy::EpistemicPlanner old_p(1.8f); old_p.set_detector_envelope(env); old_p.set_robot_radius(0.30f);
    EpistemicPlanner         new_p(1.8f); new_p.set_detector_envelope(env); new_p.set_robot_radius(0.30f);
    const float lat_rate = 0.01f, sigma_base = 0.02f;

    const auto near = [](float a, float b, float tol) { return std::abs(a - b) <= tol; };
    int fails = 0, valid = 0;
    for (int i = 0; i < static_cast<int>(cases.size()); ++i)
    {
        const auto& c = cases[i];
        const auto a = old_p.compute(c.belief, lat_rate, sigma_base, sensor, c.obstacles, c.room);
        const auto b = new_p.compute(c.belief, lat_rate, sigma_base, sensor, c.obstacles, c.room);
        bool same = a.valid == b.valid;
        if (same and a.valid)
        {
            same = near(a.epistemic_target_x_m, b.epistemic_target_x_m, 1e-4f) and near(a.epistemic_target_y_m, b.epistemic_target_y_m, 1e-4f)
               and near(a.epistemic_target_yaw_rad, b.epistemic_target_yaw_rad, 1e-4f)
               and near(a.epistemic_gain, b.epistemic_gain, 1e-5f * std::abs(a.epistemic_gain))
               and near(a.standoff_min_m, b.standoff_min_m, 1e-5f) and near(a.standoff_max_m, b.standoff_max_m, 1e-5f)
               and near(a.framing_fill, b.framing_fill, 1e-6f) and near(a.chosen_standoff_m, b.chosen_standoff_m, 1e-5f)
               and near(a.chosen_p_detect, b.chosen_p_detect, 1e-6f) and near(a.chosen_fill, b.chosen_fill, 1e-6f)
               and near(a.sensor_vfov_rad, b.sensor_vfov_rad, 1e-7f);
            for (int k = 0; k < 4; ++k) same = same and near(a.face_gains[k], b.face_gains[k], 1e-5f * std::max(1.0f, std::abs(a.face_gains[k])));
            for (int k = 0; k < 6; ++k) same = same and a.sigma_star[k] == b.sigma_star[k];
        }
        valid += a.valid ? 1 : 0;
        const char* reason = nullptr;
        for (const auto& [k, why] : kExpectedDifferences) if (k == i) reason = why;
        if (reason != nullptr and not same)
        {   // the listed difference is the GAIN only, and it never rises
            auto b2 = b; b2.epistemic_gain = a.epistemic_gain;
            bool rest = a.valid == b2.valid and b.epistemic_gain <= a.epistemic_gain * (1.0f + 1e-6f)
                and near(a.epistemic_target_x_m, b2.epistemic_target_x_m, 1e-4f) and near(a.epistemic_target_y_m, b2.epistemic_target_y_m, 1e-4f)
                and near(a.epistemic_target_yaw_rad, b2.epistemic_target_yaw_rad, 1e-4f)
                and near(a.standoff_min_m, b2.standoff_min_m, 1e-5f) and near(a.standoff_max_m, b2.standoff_max_m, 1e-5f)
                and near(a.framing_fill, b2.framing_fill, 1e-6f) and near(a.chosen_standoff_m, b2.chosen_standoff_m, 1e-5f)
                and near(a.chosen_p_detect, b2.chosen_p_detect, 1e-6f) and near(a.chosen_fill, b2.chosen_fill, 1e-6f)
                and near(a.sensor_vfov_rad, b2.sensor_vfov_rad, 1e-7f);
            for (int k = 0; k < 4; ++k) rest = rest and near(a.face_gains[k], b2.face_gains[k], 1e-5f * std::max(1.0f, std::abs(a.face_gains[k])));
            for (int k = 0; k < 6; ++k) rest = rest and a.sigma_star[k] == b2.sigma_star[k];
            if (not rest) reason = nullptr;   // something other than a lower gain ⇒ a real, unlisted difference
        }
        std::printf("  [%2d] %-40s %s%s\n", i, c.name.c_str(), same ? "same" : (reason ? "EXPECTED: " : "DIFF"), reason ? reason : "");
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
