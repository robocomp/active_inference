// planner_ab_test — Phase 0 Task 14: hood's EpistemicPlanner on the shared core (rc::nbv::plan_countable) vs a
// VERBATIM pre-task copy (tests/legacy_epistemic_planner.*, namespace rc::legacy). ≥ 20 belief states through
// HoodBelief's own API; every field the controller reads is compared. A difference FAILS unless listed in
// kExpectedDifferences with a reason. Run: LANG=es_ES.UTF-8 ./planner_ab_test
#include <clocale>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

#include "hood_belief.h"
#include "epistemic_planner.h"
#include "legacy_epistemic_planner.h"
#include "nbv/countable_planner.h"

using namespace rc;

namespace
{
struct Case { std::string name; HoodBelief belief; std::vector<nbv::Obstacle> obstacles; std::vector<Eigen::Vector2f> room; };

// ★EVERY valid state differs in epistemic_gain ONLY, and for one reason: hood publishes NO σ* (kHoodDofs all -1), and
// the pre-task planner applied the adequacy-gap bound WITHOUT the any_sigma_star guard ⇒ gap 0 ⇒ gain ≡ 0, so a hood
// never bid for a viewpoint visit. The core applies the guard (hood_dof.h states that intent) ⇒ unbounded gain. For
// these states the harness still requires every OTHER field identical and the legacy gain to be exactly 0.
constexpr const char* kGainGuard = "legacy gain 0 (unguarded adequacy gap, no sigma*); new = unbounded (guarded) — owner ruling";
const std::vector<std::pair<int, const char*>> kExpectedDifferences = {
    {0, kGainGuard}, {1, kGainGuard}, {2, kGainGuard}, {3, kGainGuard}, {4, kGainGuard}, {5, kGainGuard},
    {6, kGainGuard}, {7, kGainGuard}, {8, kGainGuard}, {9, kGainGuard}, {10, kGainGuard}, {11, kGainGuard},
    {12, kGainGuard}, {13, kGainGuard}, {14, kGainGuard}, {15, kGainGuard}, {16, kGainGuard}, {17, kGainGuard},
    {18, kGainGuard}, {19, kGainGuard}, {20, kGainGuard}, {21, kGainGuard}, {22, kGainGuard}};
constexpr float kExtent = 0.50f;   // the hood's vertical extent (HoodConcept default)

std::vector<Eigen::Vector2f> rect(float x0, float x1, float y0, float y1) { return {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}; }

// Top grid + the vertical face toward `bearing` (room frame, box→camera).
std::vector<Eigen::Vector3f> visible_points(const HoodBeliefState& s, float bearing)
{
    const float rel = std::remainder(bearing - s.yaw, 2.0f * std::numbers::pi_v<float>);
    const float hw = 0.5f * s.w, hh = 0.5f * s.h, c = std::cos(s.yaw), sn = std::sin(s.yaw);
    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 5; ++j)
        {
            const float lx = hw * (-1.0f + 2.0f * i / 6.0f), ly = hh * (-1.0f + 2.0f * j / 4.0f);
            pts.emplace_back(s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, s.H - kExtent);   // underside
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
            pts.emplace_back(s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, s.H - kExtent + kExtent * k / 3.0f);
    }
    return pts;
}

void observe(HoodBelief& b, int frames, float bearing0, float spread)
{
    for (int f = 0; f < frames; ++f)
    {
        const float bearing = bearing0 + spread * std::sin(0.7f * f);
        HoodFrame fr;
        fr.points = visible_points(b.state(), bearing);
        fr.R.assign(fr.points.size(), 0.02f * 0.02f);
        b.predict();
        b.update(fr);
    }
}

HoodBelief fresh(float x, float y, float yaw, float w = 0.7f, float h = 0.5f)
{
    HoodBeliefState s; s.cx = x; s.cy = y; s.yaw = yaw; s.w = w; s.h = h; s.H = 2.05f;
    return HoodBelief(s, HoodBeliefParams{});
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
    {   // mode-ambiguous: a square-ish box seen from one side
        auto b = fresh(0.0f, 0.0f, 0.0f, 0.9f, 0.85f); observe(b, 10, 1.5708f, 0.0f);
        cases.push_back({"square-ish, one side only", b, {}, open_room});
        auto b2 = fresh(2.0f, 1.0f, 2.0f, 1.0f, 0.95f); observe(b2, 3, 2.0f, 0.05f);
        cases.push_back({"square-ish, short look", b2, {}, open_room});
        auto b3 = fresh(-1.0f, 2.0f, -1.0f, 1.8f, 0.9f); observe(b3, 40, -1.0f, 0.0f);
        cases.push_back({"wide box, one bearing x40", b3, {}, open_room});
        auto b4 = fresh(0.0f, 0.0f, 0.4f, 0.6f, 0.6f); observe(b4, 8, 0.0f, 1.5f);
        cases.push_back({"small box, wide orbit", b4, {}, open_room});
    }
    {   // near a wall
        const auto room = rect(-6.0f, 6.0f, -8.0f, 1.0f);
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
        cases.push_back({"wall + obstacle", b2, {chair}, rect(-6.0f, 6.0f, -8.0f, 1.0f)});
    }
    // ── WALL-MOUNTED (review focus / spec G3): a hood flush against the +y wall at y = 1.0, three yaws of its
    // footprint. Indices recorded so the room-side assertion below runs on exactly these.
    const float kWallY = 1.0f;
    const auto wall_room = rect(-6.0f, 6.0f, -8.0f, kWallY);
    std::vector<int> wall_idx;
    for (const float x : {-1.5f, 0.0f, 2.0f})
    {
        auto b = fresh(x, kWallY - 0.25f, 0.0f); observe(b, 5, -1.5708f, 0.5f);
        wall_idx.push_back(static_cast<int>(cases.size()));
        cases.push_back({"wall-mounted at x=" + std::to_string(x), b, {}, wall_room});
    }
    cases.push_back({"boxed in: no usable face", fresh(0.0f, 0.0f, 0.0f), {}, rect(-0.6f, 0.6f, -0.4f, 0.4f)});

    nbv::Sensor sensor;
    sensor.vfov_rad = 55.0f * std::numbers::pi_v<float> / 180.0f;
    sensor.height_m = 1.2f;
    const detect::DetectorEnvelope env{};
    legacy::EpistemicPlanner old_p; old_p.set_detector_envelope(env); old_p.set_vertical_extent(kExtent);
    EpistemicPlanner         new_p; new_p.set_detector_envelope(env); new_p.set_vertical_extent(kExtent);
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
               and near(a.chosen_p_detect, b.chosen_p_detect, 1e-6f)
               and near(a.sensor_vfov_rad, b.sensor_vfov_rad, 1e-7f);
            for (int k = 0; k < 4; ++k) same = same and near(a.face_gains[k], b.face_gains[k], 1e-5f * std::max(1.0f, std::abs(a.face_gains[k])));
            for (int k = 0; k < 6; ++k) same = same and a.sigma_star[k] == b.sigma_star[k];
        }
        valid += a.valid ? 1 : 0;
        const char* reason = nullptr;
        for (const auto& [k, why] : kExpectedDifferences) if (k == i) reason = why;
        if (reason != nullptr and not same)
        {   // the listed difference is the GAIN only: re-compare with the new gain substituted, and legacy gain == 0
            auto b2 = b; b2.epistemic_gain = a.epistemic_gain;
            bool rest = a.valid == b2.valid and a.epistemic_gain == 0.0f and b.epistemic_gain > 0.0f
                and near(a.epistemic_target_x_m, b2.epistemic_target_x_m, 1e-4f) and near(a.epistemic_target_y_m, b2.epistemic_target_y_m, 1e-4f)
                and near(a.epistemic_target_yaw_rad, b2.epistemic_target_yaw_rad, 1e-4f)
                and near(a.standoff_min_m, b2.standoff_min_m, 1e-5f) and near(a.standoff_max_m, b2.standoff_max_m, 1e-5f)
                and near(a.framing_fill, b2.framing_fill, 1e-6f) and near(a.chosen_standoff_m, b2.chosen_standoff_m, 1e-5f)
                and near(a.chosen_p_detect, b2.chosen_p_detect, 1e-6f) and near(a.sensor_vfov_rad, b2.sensor_vfov_rad, 1e-7f);
            for (int k = 0; k < 4; ++k) rest = rest and near(a.face_gains[k], b2.face_gains[k], 1e-5f * std::max(1.0f, std::abs(a.face_gains[k])));
            if (not rest) reason = nullptr;   // something OTHER than the gain changed ⇒ a real, unlisted difference
        }
        std::printf("  [%2d] %-40s %s%s\n", i, c.name.c_str(), same ? "same" : (reason ? "EXPECTED: " : "DIFF"), reason ? reason : "");
        if (not same)
            std::printf("       legacy v=%d (%.4f,%.4f,%.4f) g=%.6f | new v=%d (%.4f,%.4f,%.4f) g=%.6f\n",
                        a.valid, a.epistemic_target_x_m, a.epistemic_target_y_m, a.epistemic_target_yaw_rad, a.epistemic_gain,
                        b.valid, b.epistemic_target_x_m, b.epistemic_target_y_m, b.epistemic_target_yaw_rad, b.epistemic_gain);
        if (not same and reason == nullptr) ++fails;
    }
    // Room side: every valid proposal for a wall-mounted hood lies inside the room, on the room side of its wall.
    for (const int i : wall_idx)
    {
        const auto b = new_p.compute(cases[i].belief, lat_rate, sigma_base, sensor, cases[i].obstacles, cases[i].room);
        const bool in_room = b.valid and nbv::detail::point_in_polygon(Eigen::Vector2f(b.epistemic_target_x_m, b.epistemic_target_y_m), cases[i].room)
                             and b.epistemic_target_y_m < kWallY;
        std::printf("  room-side [%2d] valid=%d (%.2f,%.2f) %s\n", i, b.valid, b.epistemic_target_x_m, b.epistemic_target_y_m,
                    in_room ? "inside, room side" : "FAIL: not inside on the room side");
        if (not in_room) ++fails;
    }
    // Report (spec G3): does the model stand further back and more face-on for the hood than for a chair-sized
    // target at the same wall spot? Same sensor, room and constant raw gain; only the target differs.
    {
        nbv::Sensor sen = sensor; sen.env = env;
        const nbv::CountableInputs in{sen, 0.30f, {}, wall_room};
        nbv::Target hood; hood.cx = 0.0f; hood.cy = kWallY - 0.25f; hood.w = 0.7f; hood.h = 0.5f; hood.z0 = 2.05f - kExtent; hood.z1 = 2.05f;
        nbv::Target chair = hood; chair.w = 0.6f; chair.h = 0.52f; chair.z0 = 0.0f; chair.z1 = 1.25f;
        const auto k1 = [](int, float) { return 1.0f; };
        const auto k2 = [](float) { return 1.0f; };
        const auto rh = nbv::plan_countable(hood, in, k1, k2), rc_ = nbv::plan_countable(chair, in, k1, k2);
        const auto report = [&](const char* what, const nbv::CountableResult& r, const nbv::Target& t)
        {
            const Eigen::Vector2f d(r.proposal.epistemic_target_x_m - t.cx, r.proposal.epistemic_target_y_m - t.cy);
            const float off_normal_deg = std::abs(std::atan2(d.x(), -d.y())) * 180.0f / std::numbers::pi_v<float>;
            std::printf("  G3 %-12s valid=%d face=%d standoff=%.2f m  dist=%.2f m  off-normal=%.1f deg\n",
                        what, r.proposal.valid, r.plan.best_face, r.plan.best_standoff_m, d.norm(), off_normal_deg);
            return std::pair{d.norm(), off_normal_deg};
        };
        const auto [dh, ah] = report("hood", rh, hood);
        const auto [dc, ac] = report("chair-sized", rc_, chair);
        std::printf("  G3 hood stands %s back (%.2f vs %.2f m) and %s face-on (%.1f vs %.1f deg)\n",
                    dh > dc ? "FURTHER" : "NOT further", dh, dc, ah <= ac ? "at least as" : "LESS", ah, ac);
    }
    std::printf("planner_ab_test: %zu states, %d valid proposals — %s\n", cases.size(), valid, fails == 0 ? "PASS" : "FAIL");
    if (cases.size() < 20 or valid < 15) { std::printf("planner_ab_test: harness too weak (need ≥20 states, ≥15 valid)\n"); return 1; }
    return fails == 0 ? 0 : 1;
}
