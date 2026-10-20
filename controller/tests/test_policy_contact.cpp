// Task 2 (local-motion policy): rollout + swept-body contact probability.
// Build+run (from controller/):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -I../common tests/test_policy_contact.cpp src/local_policy/contact.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tpc && /tmp/tpc
#include "../src/local_policy/contact.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

using namespace rc::policy;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)

static constexpr float kHalfPi = 1.57079633f;
static BaseLimits limits() { BaseLimits b; b.v_max = 0.5f; b.w_max = 1.5f; b.a_lin = 1.f; b.d_lin = 1.f; b.a_rot = 2.f; b.d_rot = 2.f; return b; }
static rc::RobotFootprint body(float margin = 0.05f) { auto f = rc::RobotFootprint::shadow(); f.set_safety_margin(margin); return f; }
// Wall of returns on the line x = x0 (room frame), y in [y0, y1]; robot yaw = pi/2 puts room +x on its right.
static std::vector<Return> wall_x(float x0, float y0, float y1, float sig = 0.01f)
{ std::vector<Return> r; for (float y = y0; y <= y1; y += 0.05f) r.push_back({{x0, y}, sig, 1.f}); return r; }
static std::vector<Return> wall_y(float y0, float x0, float x1, float sig = 0.01f)
{ std::vector<Return> r; for (float x = x0; x <= x1; x += 0.05f) r.push_back({{x, y0}, sig, 1.f}); return r; }

int main()
{
    const auto b = limits();
    const Pose2 origin{{0.f, 0.f}, kHalfPi};
    const float sxy = 0.005f, syaw = 0.005f;

    { // rollout_straight: 0.3 m/s for 1 s then brake at 1.0 m/s^2 -> 0.3 + 0.045
        const auto s = rollout(origin, 0.3f, 0.f, 1.f, b, 0.05f);
        CHECK("rollout_straight advance", std::abs((s.back().p.y() - 0.f) - 0.345f) < 0.005f and std::abs(s.back().p.x()) < 1e-4f);
        CHECK("rollout_straight starts at start", (s.front().p - origin.p).norm() < 1e-6f);
        // final speed 0: the last step must not still be moving
        const float last = (s.back().p - s[s.size() - 2].p).norm();
        CHECK("rollout_straight final speed 0", last < 0.3f * 0.05f * 0.5f);
    }
    { // rollout_turn_in_place: 1 rad/s for 1 s then brake at 2 rad/s^2 -> 1 + 0.25
        const auto s = rollout(origin, 0.f, 1.f, 1.f, b, 0.05f);
        const float dyaw = s.back().yaw - origin.yaw;
        CHECK("rollout_turn_in_place yaw", std::abs(dyaw - (1.f + 0.5f / b.d_rot)) < 0.02f);
        CHECK("rollout_turn_in_place no translation", (s.back().p - origin.p).norm() < 1e-5f);
    }
    { // arc closes: v=0.3,w=0.5 for 2 s lies on the circle of radius v/w about its centre
        const auto s = rollout(origin, 0.3f, 0.5f, 2.f, {b}, 0.1f);
        const Eigen::Vector2f c = origin.p + (0.3f / 0.5f) * Eigen::Vector2f(-std::sin(origin.yaw), std::cos(origin.yaw));
        // ★ only the HOLD phase (first 20 steps) is one circle: the two axes brake at different times afterwards
        float worst = 0; for (std::size_t i = 0; i < 21; ++i) { const auto& q = s[i]; worst = std::max(worst, std::abs((q.p - c).norm() - 0.6f)); }
        CHECK("rollout_arc stays on circle", worst < 1e-4f);
    }
    { // signed_distance agrees with RobotFootprint::contains_yaw (quarter-turn convention)
        // margin 0: exact agreement. margin > 0: ours is a ROUNDED (Euclidean) grow, contains() mitres the corners,
        // so ours must be inside-or-equal (sd < 0 implies contains).
        const auto f = body(0.f), fm = body(0.05f); std::mt19937 g(7); std::uniform_real_distribution<float> u(-0.6f, 0.6f), ang(-3.1f, 3.1f);
        int bad = 0, badm = 0;
        for (int i = 0; i < 2000; ++i)
        {
            const Pose2 p{{u(g), u(g)}, ang(g)}; const Eigen::Vector2f q{u(g), u(g)};
            const float sd = signed_distance(f, p, q);
            if (std::abs(sd) > 1e-4f and (sd < 0.f) != f.contains_yaw(q, p.p, p.yaw)) ++bad;
            if (signed_distance(fm, p, q) < 0.f and not fm.contains_yaw(q, p.p, p.yaw)) ++badm;
        }
        CHECK("signed_distance (margin) inside implies contains_yaw", badm == 0);
        CHECK("signed_distance matches contains_yaw", bad == 0);
        CHECK("signed_distance flank value", std::abs(signed_distance(fm, origin, {0.40f, 0.f}) - (0.40f - 0.284f - 0.05f)) < 1e-4f);
    }
    { // wall_beside_parallel_free
        const auto sw = rollout(origin, 0.3f, 0.f, 1.f, b, 0.05f);
        const auto r = wall_x(0.40f, -1.f, 2.f);
        const float lp = log_p_no_contact(body(), sw, r, sxy, syaw);
        std::printf("  parallel lp=%.5f\n", lp);
        CHECK("wall_beside_parallel_free", lp > std::log(0.99f));
    }
    { // wall_beside_turn_hits
        const auto sw = rollout(origin, 0.f, 1.f, 1.f, b, 0.05f);
        const auto r = wall_x(0.34f, -1.f, 1.f);
        const float lp = log_p_no_contact(body(), sw, r, sxy, syaw);
        std::printf("  turn lp=%.5f\n", lp);
        CHECK("wall_beside_turn_hits", lp < std::log(0.5f));
    }
    { // wall_ahead_fast_vs_slow
        const auto r = wall_y(0.8f, -1.f, 1.f);
        const float fast = log_p_no_contact(body(), rollout(origin, 0.35f, 0.f, 2.f, b, 0.05f), r, sxy, syaw);
        const float slow = log_p_no_contact(body(), rollout(origin, 0.10f, 0.f, 2.f, b, 0.05f), r, sxy, syaw);
        std::printf("  fast=%.4f slow=%.4f\n", fast, slow);
        CHECK("wall_ahead_fast_vs_slow", fast < slow and fast < std::log(0.5f) and slow > std::log(0.99f));
    }
    { // start_inside_only_separating: return 0.02 m inside the right flank (x = 0.264)
        const std::vector<Return> r{{{0.264f, 0.f}, 0.01f, 1.f}};
        const std::vector<Pose2> away{origin, {{-0.2f, 0.f}, kHalfPi}};   // body moves room -x = its left: flank leaves the point
        const std::vector<Pose2> into{origin, {{0.2f, 0.f}, kHalfPi}};    // body moves right: point sinks deeper
        CHECK("start_inside separating is free", log_p_no_contact(body(), away, r, sxy, syaw) > std::log(0.99f));
        CHECK("start_inside deeper hits", log_p_no_contact(body(), into, r, sxy, syaw) < std::log(0.5f));
    }
    { // conditional_continuous_at_surface: a return on the grown nose line at sd0 = -1 mm, 0, +1 mm; the body
      // advances 5 mm into it (a small deepening keeps P mid-range, where a step would show)
        const std::vector<Pose2> sw{origin, {{0.f, 0.005f}, kHalfPi}};
        float P[3]; int i = 0;
        for (float d : {-0.001f, 0.f, 0.001f})
        {
            const std::vector<Return> r{{{0.f, 0.28f + d}, 0.01f, 1.f}};
            P[i++] = 1.f - std::exp(log_p_no_contact(body(), sw, r, sxy, syaw));
        }
        std::printf("  P(-1mm)=%.4f P(0)=%.4f P(+1mm)=%.4f\n", P[0], P[1], P[2]);
        CHECK("conditional_continuous_at_surface", std::abs(P[0] - P[1]) < 0.05f and std::abs(P[1] - P[2]) < 0.05f and P[1] > 0.1f and P[1] < 0.9f);
    }
    { // backing_away_free: a wall exactly at the grown nose (y = 0.23 + 0.05), straight back
        const auto r = wall_y(0.28f, -0.4f, 0.4f);
        const float P = 1.f - std::exp(log_p_no_contact(body(), rollout(origin, -0.3f, 0.f, 1.f, b, 0.05f), r, sxy, syaw));
        std::printf("  backing away from a wall at the margin edge: P=%.2e\n", P);
        CHECK("backing_away_free", P < 1e-3f);
    }
    { // p_exist scales the hit; a far return is ignored
        const std::vector<Return> hit{{{0.f, 0.3f}, 0.01f, 1.f}}, ghost{{{0.f, 0.3f}, 0.01f, 0.1f}}, far{{{50.f, 0.f}, 0.01f, 1.f}};
        const auto sw = rollout(origin, 0.3f, 0.f, 1.f, b, 0.05f);
        const float a = log_p_no_contact(body(), sw, hit, sxy, syaw), g = log_p_no_contact(body(), sw, ghost, sxy, syaw);
        CHECK("p_exist=0.1 hits ~10x less", g > a and std::abs(g - std::log(0.9f)) < 0.02f);
        CHECK("far return ignored", log_p_no_contact(body(), sw, far, sxy, syaw) == 0.f);
    }
    { // timing: 21-pose sweep vs 300 returns
        std::vector<Pose2> sw; for (int i = 0; i < 21; ++i) sw.push_back({{0.f, 0.02f * i}, kHalfPi});
        std::mt19937 g(3); std::uniform_real_distribution<float> u(-1.5f, 1.5f);
        std::vector<Return> r; for (int i = 0; i < 300; ++i) r.push_back({{u(g), u(g)}, 0.02f, 1.f});
        const auto f = body(); std::vector<double> ms; volatile float sink = 0;
        for (int k = 0; k < 201; ++k)
        {
            const auto t0 = std::chrono::steady_clock::now();
            sink = sink + log_p_no_contact(f, sw, r, sxy, syaw);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        std::printf("TIMING median log_p_no_contact (21 poses x 300 returns): %.4f ms\n", ms[ms.size() / 2]);
    }
    std::printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
