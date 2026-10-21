// Build (from room_concept/):
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/room_explored_selftest.cpp src/layout_explorer.cpp \
//       src/wall_map.cpp src/wall_segmenter.cpp -o /tmp/res && /tmp/res
//
// P(room explored) — src/room_explored.h, the belief behind `until explored(room)` (Task 5, 2026-10-21).
//   A. the model's laws (Luce stop-vs-look, noisy-OR with memory, what memory restores);
//   B. THE AGENT'S EXPLORER on a real accumulator: the agent's FreeSpacePolygon fed ray-cast scans of a synthetic
//      L room along a drive, and the agent's rc::layout::plan_coverage with the agent's coverage gain
//      (visible_unknown_area over countable_mask) and price (the body's footprint area) — the same calls
//      RoomSceneGraph::dsr_update_layout_affordance makes. Checks: the belief never falls while the robot only adds
//      scans; it crosses 0.5 exactly where the explorer says `finished`; nothing standable ⇒ no belief at all;
//   C. recalled rooms: a complete memory is explored on entry and offers nothing; a partly remembered room starts at
//      its memory and needs fewer scans than a fresh one to reach the same boundary.
//   D. the agent's glue (rc::explored::Tracker): only the coverage belief is remembered; a recalled room's explorer
//      `finished` is terminal (p = 1), like promotion.
#include "free_space_polygon.h"
#include "layout_explorer.h"
#include "room_explored.h"

#include <clocale>
#include <cmath>
#include <optional>
#include <print>
#include <random>
#include <vector>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)
bool near(float a, float b, float tol = 1e-5f) { return std::abs(a - b) <= tol; }

using Poly = std::vector<Eigen::Vector2f>;
using namespace rc::explored;

// ── A: the model ─────────────────────────────────────────────────────────────────────────────────────────────────
void model_laws()
{
    CHECK(p_stop(0.5f, 0.f) == std::optional<float>(1.f));        // nothing left: explored
    CHECK(near(*p_stop(0.5f, 0.5f), 0.5f));                        // the explorer's `finished` boundary
    CHECK(*p_stop(0.5f, 5.f) < *p_stop(0.5f, 1.f));                // more left: less explored
    CHECK(not p_stop(0.f, 1.f).has_value());                       // no price: nothing to compare
    CHECK(near(*p_stop(1.f, 1.f / 9.f), 0.9f));                    // ">= 0.9" = the best look worth <= 1/9 trip
    CHECK(near(combine(0.f, 0.3f), 0.3f));                         // no memory: this visit alone
    CHECK(near(combine(1.f, 0.f), 1.f));                           // complete memory: explored whatever the visit saw
    CHECK(near(combine(0.6f, 0.f), 0.6f));                         // a visit that saw nothing yet: memory alone
    CHECK(near(recalled_gain(0.75f, 4.f), 1.f));
    CHECK(not belief(0, 0.f, 0.5f).has_value());                   // nothing assessable ⇒ no belief (never "explored")
    CHECK(near(*belief(10, 0.5f, 0.5f), 0.5f));
    CHECK(near(*belief(10, 123.f, 0.5f, 1.f), 1.f));
    // the explorer's `finished` on a recalled room (discounted gain == price) is p = 1/(2 − p_mem) >= 0.5
    for (const float pm : {0.f, 0.3f, 0.6f, 0.9f})
        CHECK(near(*belief(10, 0.5f, 0.5f, pm), 1.f / (2.f - pm)));
    CHECK(restored(0.4f, 0.8f) == 0.4f);                           // the recall's own attribute first
    CHECK(restored(std::nullopt, 0.8f) == 0.8f);                   // then this agent's record
    CHECK(restored(std::nullopt, std::nullopt) == 1.f);            // the promoted outline alone: complete
    Memory m;
    CHECK(not m.recall("room_2").has_value());
    m.remember("room_2", 0.7f);
    m.remember("room_2", 0.8f);
    CHECK(m.recall("room_2") == std::optional<float>(0.8f));
}

// ── B/C: the agent's explorer on a real accumulator ──────────────────────────────────────────────────────────────
std::optional<float> ray_segment_t(const Eigen::Vector2f& o, const Eigen::Vector2f& d, const Eigen::Vector2f& a,
                                   const Eigen::Vector2f& b)
{
    const Eigen::Vector2f e = b - a;
    const float den = d.x() * e.y() - d.y() * e.x();
    if (std::abs(den) < 1e-12f) return std::nullopt;
    const Eigen::Vector2f ao = a - o;
    const float t = (ao.x() * e.y() - ao.y() * e.x()) / den;
    const float u = (ao.x() * d.y() - ao.y() * d.x()) / den;
    if (t < 0.f or u < 0.f or u > 1.f) return std::nullopt;
    return t;
}

// L room: a 6 x 3 bottom arm and a 3 x 3 upper arm on the left (free_space_polygon_selftest's room).
const Poly kRoom = {{0.f, 0.f}, {6.f, 0.f}, {6.f, 3.f}, {3.f, 3.f}, {3.f, 6.f}, {0.f, 6.f}};
constexpr float kWidth = 0.5f, kLength = 0.6f;   // Shadow-like body
const float kPrice = static_cast<float>(M_PI) * 0.25f * (kWidth * kWidth + kLength * kLength);   // the agent's footprint

void scan(rc::freespace::FreeSpacePolygon& fs, const Eigen::Vector2f& o, std::mt19937& rng)
{
    std::normal_distribution<float> noise(0.f, 0.02f);
    std::vector<Eigen::Vector2f> ends;
    for (int b = 0; b < 720; ++b)
    {
        const float a = 2.f * static_cast<float>(M_PI) * b / 720.f;
        const Eigen::Vector2f u(std::cos(a), std::sin(a));
        float t = 8.f;
        bool hit = false;
        for (std::size_t e = 0; e < kRoom.size(); ++e)
            if (const auto tt = ray_segment_t(o, u, kRoom[e], kRoom[(e + 1) % kRoom.size()]); tt and *tt < t)
            { t = *tt; hit = true; }
        if (hit) ends.push_back(o + u * (t + noise(rng)));
    }
    fs.add_scan(o, ends);
    fs.add_footprint(o, 0.5f * std::hypot(kWidth, kLength) + fs.params().erode_m);
}

struct Assessed { std::optional<float> p; rc::layout::Plan plan; };

/// The agent's coverage plan over `outline` (the room's published outline), gain discounted by memory.
Assessed assess(const rc::freespace::FreeSpacePolygon& fs, const Poly& outline, const Eigen::Vector2f& robot, float p_mem,
                bool nothing_standable = false)
{
    rc::layout::ExplorerParams ep;
    ep.clearance = 0.5f * std::hypot(kWidth, kLength);
    rc::layout::FeasibleGrid grid(outline, ep.cell, ep.clearance);
    const auto countable = fs.countable_mask(outline, 0.40f);
    const auto standable = [&](const Eigen::Vector2f& v) { return not nothing_standable and rc::freespace::point_in_polygon(v, outline); };
    Assessed a;
    a.plan = rc::layout::plan_coverage(grid, robot, ep, standable,
        [&](const Eigen::Vector2f& v) { return recalled_gain(p_mem, fs.visible_unknown_area(v, 8.f, 180, &countable)); },
        kPrice);
    a.p = belief(a.plan.evaluated, a.plan.best_gain, a.plan.price, p_mem);   // RoomSceneGraph::publish_plan_explored
    CHECK(a.plan.price == kPrice);                                              // the plan carries its own price
    return a;
}

Poly drive()
{
    Poly path;
    for (float x = 5.0f; x >= 1.5f; x -= 0.5f) path.emplace_back(x, 1.0f);
    for (float y = 1.5f; y <= 5.0f; y += 0.5f) path.emplace_back(1.5f, y);
    return path;
}

/// Scans until the belief reaches `boundary`; returns how many it took (path size + 1 if never).
int scans_to(float boundary, float p_mem, std::vector<float>* trace = nullptr)
{
    rc::freespace::FreeSpacePolygon fs;
    std::mt19937 rng(7);
    const auto path = drive();
    for (std::size_t k = 0; k < path.size(); ++k)
    {
        scan(fs, path[k], rng);
        const auto a = assess(fs, kRoom, path[k], p_mem);
        if (trace and a.p.has_value()) trace->push_back(*a.p);
        if (a.p.has_value() and *a.p >= boundary) return static_cast<int>(k) + 1;
    }
    return static_cast<int>(path.size()) + 1;
}

void fresh_room()
{
    rc::freespace::FreeSpacePolygon fs;
    std::mt19937 rng(7);
    const auto path = drive();
    std::optional<float> prev;
    bool crossed_consistently = true, monotone = true;
    for (const auto& o : path)
    {
        scan(fs, o, rng);
        const auto a = assess(fs, kRoom, o, 0.f);
        CHECK(a.p.has_value());
        if (not a.p.has_value()) continue;
        // the boundary 0.5 IS the explorer's own `finished` (no deferrals here)
        crossed_consistently = crossed_consistently and (a.plan.finished == (*a.p > 0.5f));
        // scans only turn unknown cells known: the best look left can only shrink, the belief only grow
        monotone = monotone and (not prev.has_value() or *a.p >= *prev - 1e-6f);
        prev = a.p;
    }
    CHECK(crossed_consistently);
    CHECK(monotone);
    // first scan, deep in the bottom arm: the upper arm is unseen ⇒ far from explored
    {
        rc::freespace::FreeSpacePolygon fs1;
        std::mt19937 r1(7);
        scan(fs1, path.front(), r1);
        const auto a = assess(fs1, kRoom, path.front(), 0.f);
        CHECK(a.p.has_value() and *a.p < 0.5f and not a.plan.finished);
        std::println("  fresh room: p after 1 scan = {:.3f} (best look {:.2f} m², price {:.2f} m²)",
                     a.p.value_or(-1.f), a.plan.best_gain, kPrice);
    }
    std::println("  fresh room: p after the whole drive = {:.3f}", prev.value_or(-1.f));
    CHECK(prev.has_value() and *prev > 0.5f);
    // nothing the explorer may stand on: no belief — the predicate stays unknown, it is not "explored"
    CHECK(not assess(fs, kRoom, path.back(), 0.f, true).p.has_value());
}

void recalled_rooms()
{
    const auto path = drive();
    {   // complete memory (the promoted outline, no record): explored on entry, nothing offered
        rc::freespace::FreeSpacePolygon fs;   // the accumulator restarts empty at the crossing
        std::mt19937 rng(7);
        scan(fs, path.front(), rng);
        const float pm = restored(std::nullopt, std::nullopt);
        const auto a = assess(fs, kRoom, path.front(), pm);
        CHECK(a.p == std::optional<float>(1.f));
        CHECK(a.plan.finished and not a.plan.ok);   // no standpoint worth a trip: the robot is not sent anywhere
    }
    {   // partly remembered (this agent left it at 0.6): starts from memory, explores only the rest
        rc::freespace::FreeSpacePolygon fs;
        std::mt19937 rng(7);
        scan(fs, path.front(), rng);
        const float pm = restored(std::nullopt, 0.6f);
        const auto a = assess(fs, kRoom, path.front(), pm);
        CHECK(a.p.has_value() and *a.p >= 0.6f and *a.p < 1.f);
        CHECK(a.plan.ok);                            // it still has somewhere worth going
        const auto fresh_first = assess(fs, kRoom, path.front(), 0.f);
        CHECK(fresh_first.p.has_value() and *a.p > *fresh_first.p);
        std::println("  partly remembered (0.6): p on entry = {:.3f} (fresh room at the same scan {:.3f})",
                     a.p.value_or(-1.f), fresh_first.p.value_or(-1.f));
    }
    {   // along the same drive: never below the fresh room's belief, never below memory, and a boundary sooner
        std::vector<float> fresh_t, partial_t;
        const int fresh = scans_to(2.f, 0.f, &fresh_t), partial = scans_to(2.f, 0.6f, &partial_t);
        (void)fresh; (void)partial;
        bool above = fresh_t.size() == partial_t.size();
        for (std::size_t k = 0; above and k < fresh_t.size(); ++k)
            above = partial_t[k] >= fresh_t[k] - 1e-6f and partial_t[k] >= 0.6f - 1e-6f;
        CHECK(above);
        const int f6 = scans_to(0.6f, 0.f), p6 = scans_to(0.6f, 0.6f);
        std::println("  scans to p >= 0.60: fresh {} · partly remembered {}", f6, p6);
        CHECK(p6 < f6);
    }
}
// ── D: the agent's glue (Tracker): what is remembered, and the terminal verdicts ─────────────────────────────────
void glue_fixtures()
{
    {   // I-1: a room left while the refiner holds contested cells is REMEMBERED as surveyed — no re-survey on re-entry
        Tracker t;
        t.reset();
        CHECK(t.plan("room_2", Phase::Survey, 40, 3.f, 0.5f, false).has_value());     // surveying: p < 0.5
        CHECK(t.memory().recall("room_2").has_value() and *t.memory().recall("room_2") < 0.5f);
        CHECK(t.settled("room_2") == 1.f);                                             // promoted, nothing contested
        const auto pc = t.plan("room_2", Phase::Contested, 40, 2.f, 0.1f, false);       // the refiner's question
        CHECK(pc.has_value() and *pc < 0.1f);                                          // published as is ...
        CHECK(t.memory().recall("room_2") == std::optional<float>(1.f));               // ... but never remembered
        t.reset();                                                                     // crossed into room_3
        t.enter_recalled("room_2", std::nullopt);                                      // and back
        CHECK(t.p_mem() == 1.f and not t.recalled_partial());                          // no re-survey
    }
    {   // a room left mid-survey is remembered as far as it was seen
        Tracker t;
        t.reset();
        const auto p = t.plan("room_4", Phase::Survey, 40, 0.5f / 0.6f * 0.4f, 0.5f, false);   // p = 0.6
        CHECK(p.has_value() and near(*p, 0.6f));
        t.reset();
        t.enter_recalled("room_4", std::nullopt);
        CHECK(near(t.p_mem(), 0.6f) and t.recalled_partial());
    }
    {   // I-2: a partly recalled room's explorer `finished` is the same terminal verdict as promotion: p = 1
        Tracker t;
        t.reset();
        t.enter_recalled("room_5", 0.6f);
        const auto mid = t.plan("room_5", Phase::RecalledRest, 30, 2.f, 0.5f, false);
        CHECK(mid.has_value() and *mid >= 0.6f and *mid < 0.9f);
        const auto fin = t.plan("room_5", Phase::RecalledRest, 30, 0.4f, 0.5f, true);
        CHECK(fin == std::optional<float>(1.f));                                       // until explored >= 0.9 resolves
        CHECK(t.memory().recall("room_5") == std::optional<float>(1.f));
        CHECK(t.settled("room_5") == 1.f);                                             // the pose planner's branch after it
    }
    {   // nothing assessable: nothing published, nothing remembered
        Tracker t;
        t.reset();
        CHECK(not t.plan("room_6", Phase::Survey, 0, 0.f, 0.5f, false).has_value());
        CHECK(not t.memory().recall("room_6").has_value());
    }
}
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");   // the agent runs under es_ES (CLAUDE.md): nothing here parses, but run as it does
    model_laws();
    fresh_room();
    recalled_rooms();
    glue_fixtures();
    std::println("room_explored_selftest: {}/{} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
