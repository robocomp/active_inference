// p_stand_parity_test.cpp — K3: the countables' p_stand from the SHARED oracle vs Task 8's mirror (task8_mirror.h, which it replaces),
// on the countable's own inputs (room polygon, obstacle boxes, residual cells at residual's resolution priced by a belief field;
// margin 0 — the controller's margin is not readable by an agent). The two are different implementations of the same rule, so the
// difference is stated, not assumed away:
//   (a) DECISION (P > 0): they may disagree only on the lattice — where moving the standpoint by one residual cell flips one of
//       them. The mirror rasterised on residual's lattice with round-to-nearest; the oracle on the controller's (room bbox + pad,
//       floor), with the controller's point-in-polygon obstacle test. Everywhere else the verdict is identical.
//   (b) VALUE: the mirror is 0/1; the oracle is exp(−soft nats) — never above the mirror where both stand, and below it exactly where
//       soft (priced) residual cells lie under the turn.
//   (c) For the record (printed): the oracle at the countable's inputs vs the controller's planner at its live config (0.06 m, 0.05
//       margin) — how much more permissive the agent side is for not knowing the margin.
#include "../standpoint_oracle.h"
#include "task8_mirror.h"
#include "../../../controller/src/grid_planner.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
int failures = 0;
#define CHECK(cond, msg) do { const bool ok_ = (cond); std::printf("  [%s] %s\n", ok_ ? "ok" : "FAIL", msg); if (not ok_) ++failures; } while (0)
using V2 = Eigen::Vector2f;

struct Lcg
{
    std::uint64_t s;
    std::uint32_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return static_cast<std::uint32_t>(s >> 33); }
    float uni(float a, float b) { return a + (b - a) * static_cast<float>(next() & 0xffffff) / static_cast<float>(0x1000000); }
};

struct World
{
    std::vector<V2> room;
    std::vector<rc::nbv::Obstacle> boxes;
    std::vector<float> xyz;
    rc::residual_risk::BeliefField field;
};

// residual's lattice: origin (rx, ry) = its own grid start, centres at origin + (k + ½)·0.05
World make(std::uint64_t seed, bool with_soft)
{
    Lcg g{seed};
    World w;
    w.room = {{0, 0}, {5, 0}, {5, 2.5f}, {3, 2.5f}, {3, 4}, {0, 4}};
    w.boxes.push_back({1.2f, 1.0f, 0.6f, 0.45f, 0.35f});
    w.boxes.push_back({4.1f, 1.2f, 0.5f, 0.5f, 0.0f});
    w.boxes.push_back({1.0f, 3.1f, 0.9f, 0.35f, 1.2f});
    const float rx = -0.1f, ry = -0.1f, c = 0.05f;
    w.field.xmin = rx; w.field.ymin = ry; w.field.cell = c; w.field.width = 104; w.field.height = 66;   // covers y < 3.2 only
    w.field.prob.assign(static_cast<std::size_t>(w.field.width * w.field.height), 0.f);
    w.field.var.assign(w.field.prob.size(), 0.f);
    for (int k = 0; k < 14; ++k)
    {
        const float cx = g.uni(0.3f, 4.7f), cy = g.uni(0.3f, 3.7f);
        const int n = 2 + static_cast<int>(g.next() % 10);
        for (int j = 0; j < n; ++j)
        {
            const int ix = static_cast<int>(std::floor((cx + g.uni(-0.25f, 0.25f) - rx) / c));
            const int iy = static_cast<int>(std::floor((cy + g.uni(-0.25f, 0.25f) - ry) / c));
            w.xyz.insert(w.xyz.end(), {rx + (static_cast<float>(ix) + 0.5f) * c, ry + (static_cast<float>(iy) + 0.5f) * c, 0.3f});
            if (ix < 0 or iy < 0 or ix >= w.field.width or iy >= w.field.height) continue;   // outside the field: hard in both
            const auto i = static_cast<std::size_t>(iy * w.field.width + ix);
            const std::uint32_t kind = g.next() % 3;
            const float P = kind == 0 ? 0.f : with_soft ? g.uni(0.5f, 0.95f) : 0.f;   // collapsed, or a priced belief
            w.field.prob[i] = P;
            w.field.var[i] = P * (1.f - P) / 12.f;
        }
    }
    return w;
}

std::vector<std::vector<V2>> polygons(const std::vector<rc::nbv::Obstacle>& b)
{
    std::vector<std::vector<V2>> p;
    for (const auto& o : b) p.push_back(rc::standpoint::box_polygon(o.cx, o.cy, o.w, o.h, o.yaw));
    return p;
}
}   // namespace

int main()
{
    const rc::RobotFootprint body = rc::RobotFootprint::shadow();
    const float cell = 0.05f;
    long n = 0, agree = 0, disagree = 0, disagree_off_boundary = 0, above_mirror = 0, priced = 0, nonbinary_without_soft = 0;
    long ctrl_n = 0, agent_stands_ctrl_refuses = 0, agent_refuses_ctrl_stands = 0, with_margin_differs = 0;
    for (const bool with_soft : {false, true})
        for (std::uint64_t seed = 1; seed <= 6; ++seed)
        {
            const World w = make(seed * 7919ull + (with_soft ? 1 : 0), with_soft);
            // Task 8: mirror
            const rc::nbv::StandBody sb = rc::nbv::StandBody::of(body, cell);
            const rc::nbv::OccupiedCells oc = rc::nbv::OccupiedCells::from_xyz(w.xyz, cell, w.field);
            const rc::nbv::StandWorld old_world{&sb, &oc, w.room, w.boxes};
            // K3: the shared oracle, as countable_concept/src/epistemic_planner.cpp builds it
            rc::standpoint::Oracle o(body, cell, 0.f);
            const rc::OccupiedCells cells = rc::standpoint::cells_from_xyz(w.xyz, cell, w.field);
            const auto boxes = polygons(w.boxes);
            const std::vector<std::vector<V2>> none;
            o.set_world(w.room, none, boxes, cells);
            rc::standpoint::Oracle om(body, cell, 0.05f);   // (c) what the margin alone accounts for: the agent's lattice, the controller's margin
            om.set_world(w.room, none, boxes, cells);
            // the controller at its live config, on the same cells/boxes (its obstacle source differs live; same here)
            rc::GridPlanner gp;
            gp.params.cell_size_m = 0.06f;
            gp.params.safety_margin_m = 0.05f;
            gp.set_footprint(body);
            gp.set_world(w.room, boxes, cells);

            const auto old_p = [&](const V2& q) { return rc::nbv::p_stand(old_world, q, 0.f, true); };
            const auto new_p = [&](const V2& q) { return o.p_stand_and_turn(q); };
            for (float y = 0.05f; y < 4.0f; y += 0.037f)
                for (float x = 0.05f; x < 5.0f; x += 0.037f)
                {
                    const V2 q{x, y};
                    if (not rc::standpoint::detail::point_in_polygon(w.room, q)) continue;
                    ++n;
                    const double a = old_p(q), b = new_p(q);
                    if ((a > 0.0) == (b > 0.0))
                    {
                        ++agree;
                        if (b > a + 1e-12) ++above_mirror;
                        if (b > 0.0 and b < 1.0) ++priced;
                        if (not with_soft and b > 0.0 and b < 1.0) ++nonbinary_without_soft;
                    }
                    else
                    {
                        ++disagree;
                        bool flips_old = false, flips_new = false;
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx)
                            {
                                const V2 q2 = q + cell * V2(static_cast<float>(dx), static_cast<float>(dy));
                                flips_old = flips_old or ((old_p(q2) > 0.0) != (a > 0.0));
                                flips_new = flips_new or ((new_p(q2) > 0.0) != (b > 0.0));
                            }
                        if (not (flips_old or flips_new)) ++disagree_off_boundary;
                    }
                    ++ctrl_n;
                    const bool ctrl_stands = std::isfinite(gp.fit_nats_turning(q, /*persistent_only=*/true));   // decide_standpoint, Planning
                    if (b > 0.0 and not ctrl_stands) ++agent_stands_ctrl_refuses;
                    if (b == 0.0 and ctrl_stands) ++agent_refuses_ctrl_stands;
                    if ((om.p_stand_and_turn(q) > 0.0) != ctrl_stands) ++with_margin_differs;
                }
        }
    std::printf("p_stand: shared oracle vs Task 8 mirror over %ld standpoints in 12 worlds\n", n);
    std::printf("  decision agrees %ld (%.2f %%), differs %ld (%.2f %%), of which off the lattice boundary %ld\n", agree,
                100.0 * agree / n, disagree, 100.0 * disagree / n, disagree_off_boundary);
    std::printf("  priced (0 < P < 1) where both stand: %ld; above the mirror: %ld\n", priced, above_mirror);
    std::printf("  vs the controller at 0.06 m / 0.05 margin: agent stands where the controller hands back %ld (%.2f %%), "
                "the reverse %ld (%.2f %%)\n", agent_stands_ctrl_refuses, 100.0 * agent_stands_ctrl_refuses / ctrl_n,
                agent_refuses_ctrl_stands, 100.0 * agent_refuses_ctrl_stands / ctrl_n);
    std::printf("  ...with the controller's margin on the agent's lattice the verdicts differ at %ld (%.2f %%): the rest is the margin\n",
                with_margin_differs, 100.0 * with_margin_differs / ctrl_n);
    CHECK(disagree_off_boundary == 0, "(a) every decision difference is a lattice effect: one cell of motion flips one of them");
    CHECK(above_mirror == 0, "(b) the oracle's P never exceeds the mirror's verdict");
    CHECK(nonbinary_without_soft == 0, "(b) with no priced residual cell the oracle is 0/1, like the mirror");
    CHECK(priced > 0, "(b) with priced cells it is continuous below 1 (the mirror reads them as free)");
    CHECK(agent_refuses_ctrl_stands <= agent_stands_ctrl_refuses, "(c) the agent side (no margin) is the more permissive one");
    std::printf(failures == 0 ? "\nALL PASS (0 failures)\n" : "\nFAILURES: %d\n", failures);
    return failures == 0 ? 0 : 1;
}
