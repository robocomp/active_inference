// Task 3c (rtD1d leg 4, 2026-10-21): an occupied OBJECT standpoint is handed back, never re-sited.
// The predicate is standpoint_fit.h: P(the body fits) from the planner's occupancy belief (hard cells, soft residual
// nats); hand back iff P(fits) = 0. Pins:
//   1. a door standpoint with a stray return NEARBY — inside the old 0.62 m disc that re-sited it live — SERVES;
//   2. one genuinely occupied (a live-seen residual blob, an object polygon, a corroborated cell) is HANDED BACK;
//   3. evidence counts: soft cells are priced by their evidence, never a hand-back by themselves (P(fits) = 0 only for
//      hard cells — standpoint_fit.h says why the MAP bar ½ would reduce to "one stray hit");
//   4. a terminal facing turn is judged over every heading, a plain arrival at the cheapest one;
//   5. no world -> Unknown (no verdict before there is a belief);
//   6. the verdict is only ever Serves / HandBack / Unknown — the predicate has no "move it" answer.
// Build+run (from controller/; pure Eigen + std):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -Isrc tests/test_standpoint_fit.cpp src/grid_planner.cpp
//       src/local_policy/cost_to_go.cpp ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp
//       -o /tmp/tsf && /tmp/tsf   (one line)

#include "grid_planner.h"
#include "residual_risk.h"
#include "standpoint_fit.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace
{
int failures = 0;
void check(bool ok, const char *what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (not ok) ++failures;
}

const std::vector<Eigen::Vector2f> kRoom{{0.f, 0.f}, {6.f, 0.f}, {6.f, 6.f}, {0.f, 6.f}};
constexpr float kRes = 0.05f;   // residual's cell
constexpr float kInf = std::numeric_limits<float>::infinity();
// rtD1d: hull circumscribed 0.318 + FootprintSafetyMargin 0.05 + goal_threshold 0.25 — the disc the old repair condemned
// a turning standpoint on, for any single return inside it.
constexpr float kOldDisc = 0.318f + 0.05f + 0.25f;

// Beta moments for residual's construction: Jeffreys (½,½) + κ_obs of evidence at occupancy p (as test_grid_planner_curtain).
float nats(float p, float kobs)
{
    const double a = 0.5 + kobs * p, b = 0.5 + kobs * (1.0 - p), k = a + b;
    return rc::residual_risk::cell_risk_nats(static_cast<float>(a / k), static_cast<float>(a * b / (k * k * (k + 1.0))));
}

rc::GridPlanner planner_with(const std::vector<Eigen::Vector2f> &cells, std::vector<float> risk,
                             const std::vector<std::vector<Eigen::Vector2f>> &objects = {})
{
    rc::GridPlanner p;
    p.params.cell_size_m = 0.10f;
    p.params.safety_margin_m = 0.05f;
    p.set_world(kRoom, objects, rc::OccupiedCells{.centres = cells, .cell_size_m = kRes, .risk_nats = std::move(risk)});
    return p;
}

// A residual blob: every residual cell centre within r of c (on a lattice offset like residual's real one).
std::vector<Eigen::Vector2f> blob(const Eigen::Vector2f &c, float r)
{
    std::vector<Eigen::Vector2f> out;
    for (float x = 0.0215f; x < 6.f; x += kRes)
        for (float y = 0.0215f; y < 6.f; y += kRes)
            if ((Eigen::Vector2f(x, y) - c).norm() <= r) out.emplace_back(x, y);
    return out;
}
const char *name(rc::standpoint::Verdict v)
{
    using V = rc::standpoint::Verdict;
    return v == V::Serves ? "Serves" : v == V::HandBack ? "HandBack" : "Unknown";
}
}  // namespace

int main()
{
    using rc::standpoint::Verdict;
    const Eigen::Vector2f sp{3.f, 3.f};   // the producer's standpoint (a door approach: a Servo, so it TURNS there)

    std::printf("1. a stray return NEAR the standpoint (inside the old %.2f m disc, outside the body)\n", kOldDisc);
    {
        const auto empty = planner_with({}, {});
        check(rc::standpoint::judge(empty, sp, true) == Verdict::Serves, "empty room: a turning standpoint serves");
        // A single return 0.50 m out (rtD1d: the policy's contact set held 0-1 returns there). With no residual cell behind
        // it, the occupancy belief does not contain it at all.
        const Eigen::Vector2f stray = sp + Eigen::Vector2f(0.50f, 0.f);
        check((stray - sp).norm() < kOldDisc, "the stray return is inside the old disc (the old repair would MOVE the standpoint)");
        check(rc::standpoint::judge(empty, sp, true) == Verdict::Serves, "a lone live return with no residual cell: serves");
        // Even if residual latched it — one weakly evidenced cell, and one CONFIRMED cell — it is not under the body.
        for (const float r : {nats(0.6f, 1.5f), nats(0.993f, 40.f)})
        {
            const auto g = planner_with({stray}, {r});
            const auto f = rc::standpoint::body_fit(g, sp, true);
            std::printf("    residual cell at 0.50 m, %.2f nats: -log P(fits) %.3f -> %s\n", r, f.nats,
                        name(rc::standpoint::verdict(f)));
            check(f.nats == 0.f and rc::standpoint::verdict(f) == Verdict::Serves,
                  "a residual cell 0.50 m out is not under the turning body: P(fits) = 1, serves (not moved)");
        }
        // The same cell corroborated by a live return (hard): still outside the body's union of headings.
        const auto hard = planner_with({stray}, {kInf});
        check(rc::standpoint::judge(hard, sp, true) == Verdict::Serves, "a HARD cell 0.50 m out is not under the body: serves");
    }

    std::printf("\n2. a GENUINELY occupied standpoint is handed back\n");
    {
        // A box residual has confirmed under the body, and the LiDAR — inside the recheck window — sees it: every cell a live
        // return lands in is hard (residual_risk.h harden_corroborated), so the belief says the body cannot be there.
        const auto cells = blob(sp + Eigen::Vector2f(0.15f, 0.f), 0.20f);
        const auto seen = planner_with(cells, std::vector<float>(cells.size(), kInf));
        const auto f = rc::standpoint::body_fit(seen, sp, true);
        std::printf("    %zu corroborated cells under the body: -log P(fits) %.1f\n", cells.size(), f.nats);
        check(rc::standpoint::verdict(f) == Verdict::HandBack, "a confirmed, live-seen blob under the body: handed back");
        check(rc::standpoint::judge(seen, sp, false) == Verdict::HandBack, "...at every single heading too");

        const std::vector<std::vector<Eigen::Vector2f>> table{{{2.9f, 2.9f}, {3.3f, 2.9f}, {3.3f, 3.3f}, {2.9f, 3.3f}}};
        const auto obj = planner_with({}, {}, table);
        check(std::isinf(rc::standpoint::body_fit(obj, sp, true).nats), "an object polygon under the body: P(fits) = 0");
        check(rc::standpoint::judge(obj, sp, true) == Verdict::HandBack, "...handed back");

        const auto corroborated = planner_with({sp + Eigen::Vector2f(0.10f, 0.f)}, {kInf});
        check(rc::standpoint::judge(corroborated, sp, true) == Verdict::HandBack,
              "ONE residual cell a live return corroborates, under the body: handed back (two observations, not one hit)");
    }

    std::printf("\n3. the soft belief is PRICED (existence weights), never a hand-back by itself\n");
    {
        const Eigen::Vector2f under = sp + Eigen::Vector2f(0.10f, 0.f);
        const float weak = nats(0.6f, 1.5f), barely = nats(0.55f, 0.2f), confirmed = nats(0.993f, 40.f);
        const auto fu = rc::standpoint::body_fit(planner_with({under}, {barely}), sp, true);
        const auto fw = rc::standpoint::body_fit(planner_with({under}, {weak}), sp, true);
        const auto fc = rc::standpoint::body_fit(planner_with({under}, {confirmed}), sp, true);
        std::printf("    one cell under the body: barely-seen P(fits) %.3f, one/two-hit %.3f, confirmed %.2e\n",
                    fu.p(), fw.p(), fc.p());
        check(fu.p() > fw.p() and fw.p() > fc.p(), "P(fits) falls monotonically with the evidence behind the cell");
        // ★ why the bar is not MAP (standpoint_fit.h): every published cell has Pr(free) < 1/2 — a MAP bar would hand back on
        //   the barely-seen cell, i.e. on one stray hit.
        check(fu.p() < 0.5f, "(the MAP bar would condemn even a barely-seen cell: Pr(free) < 1/2 for every published cell)");
        check(rc::standpoint::verdict(fu) == Verdict::Serves and rc::standpoint::verdict(fw) == Verdict::Serves,
              "a stray latched cell (no live return in it) under the body: serves — one stray hit is weak evidence");
        check(rc::standpoint::verdict(fc) == Verdict::Serves,
              "a confirmed cell the LiDAR does not re-measure this cycle: priced, served (it hardens when it is seen)");
        const auto cells = blob(sp + Eigen::Vector2f(0.15f, 0.f), 0.20f);
        const auto fb = rc::standpoint::body_fit(planner_with(cells, std::vector<float>(cells.size(), confirmed)), sp, true);
        std::printf("    %zu confirmed cells, unseen this cycle: -log P(fits) %.1f (finite: priced)\n", cells.size(), fb.nats);
        check(std::isfinite(fb.nats) and fb.nats > 100.f, "a confirmed blob carries its full price in -log P(fits)");
    }

    std::printf("\n4. a terminal facing turn is judged over EVERY heading\n");
    {
        // A hard cell between the inscribed and circumscribed reach: some heading clears it, the turn does not.
        bool found = false;
        for (float d = 0.20f; d < 0.45f and not found; d += 0.01f)
            for (float a = 0.f; a < 3.1416f and not found; a += 0.19635f)
            {
                const auto g = planner_with({sp + d * Eigen::Vector2f(std::cos(a), std::sin(a))}, {kInf});
                if (rc::standpoint::judge(g, sp, false) == Verdict::Serves and rc::standpoint::judge(g, sp, true) == Verdict::HandBack)
                {
                    std::printf("    hard cell %.2f m out at %.0f deg: one heading fits, the turn does not\n", d, a * 57.29578f);
                    found = true;
                }
            }
        check(found, "an obstacle a single heading clears but a turn sweeps: plain arrival serves, a facing turn is handed back");
        // The union is counted ONCE per cell: a soft cell every heading covers costs its nats once, not eight times.
        const float r = nats(0.9f, 10.f);
        const auto g = planner_with({sp}, {r});
        const auto f = rc::standpoint::body_fit(g, sp, true);
        std::printf("    one %.3f-nat cell at the centre: turning -log P(fits) %.3f\n", r, f.nats);
        check(std::abs(f.nats - r) < 1e-4f, "a cell under every heading's footprint is counted once by a turn");
    }

    std::printf("\n5. no world -> no verdict\n");
    {
        rc::GridPlanner none;
        check(rc::standpoint::judge(none, sp, true) == Verdict::Unknown, "no planner world: Unknown (never a hand-back)");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
