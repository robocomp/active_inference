// Task 3b (2026-10-21 doorway release, rtD1c leg 5): the door leaf as a KNOWN obstacle (src/local_policy/door_leaf.h).
// Build+run (from controller/; one line):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 tests/test_door_leaf.cpp src/local_policy/cost_to_go.cpp
//     -o /tmp/tdl && /tmp/tdl
//
// The scene is DOOR_1 as test_policy_scenarios.cpp door_world(1) has it (door frame: the opening's centre x = 0, room_1 at
// y < 0, the wall's room_1 face y = -0.20 at the casings): the leaf hangs from the WEST jamb (hinge (-0.5, -0.2)) and is swung
// 90 deg into room_1 along -y, free edge at y = -1.15. West of it, between the leaf and room_1's west wall (x = -1.11), is the
// hinge-side POCKET the rtD1c leg-5 robot stood in. residual_concept sees the leaf as cells smeared over its swing (the leaf
// moved while it was mapped, +-2 sigma_phi about the hinge), and a live return corroborates them: HARD (residual_risk.h).
#include "../src/local_policy/cost_to_go.h"
#include "../src/local_policy/door_leaf.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <numbers>
#include <vector>

using namespace rc::policy;
using V = Eigen::Vector2f;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)

namespace
{
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kRins = 0.230f;      // Shadow's bare inscribed radius (test_policy_scenarios prints it)
constexpr float kCell = 0.05f;       // residual's cell

struct Box { float x0, y0, x1, y1; };
// room_1's stub before DOOR_1 + room_2 beyond it (walls only — the leaf is NOT a map polygon: residual is what saw it)
const std::vector<Box> kWalls = {
    {-3.06f, -0.20f, -0.45f, 0.f}, {0.45f, -0.20f, 3.06f, 0.f},     // the wall with the opening x in [-0.45, 0.45]
    {-1.22f, -2.40f, -1.11f, -0.20f},                                // room_1 west wall
    {0.57f, -2.40f, 0.68f, -0.20f},                                  // room_1 east wall
    {-1.22f, -2.51f, 0.68f, -2.40f},                                 // room_1 south wall
    {-3.17f, 0.f, -3.06f, 4.11f}, {3.06f, 0.f, 3.17f, 4.11f}, {-3.2f, 4.11f, 3.2f, 4.22f}};   // room_2
float box_dist(const Box& b, const V& p)
{
    const float dx = std::max({b.x0 - p.x(), 0.f, p.x() - b.x1}), dy = std::max({b.y0 - p.y(), 0.f, p.y() - b.y1});
    return std::hypot(dx, dy);
}

LeafHypothesis map_leaf(float sigma_phi, float p = 1.f)
{
    LeafHypothesis h;
    h.hinge = V(-0.5f, -0.2f);
    h.dir = V(0.f, -1.f);
    h.length = 0.95f;
    h.p = p;
    h.pivot = V(0.f, -0.1f);
    h.P_xy = Eigen::Matrix2f::Identity() * (0.02f * 0.02f);
    h.var_psi = 0.01f * 0.01f;
    h.var_phi = sigma_phi * sigma_phi;
    return h;
}
// residual's view of the leaf: cells on the centre line swept over phi +- 2 sigma_phi (the leaf moved while it was mapped)
std::vector<V> leaf_cells(const LeafHypothesis& h, float sigma_phi)
{
    std::vector<V> c;
    for (float a = -2.f * sigma_phi; a <= 2.f * sigma_phi + 1e-6f; a += 0.02f)
    {
        const V d(h.dir.x() * std::cos(a) - h.dir.y() * std::sin(a), h.dir.x() * std::sin(a) + h.dir.y() * std::cos(a));
        for (float r = 0.f; r <= h.length + 1e-6f; r += kCell)
        {
            const V q = h.hinge + r * d;
            const V k(std::round(q.x() / kCell) * kCell, std::round(q.y() / kCell) * kCell);   // residual's lattice
            bool dup = false;
            for (const auto& e : c) dup = dup or (e - k).squaredNorm() < 1e-8f;
            if (not dup) c.push_back(k);
        }
    }
    return c;
}

// the planner's two answers, emulated over the walls + residual cells: free = the inscribed disc clears every wall and
// every HARD cell (GridPlanner's distance field, field_free_from_distance's half-cell); body risk = the finite nats of the
// cells under the inscribed disc (body_risk_at, heading-free)
struct World
{
    std::vector<V> cells; std::vector<float> risk;
    bool free(const V& p) const
    {
        for (const auto& b : kWalls) if (box_dist(b, p) <= kRins) return false;
        for (std::size_t i = 0; i < cells.size(); ++i)
            if (not std::isfinite(risk[i]) and (cells[i] - p).norm() - 0.5f * kCell <= kRins) return false;
        return true;
    }
    float body_risk(const V& p) const
    {
        float s = 0.f;
        for (std::size_t i = 0; i < cells.size(); ++i)
            if (std::isfinite(risk[i]) and (cells[i] - p).norm() <= kRins) s += risk[i];
        return s;
    }
};

// the steepest-descent route over the field from `s` (8-neighbour, one cell a step): the first point where it crosses the
// leaf's line x = -0.5 (or nothing)
std::optional<V> crossing_of_leaf_line(const CostToGo& c, V s)
{
    for (int k = 0; k < 400; ++k)
    {
        V best = s; float fb = c.at(s);
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
            {
                const V q = s + kCell * V(float(dx), float(dy));
                if (c.at(q) < fb) { fb = c.at(q); best = q; }
            }
        if (best == s) return std::nullopt;
        if ((s.x() + 0.5f) * (best.x() + 0.5f) <= 0.f) return 0.5f * (s + best);
        s = best;
    }
    return std::nullopt;
}
}  // namespace

int main()
{
    const CostToGo::Grid G{V(-1.4f, -2.6f), kCell, 72, 140};   // x [-1.4, 2.15], y [-2.6, 4.35]
    std::vector<V> goal_line;
    for (float x = -0.2f; x <= 0.2f + 1e-4f; x += kCell) goal_line.push_back(V(x, 0.8f));   // past the opening
    const V pocket(-0.84f, -0.45f);    // the hinge-side pocket: 0.6 m between the leaf and the west wall, near the hinge
    const float sphi = 0.10f;           // sigma_phi (5.7 deg): the swing residual smeared the leaf over
    DoorLeaf L; L.hyps = {map_leaf(sphi)};

    // ---- 1. explanation of single cells (per hypothesis)
    {
        const LeafHypothesis& h = L.hyps.front();
        const float w_on = leaf_explained(h, V(-0.5f, -0.6f), kCell);
        const float w_far = leaf_explained(h, V(0.3f, -1.8f), kCell);
        CHECK("3b: a cell ON the leaf is explained (w > 0.9)", w_on > 0.9f);
        CHECK("3b: a cell 1 m from the leaf is not (w < 1e-6)", w_far < 1e-6f);
        CHECK("3b: a HARD cell on the leaf, under that hypothesis, is a small finite toll (< 0.1 nats)", explained_risk_nats(kInf, w_on) < 0.1f);
        CHECK("3b: a hard cell 1 m away stays (practically) hard (> 13 nats)", explained_risk_nats(kInf, w_far) > 13.f);
        CHECK("3b: an unexplained soft cell keeps its price exactly", explained_risk_nats(3.f, 0.f) == 3.f);
        CHECK("3b: explanation never ADDS risk", explained_risk_nats(3.f, 0.4f) <= 3.f and explained_risk_nats(0.f, 0.4f) == 0.f);
    }

    // ---- 2. the field from the hinge side: hard leaf cells disconnect it; taken by the door's leaf, it is finite and goes
    //         AROUND the free edge
    const auto cells = leaf_cells(L.hyps.front(), sphi);
    World raw{cells, std::vector<float>(cells.size(), kInf)};   // every leaf cell corroborated by a live return: hard
    std::vector<std::size_t> keep;
    const DoorLeafEvidence ev = take_explained_cells(L, cells, kCell, raw.risk, kRins, keep);
    World planner;   // what the planner keeps
    for (const auto i : keep) { planner.cells.push_back(cells[i]); planner.risk.push_back(raw.risk[i]); }
    std::printf("leaf cells %zu: %zu taken by the door's leaf (explained mass %.1f), %zu kept by the planner\n", cells.size(),
                ev.cells.size(), ev.mass, keep.size());
    CHECK("3b: every leaf cell is taken from the planner's unexplained residual (none stays a hard cell there)", keep.empty());
    CostToGo f_raw, f_leaf;
    f_raw.build(G, [&](const V& p) { return raw.free(p); }, [&](const V& p) { return raw.body_risk(p); }, goal_line);
    f_leaf.build(G, [&](const V& p) { return planner.free(p); },
                 [&](const V& p) { return planner.body_risk(p) + door_leaf_density(ev, p, kRins); }, goal_line);
    std::printf("   field at the pocket: unexplained %.2f, the door's leaf %.2f\n", f_raw.at(pocket), f_leaf.at(pocket));
    CHECK("3b: (the baseline) the pocket is FREE on the unexplained map (the body's centre fits there)", raw.free(pocket));
    CHECK("3b: (the baseline) ...and the leaf as hard unexplained cells disconnects it: no route (rtD1c leg 5)", not f_raw.reachable(pocket));
    CHECK("3b: with the door's leaf, the pocket is reachable: the field is FINITE", f_leaf.reachable(pocket) and std::isfinite(f_leaf.at(pocket)));
    {
        const auto c_leaf = crossing_of_leaf_line(f_leaf, pocket);
        if (c_leaf) std::printf("   descent crosses x = -0.5 at y %+.2f\n", c_leaf->y());
        CHECK("3b: the route leaves the pocket AROUND the free edge (crosses the leaf's line beyond the tip, y < -1.15)",
              c_leaf.has_value() and c_leaf->y() < -1.15f);
        // discriminating: the leaf UNSEEN (the LiDAR self-filter hides it within ~0.4 m of the body — anchor_leaf.h), no
        // residual cells at all: without the door's leaf the field cuts straight through it; with it, around the free edge
        const World empty{{}, {}};
        CostToGo f_none, f_known;
        f_none.build(G, [&](const V& p) { return empty.free(p); }, [](const V&) { return 0.f; }, goal_line);
        f_known.build(G, [&](const V& p) { return empty.free(p); }, [&](const V& p) { return leaf_contact_nats(L, p, kRins); }, goal_line);
        const auto c_none = crossing_of_leaf_line(f_none, pocket), c_known = crossing_of_leaf_line(f_known, pocket);
        if (c_none) std::printf("   unseen leaf: descent crosses x = -0.5 at y %+.2f without the door's leaf\n", c_none->y());
        if (c_known) std::printf("   unseen leaf: descent crosses x = -0.5 at y %+.2f with it\n", c_known->y());
        CHECK("3b: (discriminating) an unseen leaf, not modelled: the field cuts THROUGH it (-1.15 < y < -0.2)",
              c_none.has_value() and c_none->y() > -1.15f and c_none->y() < -0.2f);
        CHECK("3b: an unseen leaf the door models: the field goes AROUND its free edge (y < -1.15)",
              c_known.has_value() and c_known->y() < -1.15f);
    }

    // ---- 2b. THE COUPLING (the harness's leaf_known_hinge_side finding): two hypotheses, the west hinge P 0.9 and the east
    //          0.1; residual's hard cells lie on the WEST leaf. Under the east hypothesis those cells are unexplained, so the
    //          west leaf stays (all but) impassable; priced apart, "10 % it hangs from the other jamb" made it cost 2.3 nats/m.
    {
        DoorLeaf two; two.hyps = {map_leaf(0.05f, 0.9f), map_leaf(0.05f, 0.1f)};
        two.hyps[1].hinge = V(0.5f, -0.2f);
        std::vector<V> west;
        for (float y = -0.2f; y >= -1.15f - 1e-4f; y -= kCell) west.push_back(V(-0.5f, std::round(y / kCell) * kCell));
        std::vector<std::size_t> kp;
        const std::vector<float> hard(west.size(), kInf);
        const DoorLeafEvidence e2 = take_explained_cells(two, west, kCell, hard, kRins, kp);
        const float on_west = door_leaf_density(e2, V(-0.5f, -0.6f), kRins);
        const float leaf_only = leaf_contact_nats(two, V(-0.5f, -0.6f), kRins);
        const float near_east = door_leaf_density(e2, V(0.30f, -0.7f), kRins);
        std::printf("   coupling: on the seen west leaf %.1f nats (leaf mixture alone %.2f); 0.20 m from the unseen east hypothesis %.3f nats\n",
                    on_west, leaf_only, near_east);
        CHECK("3b coupling: (baseline) the leaf mixture alone prices the seen leaf at ~-log(0.1) (crossable)", leaf_only < 2.4f);
        CHECK("3b coupling: with the cells it explains, the seen leaf is (all but) impassable (> 20 nats)", on_west > 20.f);
        CHECK("3b coupling: where only the 0.1 hypothesis puts a leaf and nothing was seen, ~-log(0.9) at most",
              near_east <= -std::log(0.9f) + 0.01f);
        const std::vector<float> none_soft(west.size(), 0.f);
        const DoorLeafEvidence e3 = take_explained_cells(two, west, kCell, none_soft, kRins, kp);
        CHECK("3b coupling: cells residual believes FREE (0 nats) add nothing: the leaf mixture alone",
              std::abs(door_leaf_density(e3, V(-0.5f, -0.6f), kRins) - leaf_only) < 1e-4f);
    }

    // ---- 3. sigma_phi widens the swept region the leaf is priced over
    {
        DoorLeaf sharp; sharp.hyps = {map_leaf(0.02f)};
        DoorLeaf wide;  wide.hyps = {map_leaf(0.30f)};
        // the body's centre 0.40 m east of the MAP free edge's line, level with the tip: where the leaf would be at phi + 25 deg
        const V p(-0.5f + 0.40f, -1.05f);
        const float ns = leaf_contact_nats(sharp, p, kRins), nw = leaf_contact_nats(wide, p, kRins);
        std::printf("   at (%.2f, %.2f): sigma_phi 0.02 -> %.4f nats, sigma_phi 0.30 -> %.3f nats\n", p.x(), p.y(), ns, nw);
        CHECK("3b: a sharp leaf leaves a body 0.40 m off its free edge ~free (< 0.01 nats)", ns < 0.01f);
        CHECK("3b: a leaf with a large sigma_phi prices the same place by its swept uncertainty (> 0.3 nats)", nw > 0.3f);
        // and the swept region is a property of the free edge, not of the hinge: near the hinge sigma_phi barely matters
        const V q(-0.5f + 0.33f, -0.25f);
        CHECK("3b: near the hinge sigma_phi barely matters (r sigma_phi -> 0)",
              std::abs(leaf_contact_nats(wide, q, kRins) - leaf_contact_nats(sharp, q, kRins)) < 0.05f);
        CHECK("3b: the body on the leaf is all but certain contact (> 20 nats)", leaf_contact_nats(sharp, V(-0.5f, -0.6f), kRins) > 20.f);
        DoorLeaf none;
        std::vector<std::size_t> kp;
        const std::vector<V> one{V(-0.5f, -0.6f)};
        const std::vector<float> hard{kInf};
        const DoorLeafEvidence en = take_explained_cells(none, one, kCell, hard, kRins, kp);
        CHECK("3b: no leaf known -> no price, and the planner keeps every cell", leaf_contact_nats(none, V(-0.5f, -0.6f), kRins) == 0.f
              and en.cells.empty() and kp.size() == 1);
    }

    std::printf("%s (%d failure%s)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails, fails == 1 ? "" : "s");
    return fails == 0 ? 0 : 1;
}
