// test_standpoint_oracle_parity.cpp — K3: the controller's standpoint decisions are UNCHANGED by moving its occupancy raster and
// its fit (-log P(the body fits)) into the shared oracle (common/standpoint/standpoint_oracle.h).
//
// HOW. A golden file recorded from the controller AS IT WAS before K3 (BASE aea54896): for a fixed set of synthetic worlds
// (room polygons, passages, rotated object boxes, residual cells hard / soft / live-hard / collapsed, an unusable room mask,
// three cell-size/margin pairs incl. the live config 0.06/0.05), every planner cell centre plus off-centre points is queried for
// every decision the controller makes from the raster: pose_free at the 8 headings, can_turn_here, fit_nats_some_heading and
// fit_nats_turning (all hard / persistent only — the Planning and ApproachWindow stages of decide_standpoint), the decision itself
// (decide_standpoint's action for a Servo and a Reach contract), distance_at, why_cannot_sweep, and one plan() per world. The
// lines are hashed per world (FNV-1a over the exact text, floats by std::to_chars shortest round-trip) and every 97th line is
// stored verbatim for diagnosis. Check mode recomputes and compares; any difference is a changed decision.
// Part 2 (added with the oracle): the SHARED oracle, rasterising the same inputs itself, gives the controller's fit at every query.
//
// Build+run (from controller/, one line):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -Isrc tests/test_standpoint_oracle_parity.cpp src/grid_planner.cpp
//       src/local_policy/cost_to_go.cpp ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp
//       -o /tmp/tsop && /tmp/tsop tests/data/standpoint_oracle_golden.txt          (--record <file> rewrites the golden)
// Registered with ctest in common/standpoint/tests/CMakeLists.txt.

#include "grid_planner.h"
#include "standpoint_fit.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
using V2 = Eigen::Vector2f;
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kPi = 3.14159265358979f;

// deterministic generator (no std distributions: their output is implementation-defined)
struct Lcg
{
    std::uint64_t s;
    std::uint32_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return static_cast<std::uint32_t>(s >> 33); }
    float uni(float a, float b) { return a + (b - a) * static_cast<float>(next() & 0xffffff) / static_cast<float>(0x1000000); }
};

std::string fnum(float v)
{
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    if (std::isnan(v)) return "nan";
    std::array<char, 48> b{};
    const auto r = std::to_chars(b.data(), b.data() + b.size(), v);
    return std::string(b.data(), r.ptr);
}

std::vector<V2> box(float cx, float cy, float w, float h, float yaw)
{
    const float c = std::cos(yaw), s = std::sin(yaw);
    std::vector<V2> p;
    for (const auto& [u, v] : {std::pair{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}, {-0.5f, 0.5f}})
        p.emplace_back(cx + c * u * w - s * v * h, cy + s * u * w + c * v * h);
    return p;
}

struct World
{
    std::string name;
    float cell = 0.06f, margin = 0.05f;
    std::vector<V2> room;
    std::vector<std::vector<V2>> passages, obstacles;
    rc::OccupiedCells cells;
};

// residual cells: clusters on a 0.05 m lattice, each with a risk drawn from the four kinds the controller sees
void add_cells(World& w, Lcg& g, int clusters, bool priced, bool flagged)
{
    w.cells.cell_size_m = 0.05f;
    for (int c = 0; c < clusters; ++c)
    {
        const float cx = g.uni(0.3f, 3.7f), cy = g.uni(0.3f, 3.7f);
        const int n = 3 + static_cast<int>(g.next() % 12);
        for (int k = 0; k < n; ++k)
        {
            const float x = 0.025f + 0.05f * std::round((cx + g.uni(-0.3f, 0.3f)) / 0.05f);
            const float y = 0.025f + 0.05f * std::round((cy + g.uni(-0.3f, 0.3f)) / 0.05f);
            w.cells.centres.emplace_back(x, y);
            const std::uint32_t kind = g.next() % 4;
            const float r = kind == 0 ? kInf : kind == 1 ? 0.f : g.uni(0.2f, 27.f);
            if (priced) w.cells.risk_nats.push_back(r);
            if (flagged) w.cells.live_hard.push_back(kind == 0 and (g.next() % 2) == 0 ? 1 : 0);
        }
    }
}

std::vector<World> worlds()
{
    std::vector<World> ws;
    Lcg g{20261126};
    const std::vector<V2> square{{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    const std::vector<V2> ell{{0, 0}, {4, 0}, {4, 2}, {2.2f, 2}, {2.2f, 4}, {0, 4}};
    for (const auto& [cell, margin] : {std::pair{0.06f, 0.05f}, {0.10f, 0.05f}, {0.05f, 0.0f}})
    {
        const std::string tag = "_c" + fnum(cell) + "_m" + fnum(margin);
        ws.push_back({"empty" + tag, cell, margin, square, {}, {}, {}});
        World a{"objects" + tag, cell, margin, ell, {}, {}, {}};
        a.obstacles.push_back(box(1.0f, 1.0f, 0.6f, 0.4f, 0.3f));
        a.obstacles.push_back(box(3.2f, 1.2f, 0.5f, 0.5f, 0.0f));
        a.obstacles.push_back(box(1.1f, 3.0f, 0.9f, 0.3f, 1.1f));
        ws.push_back(a);
        World r{"residual" + tag, cell, margin, square, {}, {}, {}};
        add_cells(r, g, 9, true, true);
        r.obstacles.push_back(box(2.0f, 2.0f, 0.4f, 0.4f, 0.7f));
        ws.push_back(r);
        World h{"allhard" + tag, cell, margin, square, {}, {}, {}};
        add_cells(h, g, 6, false, false);
        ws.push_back(h);
        World p{"passage" + tag, cell, margin, square, {{{3.6f, 1.6f}, {5.2f, 1.6f}, {5.2f, 2.4f}, {3.6f, 2.4f}}}, {}, {}};
        add_cells(p, g, 4, true, false);
        ws.push_back(p);
    }
    // a degenerate sliver room: almost every cell is outside it, so set_world DISCARDS the mask and plans on obstacles alone
    World u{"unusable_room", 0.06f, 0.05f, {{0, 0}, {4, 0}, {4, 0.02f}}, {}, {}, {}};
    u.obstacles.push_back(box(2.0f, 0.0f, 0.5f, 0.5f, 0.0f));
    ws.push_back(u);
    return ws;
}

std::uint64_t fnv(std::uint64_t h, std::string_view s)
{
    for (const unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

struct Result { std::uint64_t hash = 1469598103934665603ull; long lines = 0; std::vector<std::string> samples; };

Result run_world(const World& w)
{
    rc::GridPlanner p;
    p.params.cell_size_m = w.cell;
    p.params.safety_margin_m = w.margin;
    p.set_passages(w.passages);
    p.set_world(w.room, w.obstacles, w.cells);
    Result res;
    const auto emit = [&](const std::string& line)
    {
        res.hash = fnv(res.hash, line);
        res.hash = fnv(res.hash, "\n");
        if (res.lines % 97 == 0) res.samples.push_back(line);
        ++res.lines;
    };
    {
        std::ostringstream os;
        os.imbue(std::locale::classic());
        p.write_grid(os);
        emit("grid " + std::to_string(fnv(1469598103934665603ull, os.str())) + " w " + std::to_string(p.width()) + " h " +
             std::to_string(p.height()) + " occ " + std::to_string(p.occupied_cells()) + " soft " + std::to_string(p.soft_cells()) +
             " in " + std::to_string(p.residual_soft_in()) + "/" + std::to_string(p.residual_hard_in()) + " hash " +
             std::to_string(p.world_hash()) + " o " + fnum(p.grid_origin().x()) + "," + fnum(p.grid_origin().y()));
    }
    std::vector<V2> qs;
    const V2 o = p.grid_origin();
    for (int iy = 0; iy < p.height(); ++iy)
        for (int ix = 0; ix < p.width(); ++ix)
            qs.emplace_back(o.x() + static_cast<float>(ix) * p.cell_size(), o.y() + static_cast<float>(iy) * p.cell_size());
    Lcg g{static_cast<std::uint64_t>(qs.size()) * 7919ull + 17ull};
    for (int k = 0; k < 400; ++k) qs.emplace_back(g.uni(-0.8f, 5.6f), g.uni(-0.8f, 4.8f));   // off-centre, some off the grid
    for (const V2& q : qs)
    {
        int mask = 0;
        for (int h = 0; h < 8; ++h) mask |= p.pose_free(q, static_cast<float>(h) * kPi / 4.f) ? (1 << h) : 0;
        mask |= p.pose_free(q, 0.37f) ? 256 : 0;   // a yaw between buckets
        const auto servo = rc::standpoint::decide_standpoint(p, q, true, true, true, false, rc::standpoint::Stage::Planning);
        const auto servo_w = rc::standpoint::decide_standpoint(p, q, true, true, true, false, rc::standpoint::Stage::ApproachWindow);
        const auto reach = rc::standpoint::decide_standpoint(p, q, false, true, true, true, rc::standpoint::Stage::ApproachWindow);
        emit(fnum(q.x()) + " " + fnum(q.y()) + " m" + std::to_string(mask) + " t" + (p.can_turn_here(q) ? "1" : "0") +
             " s" + fnum(p.fit_nats_some_heading(q, false)) + " sp" + fnum(p.fit_nats_some_heading(q, true)) +
             " u" + fnum(p.fit_nats_turning(q, false)) + " up" + fnum(p.fit_nats_turning(q, true)) +
             " d" + std::to_string(static_cast<int>(servo.action)) + std::to_string(static_cast<int>(servo_w.action)) +
             std::to_string(static_cast<int>(reach.action)) + " " + fnum(servo.fit.nats) + " " + fnum(reach.fit.nats) +
             " b" + fnum(p.body_risk_at(q)) + " e" + fnum(p.distance_at(q)));
    }
    for (int k = 0; k < 60; ++k)
    {
        const V2 q{g.uni(0.2f, 3.8f), g.uni(0.2f, 3.8f)};
        const float a = g.uni(-kPi, kPi), b = g.uni(-kPi, kPi);
        const auto t = p.why_cannot_sweep(q, a, b);
        const auto r = p.rotation_sweep(q, a, b);
        emit("sweep " + fnum(q.x()) + " " + fnum(q.y()) + " " + std::to_string(t.blocked) + " " + std::to_string(t.headings_blocked) +
             " " + std::to_string(t.cells_blocked) + " " + std::to_string(t.off_map) + " " + fnum(t.nearest_m) + " rot " +
             std::to_string(r.feasible) + " " + fnum(r.min_clearance_m));
    }
    {
        const auto path = p.plan({0.5f, 0.5f}, {3.5f, 3.5f});
        std::string line = "plan " + std::to_string(path.has_value()) + " " + p.last_failure();
        if (path) for (const V2& v : *path) line += " " + fnum(v.x()) + "," + fnum(v.y());
        line += " cost " + fnum(p.last_route_cost_m()) + " risk " + fnum(p.last_route_risk_nats());
        emit(line);
        const auto rot = p.nearest_rotatable({2.0f, 2.0f});
        emit("rotatable " + (rot ? fnum(rot->x()) + "," + fnum(rot->y()) : std::string("none")));
    }
    return res;
}

int record(const char* path)
{
    std::ofstream f(path);
    f.imbue(std::locale::classic());
    for (const auto& w : worlds())
    {
        const Result r = run_world(w);
        f << "world " << w.name << ' ' << r.hash << ' ' << r.lines << '\n';
        for (const auto& s : r.samples) f << "  " << s << '\n';
    }
    std::printf("recorded %s\n", path);
    return 0;
}

int check(const char* path)
{
    std::ifstream f(path);
    if (not f) { std::printf("  [FAIL] cannot open golden %s\n", path); return 1; }
    int failures = 0, nworlds = 0;
    std::string line;
    const auto ws = worlds();
    std::size_t wi = 0;
    Result cur;
    std::size_t si = 0;
    while (std::getline(f, line))
    {
        if (line.starts_with("world "))
        {
            if (wi >= ws.size()) { std::printf("  [FAIL] golden has more worlds than the test\n"); return 1; }
            const World& w = ws[wi++];
            ++nworlds;
            std::istringstream is(line.substr(6));
            is.imbue(std::locale::classic());
            std::string name, hs, ls;
            is >> name >> hs >> ls;
            std::uint64_t h = 0; long n = 0;
            std::from_chars(hs.data(), hs.data() + hs.size(), h);
            std::from_chars(ls.data(), ls.data() + ls.size(), n);
            cur = run_world(w);
            si = 0;
            const bool ok = name == w.name and h == cur.hash and n == cur.lines;
            std::printf("  [%s] %-24s %ld decisions %s\n", ok ? "ok" : "FAIL", w.name.c_str(), cur.lines,
                        ok ? "identical to BASE" : "DIFFER from BASE");
            if (not ok) ++failures;
            continue;
        }
        if (line.size() > 2 and si < cur.samples.size())
        {
            const std::string want = line.substr(2);
            if (want != cur.samples[si])
            {
                std::printf("    first differing sample:\n      BASE %s\n      NOW  %s\n", want.c_str(), cur.samples[si].c_str());
                si = cur.samples.size();   // one per world
            }
            else ++si;
        }
    }
    if (nworlds != static_cast<int>(ws.size())) { std::printf("  [FAIL] golden has %d worlds, test %zu\n", nworlds, ws.size()); ++failures; }
    return failures;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc >= 3 and std::string_view(argv[1]) == "--record") return record(argv[2]);
    const char* path = argc >= 2 ? argv[1] : "tests/data/standpoint_oracle_golden.txt";
    std::printf("K3 part 1: the controller's standpoint decisions vs the golden recorded at BASE\n");
    int failures = check(path);
    std::printf(failures == 0 ? "\nALL PASS (0 failures)\n" : "\nFAILURES: %d\n", failures);
    return failures == 0 ? 0 : 1;
}
