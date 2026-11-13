// Build (from room_concept/):
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/refine_crossing_test.cpp -o /tmp/rct && /tmp/rct
//
// ONE OBSERVATION PER CROSSING (clean33, 2026-11-10). A 2 cm slit at a shut door's jamb let beams into the next room;
// scored once per CELL, ~1350 swept cells outweighed 20 vertices and room_1 annexed the next room. The free cells
// behind one crossing of the outline repeat one fact ("the wall is open here"), so they count once per crossing.
// The synthetic world is ray-cast into the real FreeSpacePolygon and scored by the real LayoutRefiner:
//   slit  room A [0,6]x[0,4], a 2 cm slit in its east wall, room B beyond: A must beat A + a strip through the slit,
//         and propose() from A must not grow through it. Without origins (the old per-cell count) the strip wins —
//         the test reproduces the bug before it shows the cure.
//   wing  an L room, A plus an open wing [6,8]x[2.5,4], seen only from inside A: the L must still beat A, and
//         propose() from A must take the wing in. Real extensions are crossed along their whole junction.
#include "layout_refiner.h"

#include <print>
#include <random>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, const char* what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)
using V = Eigen::Vector2f;
struct Seg { V a, b; };

std::vector<Seg> ring(const std::vector<V>& P)
{
    std::vector<Seg> s;
    for (std::size_t k = 0; k < P.size(); ++k) s.push_back({P[k], P[(k + 1) % P.size()]});
    return s;
}

// nearest hit of the ray o + t u (t > 0) on the walls, or nullopt within `range`
std::optional<V> cast(const V& o, const V& u, const std::vector<Seg>& walls, float range)
{
    float best = range; bool hit = false;
    for (const auto& w : walls)
    {
        const V e = w.b - w.a; const float den = u.x() * e.y() - u.y() * e.x();
        if (std::abs(den) < 1e-9f) continue;
        const V d = w.a - o;
        const float t = (d.x() * e.y() - d.y() * e.x()) / den, s = (d.x() * u.y() - d.y() * u.x()) / den;
        if (t > 1e-4f and t < best and s >= 0.f and s <= 1.f) { best = t; hit = true; }
    }
    return hit ? std::optional<V>(o + u * best) : std::nullopt;
}

struct Evidence { std::vector<V> surf, fr, fo; std::vector<float> cs, cf; };

Evidence survey(const std::vector<Seg>& walls, const std::vector<V>& stands)
{
    rc::freespace::FreeSpacePolygon fs;
    std::mt19937 rng(7); std::normal_distribution<float> noise(0.f, 0.01f);
    for (const auto& o : stands)
    {
        std::vector<V> ends;
        for (int b = 0; b < 1440; ++b)
        {
            const float a = 2.f * float(M_PI) * float(b) / 1440.f;
            const V u(std::cos(a), std::sin(a));
            if (const auto h = cast(o, u, walls, 12.f)) ends.push_back(*h + u * noise(rng));
        }
        fs.add_scan(o, ends);
        fs.add_footprint(o, 0.25f);
    }
    Evidence E;
    fs.evidence_cells(E.surf, E.fr, &E.cs, &E.cf, &E.fo);
    return E;
}

bool covers(const std::vector<V>& P, const V& q) { return rc::freespace::point_in_polygon(q, P); }

void slit()
{
    // room A [0,6]x[0,4]; the east wall x=6 has a slit y in [2.00, 2.02]; room B [6,10]x[0,4] beyond
    std::vector<Seg> walls = {{{0, 0}, {6, 0}}, {{6, 0}, {10, 0}}, {{10, 0}, {10, 4}}, {{10, 4}, {6, 4}}, {{6, 4}, {0, 4}},
                              {{0, 4}, {0, 0}}, {{6, 0}, {6, 2.00f}}, {{6, 2.02f}, {6, 4}}};
    std::vector<V> stands;
    for (float x : {1.0f, 2.5f, 4.0f}) for (float y : {1.0f, 3.0f}) stands.emplace_back(x, y);   // the survey
    for (int k = 0; k < 12; ++k) stands.emplace_back(5.0f + 0.03f * float(k % 4), 1.90f + 0.02f * float(k));   // the dwell at the slit
    const auto E = survey(walls, stands);
    const std::vector<V> A = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
    const std::vector<V> strip = {{0, 0}, {6, 0}, {6, 1.5f}, {10, 1.5f}, {10, 2.5f}, {6, 2.5f}, {6, 4}, {0, 4}};
    long behind = std::ranges::count_if(E.fr, [](const V& q) { return q.x() > 6.1f; });
    std::println("slit: {} free cells, {} of them behind the wall", E.fr.size(), behind);
    CHECK(behind > 100);   // the slit really leaks (else the case tests nothing)

    rc::refine::LayoutRefiner R;
    const auto dl_old = R.description_lengths({A, strip}, A, E.surf, E.fr, E.cs, E.cf);
    const auto dl_new = R.description_lengths({A, strip}, A, E.surf, E.fr, E.cs, E.cf, E.fo);
    std::println("slit  per cell:     A {:.1f}  A+strip {:.1f}", dl_old[0], dl_old[1]);
    std::println("slit  per crossing: A {:.1f}  A+strip {:.1f}", dl_new[0], dl_new[1]);
    CHECK(dl_old[1] < dl_old[0]);   // the bug, reproduced: per cell the strip through the slit pays its 4 vertices
    CHECK(dl_new[0] < dl_new[1]);   // the cure: the room stays A
    const auto P = R.propose(A, E.surf, E.fr, E.cs, E.cf, {}, E.fo);
    std::println("slit  propose: {} verts, {} edits", P.polygon.size(), P.edits.size());
    CHECK(not covers(P.polygon, V(7.5f, 2.01f)));
    CHECK(not covers(P.polygon, V(6.5f, 2.01f)));
}

void wing()
{
    // L room: A [0,6]x[0,4] plus an open wing [6,8]x[2.5,4]; seen only from inside A
    const std::vector<V> L = {{0, 0}, {6, 0}, {6, 2.5f}, {8, 2.5f}, {8, 4}, {0, 4}};
    const auto walls = ring(L);
    std::vector<V> stands;
    for (float x : {1.0f, 2.5f, 4.0f, 5.3f}) for (float y : {1.0f, 3.0f}) stands.emplace_back(x, y);
    const auto E = survey(walls, stands);
    const std::vector<V> A = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
    rc::refine::LayoutRefiner R;
    const auto dl_new = R.description_lengths({A, L}, A, E.surf, E.fr, E.cs, E.cf, E.fo);
    std::println("wing  per crossing: A {:.1f}  L {:.1f}", dl_new[0], dl_new[1]);
    CHECK(dl_new[1] < dl_new[0]);   // a real wing is still taken in
    const auto P = R.propose(A, E.surf, E.fr, E.cs, E.cf, {}, E.fo);
    std::println("wing  propose: {} verts, {} edits", P.polygon.size(), P.edits.size());
    CHECK(covers(P.polygon, V(7.0f, 3.25f)));
    CHECK(not covers(P.polygon, V(7.0f, 1.5f)));
}
}  // namespace

int main()
{
    slit();
    wing();
    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
