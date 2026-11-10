// Build (from room_concept/):
//   g++ -std=c++23 -O2 -w -Isrc -I/usr/include/eigen3 tools/doorway_necks_test.cpp -o /tmp/dnt && /tmp/dnt [refine_input.bin ...]
// "The room is bigger" against "there is a door here" (spec 2026-11-06-unrecognised-doorway-hypothesis-design.md).
// With arguments, it also prints what the comparison does to the refiner's recorded inputs, last frame of each room.
#include "doorway_necks.h"
#include <clocale>
#include <chrono>
#include <fstream>
#include <print>
#include <random>
namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, const char* what) { ++checks; if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); } }
#define CHECK(x) check((x), __LINE__, #x)
using V = Eigen::Vector2f; using Poly = std::vector<V>;
struct Scene { Poly surf, fr; };
// Surface cells along every edge of `walls`, except where `gap` says; free cells on a raster inside `poly`.
Scene scene(const Poly& poly, auto gap)
{
    Scene s; std::mt19937 rng(7); std::normal_distribution<float> nz(0.f, 0.01f);
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
    {
        const V e = poly[i] - poly[j]; const float L = e.norm();
        for (float t = 0.025f; t < L; t += 0.05f) { const V q = poly[j] + e * (t / L); if (not gap(q)) s.surf.emplace_back(q.x() + nz(rng), q.y() + nz(rng)); }
    }
    for (float x = -1.f; x < 8.f; x += 0.05f) for (float y = -1.f; y < 8.f; y += 0.05f)
        if (const V q(x, y); rc::freespace::point_in_polygon(q, poly) and rc::refine::detail::dist_to_boundary(poly, q) > 0.1f) s.fr.push_back(q);
    return s;
}
struct Input { std::int64_t ts; Poly poly, surf, fr; std::vector<float> cs, cf; };
std::vector<Input> read_inputs(const char* path)
{
    std::vector<Input> out; std::ifstream f(path, std::ios::binary);
    auto get = [&](auto& v) { f.read(reinterpret_cast<char*>(&v), sizeof(v)); return bool(f); };
    auto pts = [&](Poly& P, std::vector<float>* c) { std::uint32_t n = 0; if (not get(n)) return false;
        for (std::uint32_t i = 0; i < n; ++i) { float x, y, w = 1.f; get(x); get(y); if (c) { get(w); c->push_back(w); } P.emplace_back(x, y); } return bool(f); };
    while (true) { Input in; if (not get(in.ts) or not pts(in.poly, nullptr) or not pts(in.surf, &in.cs) or not pts(in.fr, &in.cf)) break; out.push_back(std::move(in)); }
    return out;
}
float area(const Poly& P) { return std::abs(rc::freespace::signed_area(P)); }
}
int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    rc::refine::LayoutRefiner refiner;
    {   // a strip of the next room seen through a 1 m gap in a measured wall: a door, not the room
        const Poly P{{0, 0}, {6, 0}, {6, 4}, {3, 4}, {3, 6.5}, {2, 6.5}, {2, 4}, {0, 4}};
        const auto s = scene(P, [](const V& q) { return std::abs(q.y() - 4.f) < 0.03f and q.x() > 2.f and q.x() < 3.f; });
        const auto r = rc::refine::explain_openings(refiner, P, s.surf, s.fr, {}, {});
        std::println("strip: {} verts {:.2f} m2, {} opening(s)", r.polygon.size(), area(r.polygon), r.openings.size());
        CHECK(std::abs(area(r.polygon) - 24.f) < 0.1f);
        CHECK(r.openings.size() == 1);
        if (r.openings.size() == 1) { const V m = 0.5f * (r.openings[0].a + r.openings[0].b); CHECK((m - V(2.5f, 4.f)).norm() < 0.05f); CHECK(std::abs((r.openings[0].a - r.openings[0].b).norm() - 1.f) < 0.05f); }
    }
    {   // a real bay, 2.5 m wide at its mouth: the room
        const Poly P{{0, 0}, {6, 0}, {6, 4}, {3.5, 4}, {3.5, 6}, {1, 6}, {1, 4}, {0, 4}};
        const auto s = scene(P, [](const V& q) { return std::abs(q.y() - 4.f) < 0.03f and q.x() > 1.f and q.x() < 3.5f; });
        const auto r = rc::refine::explain_openings(refiner, P, s.surf, s.fr, {}, {});
        CHECK(r.openings.empty());
        CHECK(std::abs(area(r.polygon) - area(P)) < 1e-3f);
    }
    {   // nothing to cut
        const Poly P{{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        const auto s = scene(P, [](const V&) { return false; });
        const auto r = rc::refine::explain_openings(refiner, P, s.surf, s.fr, {}, {});
        CHECK(r.openings.empty() and r.polygon.size() == 4);
    }
    {   // a kink of the wall, door-wide and 8 cm deep: no floor beyond it, so no door
        const Poly P{{0, 0}, {6, 0}, {6, 4}, {3, 4}, {3, 4.08f}, {2, 4.08f}, {2, 4}, {0, 4}};
        const auto s = scene(P, [](const V&) { return false; });
        const auto r = rc::refine::explain_openings(refiner, P, s.surf, s.fr, {}, {});
        CHECK(r.openings.empty());
    }
    for (int a = 1; a < argc; ++a)
    {
        const auto ins = read_inputs(argv[a]);
        std::println("== {}", argv[a]);
        for (std::size_t i = 0; i < ins.size(); ++i)
            if (i + 1 == ins.size() or ins[i + 1].ts - ins[i].ts > 12000)
            {
                const auto t0 = std::chrono::steady_clock::now();
                const auto r = rc::refine::explain_openings(refiner, ins[i].poly, ins[i].surf, ins[i].fr, ins[i].cs, ins[i].cf);
                const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                std::println(" verts {} area {:.1f} -> verts {} area {:.1f}, {:.0f} ms", ins[i].poly.size(), area(ins[i].poly), r.polygon.size(), area(r.polygon), ms);
                for (const auto& o : r.openings) std::println("   opening ({:+.2f},{:+.2f})-({:+.2f},{:+.2f}) width {:.2f} m, cut {:.1f} m2, score {:+.1f}", o.a.x(), o.a.y(), o.b.x(), o.b.y(), (o.a - o.b).norm(), o.cut_area, o.score);
            }
    }
    std::println("doorway_necks_test: {}/{} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
