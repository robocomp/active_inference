// Build (from room_concept/):
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/refine_doorway_test.cpp -o /tmp/rdt && /tmp/rdt [refine_input.bin ts_ms door_ax door_ay door_bx door_by]
//
// explore_tour15: 5 s after room_1 was promoted the refiner adopted an 'extend' 0.3 m wide and 2.6 m long through DOOR_0
// into the bathroom — free cells seen through a doorway door_concept believed shut (p_open 0.01). A doorway is where a
// room ends: an edit that grows the outline through a known door's opening is not a candidate.
// Synthetic case always; with arguments, also the live refiner's exact input (tmp/refine_input.bin) at one timestamp.
#include "layout_refiner.h"

#include <charconv>
#include <clocale>
#include <cstring>
#include <fstream>
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

float max_y(const std::vector<V>& P) { float m = -1e9f; for (const auto& v : P) m = std::max(m, v.y()); return m; }

struct Input { std::int64_t ts; std::vector<V> poly, surf, fr; std::vector<float> cs, cf; };
std::vector<Input> read_inputs(const char* path)
{
    std::vector<Input> out;
    std::ifstream f(path, std::ios::binary);
    auto get = [&](auto& v) { f.read(reinterpret_cast<char*>(&v), sizeof(v)); return bool(f); };
    auto pts = [&](std::vector<V>& P, std::vector<float>* c)
    {
        std::uint32_t n = 0; if (not get(n)) return false;
        for (std::uint32_t i = 0; i < n; ++i)
        {
            float x, y, w = 1.f; get(x); get(y); if (c) { get(w); c->push_back(w); }
            P.emplace_back(x, y);
        }
        return bool(f);
    };
    while (true)
    {
        Input in;
        if (not get(in.ts) or not pts(in.poly, nullptr) or not pts(in.surf, &in.cs) or not pts(in.fr, &in.cf)) break;
        out.push_back(std::move(in));
    }
    return out;
}
}

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    {   // synthetic: a 6 x 4 m room, a 1 m doorway in the top wall at x 2..3, and a strip of floor seen through it
        const std::vector<V> room{{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        std::vector<V> surf, fr;
        std::mt19937 rng(7); std::normal_distribution<float> nz(0.f, 0.01f);
        for (float t = 0.025f; t < 6.f; t += 0.05f)
        {
            surf.emplace_back(t, nz(rng));
            if (t < 2.f or t > 3.f) surf.emplace_back(t, 4.f + nz(rng));
        }
        for (float t = 0.025f; t < 4.f; t += 0.05f) { surf.emplace_back(nz(rng), t); surf.emplace_back(6.f + nz(rng), t); }
        for (float x = 0.125f; x < 5.9f; x += 0.05f) for (float y = 0.125f; y < 3.9f; y += 0.05f) fr.emplace_back(x, y);
        for (float x = 2.625f; x < 2.95f; x += 0.05f) for (float y = 4.025f; y < 6.6f; y += 0.05f) fr.emplace_back(x, y);
        rc::refine::LayoutRefiner blind, sighted;
        const auto r0 = blind.propose(room, surf, fr);
        std::println("synthetic, no doorway: {} edits, max y {:.2f}", r0.edits.size(), max_y(r0.polygon));
        CHECK(max_y(r0.polygon) > 4.5f);   // the defect reproduced: the outline grows through the opening
        const auto r1 = sighted.propose(room, surf, fr, {}, {}, {{.a = V(2.f, 4.f), .b = V(3.f, 4.f), .depth = 0.05f}});
        std::println("synthetic, doorway known: {} edits, max y {:.2f}", r1.edits.size(), max_y(r1.polygon));
        CHECK(max_y(r1.polygon) < 4.2f);   // the room ends at its doorway
    }
    if (argc >= 7)
    {   // the live input
        const auto ins = read_inputs(argv[1]);
        const auto num = [](const char* s, auto v) { std::from_chars(s, s + std::strlen(s), v); return v; };   // locale-free
        const std::int64_t ts = num(argv[2], std::int64_t{0});
        const V a(num(argv[3], 0.f), num(argv[4], 0.f)), b(num(argv[5], 0.f), num(argv[6], 0.f));
        for (const auto& in : ins)
            if (in.ts == ts)
            {
                rc::refine::LayoutRefiner blind, sighted;
                const auto r0 = blind.propose(in.poly, in.surf, in.fr, in.cs, in.cf);
                const auto r1 = sighted.propose(in.poly, in.surf, in.fr, in.cs, in.cf, {{.a = a, .b = b, .depth = 0.05f}});
                const auto area = [](const std::vector<V>& P) { return rc::freespace::signed_area(P); };
                std::println("live ts {}: no doorway {} edits {} verts {:.2f} m2 | doorway {} edits {} verts {:.2f} m2 (given {:.2f} m2)",
                             ts, r0.edits.size(), r0.polygon.size(), area(r0.polygon), r1.edits.size(), r1.polygon.size(),
                             area(r1.polygon), area(in.poly));
                for (const auto& e : r0.edits) std::println("  blind   {} edge {} value {:.2f} gain {:.1f}", e.kind, e.edge, e.value, e.gain);
                for (const auto& e : r1.edits) std::println("  sighted {} edge {} value {:.2f} gain {:.1f}", e.kind, e.edge, e.value, e.gain);
            }
    }
    std::println("refine_doorway_test: {}/{} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
