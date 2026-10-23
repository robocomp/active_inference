// test_door_ricoh_windows.cpp — Task Q2, the pure parts of the ricoh birth model (src/core/ricoh_door_birth.h):
//   A. a DOORWAY NICHE in the outline: a door whose LiDAR range lands on the wall LINE the niche interrupts is that wall's door, not
//      a disagreement with the niche's back face (the nearest crossing); without niche awareness it scores ~0.
//   B. ricoh_looked(): the processed strips' azimuth windows (incl. wrap at +-pi); EMPTY = unknown = looked at.
//   C. RicohBirthBank: a place seen on a third of the LOOKED frames ripens; the same place with the other frames NOT looked at
//      ripens at least as fast (an unlooked frame is no absence); a place seen once decays on looked absences and is dropped;
//      scattered hits never ripen.
//   D. RicohViewPool::end_frame(looked): a view whose strip was not processed is HELD, not expired.
// Build + run (from door_concept/): g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_ricoh_windows.cpp -o /tmp/t && /tmp/t
#include <cmath>
#include <numbers>
#include <print>
#include <vector>
#include "../src/core/ricoh_door_birth.h"

using namespace rc::door::core;
using V2 = Eigen::Vector2f;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

int main()
{
    const float pi = std::numbers::pi_v<float>;
    // ── A. niche ───────────────────────────────────────────────────────────────────────────────────────────────
    const std::vector<V2> poly{{-5, -5}, {5, -5}, {5, 5}, {3.5f, 5}, {3.5f, 5.3f}, {2.5f, 5.3f}, {2.5f, 5}, {-5, 5}};
    const V2 cam(0.f, 0.f), door_on_line(3.0f, 5.0f);
    const auto plain = ricoh_range(poly, cam, door_on_line, 0.001f, 0.03f, 0.01f);
    const auto niche = ricoh_range(poly, cam, door_on_line, 0.001f, 0.03f, 0.01f, 0.0f, 0.0f, true);
    std::print("A: plain agree {:.3f}, niche-aware agree {:.3f}, fused xy ({:.2f},{:.2f})\n", plain.agree, niche.agree, niche.xy.x(), niche.xy.y());
    CHECK(plain.agree < 0.2f);
    CHECK(niche.agree > 0.9f);
    CHECK(std::abs(niche.xy.y() - 5.0f) < 0.05f);
    // a door on the BACK face is still that wall's door
    const auto back = ricoh_range(poly, cam, V2(3.0f, 5.3f), 0.001f, 0.03f, 0.01f, 0.0f, 0.0f, true);
    CHECK(back.agree > 0.9f);
    // mid-room (no wall): niche awareness does not invent one
    const auto mid = ricoh_range(poly, cam, V2(1.0f, 2.0f), 0.001f, 0.03f, 0.01f, 0.0f, 0.0f, true);
    CHECK(mid.agree < 0.01f);
    // ── B. windows ─────────────────────────────────────────────────────────────────────────────────────────────
    const std::vector<Eigen::Matrix<float, 2, 1>> none;
    CHECK(ricoh_looked(none, cam, V2(1, 1)));
    const std::vector<Eigen::Matrix<float, 2, 1>> w{{0.0f, 2.0f * pi / 3.0f}};          // 0..120 deg
    CHECK(ricoh_looked(w, cam, V2(std::cos(1.0f), std::sin(1.0f))));
    CHECK(not ricoh_looked(w, cam, V2(std::cos(3.0f), std::sin(3.0f))));
    CHECK(not ricoh_looked(w, cam, V2(std::cos(-1.0f), std::sin(-1.0f))));
    const std::vector<Eigen::Matrix<float, 2, 1>> across{{pi - 0.5f, -pi + 0.5f}};       // 1 rad window across the seam
    CHECK(ricoh_looked(across, cam, V2(-1.0f, 0.0f)));
    CHECK(ricoh_looked(across, cam, V2(std::cos(pi - 0.3f), std::sin(pi - 0.3f))));
    CHECK(not ricoh_looked(across, cam, V2(1.0f, 0.0f)));
    // ── C. bank ────────────────────────────────────────────────────────────────────────────────────────────────
    const float h = 0.35f, pc = 0.05f, gate = 9.0f, bar = 6.6f * std::log(h / pc);
    const V2 place(3.0f, 5.0f);
    const std::vector<V2> no_tracks;
    const auto run = [&](bool unlooked_frames_are_looked, int frames) -> int      // frames until ripe, -1 never
    {
        RicohBirthBank b;
        for (int k = 1; k <= frames; ++k)
        {
            const bool seen = (k % 3 == 0);
            std::vector<RicohBirthBank::Look> hits;
            if (seen) hits.push_back({place, 0.01f, 0.8f});
            // q: the detectability of the place; an unlooked frame withholds its absence unless we pretend the strip was processed
            b.update(hits, h, pc, gate, [&](const V2&) { return (seen or unlooked_frames_are_looked) ? 0.85f : 0.0f; }, no_tracks, 0.7f);
            if (b.ripe_at(place, 0.01f, gate, bar)) return k;
        }
        return -1;
    };
    const int blind = run(true, 600), aware = run(false, 600);
    std::print("C: ripe after {} frames counting unlooked frames as misses, {} when they are not looked at\n", blind, aware);
    CHECK(aware > 0 and (blind < 0 or aware <= blind));
    {   // seen once, then looked-at absences: dropped (the prior), never ripe
        RicohBirthBank b;
        b.update({{place, 0.01f, 0.8f}}, h, pc, gate, [](const V2&) { return 0.85f; }, no_tracks, 0.7f);
        CHECK(b.entries().size() == 1);
        for (int k = 0; k < 20; ++k) b.update({}, h, pc, gate, [](const V2&) { return 0.85f; }, no_tracks, 0.7f);
        CHECK(b.entries().empty());
    }
    {   // seen once, then NOT looked at: held
        RicohBirthBank b;
        b.update({{place, 0.01f, 0.8f}}, h, pc, gate, [](const V2&) { return 0.85f; }, no_tracks, 0.7f);
        for (int k = 0; k < 200; ++k) b.update({}, h, pc, gate, [](const V2&) { return 0.0f; }, no_tracks, 0.7f);
        CHECK(b.entries().size() == 1);
    }
    {   // scattered hits (a new place each time) never ripen
        RicohBirthBank b;
        for (int k = 0; k < 300; ++k)
        {
            const V2 p(-4.0f + 8.0f * std::fmod(0.618034f * static_cast<float>(k), 1.0f), 5.0f);
            b.update({{p, 0.01f, 0.8f}}, h, pc, gate, [](const V2&) { return 0.85f; }, no_tracks, 0.7f);
            for (const auto& e : b.entries()) CHECK(e.L < bar);
        }
    }
    {   // an entry that lands on a live track is dropped
        RicohBirthBank b;
        b.update({{place, 0.01f, 0.8f}}, h, pc, gate, [](const V2&) { return 0.85f; }, std::vector<V2>{V2(3.1f, 5.0f)}, 0.7f);
        CHECK(b.entries().empty());
    }
    // ── D. view pool holds an unlooked candidate ─────────────────────────────────────────────────────────────
    {
        RicohViewPool vp;
        const V2 cam2(0, 0), tgt(3, 5);
        CHECK(vp.credit(cam2, tgt, 0.45f, 0.7f) > 0.99f);
        vp.end_frame([](const V2&) { return false; });                         // its strip was not processed: held
        CHECK(vp.peek(cam2, tgt, 0.45f, 0.7f) < 0.01f);                         // still remembered: the same view earns ~nothing
        vp.end_frame([](const V2&) { return true; });                           // processed, not seen (seen flag was cleared above): expired
        CHECK(vp.peek(cam2, tgt, 0.45f, 0.7f) > 0.99f);
    }
    if (fails == 0) std::print("test_door_ricoh_windows: ALL PASS\n"); else std::print("test_door_ricoh_windows: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
