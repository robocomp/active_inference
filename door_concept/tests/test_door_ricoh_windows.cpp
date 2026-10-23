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
    {   // the three states as the helpers see them
        struct Sl { int source; };
        struct Pk { std::vector<Eigen::Matrix<float, 2, 1>> ricoh_windows; std::vector<Sl> slices; };
        Pk a{{{0.0f, 2.0f * pi / 3.0f}}, {}};                  // (a) fresh, windows (even with no detection)
        Pk b{{}, {{1}}};                                       // (b) fresh ricoh slice, no attribute
        Pk c{{}, {{0}}};                                       // (c) only a ZED slice / nothing: no fresh ricoh frame
        Pk d{{}, {}};
        CHECK(ricoh_fresh(a) and ricoh_fresh(b) and not ricoh_fresh(c) and not ricoh_fresh(d));
        const V2 in_w(std::cos(1.0f), std::sin(1.0f)), out_w(std::cos(-2.0f), std::sin(-2.0f));
        CHECK(ricoh_looked_in(a, cam, in_w) and not ricoh_looked_in(a, cam, out_w));
        CHECK(ricoh_looked_in(b, cam, in_w) and ricoh_looked_in(b, cam, out_w));            // legacy: unknown => looked
        CHECK(not ricoh_looked_in(c, cam, in_w) and not ricoh_looked_in(d, cam, out_w));    // nothing fresh => nothing looked
    }
    // ── B2. full circle and three adjacent strips ───────────────────────────────────────────────────────────────
    {
        const std::vector<Eigen::Matrix<float, 2, 1>> full{{0.0f, 2.0f * pi}};
        CHECK(ricoh_looked(full, cam, V2(-1, 0)) and ricoh_looked(full, cam, V2(0, 1)));
        const float t3 = 2.0f * pi / 3.0f;
        const std::vector<Eigen::Matrix<float, 2, 1>> three{{-pi, -pi + t3}, {-pi + t3, -pi + 2 * t3}, {-pi + 2 * t3, pi}};
        for (int k = 0; k < 72; ++k)
        {
            const float a = -pi + 2.0f * pi * (static_cast<float>(k) + 0.5f) / 72.0f;
            CHECK(ricoh_looked(three, cam, V2(std::cos(a), std::sin(a))));
        }
        const std::vector<Eigen::Matrix<float, 2, 1>> one_of_three{{-pi + t3, -pi + 2 * t3}};
        int inside = 0;
        for (int k = 0; k < 72; ++k)
        {
            const float a = -pi + 2.0f * pi * (static_cast<float>(k) + 0.5f) / 72.0f;
            inside += ricoh_looked(one_of_three, cam, V2(std::cos(a), std::sin(a))) ? 1 : 0;
        }
        CHECK(inside == 24);                                                                 // exactly a third of the bearings
    }
    // ── C. bank ────────────────────────────────────────────────────────────────────────────────────────────────
    RicohBirthBank::Params prm;
    prm.hit_rate = 0.8f; prm.clutter = 0.05f; prm.gate_mahalanobis = 9.0f; prm.min_sep_m = 0.7f; prm.tau_s = 5.0f; prm.forget_s = 120.0f;
    const float bar = 6.6f * RicohBirthBank::llr_hit(prm);
    const V2 place(3.0f, 5.0f);
    const std::vector<V2> no_tracks;
    const float dt = 0.1f;
    // seconds until ripe; `hit_every`: a detection every n-th frame (looked frames only when `unlooked_are_looks` is false)
    const auto run = [&](bool moving, int hit_every, bool unlooked_are_looks, float t_max) -> float
    {
        RicohBirthBank b;
        for (int k = 1; k * dt <= t_max; ++k)
        {
            const float t = static_cast<float>(k) * dt;
            const V2 c = moving ? V2(3.0f + 3.0f * std::cos(0.01f * static_cast<float>(k)), 5.0f - 3.0f * std::sin(0.01f * static_cast<float>(k)) - 1.0f) : V2(0.f, 0.f);
            const bool seen = (k % hit_every == 0);
            std::vector<RicohBirthBank::Look> hits;
            if (seen) hits.push_back({place, 0.01f, 0.5f});
            b.update(t, c, hits, prm, [&](const V2&) { return (seen or unlooked_are_looks) ? 0.85f : 0.0f; }, no_tracks);
            if (b.ripe_at(place, 0.01f, prm.gate_mahalanobis, bar)) return t;
        }
        return -1.0f;
    };
    const float t_parked = run(false, 1, true, 600.f), t_moving = run(true, 1, true, 600.f);
    const float t_blind = run(false, 3, true, 1200.f), t_aware = run(false, 3, false, 1200.f);
    std::print("C: time to ripe: parked, every frame a hit {:.1f} s; moving at 0.3 m/s on a 3 m circle {:.1f} s; parked 1-in-3 hits: unlooked frames as misses {:.1f} s, not looked at {:.1f} s\n",
               t_parked, t_moving, t_blind, t_aware);
    CHECK(t_parked > 0.0f and t_moving > 0.0f);
    CHECK(t_parked > 10.0f * dt * 9.0f);                 // NOT ~9 looked hits (1 s): the parked stare is a few effective samples, at tau = 5 s each
    CHECK(t_moving < t_parked);                          // new viewpoints are worth more than a stare
    CHECK(t_aware > 0.0f and (t_blind < 0.0f or t_aware <= t_blind));
    {   // seen once, then looked-at absences from other views: dropped (the prior), never ripe
        RicohBirthBank b;
        b.update(0.0, V2(0, 0), {{place, 0.01f, 0.5f}}, prm, [](const V2&) { return 0.85f; }, no_tracks);
        CHECK(b.entries().size() == 1);
        for (int k = 1; k <= 60; ++k)
            b.update(5.0 * k, V2(std::cos(0.7f * static_cast<float>(k)) * 4.0f, 1.0f + std::sin(0.7f * static_cast<float>(k)) * 2.0f), {}, prm, [](const V2&) { return 0.85f; }, no_tracks);
        CHECK(b.entries().empty());
    }
    {   // seen once, then NOT observed: relaxes toward the prior with forget_s, and is eventually dropped; not before
        RicohBirthBank b;
        b.update(0.0, V2(0, 0), {{place, 0.01f, 0.5f}}, prm, [](const V2&) { return 0.0f; }, no_tracks);
        const float L0 = b.entries().front().L;
        b.update(60.0, V2(0, 0), {}, prm, [](const V2&) { return 0.0f; }, no_tracks);
        CHECK(b.entries().size() == 1 and b.entries().front().L < L0 and b.entries().front().L > 0.4f * L0);   // exp(-60/120) = 0.61
        for (int k = 2; k < 400; ++k) b.update(60.0 * k, V2(0, 0), {}, prm, [](const V2&) { return 0.0f; }, no_tracks);
        CHECK(b.entries().empty());
    }
    {   // two detections of one place in ONE frame are one hit
        RicohBirthBank a, c;
        a.update(0.0, V2(0, 0), {{place, 0.01f, 0.5f}}, prm, [](const V2&) { return 0.85f; }, no_tracks);
        c.update(0.0, V2(0, 0), {{place, 0.01f, 0.5f}, {place + V2(0.05f, 0.f), 0.01f, 0.5f}}, prm, [](const V2&) { return 0.85f; }, no_tracks);
        CHECK(std::abs(a.entries().front().L - c.entries().front().L) < 1e-6f);
    }
    {   // scattered hits (a new place each time) never ripen; the bank stays capped
        RicohBirthBank b;
        for (int k = 0; k < 600; ++k)
        {
            const V2 p(-4.0f + 8.0f * std::fmod(0.618034f * static_cast<float>(k), 1.0f), 5.0f);
            b.update(0.1 * k, V2(0, 0), {{p, 0.01f, 0.5f}}, prm, [](const V2&) { return 0.85f; }, no_tracks);
            for (const auto& e : b.entries()) CHECK(e.L < bar);
            CHECK(b.entries().size() <= prm.max_entries);
        }
    }
    {   // an entry that lands on a live track is dropped
        RicohBirthBank b;
        b.update(0.0, V2(0, 0), {{place, 0.01f, 0.5f}}, prm, [](const V2&) { return 0.85f; }, std::vector<V2>{V2(3.1f, 5.0f)});
        CHECK(b.entries().empty());
    }
    {   // a detector that would not tell a door from clutter (learned h <= pc): nothing ripens
        RicohBirthBank b; auto bad = prm; bad.hit_rate = 0.04f;
        for (int k = 0; k < 100; ++k) b.update(1.0 * k, V2(std::cos(0.3f * k) * 3, std::sin(0.3f * k) * 3), {{place, 0.01f, 0.5f}}, bad, [](const V2&) { return 0.85f; }, no_tracks);
        CHECK(b.entries().empty() and RicohBirthBank::llr_hit(bad) == 0.0f);
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
