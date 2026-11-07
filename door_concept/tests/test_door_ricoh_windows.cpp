// test_door_ricoh_windows.cpp — Task Q2, the pure parts of the ricoh birth model (src/core/ricoh_door_birth.h):
//   A. a DOORWAY NICHE in the outline: a door whose LiDAR range lands on the wall LINE the niche interrupts is that wall's door, not
//      a disagreement with the niche's back face (the nearest crossing); without niche awareness it scores ~0.
//   B. ricoh_looked(): the processed strips' azimuth windows (incl. wrap at +-pi); EMPTY = unknown = looked at.
//   (C. the ricoh birth BANK and D. the ricoh VIEW POOL are retired with the existence patches: a ricoh frame is a Bernoulli trial
//   in the view ledger - tests/test_view_ledger.cpp, test_door_visibility.cpp `ricoh_unknown_windows_no_miss`.)
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
    if (fails == 0) std::print("test_door_ricoh_windows: ALL PASS\n"); else std::print("test_door_ricoh_windows: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
