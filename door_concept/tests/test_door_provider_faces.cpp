// Offline harness for door_provider_faces.h: ONE PHYSICAL DOOR, ONE LEAF (warm34). Two faces of one provider door (door_1 in
// room_1, door_24 in room_2): closing through one must reach the other's remembered leaf, and an unobserved face's leaf must
// relax toward "unknown" at the stated hazard instead of being held for ever.
// Runs under the agent's locale on purpose (CLAUDE.md).
// Build + run:
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_provider_faces.cpp -o /tmp/test_door_provider_faces && /tmp/test_door_provider_faces
#include <clocale>
#include <cmath>
#include <numbers>
#include <print>
#include "../src/door_provider_faces.h"
#include "../src/door_leaf_channel.h"

using namespace rc::door;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)
static bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

static constexpr std::int64_t kT0 = 1'792'612'380'153;
static constexpr float kPhiMax = 120.f * std::numbers::pi_v<float> / 180.f;
static constexpr float kDeg = std::numbers::pi_v<float> / 180.f;
static constexpr double kHaz = 1.0 / 600.0;   // the place table's prior-mean hazard (PlaceTable::hazard_per_s() before any revisit)

static IdentityRow open_row(const char* room, const char* name, float x, float y)
{
    IdentityRow r;
    r.room = room; r.name = name; r.xy = {x, y};
    r.pose = RestorePose{3.1f, 1.0f, 2.1f, 0.05f};
    r.exist_logodds = 4.0f;
    r.leaf = LeafMemory{{x - 0.5f, y}, {0.f, -1.f}, 19.f, 9.f, 105.f * kDeg, 0.f};
    r.seen_wall_ms = kT0;
    return r;
}

int main()
{
    std::setlocale(LC_ALL, "");

    // (a) CLOSE through face B (room_2) updates face A's (room_1) remembered leaf toward shut.
    {
        std::vector<IdentityRow> rows{open_row("room_1", "door_1", -0.5f, 4.6f), open_row("room_2", "door_24", 0.1f, 2.9f)};
        ProviderFaces pf;
        pf.note("DOOR_1", "room_1", {-0.5f, 4.6f});          // leg 1: OPEN door_1 as DOOR_1
        pf.note("DOOR_1", "room_2", {0.1f, 2.9f});           // CLOSE door_24 as DOOR_1
        const auto others = pf.others("DOOR_1", "room_2");
        CHECK(others.size() == 1 and others[0].room == "room_1");
        CHECK(pf.others("DOOR_1", "room_3").size() == 2);
        CHECK(pf.others("DOOR_2", "room_2").empty());

        const std::int64_t now = kT0 + 60'000;               // closed a minute after it was last seen open
        const float before = rows[0].leaf->phi;
        bool hit = false;
        for (const auto& f : others) hit |= apply_actuation_to_face(rows, f, 0.0f, kPhiMax, now, 0.5f, kHaz);
        CHECK(hit);
        // the restore of door_1 in leg 4 (68 s later): no longer the open leaf the stale row would have restored
        const auto [phi, sd] = restore_leaf_phi(rows[0].leaf->phi, rows[0].leaf->phi_sigma, rows[0].seen_wall_ms, now + 8'000, kPhiMax);
        const auto [phi_stale, sd_stale] = restore_leaf_phi(before, 0.f, kT0, now + 8'000, kPhiMax);   // what restore gave before
        std::print("  door_1 restored leaf: {:.0f} deg sigma {:.0f}  (stale row: {:.0f} deg sigma {:.0f})\n", phi / kDeg, sd / kDeg,
                   phi_stale / kDeg, sd_stale / kDeg);
        CHECK(phi_stale > 95.f * kDeg);
        CHECK(phi < 0.25f * phi_stale);                      // toward shut
        CHECK(rows[0].seen_wall_ms == now);                   // stamp moved: the restore does not age the evidence twice
        CHECK(rows[1].leaf->phi == 105.f * kDeg);             // the actuated face's own row is untouched (its instance owns it)
        CHECK(rows[0].leaf->hinge_llr == 19.f and rows[0].leaf->swing_llr == 9.f);   // side/hinge are face-relative: not touched
        // existence did not lose a minute of age by being rewritten with a new stamp: aged, not refreshed
        const double s60 = std::exp(-kLeafRestoreHazardPerS * 60.0);
        CHECK(near(rows[0].exist_logodds.value(), 4.0 * s60, 1e-4));   // Task 9: aged toward 0 (ignorance), not toward a birth seed
    }

    // an OPEN through the other face moves a shut face toward open
    {
        std::vector<IdentityRow> rows{open_row("room_1", "door_1", -0.5f, 4.6f)};
        rows[0].leaf->phi = 0.f;
        CHECK(apply_actuation_to_face(rows, {"room_1", {-0.45f, 4.65f}}, 100.f * kDeg, kPhiMax, kT0, 0.5f, kHaz));
        CHECK(near(rows[0].leaf->phi, kActuationObeyed * 100.f * kDeg, 1e-4));
        CHECK(not apply_actuation_to_face(rows, {"room_1", {-5.f, 9.f}}, 0.f, kPhiMax, kT0, 0.5f, kHaz));   // another hole
        CHECK(not apply_actuation_to_face(rows, {"room_9", {-0.5f, 4.6f}}, 0.f, kPhiMax, kT0, 0.5f, kHaz)); // another room
    }

    // (b) an unobserved face's leaf relaxes toward the uniform prior at the stated hazard (not held for ever).
    {
        const double h = kLeafRestoreHazardPerS;
        const float phi0 = 105.f * kDeg, sd0 = 3.f * kDeg;
        double prev_mean = 1e9;
        for (const double dt_s : {0.0, 60.0, 600.0, 3600.0, 36000.0})
        {
            const auto [m, sd] = restore_leaf_phi(phi0, sd0, kT0, kT0 + std::int64_t(dt_s * 1e3), kPhiMax);
            const double surv = std::exp(-h * dt_s);
            CHECK(near(m, surv * phi0 + (1.0 - surv) * 0.5 * kPhiMax, 1e-4));    // the change-point mixture's mean
            CHECK(m < prev_mean);                                                 // monotone toward the uniform's mean
            prev_mean = m; (void)sd;   // sigma is NOT monotone: a bimodal mixture is wider than its uniform limit
        }
        const auto [mu, su] = restore_leaf_phi(phi0, sd0, kT0, kT0 + 10'000'000'000LL, kPhiMax);
        CHECK(near(mu, 0.5 * kPhiMax, 1e-3) and near(su, kPhiMax / std::sqrt(12.0), 1e-3));   // long gap = uniform on [0, phi_max]
    }

    std::print("{}\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails == 0 ? 0 : 1;
}
