// test_door_remembered_height.cpp — clean60 (2026-11-13), door_7: born and removed 18 times in a row at one place in room_3.
//
// The place is the GlassBay in Webots' room_1 (a 3 m tall glass shower partition, panes 0.99 / 0.88 m between mullions at z 1 and
// 2 m): NOT a door. Its fit read 1.00 x 2.62 m (σ_w 0.043, σ_h 0.024) at 6.1 m, and the size term under the fit's prior
// 2.10 ± 0.08 m (place_prior.h normalised_log_predictive) was −19.67 against vision's +1.5: removed, correctly. The defect is the
// LOOP: the removed row keeps L_evidence, which excludes the size term (it lives in the structural prior), so the row remembers
// "door-shaped" (+); and a pending candidate measures its blob's width but NO height (its height is the prior's, the height half
// of the size term is 0), so the next detection at the place was born again and refuted again by its first fit.
// The cure (place_table.h PlaceTable::remembered_height): the place's measured height is the candidate's height measurement, with
// its precision aged by the change hazard that ages the row's L.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I/usr/include/opencv4 -I../common tests/test_door_remembered_height.cpp -o /tmp/test_door_remembered_height && /tmp/test_door_remembered_height
#include <clocale>
#include <cmath>
#include <cstdio>
#include <vector>
#include "../src/core/place_table.h"
#include "../src/core/place_prior.h"
using namespace rc::door;
using namespace rc::door::core;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// clean60's numbers: Door.PriorWidthM/Std 1.00/0.06, PriorHeightM/Std 2.10/0.08 (etc/config.toml); as logged for door_7:
// base −0.41 (the first door on its wall), band ≈ 0, delete line −1.08, vision (zed + sup) ≈ +1.5.
static const SizePrior kPrior{1.00f, 0.06f, 2.10f, 0.08f};
static constexpr float kBase = -0.41f, kDeleteLine = -1.08f, kVision = 1.5f;

static float size_term(float w, float w_sd, float h, float h_sd)
{
    const std::vector<Vec2f> none;   // no polygon: band and extent are 0, the size term alone
    PlacePriorInput in{none, true, rc::door::RoomBand{}, Vec2f::Zero(), Vec2f::UnitX(), w, h, w_sd, h_sd, true, 0, kPrior, 0.f};
    return place_prior(in).size;
}

int main()
{
    std::setlocale(LC_ALL, "");
    constexpr std::int64_t kRemovedMs = 1'794'526'950'000;

    // (1) the born door's verdict, as logged: size −19.67 (h 2.6234 σ 0.0237, w 1.00258 σ 0.0431)
    const float born = size_term(1.00258f, 0.0431f, 2.6234f, 0.0237f);
    std::printf("born door_7: size %+.2f, L %+.2f\n", born, kBase + born + kVision);
    CHECK(std::abs(born - (-19.67f)) < 0.3f);
    CHECK(kBase + born + kVision < kDeleteLine);

    // (2) BEFORE: the candidate at the place scored the measured width and the PRIOR's height — the refutation vanished
    const float cand_blind = size_term(1.00f, 0.20f, kPrior.h_m, kPrior.h_sd);
    std::printf("candidate without the place's height: size %+.2f, L %+.2f -> born again\n", cand_blind, kBase + cand_blind + kVision);
    CHECK(kBase + cand_blind + kVision > 0.0f);   // the loop: born, then refuted by its first fit

    // (3) AFTER: the removed row's height is the candidate's height measurement
    PlaceTable t(false);
    PlaceRow r; r.room = "room_3"; r.name = "door_7"; r.xy = {-0.16f, 8.46f}; r.cov = Eigen::Matrix2f::Identity() * 0.04f;
    r.width = 1.00f; r.height = 2.6234f; r.height_sd = 0.0237f; r.depth = 0.05f; r.L_evidence = kVision;
    r.last_seen_wall_ms = r.sighted_wall_ms = kRemovedMs; r.removed_wall_ms = kRemovedMs;
    t.upsert(r);
    const PlaceRow& row = *t.find("room_3", "door_7");
    {
        const auto mh = t.remembered_height(row, kRemovedMs + 300, kPrior.h_sd);   // the next detection, 0.3 s later
        CHECK(mh.has_value());
        const float s = mh ? size_term(1.00f, 0.20f, mh->height, mh->sd) : 0.f;
        const float L = kBase + s + t.aged_L(row, kRemovedMs + 300);
        std::printf("candidate 0.3 s after the removal: height %.2f sd %.3f -> size %+.2f, L %+.2f -> %s\n", mh ? mh->height : NAN,
                    mh ? mh->sd : NAN, s, L, L < kDeleteLine ? "not born" : "BORN");
        CHECK(L < kDeleteLine and s < -19.0f);
    }
    {   // a memory fades with the hazard (prior mean 1/600 s): an hour later the remembered height says almost nothing
        const auto mh = t.remembered_height(row, kRemovedMs + 3'600'000, kPrior.h_sd);
        const float s = mh ? size_term(1.00f, 0.20f, mh->height, mh->sd) : 0.f;
        std::printf("one hour later: sd %.3f -> size %+.2f\n", mh ? mh->sd : NAN, s);
        CHECK(mh.has_value() and s > -1.0f and s < 0.0f);
    }
    {   // a row loaded from the identity CSV carries no sd: the unknown sd is the prior's, the refutation is still strong
        PlaceRow loaded = row; loaded.height_sd = 0.f;
        const auto mh = t.remembered_height(loaded, kRemovedMs, kPrior.h_sd);
        const float s = mh ? size_term(1.00f, 0.20f, mh->height, mh->sd) : 0.f;
        std::printf("loaded row (no sd): size %+.2f\n", s);
        CHECK(mh.has_value() and s < -9.0f);
    }
    {   // a door-tall remembered place (a real door's row) costs nothing, and a row without geometry supplies nothing
        PlaceRow door = row; door.height = 2.10f;
        const auto mh = t.remembered_height(door, kRemovedMs, kPrior.h_sd);
        CHECK(mh.has_value() and std::abs(size_term(1.00f, 0.20f, mh->height, mh->sd)) < 1e-4f);
        PlaceRow bare = row; bare.height = 0.f;
        CHECK(not t.remembered_height(bare, kRemovedMs, kPrior.h_sd).has_value());
    }
    std::printf(fails == 0 ? "test_door_remembered_height: ALL PASS\n" : "test_door_remembered_height: %d FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
