// door_memory_note.h: IdentityRow <-> the attributes a door node / LTSM note carries (2026-11-13, door memory through ltsm).
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_memory_note.cpp -o /tmp/tdmn && /tmp/tdmn
#include "../src/door_memory_note.h"

#include <cmath>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, const char* what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL: {}", what); }
}
bool near(float a, float b, float tol = 1e-5f) { return std::abs(a - b) <= tol; }
bool near2(const Eigen::Vector2f& a, const Eigen::Vector2f& b, float tol = 1e-5f) { return (a - b).norm() <= tol; }

// a row shaped like specificworker identity_of(): hinge_xy = a jamb, open_dir = swing * the wall normal, llr >= 0 relative to the MAP side
rc::door::IdentityRow make_row(float yaw, bool hinge_plus, float swing, float hinge_llr, float swing_llr)
{
    rc::door::IdentityRow r;
    r.room = "room_1";  r.name = "door_5";
    r.xy = {1.5f, -2.0f};
    const float w = 0.9f;
    r.pose = rc::door::RestorePose{yaw, w, 2.05f, 0.12f};
    const Eigen::Vector2f x(std::cos(yaw), std::sin(yaw)), n(-std::sin(yaw), std::cos(yaw));
    r.exist_logodds = 2.3f;  r.exist_physical = 0.7f;  r.crossed = true;
    r.seen_wall_ms = 1'700'000'000'123;  r.sighted_ms = 1'700'000'000'456;
    r.place_sd = 0.04f;  r.ledger = "v1;v2";  r.visit = 3;
    r.leaf = rc::door::LeafMemory{r.xy + (hinge_plus ? 1.f : -1.f) * 0.5f * w * x, swing * n, hinge_llr, swing_llr, 0.35f, 0.08f};
    return r;
}
}  // namespace

int main()
{
    namespace mn = rc::door::memnote;
    namespace lm = rc::door::leafmem;

    // memnote.row_round_trip_every_field: both MAP hinge sides x both MAP swing sides, a rotated and an axis-aligned frame
    for (const float yaw : {0.0f, 0.7f, -2.4f})
        for (const bool hp : {true, false})
            for (const float sw : {1.0f, -1.0f})
            {
                const auto r = make_row(yaw, hp, sw, 2.2f, 1.4f);
                const auto b = mn::row_of(mn::note_of(r, 0.03f), r.room, r.name);
                check(b.room == r.room and b.name == r.name, "room/name passed through");
                check(near2(b.xy, r.xy) and b.pose.has_value(), "xy and pose");
                check(near(b.pose->yaw, yaw) and near(b.pose->width, 0.9f) and near(b.pose->height, 2.05f) and near(b.pose->depth, 0.12f), "yaw w h d");
                check(b.exist_logodds == r.exist_logodds and b.exist_physical == r.exist_physical and b.crossed == r.crossed, "existence, physical, crossed");
                check(b.seen_wall_ms == r.seen_wall_ms and b.sighted_ms == r.sighted_ms, "seen_ms, sighted_ms");
                check(b.place_sd == r.place_sd and b.ledger == r.ledger and b.visit == r.visit, "place_sd, ledger, visit");
                check(b.leaf.has_value(), "leaf present");
                if (not b.leaf) continue;
                check(near2(b.leaf->hinge_xy, r.leaf->hinge_xy, 1e-5f), "hinge_xy within 1e-5 m");
                check(near2(b.leaf->open_dir, r.leaf->open_dir, 1e-5f), "open_dir within 1e-5");
                check(near(b.leaf->hinge_llr, 2.2f, 1e-4f) and near(b.leaf->swing_llr, 1.4f, 1e-4f), "llr relative to MAP side, >= 0");
                check(b.leaf->phi == r.leaf->phi and b.leaf->phi_sigma == r.leaf->phi_sigma, "phi, phi_sigma equal");
            }

    // memnote.leaf_llr_beyond_float_range
    for (const bool hp : {true, false})
        for (const float sw : {1.0f, -1.0f})
        {
            const auto r = make_row(0.7f, hp, sw, 95.0f, 70.0f);
            const auto b = mn::row_of(mn::note_of(r, std::nullopt), r.room, r.name);
            check(b.leaf.has_value(), "saturated leaf present");
            if (not b.leaf) continue;
            check(near(b.leaf->hinge_llr, lm::logit(1.0f - 1e-6f), 0.02f) and near(b.leaf->swing_llr, lm::logit(1.0f - 1e-6f), 0.02f), "|llr| == logit clamp (13.8; float 1-1e-6 is asymmetric by 0.013)");
            check(near2(b.leaf->hinge_xy, r.leaf->hinge_xy) and near2(b.leaf->open_dir, r.leaf->open_dir), "MAP sides kept at saturation");
        }

    // memnote.absent_fields_stay_absent
    {
        auto r = make_row(0.3f, true, 1.0f, 1.0f, 1.0f);
        r.ledger.reset(); r.visit.reset(); r.seen_wall_ms.reset(); r.sighted_ms.reset();
        r.place_sd.reset(); r.exist_physical.reset(); r.crossed.reset(); r.exist_logodds.reset(); r.leaf.reset();
        const auto n = mn::note_of(r, std::nullopt);
        check(not n.ledger and not n.visit and not n.seen_ms and not n.sighted_ms and not n.place_sd and not n.height_sd, "note keeps them absent");
        check(not n.leaf_into_room_prob and not n.leaf_hinge_plus_prob and not n.phi and not n.phi_sd and not n.crossed, "no leaf in the note");
        const auto b = mn::row_of(n, "room_1", "door_5");
        check(not b.ledger and not b.visit and not b.seen_wall_ms and not b.sighted_ms and not b.place_sd, "row keeps them absent");
        check(not b.exist_logodds and not b.exist_physical and not b.crossed and not b.leaf, "row: no belief, no leaf");
    }

    // memnote.removed_never_encoded
    {
        auto r = make_row(0.3f, true, 1.0f, 1.0f, 1.0f);
        r.removed_ms = 12345;
        const auto b = mn::row_of(mn::note_of(r, std::nullopt), r.room, r.name);
        check(not b.removed_ms.has_value(), "removed_ms never survives the codec");
    }

    // memnote.face_live_name_strips_prefix
    check(mn::face_live_name("remembered_door_5") == "door_5", "remembered_door_5 -> door_5");
    check(mn::face_live_name("door_5") == "door_5", "a live name is unchanged");

    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
