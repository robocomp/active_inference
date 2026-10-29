// Offline harness for the room-recognition swap in door_identity_table.h: a LATE recall (ltsm's recall_<K>, recall_late)
// for the room being entered, right after a PROVISIONAL room of another name was left, is a recalled entry (restore of
// K's doors), never "entered FRESH", and the provisional name's identity rows are DROPPED (room names are never reused).
// Build + run:
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_room_swap.cpp -o /tmp/test_door_room_swap && /tmp/test_door_room_swap
#include <clocale>
#include <print>
#include <string>
#include <vector>
#include "../src/door_identity_table.h"

using namespace rc::door;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)

static IdentityRow row(std::string room, std::string name, float x, float y, bool full)
{
    IdentityRow r;  r.room = std::move(room); r.name = std::move(name); r.xy = {x, y};
    if (full) { r.pose = RestorePose{3.1f, 0.95f, 2.13f, 0.05f}; r.exist_logodds = 3.0f; }
    return r;
}

int main()
{
    std::setlocale(LC_ALL, "");
    // room_3 remembered (full rows); provisional room_4 earned two doors while it was believed new.
    std::vector<IdentityRow> rows{row("room_3", "door_7", 1, 2, true), row("room_4", "door_9", 0.5f, 1, true),
                                  row("room_4", "door_10", -2, 3, false), row("room_1", "door_1", 4, 4, true)};

    // swap: room_4 left, room_3 entered, recall_room_3 announced LATE
    const auto p = plan_room_entry(false, true, true, "room_4", "room_3");
    std::print("door_concept: [restore] '{}' {}\n", "room_3", p.recalled ? "recalled" : "entered FRESH");
    CHECK(p.recalled);
    CHECK(p.drop_left);
    const auto dropped = drop_room(rows, "room_4");
    CHECK(dropped.size() == 2);
    for (const auto& r : rows) CHECK(r.room != "room_4");
    CHECK(rows.size() == 2);                                  // room_3 and room_1 untouched
    CHECK(rows[0].pose.has_value());

    // an ordinary (early) recall never drops what it leaves behind
    CHECK(not plan_room_entry(false, true, false, "room_2", "room_3").drop_left);
    CHECK(plan_room_entry(false, true, false, "room_2", "room_3").recalled);
    // no recall: FRESH, nothing dropped, even if "late" were stale
    const auto f = plan_room_entry(false, false, true, "room_4", "room_5");
    CHECK(not f.recalled and not f.drop_left);
    // first resolution is never a recall, no left room
    CHECK(not plan_room_entry(true, true, true, "", "room_1").recalled);
    // Undo (turn-around): back to the provisional name, announced by nobody: FRESH, and the dropped rows stay dropped
    const auto u = plan_room_entry(false, false, false, "room_3", "room_4");
    CHECK(not u.recalled and not u.drop_left);
    CHECK(drop_room(rows, "room_4").empty());
    // same-name late recall (no swap) drops nothing
    CHECK(not plan_room_entry(false, true, true, "room_3", "room_3").drop_left);

    // C1.4: an announced room restores its remembered rows, the crossed door's face included, and gets no mirror
    {
        std::vector<IdentityRow> rr{row("room_1", "door_1", 2, 0, true), row("room_1", "door_2", -3, 1, true)};   // door_1 = crossed face
        CHECK(plan_room_entry(false, true, false, "room_2", "room_1").recalled);
        CHECK(not entry_mirror_wanted(true, true));            // announced proto room: no mirror
        const auto d = plan_restore(rr, "room_1", true, {}, 0.3f);
        CHECK(d.restore.size() == 2 and d.restore[0].name == "door_1");
        // a face already live at its place is not duplicated (the recreated/carried crossed door)
        std::vector<LiveDoor> live{LiveDoor{"door_9", Eigen::Vector2f(2.05f, 0.f)}};
        const auto d2 = plan_restore(rr, "room_1", true, live, 0.3f);
        CHECK(d2.restore.size() == 1 and d2.restore[0].name == "door_2");
    }
    CHECK(entry_mirror_wanted(true, false));                   // fresh_room_still_mirrors
    CHECK(not entry_mirror_wanted(false, false));              // not proto: never

    std::print("{}\n", fails == 0 ? "ALL PASS" : "FAILED");
    return fails != 0;
}
