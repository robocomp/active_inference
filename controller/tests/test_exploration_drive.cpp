// test_exploration_drive — the exploration drive's decision core (controller/src/exploration_drive.h; spec
// docs/superpowers/specs/2026-11-05-exploration-tour-design.md). Pure: no DSR.
// Build + run (from controller/):
//   g++ -std=c++23 -O1 -Isrc tests/test_exploration_drive.cpp -o /tmp/ted && /tmp/ted
#include <cmath>
#include <print>
#include <string>
#include <vector>

#include "../src/exploration_drive.h"

namespace
{
int fails = 0, checks = 0;
void check(bool ok, const char* what) { ++checks; if (not ok) { ++fails; std::println("FAIL: {}", what); } }
using rc::explore::Cause;
using rc::explore::DoorCand;
using rc::explore::Drive;
DoorCand found(std::string d, double dist) { return {.door = std::move(d), .found = true, .crossed = false, .dist_m = dist}; }
}  // namespace

int main()
{
    {   // waits_while_unpromoted
        Drive d(300.0);
        check(d.step("room_1", false, {found("door_1", 2.0)}, false, 0.0).cause == Cause::None, "t0: nothing");
        check(d.step("room_1", false, {found("door_1", 2.0)}, false, 10.0).cause == Cause::None, "waits_while_unpromoted");
    }
    {   // max_time_leaves
        Drive d(300.0);
        (void)d.step("room_1", false, {}, false, 0.0);
        const auto r = d.step("room_1", false, {found("door_1", 2.0)}, false, 301.0);
        check(r.cause == Cause::MaxTime and r.door == "door_1" and not r.is_return, "max_time_leaves");
    }
    {   // promoted_picks_nearest_uncrossed_found
        Drive d(300.0);
        const std::vector<DoorCand> doors{found("a", 3.0), found("b", 1.0), {.door = "c", .found = false, .crossed = false, .dist_m = 0.5},
                                          {.door = "d", .found = true, .crossed = true, .dist_m = 0.2}};
        const auto r = d.step("room_1", true, doors, false, 0.0);
        check(r.cause == Cause::Promoted and r.door == "b", "promoted_picks_nearest_uncrossed_found");
    }
    {   // entry_door_never_a_candidate
        Drive d(300.0);
        (void)d.step("room_1", true, {}, true, 0.0);   // start room seen (passage running)
        d.note_entered("room_2", "door_7", 10.0);
        const auto r = d.step("room_2", true, {found("door_7", 0.5)}, false, 20.0);
        check(r.cause == Cause::Return and r.door == "door_7" and r.is_return, "entry_door_never_a_candidate (returns through it instead)");
    }
    {   // start_room_nothing_left_is_done
        Drive d(300.0);
        check(d.step("room_1", true, {}, false, 0.0).cause == Cause::Done, "start_room_nothing_left_is_done");
    }
    {   // dfs_return_uses_first_entry_door
        Drive d(300.0);
        (void)d.step("room_1", true, {}, true, 0.0);
        d.note_entered("room_2", "d1", 1.0);
        d.note_entered("room_3", "d2", 2.0);
        d.note_entered("room_2", "d3", 3.0);   // came back from room_3 through room_2's mirror of d2
        check(d.entry_door("room_2") == "d1", "first entry door kept");
        const auto r = d.step("room_2", true, {}, false, 4.0);
        check(r.cause == Cause::Return and r.door == "d1", "dfs_return_uses_first_entry_door");
    }
    {   // failed_door_is_not_retried_immediately
        Drive d(300.0);
        (void)d.step("room_1", true, {}, true, 0.0);
        d.note_failed("door_1");
        const auto r = d.step("room_1", true, {found("door_1", 1.0), found("door_2", 4.0)}, false, 1.0);
        check(r.door == "door_2", "failed_door_is_not_retried_immediately");
    }
    {   // passage_active_holds
        Drive d(300.0);
        check(d.step("room_1", true, {found("door_1", 1.0)}, true, 400.0).cause == Cause::None, "passage_active_holds");
    }
    {   // start_room_never_gets_entry_door
        Drive d(300.0);
        (void)d.step("room_1", true, {}, true, 0.0);
        d.note_entered("room_1", "door_0", 50.0);   // the ring R1->R2->R3->R1 re-enters the start room
        check(not d.entry_door("room_1").has_value(), "start_room_never_gets_entry_door");
        check(d.step("room_1", true, {}, false, 60.0).cause == Cause::Done, "start room, nothing left: Done");
    }
    {   // reentered_face_is_crossed (explore_tour1: back in the START room through the mirror door_4 of door_3, the drive
        // chose door_4 — the doorway it had just come through — and the passage held 7.5 min before failing)
        Drive d(300.0);
        (void)d.step("room_1", true, {}, true, 0.0);
        d.note_crossed("door_2");
        d.note_entered("room_2", "door_3", 10.0);
        d.note_crossed("door_3");
        d.note_entered("room_1", "door_4", 40.0);
        const auto r = d.step("room_1", true, {found("door_4", 1.4), found("door_1", 3.7)}, false, 50.0);
        check(r.cause == Cause::Promoted and r.door == "door_1", "reentered_face_is_crossed (door_4 skipped, door_1 chosen)");
        const auto r2 = d.step("room_1", true, {found("door_2", 3.6)}, false, 60.0);
        check(r2.cause == Cause::Done, "a door the drive itself crossed is never chosen again");
    }
    {   // C2: a failed entry door is never retried in a loop; an empty entry door is never stored
        Drive d(300.0);
        (void)d.step("room_1", true, {}, true, 0.0);
        d.note_entered("room_2", "", 1.0);   // no mirror resolved yet
        check(not d.entry_door("room_2").has_value(), "empty_entry_never_stored");
        d.note_entered("room_2", "door_3", 2.0);   // resolved later: still the first REAL entry
        check(d.entry_door("room_2") == "door_3", "a later real entry is kept when the first was empty");
        const auto r = d.step("room_2", true, {}, false, 3.0);
        check(r.cause == Cause::Return and r.door == "door_3", "returns through it");
        d.note_failed("door_3");
        const auto r2 = d.step("room_2", true, {}, false, 3.2);
        check(r2.cause == Cause::Stuck and not r2.door.has_value(), "failed_return_is_stuck_not_retried");
    }
    {   // C1: a re-entered promoted room whose remembered doors are not yet seen again waits (bounded by MaxRoomS)
        Drive d(300.0);
        (void)d.step("room_1", true, {}, true, 0.0);
        d.note_entered("room_2", "door_3", 10.0);
        d.note_entered("room_1", "door_4", 20.0);
        check(d.step("room_1", true, {}, false, 30.0, /*remembered_uncrossed=*/1).cause == Cause::None,
              "remembered_uncrossed_waits (not Done while the LTSM remembers an uncrossed door of this room)");
        check(d.step("room_1", true, {}, false, 330.0, 1).cause == Cause::Done, "the wait is bounded by MaxRoomS");
        check(d.step("room_1", true, {found("door_9", 2.0)}, false, 31.0, 1).door == "door_9", "a re-found door is taken at once");
    }
    check(std::string(rc::explore::to_string(Cause::Stuck)) == "stuck", "to_string stuck");
    check(std::string(rc::explore::to_string(Cause::MaxTime)) == "max-time", "to_string");
    std::println("{}/{} checks passed", checks - fails, checks);
    return fails == 0 ? 0 : 1;
}
