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
    check(std::string(rc::explore::to_string(Cause::MaxTime)) == "max-time", "to_string");
    std::println("{}/{} checks passed", checks - fails, checks);
    return fails == 0 ? 0 : 1;
}
