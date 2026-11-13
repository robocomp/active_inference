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
    {   // clean13: the room's identity is open while a remembered room is the best explanation
        using rc::explore::recognition_leans_known;
        const std::vector<float> known{0.1f, 0.8f, 0.1f}, fresh{0.0f, 0.0f, 1.0f}, tie{0.33f, 0.33f, 0.33f}, none{1.0f};
        if (not recognition_leans_known(known) or recognition_leans_known(fresh) or not recognition_leans_known(tie) or recognition_leans_known(none))
        { std::println("FAIL recognition_leans_known"); return 1; }
        if (not rc::explore::door_believed(std::nullopt) or rc::explore::door_believed(0.31f) or not rc::explore::door_believed(0.5f))
        { std::println("FAIL door_believed"); return 1; }
        rc::explore::Drive d(300.0);
        if (not d.within_room_time("r", 0.0)) { std::println("FAIL within_room_time unknown room"); return 1; }
    }

    {   // clean23: an undecided candidate is unfinished business. Return and Done wait for it, bounded by MaxRoomS; a found door does not
        using rc::explore::doors_settled;
        check(doors_settled(std::nullopt, 7), "no statement does not hold the drive");
        check(doors_settled(rc::explore::DoorQuestions{7, "settled"}, 7), "settled in this room");
        check(not doors_settled(rc::explore::DoorQuestions{7, "open"}, 7), "open in this room");
        check(not doors_settled(rc::explore::DoorQuestions{5, "settled"}, 7), "a statement about another room is no answer");
        Drive done(300.0);
        check(done.step("room_1", true, {}, false, 0.0, 0, false).cause == Cause::None, "Done waits on an open room");
        check(done.step("room_1", true, {}, false, 10.0, 0, true).cause == Cause::None, "a start room with NO door at all is still an open question (clean43)");
        check(done.step("room_1", true, {}, false, 301.0, 0, true).cause == Cause::Done, "...bounded by MaxRoomS");
        Drive bounded(300.0);
        bounded.step("room_1", true, {}, false, 0.0, 0, false);
        check(bounded.step("room_1", true, {}, false, 301.0, 0, false).cause == Cause::Done, "MaxRoomS bounds the wait");
        Drive ret(300.0);
        ret.step("room_1", true, {}, false, 0.0);
        ret.note_entered("room_2", "door_2", 1.0);
        check(ret.step("room_2", true, {}, false, 2.0, 0, false).cause == Cause::None, "Return waits on an open room");
        check(ret.step("room_2", true, {}, false, 3.0, 0, true).cause == Cause::Return, "Return once settled");
        Drive go(300.0);
        check(go.step("room_1", true, {found("door_1", 1.0)}, false, 0.0, 0, false).door == std::optional<std::string>("door_1"), "a found door is not held");
    }

    {   // clean46: a found door that dips below 1/2 is skipped while it stays there and is taken again when its belief recovers
        Drive d(300.0);
        d.step("room_1", true, {found("door_0", 1.0)}, false, 0.0);
        d.note_entered("room_3", "door_6", 10.0);
        DoorCand dip = found("door_7", 2.8); dip.believed = false;
        const auto during = d.step("room_3", true, {found("door_6", 4.0), dip}, false, 11.0);
        check(during.door != std::optional<std::string>{"door_7"}, "a disbelieved found door is not chosen");
        const auto after = d.step("room_3", true, {found("door_6", 4.0), found("door_7", 2.8)}, false, 12.0);
        check(after.cause == Cause::Promoted and after.door == std::optional<std::string>{"door_7"}, "the same door is chosen again once believed");
    }

    {   // clean28: a door just born is not offered yet. A believed door with no approach is unfinished; a disbelieved one is nothing
        Drive d(300.0);
        const DoorCand fresh{.door = "door_1", .found = false, .crossed = false, .dist_m = 2.5};
        check(d.step("room_1", true, {fresh}, false, 0.0).cause == Cause::None, "Done waits for a door not offered yet");
        check(d.step("room_1", true, {fresh}, false, 301.0).cause == Cause::Done, "MaxRoomS bounds that wait");
        Drive no(300.0);
        const DoorCand ghost{.door = "door_2", .found = false, .crossed = false, .dist_m = 2.5, .believed = false};
        check(no.step("room_1", true, {ghost}, false, 0.0).cause == Cause::Done, "a disbelieved door does not hold the drive");
        Drive off(300.0);
        off.step("room_1", true, {found("door_1", 1.0)}, false, 0.0);
        off.note_failed("door_1");
        check(off.step("room_1", true, {fresh}, false, 1.0).cause == Cause::Done, "a door that failed this visit does not hold the drive");
    }

    {   // withdrawn crossing (clean16): back in the room left; the door is failed there, the false entry forgotten, the drive goes on
        Drive d(300.0);
        d.step("room_1", true, {found("door_9", 1.0), found("door_3", 5.0)}, false, 0.0);
        d.note_crossed("door_9");
        d.note_entered("room_4", "door_11", 10.0);
        d.note_crossing_withdrawn("door_9", "door_11", "room_4", "room_1");
        check(not d.entry_door("room_4").has_value(), "the false room has no entry door any more");
        const auto s = d.step("room_1", true, {found("door_3", 5.0), found("door_9", 1.0)}, false, 20.0);
        check(s.door == std::optional<std::string>("door_3"), "the phantom is failed there: the next door is the nearest other one, no wait for a room");
        check(d.step("room_1", true, {found("door_9", 1.0)}, false, 21.0).cause == Cause::Done, "and the door stays failed this visit");
    }
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
        // clean43: a start room with NO door known at all is an open question (the robot has not looked yet), MaxRoomS bounds it
        check(d.step("room_1", true, {}, false, 0.0).cause == Cause::None, "start room, no door at all: not Done yet");
        check(d.step("room_1", true, {}, false, 301.0).cause == Cause::Done, "start room, no door at all: Done after MaxRoomS");
        Drive d2(300.0);
        const DoorCand crossed{.door = "door_1", .found = true, .crossed = true, .dist_m = 1.0};
        check(d2.step("room_1", true, {crossed}, false, 0.0).cause == Cause::Done, "start_room_nothing_left_is_done (its doors all crossed)");
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
        const DoorCand came_in{.door = "door_0", .found = true, .crossed = true, .dist_m = 0.5};   // the door the ring re-entered by
        check(d.step("room_1", true, {came_in}, false, 60.0).cause == Cause::Done, "start room, nothing left: Done");
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
    {   // explore_tour11: a door found AFTER a Return was decided (room_3 promoted, DOOR_0 born 2.6 s later) re-decides the leave
        Drive d(300.0);
        (void)d.step("room_1", true, {}, true, 0.0);
        d.note_entered("room_3", "door_5", 10.0);
        const auto r = d.step("room_3", true, {found("door_5", 2.0)}, false, 40.0);
        check(r.cause == Cause::Return and r.door == "door_5", "late door: the first decision is the Return");
        const auto late = d.reconsider_return("room_3", {found("door_5", 2.0), found("door_6", 2.5)});
        check(late.has_value() and late->door == "door_6" and late->cause == Cause::Promoted and not late->is_return,
              "late door: a newly found uncrossed door supersedes the Return");
        check(not d.reconsider_return("room_3", {found("door_5", 2.0)}).has_value(), "late door: nothing new, the Return stands");
        d.note_failed("door_6");
        check(not d.reconsider_return("room_3", {found("door_5", 2.0), found("door_6", 2.5)}).has_value(),
              "late door: a door that failed this visit never supersedes");
        d.note_crossed("door_7");
        check(not d.reconsider_return("room_3", {found("door_5", 2.0), found("door_7", 1.0)}).has_value(),
              "late door: a crossed door never supersedes");
    }
    {   // clean58 (2026-11-12): a passage FAILED because door_concept never offered door_3's cross. The drive must DO something else —
        //   another door, else the way back — and must not take door_3 again until there is NEW EVIDENCE about it (a model event).
        using rc::explore::DoorEvidence;
        Drive d(300.0);
        (void)d.step("room_1", true, {}, true, 0.0);
        d.note_entered("room_2", "door_2", 10.0);
        const DoorEvidence at_failure{.node_id = 33, .exists_map = true, .step_offered = false};
        auto door3 = found("door_3", 0.6); door3.evidence = at_failure;   // the nearest
        auto door4 = found("door_4", 3.0); door4.evidence = {.node_id = 44};
        const auto first = d.step("room_2", true, {door3, door4}, false, 20.0);
        check(first.door == std::optional<std::string>{"door_3"}, "clean58: the nearest door is taken first");
        d.note_failed("door_3", at_failure);                               // ... and its passage fails (cross_not_offered)
        const auto next = d.step("room_2", true, {door3, door4}, false, 21.0);
        check(next.cause == Cause::Promoted and next.door == std::optional<std::string>{"door_4"},
              "clean58: after the failure the drive picks ANOTHER door, not the failed one");
        for (double t = 22.0; t < 400.0; t += 0.2)                          // the same evidence, for as long as it stays the same
            if (d.step("room_2", true, {door3}, false, t).door == std::optional<std::string>{"door_3"})
            { check(false, "clean58: the failed door is NEVER re-picked without new evidence (no loop)"); break; }
        const auto back = d.step("room_2", true, {door3}, false, 30.0);
        check(back.cause == Cause::Return and back.door == std::optional<std::string>{"door_2"} and back.is_return,
              "clean58: no other door -> the way back through the entry door, not a stand");
        check(d.take_retried().empty(), "no retry without new evidence");
        auto offered_now = door3; offered_now.evidence.step_offered = true;   // door_concept now offers its open/cross
        const auto retry = d.step("room_2", true, {offered_now}, false, 31.0);
        check(retry.door == std::optional<std::string>{"door_3"}, "new evidence (door_concept offers its step): the door may be retried");
        const auto retried = d.take_retried();
        check(retried.size() == 1 and retried.front() == "door_3", "the retry is reported once");
        d.note_failed("door_3", offered_now.evidence);                     // fails again: the new evidence is recorded anew
        check(d.step("room_2", true, {offered_now}, false, 32.0).door != std::optional<std::string>{"door_3"},
              "a second failure under that evidence skips it again");
        auto reborn = door3; reborn.evidence.node_id = 99;                  // door_concept re-born door_3 (a new hypothesis)
        check(d.step("room_2", true, {reborn}, false, 33.0).door == std::optional<std::string>{"door_3"}, "a re-born door may be retried");
    }
    {   // a failed WAY BACK is Stuck (no loop), and new evidence about the entry door lifts it — the drive does not stop for ever
        using rc::explore::DoorEvidence;
        Drive d(300.0);
        (void)d.step("room_1", true, {}, true, 0.0);
        d.note_entered("room_2", "door_2", 1.0);
        auto entry = found("door_2", 1.0); entry.evidence = {.node_id = 22};
        check(d.step("room_2", true, {entry}, false, 2.0).cause == Cause::Return, "returns through the entry door");
        d.note_failed("door_2", entry.evidence);
        check(d.step("room_2", true, {entry}, false, 3.0).cause == Cause::Stuck, "a failed way back with no new evidence: Stuck");
        auto entry_new = entry; entry_new.evidence.step_offered = true;
        const auto r = d.step("room_2", true, {entry_new}, false, 4.0);
        check(r.cause == Cause::Return and r.door == std::optional<std::string>{"door_2"}, "new evidence about the entry door: the way back again");
    }
    check(std::string(rc::explore::to_string(Cause::Stuck)) == "stuck", "to_string stuck");
    check(std::string(rc::explore::to_string(Cause::MaxTime)) == "max-time", "to_string");
    std::println("{}/{} checks passed", checks - fails, checks);
    return fails == 0 ? 0 : 1;
}
