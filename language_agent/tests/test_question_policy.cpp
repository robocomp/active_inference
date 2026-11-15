// question_policy.h + place_words.h: ask or report a moved object, its words, the reply. Build+run: tests/run_tests.sh
#include "../src/question_policy.h"
#include "../src/place_words.h"
#include <cassert>
#include <print>

using namespace rc::chat;

int main()
{
    int checks = 0;
    assert(choose_mode(0, 0, 0.4f, 1.f) == QMode::ask); ++checks;      // P = 1/2: a wrong "I assume" costs 0.5 > 0.4
    assert(choose_mode(1, 0, 0.4f, 1.f) == QMode::report); ++checks;   // P = 2/3: 0.33 < 0.4, one yes tips it
    assert(choose_mode(1, 1, 0.4f, 1.f) == QMode::ask); ++checks;

    assert(classify_reply("yes, I moved it") == Reply::yes); ++checks;
    assert(classify_reply("Yes.") == Reply::yes); ++checks;
    assert(classify_reply("sí, lo moví yo") == Reply::yes); ++checks;
    assert(classify_reply("no, I didn't") == Reply::no); ++checks;
    assert(classify_reply("No lo moví") == Reply::no); ++checks;
    assert(classify_reply("no sé") == Reply::unknown); ++checks;
    assert(classify_reply("I don't know") == Reply::unknown); ++checks;
    assert(classify_reply("go to the kitchen") == Reply::other); ++checks;
    assert(classify_reply("nothing to do with chairs: what is in room 3?") == Reply::other); ++checks;

    OpenQuestion q{.stamp = 1000, .memory_uid = 50, .live_uid = 57, .room = "room_2", .cls = "chair", .colour = "magenta",
                   .memory_name = "chair_50", .old_xy = {-2.57f, -0.61f}, .new_xy = {1.2f, 1.5f}, .again = false};
    assert(question_text(q, QMode::ask, false, "by door_25", "by the window")
           == "The magenta chair in room 2 is by the window now; it was by door_25. Did you move it?"); ++checks;
    assert(question_text(q, QMode::ask, true, "by door_25", "by the window")
               .starts_with("By the way, while you were away the magenta chair in room 2 moved ")); ++checks;
    assert(question_text(q, QMode::ask, true, "by door_25", "by the window")
               .ends_with("I've assumed it's the same chair. Did you move it?")); ++checks;
    q.again = true;
    assert(question_text(q, QMode::report, false, "by door_25", "in the corner")
           == "The magenta chair has moved again, in the corner. I assume you moved it."); ++checks;
    q.colour.clear();
    assert(question_text(q, QMode::ask, false, "by door_25", "in the corner").starts_with("The chair in room 2 is in the corner")); ++checks;

    assert(moved_claim_value(Reply::no, 1763200000000ull) == "no:1763200000000"); ++checks;
    assert(moved_claim_value(Reply::yes, 7) == "yes:7"); ++checks;
    assert(moved_claim_value(Reply::unknown, 7) == "unknown:7"); ++checks;

    // place words: the nearest landmark, no cutoff. A 4 x 6 room centred on the origin, one door on the x = -2 wall.
    const std::vector<P2> room{{-2.f, -3.f}, {2.f, -3.f}, {2.f, 3.f}, {-2.f, 3.f}};
    const std::vector<std::pair<std::string, P2>> doors{{"door_25", P2{-2.f, 0.f}}};
    assert(place_words({-1.6f, 0.f}, room, doors) == "by door_25"); ++checks;
    assert(place_words({1.8f, 2.8f}, room, doors) == "in the corner"); ++checks;
    assert(place_words({0.f, 0.1f}, room, doors) == "near the middle of the room"); ++checks;
    assert(place_words({0.f, 0.f}, {}, {}) == "near the middle of the room"); ++checks;

    std::println("question_policy: {} checks passed", checks);
}
