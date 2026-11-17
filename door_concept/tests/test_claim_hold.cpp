// claim_hold.h: a door is held while a claim executes on one of its affordances, and freed on the first look without one.
//   g++ -std=c++23 -O1 -Isrc tests/test_claim_hold.cpp -o /tmp/test_claim_hold && /tmp/test_claim_hold     (from door_concept/)
#include "claim_hold.h"
#include <cstdio>

static int checks = 0, failures = 0;
#define CHECK(c) do { ++checks; if (not (c)) { ++failures; std::printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)

int main()
{
    using rc::door::claim_executes_on;
    using rc::door::ClaimHolds;
    const std::vector<std::string> none;
    const std::vector<std::string> cross2{"aff_door_2_cross"};
    const std::vector<std::string> other{"afford_room", "aff_door_21_cross", "aff_chair_3_approach"};
    const std::vector<std::string> look2{"aff_door_2"};

    // the helper: name-exact on the door, any verb, nothing on a longer door name, nothing on no edges
    CHECK(claim_executes_on("door_2", cross2));
    CHECK(claim_executes_on("door_2", look2));
    CHECK(not claim_executes_on("door_2", other));        // door_21's cross is not door_2's
    CHECK(claim_executes_on("door_21", other));
    CHECK(not claim_executes_on("door_2", none));
    CHECK(not claim_executes_on("", cross2));

    // the deferral: held once, kept while the claim executes, freed on the first look without it, exactly once
    ClaimHolds h;
    CHECK(h.hold(7, "door_2", 70, /*delete_door=*/false));      // first time: the caller logs
    CHECK(not h.hold(7, "door_2", 70, false));                  // again: no second log
    CHECK(h.holds(7) and h.size() == 1);
    CHECK(h.holds_affordance(70) and not h.holds_affordance(71) and not h.holds_affordance(0));
    CHECK(h.take_free(cross2).empty());                         // the claim executes: nothing is freed, any number of cycles
    CHECK(h.take_free(cross2).empty());
    CHECK(h.holds(7));
    CHECK(not h.hold(7, "door_2", 0, /*delete_door=*/true));    // its room goes meanwhile: the deletion is ADDED, the aff id kept
    CHECK(h.take_free(look2).empty());                          // a claim on another of its affordances still holds it
    {
        const auto freed = h.take_free(other);                  // the consumer executes something else: freed
        CHECK(freed.size() == 1 and freed[0].door_id == 7 and freed[0].hold.delete_door and freed[0].hold.cross_aff_id == 70
              and freed[0].hold.door_name == "door_2");
    }
    CHECK(h.empty() and h.take_free(none).empty());             // ...once

    // two doors: only the one whose claim ended is freed; a door deleted by somebody else is forgotten
    CHECK(h.hold(1, "door_2", 10, true));
    CHECK(h.hold(2, "door_21", 20, false));
    {
        const auto freed = h.take_free(cross2);                 // door_2 still claimed, door_21 not
        CHECK(freed.size() == 1 and freed[0].door_id == 2 and not freed[0].hold.delete_door);
    }
    h.forget(1);
    CHECK(h.empty());

    std::printf("test_claim_hold: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
