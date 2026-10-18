// What the controller is doing now, for the STATE (src/controller_facts.h): the affordance it executes and the mission.
#include "../src/controller_facts.h"

#include <clocale>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)

using namespace rc::chat;
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");   // es_ES here: numbers must still print with a point

    // ── an affordance's name in words ────────────────────────────────────────────────────────────────────────────
    CHECK(affordance_words("aff_door_1_cross") == "cross door_1");
    CHECK(affordance_words("aff_table_1_epistemic") == "look at table_1");   // epistemic = go and see it better
    CHECK(affordance_words("aff_door_12_approach") == "approach door_12");
    CHECK(affordance_words("aff_door_1") == "look at door_1");      // door_concept's viewpoint: no step kind
    CHECK(affordance_words("afford_room") == "explore the room");   // room_concept's next viewpoint of the room
    CHECK(affordance_words("something_else") == "something_else");

    // ── the target line ──────────────────────────────────────────────────────────────────────────────────────────
    CHECK(target_line(std::nullopt) == "controller target now: none (it is executing no affordance)");
    {
        const std::string s = target_line(TargetFact{.affordance = "aff_door_1_cross", .dist_m = 2.34, .bearing_deg = 35});
        CHECK(s == "controller target now: cross door_1 (aff_door_1_cross), its goal 2.3 m away, ahead-left");
        const std::string u = target_line(TargetFact{.affordance = "aff_door_1_open"});
        CHECK(u == "controller target now: open door_1 (aff_door_1_open)");   // goal pose unknown: said without it
    }

    // ── the mission line: only while one runs ───────────────────────────────────────────────────────────────────
    {
        rc::mission::Progress p{.source = "chat", .title = "approach table_1", .status = "in_progress",
                                .line = "go_to pose(room_1 0.95 -0.25)"};
        CHECK(mission_now_line(p) ==
              std::optional<std::string>("controller mission now: \"approach table_1\" (from chat), running: go_to pose(room_1 0.95 -0.25)"));
        p.status = "completed";
        CHECK(not mission_now_line(p).has_value());   // finished: the last-command line reports how it ended
        CHECK(not mission_now_line(rc::mission::Progress{}).has_value());
    }

    std::println("test_controller_facts: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
