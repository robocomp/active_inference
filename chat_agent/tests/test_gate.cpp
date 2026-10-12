// CommandGate rejections / acceptances and the fast paths. Build+run: tests/run_tests.sh
#include "../src/command_gate.h"
#include <cassert>
#include <print>

int main()
{
    using namespace rc::chat;
    int checks = 0;
    GateFacts f{.room = "room_1", .doors = {"door_19", "door_7"}, .nearest_door = "door_19", .controller_present = true};

    auto r = validate({.name = "cross_door", .door = "door_7"}, f);
    assert(r.ok and r.needs_confirmation and r.command == "cross door_7"); ++checks;
    r = validate({.name = "cross_door", .door = "nearest"}, f);
    assert(r.ok and r.command == "cross door_19"); ++checks;                 // nearest resolved to an explicit name
    r = validate({.name = "cross_door", .door = "door_99"}, f);
    assert(not r.ok and r.reason.find("not a door of the current room") != std::string::npos); ++checks;
    auto g = f; g.controller_present = false;
    r = validate({.name = "cross_door", .door = "door_7"}, g);
    assert(not r.ok and r.reason.find("controller") != std::string::npos); ++checks;
    g = f; g.nearest_door.reset(); g.doors.clear();
    r = validate({.name = "cross_door", .door = "nearest"}, g);
    assert(not r.ok and r.reason.find("no door") != std::string::npos); ++checks;
    g = f; g.room.clear();
    r = validate({.name = "cross_door", .door = "door_7"}, g);
    assert(not r.ok); ++checks;
    r = validate({.name = "stop"}, f);
    assert(r.ok and not r.needs_confirmation and r.command == "stop"); ++checks;
    g = f; g.controller_present = false;
    r = validate({.name = "stop"}, g);
    assert(r.ok and not r.reason.empty()); ++checks;                         // stop always goes, with a warning
    r = validate({.name = "cancel_passage"}, f);
    assert(r.ok and not r.needs_confirmation and r.command == "uncross"); ++checks;
    r = validate({.name = "cancel_passage"}, g);
    assert(not r.ok); ++checks;
    r = validate({.name = "open_fridge"}, f);
    assert(not r.ok and r.reason.find("unknown tool") != std::string::npos); ++checks;

    assert(is_stop("stop") and is_stop("Stop!") and is_stop("para ya") and is_stop("detente") and is_stop("alto")); ++checks;
    assert(not is_stop("parar") and not is_stop("please stop") and not is_stop("stopwatch")); ++checks;
    assert(is_yes("yes") and is_yes("sí") and is_yes("Si.") and is_yes("ok") and not is_yes("yes but later")); ++checks;
    assert(is_no("no") and is_no("cancel") and not is_no("now")); ++checks;

    auto d = parse_direct("cross door_19");
    assert(d and d->name == "cross_door" and d->door == "door_19"); ++checks;
    d = parse_direct("Cross the door");
    assert(d and d->door == "nearest"); ++checks;
    d = parse_direct("cruza la puerta");
    assert(d and d->door == "nearest"); ++checks;
    d = parse_direct("uncross");
    assert(d and d->name == "cancel_passage"); ++checks;
    assert(not parse_direct("can you cross the door please?")); ++checks;
    std::println("test_gate: {} checks passed", checks);
    return 0;
}
