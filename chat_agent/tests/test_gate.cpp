// CommandGate rejections / acceptances and the fast paths. Build+run: tests/run_tests.sh
#include "../src/command_gate.h"
#include <cassert>
#include <clocale>
#include <cmath>
#include <print>

int main()
{
    using namespace rc::chat;
    std::setlocale(LC_ALL, "");   // the agent runs under es_ES (decimal comma): parsing must not care (CLAUDE.md)
    int checks = 0;
    GateFacts f{.room = "room_1", .doors = {"door_19", "door_7"}, .nearest_door = "door_19", .controller_present = true};

    auto r = validate({.name = "cross_door", .door = "door_7"}, f);
    assert(r.ok and r.command == "cross door_7"); ++checks;
    r = validate({.name = "cross_door", .door = "nearest"}, f);
    assert(not r.ok and r.reason.find("2 doors") != std::string::npos); ++checks;   // ambiguous: two doors, ask which
    {
        auto one = f; one.doors = {"door_19"};
        r = validate({.name = "cross_door", .door = "nearest"}, one);
        assert(r.ok and r.command == "cross door_19"); ++checks;            // one door: resolved to an explicit name
    }
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
    assert(r.ok and r.command == "stop"); ++checks;
    g = f; g.controller_present = false;
    r = validate({.name = "stop"}, g);
    assert(r.ok and not r.reason.empty()); ++checks;                         // stop always goes, with a warning
    r = validate({.name = "cancel_passage"}, f);
    assert(r.ok and r.command == "uncross"); ++checks;
    r = validate({.name = "cancel_passage"}, g);
    assert(not r.ok); ++checks;
    r = validate({.name = "open_fridge"}, f);
    assert(not r.ok and r.reason.find("unknown tool") != std::string::npos); ++checks;

    assert(is_stop("stop") and is_stop("Stop!") and is_stop("para ya") and is_stop("detente") and is_stop("alto")); ++checks;
    assert(not is_stop("parar") and not is_stop("please stop") and not is_stop("stopwatch")); ++checks;

    auto d = parse_direct("cross door_19");
    assert(d and d->name == "cross_door" and d->door == "door_19"); ++checks;
    d = parse_direct("Cross the door");
    assert(d and d->door == "nearest"); ++checks;
    d = parse_direct("cruza la puerta");
    assert(d and d->door == "nearest"); ++checks;
    d = parse_direct("uncross");
    assert(d and d->name == "cancel_passage"); ++checks;
    assert(not parse_direct("can you cross the door please?")); ++checks;
    // ── move_relative ──
    {
        const GateLimits lim;   // default max 3.0 m; every accepted move is sent at once (no card, 2026-10-18)
        r = validate({.name = "move_relative", .distance_m = 0.3}, f, lim);
        assert(r.ok and r.command == "move 0.30"); ++checks;
        r = validate({.name = "move_relative", .distance_m = -0.5}, f, lim);
        assert(r.ok and r.command == "move -0.50"); ++checks;     // backing up a bit
        r = validate({.name = "move_relative", .distance_m = 1.5}, f, lim);
        assert(r.ok and r.command == "move 1.50"); ++checks;
        r = validate({.name = "move_relative", .distance_m = 3.5}, f, lim);
        assert(not r.ok and r.reason.find("at most 3.0 m") != std::string::npos); ++checks;   // beyond max: refused
        r = validate({.name = "move_relative", .turn_deg = 180}, f, lim);
        assert(r.ok and r.command == "turn 180"); ++checks;
        r = validate({.name = "move_relative", .turn_deg = -90}, f, lim);
        assert(r.ok and r.command == "turn -90"); ++checks;
        r = validate({.name = "move_relative", .turn_deg = 360}, f, lim);
        assert(r.ok and r.command == "turn 360"); ++checks;
        r = validate({.name = "move_relative", .turn_deg = 400}, f, lim);
        assert(not r.ok); ++checks;
        r = validate({.name = "move_relative", .distance_m = 0.5, .turn_deg = 90}, f, lim);
        assert(r.ok and r.command == "turn 90 move 0.50"); ++checks;   // turn THEN move
        r = validate({.name = "move_relative", .distance_m = 0.5, .turn_deg = 180}, f, lim);
        assert(r.ok and r.command == "turn 180 move 0.50"); ++checks;
        r = validate({.name = "move_relative"}, f, lim);
        assert(not r.ok and r.reason.find("nothing to do") != std::string::npos); ++checks;
        r = validate({.name = "move_relative", .distance_m = 0.004}, f, lim);
        assert(not r.ok); ++checks;                                                            // rounds to 0 cm
        r = validate({.name = "move_relative", .distance_m = std::nan("")}, f, lim);
        assert(not r.ok); ++checks;
        auto nc = f; nc.controller_present = false;
        r = validate({.name = "move_relative", .distance_m = 0.3}, nc, lim);
        assert(not r.ok and r.reason.find("controller") != std::string::npos); ++checks;
        nc = f; nc.room.clear();
        r = validate({.name = "move_relative", .distance_m = 0.3}, nc, lim);
        assert(not r.ok); ++checks;
        const GateLimits tight{.max_move_m = 1.0};
        r = validate({.name = "move_relative", .distance_m = 1.5}, f, tight);
        assert(not r.ok and r.reason.find("at most 1.0 m") != std::string::npos); ++checks;   // limit comes from config

        d = parse_direct("move 0.5");
        assert(d and d->name == "move_relative" and d->distance_m == 0.5 and d->turn_deg == 0); ++checks;
        d = parse_direct("Move -0.3 m");
        assert(d and d->distance_m == -0.3); ++checks;
        d = parse_direct("turn 90");
        assert(d and d->turn_deg == 90 and d->distance_m == 0); ++checks;
        d = parse_direct("turn -45 then move 1");
        assert(d and d->turn_deg == -45 and d->distance_m == 1); ++checks;
        assert(not parse_direct("move a bit")); ++checks;                                      // a sentence: the model's
        assert(not parse_direct("turn around")); ++checks;
        assert(parse_number("0.25") == 0.25 and parse_number("+1") == 1.0 and not parse_number("1,5")); ++checks;
        assert(explain_relative_command("move -0.30") == "move -0.30 (move backward 0.30 m)"); ++checks;
        assert(explain_relative_command("turn 90 move 0.50") == "turn 90 move 0.50 (turn left 90 deg, then move forward 0.50 m)"); ++checks;
    }
    {
        // The end-of-command line the model reads (live 2026-10-18: a relative move failed against a wall, and the model
        // then refused "cross it into next room" for ever, the move's failure read as blocking the door).
        const std::string fail = robot_status_text("turn -126 move 3.00",
            "failed|a wall or obstacle blocks the way ahead: only 1.95 m of the 3.00 m are clear",
            "a wall or obstacle blocks the way ahead: only 1.95 m of the 3.00 m are clear");
        assert(fail.starts_with("The command 'turn -126 move 3.00' (turn right 126 deg, then move forward 3.00 m) "
                                "ended: failed — a wall or obstacle blocks the way ahead: only 1.95 m of the 3.00 m "
                                "are clear.")); ++checks;
        assert(fail.find("1.95 m of the 3.00 m are clear", fail.find("clear") + 1) == std::string::npos); ++checks;   // said once
        assert(fail.find("only this straight-line move") != std::string::npos
               and fail.find("cross_door") != std::string::npos); ++checks;
        const std::string cross = robot_status_text("cross door_1", "failed|door stayed shut", "door stayed shut");
        assert(cross == "The command 'cross door_1' ended: failed — door stayed shut."); ++checks;   // not a relative move
        assert(robot_status_text("cross door_1", "completed", "") == "The command 'cross door_1' ended: completed."); ++checks;
        assert(robot_status_text("move 0.30", "completed", "arrived") ==
               "The command 'move 0.30' (move forward 0.30 m) ended: completed — arrived."); ++checks;   // success: no caveat
    }
    {
        // A door deleted from the map (live 2026-10-18: the person deleted door_3 while a crossing of it was running;
        // the model kept describing door_3 from its own earlier reply, and the STATE's command line still named it).
        const std::vector<std::string> before{"door_1", "door_3"}, after{"door_1"};
        assert(doors_gone(before, after) == std::vector<std::string>{"door_3"}); ++checks;
        assert(doors_gone(after, before).empty()); ++checks;                       // a new door is not a gone one
        assert(command_target_gone("cross door_3", after) == std::optional<std::string>("door_3")); ++checks;
        assert(not command_target_gone("cross door_1", after)); ++checks;
        assert(not command_target_gone("move 0.30", after)); ++checks;             // not a door command
    }
    std::println("test_gate: {} checks passed", checks);
    return 0;
}
