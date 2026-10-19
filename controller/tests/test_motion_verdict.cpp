// test_motion_verdict.cpp — the WHY-IS-THE-ROBOT-NOT-MOVING verdict (src/motion_verdict.h).
// Build + run (from controller/):
//   g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-missing-field-initializers -Isrc tests/test_motion_verdict.cpp -o /tmp/tmv && /tmp/tmv
#include <print>
#include <string>

#include "../src/motion_verdict.h"

using namespace rc::why;

static int checks = 0, failures = 0;
static void check(bool ok, const std::string &what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL: {}", what);
    }
}
static bool has(const std::string &s, const std::string &sub) { return s.find(sub) != std::string::npos; }

int main()
{
    // 1. Precedence: several causes in one cycle — the earliest-declared binds, whatever the noting order.
    {
        Evidence ev;
        ev.begin();
        ev.note(Cause::NoTarget, "nothing eligible");
        ev.note(Cause::PassageWaiting, "door passage step cross: …");
        ev.note(Cause::ReturnsHold, "front return 0.20 m");
        const auto v = select(ev, false);
        check(v.cause == Cause::PassageWaiting, "passage waiting beats no-target and returns hold");
        check(v.klass() == Klass::WaitingOnAgent, "passage waiting is waiting-on-agent");
        check(not v.site.empty() and has(v.site, ":"), "the noting site is recorded");
    }
    {
        Evidence ev;
        ev.begin();
        ev.note(Cause::PassageWaiting, "x");
        ev.note(Cause::Disarmed, "output disarmed");
        ev.note(Cause::Halted, "Run not pressed");
        check(select(ev, false).cause == Cause::Halted, "halted beats disarmed beats passage");
        ev.note(Cause::LidarStalled, "stale");
        check(select(ev, false).cause == Cause::LidarStalled, "a stalled stream beats an operator halt");
        ev.note(Cause::PresenceDegraded, "missing room_concept");
        check(select(ev, false).cause == Cause::PresenceDegraded, "presence beats everything");
    }
    // Same cause twice: the FIRST note's detail is kept (stable).
    {
        Evidence ev;
        ev.begin();
        ev.note(Cause::NoRoute, "first");
        ev.note(Cause::NoRoute, "second");
        check(select(ev, false).detail == "first", "ties keep the first note");
    }
    // 2. Nothing noted: driving ⇒ Driving; not driving ⇒ UNEXPLAINED naming the stop site.
    {
        Evidence ev;
        ev.begin();
        const auto d = select(ev, true, "", "to 'aff_room_2'");
        check(d.cause == Cause::Driving and d.driving and d.detail == "to 'aff_room_2'", "driving with no note");
        ev.note_zero_site("execute_plan:5519");
        const auto u = select(ev, false, "stop:7640");
        check(u.cause == Cause::Unexplained and u.klass() == Klass::Fault, "unexplained zero command is a fault");
        check(has(u.detail, "UNEXPLAINED zero command at execute_plan:5519"), "names the branch's own site first");
        ev.begin();
        check(has(select(ev, false, "stop:7640").detail, "at stop:7640"), "falls back to the commander's stop site");
        check(has(select(ev, false).detail, "<no stop site recorded>"), "says so when no site is known");
    }
    // A limiting cause while driving keeps driving=true but names the limiter.
    {
        Evidence ev;
        ev.begin();
        ev.note(Cause::Escape, "reverse + turn LEFT");
        const auto v = select(ev, true);
        check(v.driving and v.cause == Cause::Escape, "escape: moving, cause escape");
    }
    // Context (the running mission step) rides on every verdict.
    {
        Evidence ev;
        ev.begin();
        ev.set_context("mission 2:cross door_1");
        ev.note(Cause::NoTarget, "nothing eligible");
        check(has(select(ev, false).detail, "nothing eligible · mission 2:cross door_1"), "context appended");
    }

    // 3. The deleted affordance keeps its last precondition — the try97 leg-2 case.
    {
        AffordanceMemory mem;
        const std::string cross = "aff_door_1_cross";
        mem.seen(cross, {.state = "offered", .outcome = "",
                         .precondition = "P(passable)=0.12, clear span 0.65 m at phi=1.21 rad (aperture 0.92 m)",
                         .t_ms = 1000});
        mem.absent(cross, 2000);
        mem.absent(cross, 9000);   // still gone: the "since" must not move
        check(mem.gone_since(cross) == 2000u, "gone_since is the first absent cycle");
        PassageFacts f{.door = "door_1", .step = "cross", .required = cross, .state = "not offered",
                       .reached_door = true, .opened = true, .clearing = false, .attempts = 0, .last_outcome = "",
                       .door_now = "P(passable)=0.12, clear span 0.65 m at phi=1.21 rad (aperture 0.92 m)"};
        const auto n = passage_note(f, mem, 2000 + 412000);
        check(n.has_value() and n->cause == Cause::PassageWaiting, "a withdrawn required affordance is a passage wait");
        const std::string &d = n->detail;
        check(has(d, "door passage step cross"), "names the step");
        check(has(d, "'aff_door_1_cross' withdrawn by door_concept 6m52s ago"), "names the affordance, producer, since");
        check(has(d, "last precondition: P(passable)=0.12, clear span 0.65 m"), "keeps the last precondition text");
        check(has(d, "passage will not fall back (reached door, open already done)"), "says why it will not fall back");
        std::println("  try97 leg 2 verdict: STOPPED {} — {}", cause_name(n->cause), d);

        // Re-offered: the wait clears, and the memory forgets the absence.
        mem.seen(cross, {.state = "offered", .outcome = "", .precondition = "P(passable)=0.80", .t_ms = 500000});
        check(not mem.gone_since(cross).has_value(), "seen again clears gone_since");
        f.state = "offered";
        check(not passage_note(f, mem, 500000).has_value(), "an offered step is not a passage wait");
        f.state = "executing";
        check(not passage_note(f, mem, 500000).has_value(), "an executing step is not a passage wait");
    }
    // Never offered at all.
    {
        AffordanceMemory mem;
        PassageFacts f{.door = "door_7", .step = "approach", .required = "aff_door_7_approach", .state = "not offered"};
        const auto n = passage_note(f, mem, 0);
        check(n.has_value() and has(n->detail, "has not offered it since this controller started"), "never offered");
        check(has(n->detail, "falls back once an earlier step is on offer"), "not at the door: the fallback is open");
    }
    // Completed without success: waiting for the re-offer.
    {
        AffordanceMemory mem;
        PassageFacts f{.door = "door_1", .step = "cross", .required = "aff_door_1_cross", .state = "completed",
                       .reached_door = true, .attempts = 2, .last_outcome = "unreachable"};
        const auto n = passage_note(f, mem, 0);
        check(n.has_value() and has(n->detail, "completed unreachable (attempt 2)"), "completed: outcome + attempt");
    }
    // Clearing: deliberate, not a wait.
    {
        AffordanceMemory mem;
        PassageFacts f{.door = "door_1", .step = "cross", .required = "aff_door_1_cross#through", .state = "not offered",
                       .clearing = true};
        const auto n = passage_note(f, mem, 0);
        check(n.has_value() and n->cause == Cause::PassageClearing and klass_of(n->cause) == Klass::Deliberate,
              "clearing is deliberate");
    }

    // 4. The tracker: log on change, heartbeat every 10 s while stopped, never while driving; since-when holds.
    {
        Tracker t;
        Verdict stop{.driving = false, .cause = Cause::PassageWaiting, .detail = "a"};
        auto e = t.update(stop, 1000);
        check(e.changed and has(e.log_line, "[why] STOPPED WAITING — a"), "first verdict logs");
        e = t.update(stop, 5000);
        check(not e.changed and e.log_line.empty(), "unchanged within 10 s: silent");
        stop.detail = "b";
        e = t.update(stop, 11000);
        check(not e.changed and has(e.log_line, "still STOPPED WAITING for 10.0 s — b"), "heartbeat after 10 s");
        check(t.view(13000).duration_ms == 12000 and t.view(13000).detail == "b", "duration from the first cycle");
        Verdict go{.driving = true, .cause = Cause::Driving, .detail = "to x"};
        e = t.update(go, 20000);
        check(e.changed and has(e.log_line, "MOVING DRIVING — to x (was WAITING for 19.0 s)"), "change says what it was");
        e = t.update(go, 60000);
        check(e.log_line.empty(), "no heartbeat while driving");
    }
    // 5. JSON row: escaped, one line.
    {
        Verdict v{.driving = false, .cause = Cause::NoTarget, .detail = "say \"hi\"\n", .site = "f:1"};
        const auto row = json_row(v, 42);
        check(has(row, R"("detail":"say \"hi\"\n")") and not has(row, "\n\""), "json escaped");
        check(has(row, R"("class":"waiting-on-agent")"), "json carries the class");
    }

    std::println("test_motion_verdict: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
