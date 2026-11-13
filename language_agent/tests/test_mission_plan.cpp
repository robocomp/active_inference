// mission_plan.h: parse/validate for the planner, name check, progress lines, reports.
#include "../src/mission_plan.h"
#include "../src/remembered_goto.h"
#include "../src/passage_refs.h"

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
using namespace rc::mission;
using SS = Interpreter::StepState;

const std::string kText = "mission \"t\"\n  cross door_1 bind=n\n  explore $n budget_s=5s\n  report \"objects here\"\n";

Progress snap(std::string status, std::vector<Interpreter::StepView> steps)
{
    Progress p;
    p.status = std::move(status);
    p.steps = std::move(steps);
    return p;
}
Interpreter::StepView view(std::string id, std::string op, SS st, std::uint64_t t0 = 0, std::uint64_t t1 = 0)
{
    return {.id = std::move(id), .op = std::move(op), .state = st, .t0 = t0, .t1 = t1, .attempt = 1};
}
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");

    // ── parse_planned ────────────────────────────────────────────────────────────────────────────────────────────
    const auto m = parse_planned(kText);
    CHECK(m.has_value());
    if (not m.has_value()) return 1;
    CHECK(m->steps.size() == 3);
    const auto from_json = parse_planned("  " + encode_json(*m));
    CHECK(from_json.has_value() and *from_json == *m);
    const auto unbound = parse_planned("mission\n  explore $x budget_s=5s\n");
    CHECK(not unbound.has_value() and unbound.error().contains("unbound"));
    CHECK(not parse_planned("mission\n  fly\n").has_value());

    // ── unknown_name ─────────────────────────────────────────────────────────────────────────────────────────────
    const auto exists = [](std::string_view n) { return n == "door_1"; };
    const auto bad = parse_planned("mission\n  explore room_9 budget_s=5s\n");
    CHECK(bad.has_value() and unknown_name(*bad, exists) == "room_9");
    CHECK(not unknown_name(*m, exists).has_value());
    const auto in_pred = parse_planned("mission\n  wait s=5s until=door_open(door_7)<0.5\n");
    CHECK(in_pred.has_value() and unknown_name(*in_pred, exists) == "door_7");
    const auto nested = parse_planned("mission\n  repeat 2\n    cross door_1\n    open door_8\n");
    CHECK(nested.has_value() and unknown_name(*nested, exists) == "door_8");
    // final review I4: `expect=ROOM` (a String field) and a passage's `@ROOM` are names too
    const auto known = [](std::string_view n) { return n == "door_1" or n == "room_1" or n == "room_2" or n == "passage_0"; };
    const auto bad_expect = parse_planned("mission\n  cross door_1 expect=room_9\n");
    CHECK(bad_expect.has_value() and unknown_name(*bad_expect, known) == "room_9");
    const auto good_expect = parse_planned("mission\n  cross door_1 expect=room_2\n");
    CHECK(good_expect.has_value() and not unknown_name(*good_expect, known).has_value());
    const auto nested_expect = parse_planned("mission\n  repeat 2\n    cross door_1 expect=room_7\n");
    CHECK(nested_expect.has_value() and unknown_name(*nested_expect, known) == "room_7");
    const auto bad_face = parse_planned("mission\n  cross passage:passage_0@room_8 expect=room_1\n");
    CHECK(bad_face.has_value() and unknown_name(*bad_face, known) == "room_8");
    const auto good_face = parse_planned("mission\n  cross passage:passage_0@room_2 expect=room_1\n");
    CHECK(good_face.has_value() and not unknown_name(*good_face, known).has_value());
    CHECK(not check_planned("mission\n  cross door_1 expect=room_9\n", known).has_value());   // never reaches a card

    // ── progress_lines ───────────────────────────────────────────────────────────────────────────────────────────
    const Progress idle = snap("in_progress", {view("1", "cross", SS::Pending), view("2", "explore", SS::Pending),
                                               view("3", "report", SS::Pending)});
    const Progress run1 = snap("in_progress", {view("1", "cross", SS::Running, 1000), view("2", "explore", SS::Pending),
                                               view("3", "report", SS::Pending)});
    const auto l1 = progress_lines(idle, run1, *m);
    CHECK(l1 == std::vector<std::string>{"→ 1/3 1:cross door_1"});
    const Progress done1 = snap("in_progress", {view("1", "cross", SS::Satisfied, 1000, 43000),
                                                view("2", "explore", SS::Pending), view("3", "report", SS::Pending)});
    const auto l2 = progress_lines(run1, done1, *m);
    CHECK(l2.size() == 1 and l2[0].starts_with("✓ 1:cross door_1 (42 s)"));
    const Progress paused = snap("paused|2", run1.steps);
    const auto l3 = progress_lines(run1, paused, *m);
    CHECK(l3.size() == 1 and l3[0].starts_with("⏸ paused at 2/3"));
    const auto l4 = progress_lines(paused, run1, *m);
    CHECK(l4.size() == 1 and l4[0].starts_with("▶ resumed: 2/3"));
    Progress failed = run1;
    failed.steps[0].state = SS::Failed;
    failed.steps[0].outcome = "door shut";
    CHECK(progress_lines(run1, failed, *m) == std::vector<std::string>{"✗ 1:cross door_1: door shut"});
    const auto mr = parse_planned("mission\n  cross door_1 retry=2\n");
    CHECK(mr.has_value());
    if (mr.has_value())
    {
        const Progress r1 = snap("in_progress", {view("1", "cross", SS::Running, 1000)});
        Progress r2 = r1;
        r2.steps[0].attempt = 2;
        CHECK(progress_lines(r1, r2, *mr) == std::vector<std::string>{"⟳ retry 1/2 1:cross door_1"});
        // the reason for the retry and why a step waits are told (chat 2026-11-12: minutes of exploring, no reason)
        Progress r3 = r2;
        r3.steps[0].after = "door_unresolved";
        CHECK(progress_lines(r1, r3, *mr) == std::vector<std::string>{"⟳ retry 1/2 1:cross door_1 (the last try: door_unresolved)"});
        Progress w1 = r1;
        w1.steps[0].note = "no door I know of in room_1 leads to room_3 yet";
        CHECK(progress_lines(r1, w1, *mr) ==
              std::vector<std::string>{"… 1:cross door_1 waits: no door I know of in room_1 leads to room_3 yet"});
        CHECK(progress_lines(w1, w1, *mr).empty());                  // told once, not every snapshot
        CHECK(progress_lines(w1, r1, *mr).empty());                  // the wait ending is not news (the step goes on)
        Progress w0 = snap("in_progress", {view("1", "cross", SS::Pending)});
        CHECK(progress_lines(w0, w1, *mr) == (std::vector<std::string>{"→ 1/1 1:cross door_1",
                                                                       "… 1:cross door_1 waits: no door I know of in room_1 leads to room_3 yet"}));
    }
    {
        Progress back = snap("in_progress", {view("1", "cross", SS::Satisfied, 5000, 1000)});
        const auto l = progress_lines(run1, back, *m);
        CHECK(l.size() == 1 and l[0] == "✓ 1:cross door_1 (0 s)");   // t1 < t0 must not underflow
    }
    CHECK(progress_lines(paused, snap("completed", run1.steps), *m).empty());   // paused -> completed is not a resume
    const auto sup = progress_lines(run1, snap("aborted|superseded", run1.steps), *m);
    CHECK(sup == std::vector<std::string>{"■ mission aborted: superseded"});
    CHECK(progress_lines(run1, snap("aborted|cancelled", run1.steps), *m) ==
          std::vector<std::string>{"■ mission cancelled"});
    CHECK(progress_lines(run1, run1, *m).empty());

    // ── new_reports ──────────────────────────────────────────────────────────────────────────────────────────────
    const Progress rep = snap("in_progress", {view("1", "cross", SS::Satisfied, 1000, 2000),
                                              view("2", "explore", SS::Satisfied, 2000, 3000),
                                              view("3", "report", SS::Satisfied, 3000, 3000)});
    CHECK(new_reports(done1, rep, *m) == std::vector<std::string>{"objects here"});
    CHECK(new_reports(rep, rep, *m).empty());

    // ── strip_fence ──────────────────────────────────────────────────────────────────────────────────────────────
    CHECK(strip_fence("```\nmission \"t\"\n  stop\n```") == "mission \"t\"\n  stop");
    CHECK(strip_fence("  ```text\nmission \"t\"\n  stop\n```\n") == "mission \"t\"\n  stop");
    CHECK(strip_fence("```json\n{\"v\":1}\n```") == "{\"v\":1}");
    CHECK(strip_fence("\n mission \"t\"\n  stop \n") == "mission \"t\"\n  stop");   // no fence: only trimmed
    CHECK(strip_fence("```\nask: which door?\n```") == "ask: which door?");
    CHECK(strip_fence("```\nmission \"t\"\n  stop") == "mission \"t\"\n  stop");   // cut before its closing fence
    CHECK(strip_fence("Here it is:\n```\nmission \"t\"\n  stop\n```\nSay yes.") == "mission \"t\"\n  stop");
    CHECK(strip_fence("").empty());
    {
        const auto fenced = parse_planned(strip_fence("```\n" + kText + "```"));
        CHECK(fenced.has_value() and *fenced == *m);
    }

    // ── planned_ask ──────────────────────────────────────────────────────────────────────────────────────────────
    CHECK(planned_ask("ask: which door?") == "which door?");
    CHECK(planned_ask("  ASK:   which door?  ") == "which door?");
    CHECK(planned_ask("```\nask: which door?\n```") == "which door?");
    CHECK(not planned_ask(kText).has_value());
    CHECK(not planned_ask("mission \"ask: x\"\n  stop").has_value());

    // ── check_planned: fence → parse → validate → names ──────────────────────────────────────────────────────────
    {
        const auto ok = check_planned("```\n" + kText + "```", exists);
        CHECK(ok.has_value() and *ok == *m);
        const auto bad_name = check_planned("mission \"t\"\n  cross door_7\n", exists);
        CHECK(not bad_name.has_value() and bad_name.error().contains("door_7"));
        const auto bad_syntax = check_planned("mission \"t\"\n  fly to the moon\n", exists);
        CHECK(not bad_syntax.has_value() and not bad_syntax.error().empty());
    }

    // ── card_still_valid: the yes is the person's NEXT message ───────────────────────────────────────────────────
    CHECK(card_still_valid(5, 6));
    CHECK(not card_still_valid(5, 7));
    CHECK(not card_still_valid(5, 5));

    // ── plan_still_wanted: a planner reply counts only if the person has not spoken since plan_mission ───────────
    CHECK(plan_still_wanted(5, 5));
    CHECK(not plan_still_wanted(5, 6));

    // ── WakeQueue: a robot event reaches the model only while it is idle, and is never dropped ───────────────────
    {
        WakeQueue w;
        CHECK(not w.pop_if_idle(false).has_value());   // nothing queued
        w.push("a");
        CHECK(not w.pop_if_idle(true).has_value());    // busy: kept
        CHECK(w.pop_if_idle(false) == "a");
        CHECK(not w.pop_if_idle(false).has_value());   // popped once
        w.push("b");
        w.push("c", "m7");
        CHECK(w.pop_if_idle(false) == "b" and w.last_fixes().empty());
        CHECK(w.pop_if_idle(false) == "c" and w.last_fixes() == "m7");   // the failure it came from
        CHECK(wake_message("x") == "(robot event, not the person) x");
    }

    // ── HeldMessages: the person typed while the model was busy (chat_e2e11: "close the door" arrived 4 s into the
    //    mission close-out and was thrown away). Held, never dropped, and answered BEFORE any robot event. ─────────
    {
        HeldMessages held;
        WakeQueue w;
        w.push("mission completed");
        held.hold("close the door");
        CHECK(not next_idle_turn(held, w, true).has_value());   // busy: both wait
        const auto t1 = next_idle_turn(held, w, false);
        CHECK(t1.has_value() and t1->from_person and t1->text == "close the door");   // the person outranks the wake
        const auto t2 = next_idle_turn(held, w, false);
        CHECK(t2.has_value() and not t2->from_person and t2->text == "mission completed");
        CHECK(not next_idle_turn(held, w, false).has_value());
        held.hold("a");
        held.hold("b");
        CHECK(held.size() == 2);
        CHECK(next_idle_turn(held, w, false)->text == "a");   // in the order typed
        held.clear();
        CHECK(held.empty());
    }

    // ── passages_to_rooms (chat_e2e11): "go to my chair" from room_2 was refused passage_unresolved — the controller
    //    resolves passage:ID@ROOM only through ltsm's LIVE passage node (alive only while the room beyond is proto).
    //    A passage into a REMEMBERED room becomes `cross <that room>`; the model still writes passages (it copies
    //    them 10/10, and `cross room_3` only 4/10). ─────────────────────────────────────────────────────────────────
    {
        rc::topo::Topology t;
        t.rooms = {{"room_1", false}, {"room_2", true}, {"room_3", false}};
        rc::topo::TopoPassageInfo p0; p0.id = "passage_0"; p0.room_a = "room_1"; p0.room_b = "room_2";
        rc::topo::TopoPassageInfo p1; p1.id = "passage_1"; p1.room_a = "room_2"; p1.room_b = "room_3";
        t.passages = {p0, p1};
        auto m = parse_planned("mission\n  cross passage:passage_0@room_2 expect=room_1\n"
                               "  repeat 2\n    cross passage:passage_1@room_2 expect=room_3\n"   // nested: rewritten too
                               "  cross passage:passage_1@room_1 expect=room_3\n"   // room_1 is not an end of passage_1
                               "  cross passage:passage_9@room_2 expect=room_1\n");               // unknown: left as is
        CHECK(m.has_value());
        if (m.has_value())
        {
            CHECK(rc::chat::passages_to_rooms(m->steps, t) == 2);
            const std::string out = rc::mission::print_text(*m);
            CHECK(out.contains("cross room_1 expect=room_1"));
            CHECK(out.contains("cross room_3 expect=room_3"));
            CHECK(out.contains("passage:passage_1@room_1"));   // passage_1 does not touch room_1: not rewritten
            CHECK(out.contains("passage:passage_9@room_2"));
        }
    }

    // ── the wakes' text ──────────────────────────────────────────────────────────────────────────────────────────
    {
        const std::string r = report_wake("objects here", "room_2", "2 objects perceived in room_2:\n- chair_1\n");
        CHECK(r.contains("objects here") and r.contains("room_2") and r.contains("chair_1"));
        Progress f = failed;   // step 1 failed, 2 and 3 never ran
        f.status = "failed|1|door shut";
        f.title = "t";
        f.serial = 12;
        const auto undone = undone_steps(f, *m);
        CHECK(undone.size() == 3 and undone[0] == "1:cross door_1" and undone[1] == "2:explore $n"
              and undone[2].starts_with("3:report"));
        CHECK(undone_steps(done1, *m).size() == 2);   // a satisfied step is done
        const auto fw = failure_wake(f, *m);
        CHECK(fw.has_value() and fw->contains("1:cross door_1") and fw->contains("door shut")
              and fw->contains("2:explore $n") and fw->contains("plan_mission"));
        CHECK(not failure_wake(run1, *m).has_value());   // not failed
        Progress c = rep;
        c.status = "completed";
        c.title = "t";
        c.t0 = 1000;
        c.t1 = 61000;
        CHECK(undone_steps(c, *m).empty());
        const auto sw = summary_wake(c, *m);
        CHECK(sw.contains("\"t\"") and sw.contains("60 s") and sw.contains("every step done"));
        // final review I7: a skipped step is named, never "every step done"
        Progress k = c;
        k.steps = {view("1", "cross", SS::Skipped), view("2", "explore", SS::Satisfied), view("3", "report", SS::Satisfied)};
        k.steps[0].outcome = "door_wont_open";
        CHECK(skipped_steps(k, *m) == std::vector<std::string>{"1:cross door_1 (door_wont_open)"});
        const auto sk = summary_wake(k, *m);
        CHECK(not sk.contains("every step done") and sk.contains("SKIPPED") and sk.contains("1:cross door_1 (door_wont_open)"));
    }

    // ── skipped steps get a line (final review I7) ───────────────────────────────────────────────────────────────
    {
        Progress a = snap("in_progress", {view("1", "cross", SS::Running), view("2", "explore", SS::Pending)});
        Progress b = snap("in_progress", {view("1", "cross", SS::Skipped), view("2", "explore", SS::Running)});
        b.steps[0].outcome = "door_wont_open";
        const auto l = progress_lines(a, b, *m);
        CHECK(l.size() == 2 and l[0] == "↷ skipped: 1:cross door_1 (door_wont_open)" and l[1].starts_with("→ 2/3"));
        CHECK(progress_lines(b, b, *m).empty());   // told once
    }

    // ── allowed_in_wake: a robot event's turn may look, stop and propose, never act (final review C1) ───────────
    {
        for (const char* t : {"stop", "cancel_passage", "plan_mission", "look", "find_objects", "where_is", "known_rooms",
                              "route_to", "can_see", "imagine_view", "imagine", "agents", "sensors", "graph_summary", "inspect", "use_skill", "episodes",
                              "episode", "outcome_stats", "last_episode", "show", "draw", "table", "hide_canvas"})
            CHECK(allowed_in_wake(t));
        for (const char* t : {"cross_door", "approach_object", "go_around_object", "move_relative", "sim_door",
                              "agent_control", "some_future_tool", ""})
            CHECK(not allowed_in_wake(t));
        CHECK(kWakeRefusal.starts_with("REFUSED: during a robot event only plan_mission/stop/cancel_passage may act"));
    }

    // ── mission_refused_note: refused before its first step (K::Rejected) vs failed while running (final review I2) ─
    {
        const auto r = mission_refused_note("failed|bad_value: unknown op 'report'", "-", "look around");
        CHECK(r.has_value() and r->contains("REFUSED") and r->contains("\"look around\"") and r->contains("unknown op 'report'"));
        CHECK(not mission_refused_note("failed|1|door_wont_open", "1:cross door_1", "t").has_value());   // ran, then failed
        CHECK(not mission_refused_note("in_progress", "-", "t").has_value());
        CHECK(not mission_refused_note("completed", "", "t").has_value());
        CHECK(mission_refused_note("failed", "-", "t").has_value());   // no reason given still says something
    }

    // ── mission_control_action: the fast path only where it means something ─────────────────────────────────────
    {
        CHECK(not mission_control_action("sigue recto", "", "").has_value());
        CHECK(not mission_control_action("espera, ¿qué ves?", "", "").has_value());
        CHECK(not mission_control_action("continue to the kitchen", "completed", "").has_value());
        const auto p = mission_control_action("pausa", "in_progress", "cross");
        CHECK(p.has_value() and p->word == "pause");
        CHECK(not mission_control_action("pausa", "paused|2", "").has_value());   // already paused: conversation
        const auto u = mission_control_action("sigue", "paused|2", "");
        CHECK(u.has_value() and u->word == "unpause");
        CHECK(not mission_control_action("sigue", "in_progress", "explore").has_value());   // moving already
        const auto r = mission_control_action("continue", "in_progress", "wait");
        CHECK(r.has_value() and r->word == "resume");
        const auto c0 = mission_control_action("cancel the mission", "", "");
        CHECK(c0.has_value() and c0->word.empty() and c0->reply == "There is no mission to cancel.");
        const auto c1 = mission_control_action("abort", "paused|1", "");
        CHECK(c1.has_value() and c1->word == "cancel");
        CHECK(not mission_control_action("what doors do you see?", "in_progress", "wait").has_value());
        Progress rw = snap("in_progress", {view("1", "repeat", SS::Running), view("1.1", "wait", SS::Running),
                                           view("2", "cross", SS::Pending)});
        CHECK(running_op(rw) == "wait");
        CHECK(running_op(idle).empty());
    }

    // ── fix_still_open: a failure's fix is planned in its own wake turn or in answer to the person's next message ─
    CHECK(fix_still_open(4, 4));
    CHECK(fix_still_open(4, 5));
    CHECK(not fix_still_open(4, 6));

    // ── planner_request: the person's VERBATIM words lead; the chat's summary only when it differs (live 10-24) ────
    {
        const std::string said = "Cross door_1 into the next room, tell me what objects are there, then cross back to room_1";
        const std::string summ = "cross door_1, explore the next room, and then cross door_1 back to room_1";
        const std::string both = planner_request(said, summ, false);
        CHECK(both.starts_with(said + "\n\n"));
        CHECK(both.contains("(the chat's request: \"" + summ + "\")"));
        CHECK(both.find(said) < both.find(summ));
        // the same words (any case, any surrounding blanks): no second line
        CHECK(planner_request(said, "  " + said + " ", false) == said);
        CHECK(planner_request(" Cross door_0. ", "cross door_0", false) == "Cross door_0.");
        // planned in a robot event's turn (a failure's fix): the person's last words are about the old mission
        CHECK(planner_request(said, "finish the report, then back to room_1", true) == "finish the report, then back to room_1");
        CHECK(planner_request("", "cross door_0", false) == "cross door_0");   // no message from the person yet
        // the person's ANSWER to the planner's ask: the words lead, the chat's request says what they answer
        CHECK(planner_request("door_1", "cross the door, report the objects, come back", false) ==
              "door_1\n\n(the chat's request: \"cross the door, report the objects, come back\")");
    }

    // ── check_planned: an object that is only REMEMBERED (live 2026-11-10: the planner wrote a bare `approach chair_34`
    //    and the controller refused it — a remembered object offers no approach until it is seen again). The lookup
    //    gives its last pose and the exact steps that bring it into view; the error IS the fix. ─────────────────────
    {
        const auto names = [](std::string_view n) { return n == "door_1" or n == "room_1" or n == "room_2" or n == "chair_34"
                                                           or n == "chair_39"; };
        // chair_34 is in the robot's room, behind it: TURN to it and wait (chat_e2e2: a go_to onto its centre had no route
        // and the policy backed up 2 m); chair_39 is in another room: drive there
        const auto remembered = [](std::string_view n) -> std::optional<RememberedPlace>
        {
            if (n == "chair_34") return RememberedPlace{.pose = {.room = "room_1", .x = -1.72, .y = -1.34},
                                                        .view_steps = {"move turn_deg=145deg", "wait s=3s"}};
            if (n == "chair_39") return RememberedPlace{.pose = {.room = "room_2", .x = -2.49, .y = -0.66},
                                                        .view_steps = {"go_to pose:room_2(-2.49, -0.66) on_fail=skip"}};
            return std::nullopt;
        };
        const auto bare = check_planned("mission \"g\"\n  approach chair_34\n", names, remembered);
        CHECK(not bare.has_value());
        CHECK(not bare.has_value() and bare.error().contains("move turn_deg=145deg"));
        CHECK(not bare.has_value() and bare.error().contains("wait s=3s"));
        CHECK(not bare.has_value() and bare.error().contains("approach chair_34 on_fail=skip"));
        const auto far = check_planned("mission \"g\"\n  approach chair_39\n", names, remembered);
        CHECK(not far.has_value() and far.error().contains("go_to pose:room_2(-2.49, -0.66) on_fail=skip"));
        // turned and waited: in view; turned without waiting: the tracker has had no time to see it
        CHECK(check_planned("mission\n  move turn_deg=145deg\n  wait s=3s\n  approach chair_34 on_fail=skip\n", names,
                            remembered).has_value());
        CHECK(not check_planned("mission\n  move turn_deg=145deg\n  approach chair_34 on_fail=skip\n", names,
                                remembered).has_value());
        // a go_to in its room also brings it into view
        CHECK(check_planned("mission\n  cross door_1 expect=room_2 retry=2\n  go_to pose:room_2(-2.49, -0.66) on_fail=skip\n"
                            "  approach chair_39 on_fail=skip\n  report \"whether chair_39 is still here\"\n", names,
                            remembered).has_value());
        // in view, but the approach would abort the mission when the chair has gone
        CHECK(not check_planned("mission\n  go_to pose:room_2(-2.49, -0.66) on_fail=skip\n  approach chair_39\n", names,
                                remembered).has_value());
        // a go_to AFTER the approach does not count; a go_to into another room does not either
        CHECK(not check_planned("mission\n  approach chair_39 on_fail=skip\n  go_to pose:room_2(-2.49, -0.66)\n", names,
                                remembered).has_value());
        CHECK(not check_planned("mission\n  go_to pose:room_1(0.5, 0.5) on_fail=skip\n  approach chair_39 on_fail=skip\n",
                                names, remembered).has_value());
        CHECK(not check_planned("mission\n  repeat 2\n    approach chair_34\n", names, remembered).has_value());   // in a body
        // a LIVE object (no remembered place) is approached as before; and without the lookup nothing changes
        const auto live = [](std::string_view) -> std::optional<RememberedPlace> { return std::nullopt; };
        CHECK(check_planned("mission\n  approach chair_34\n", names, live).has_value());
        CHECK(check_planned("mission\n  approach chair_34\n", names).has_value());
    }

    // ── go_to_object_name: "go to <name>" in the person's words → the name; anything else → nullopt ────────────────
    {
        CHECK(go_to_object_name("go to chair_34") == "chair_34");
        CHECK(go_to_object_name("Go to chair_34.") == "chair_34");
        CHECK(go_to_object_name("go back to the chair_34") == "chair_34");
        CHECK(go_to_object_name("go to where chair_34 was last seen") == "chair_34");
        CHECK(go_to_object_name("go to where chair_34 was") == "chair_34");
        CHECK(go_to_object_name("ve a chair_34") == "chair_34");
        CHECK(go_to_object_name("vuelve donde estaba chair_34") == "chair_34");
        CHECK(not go_to_object_name("go to the kitchen and look for a chair").has_value());   // more than a name
        CHECK(not go_to_object_name("where is chair_34").has_value());
        CHECK(not go_to_object_name("go to").has_value());
    }

    // ── turn_to_face_deg: the robot-frame bearing (world_digest's convention: y ahead, x right, + = to the left) ────
    {
        CHECK(turn_to_face_deg(0.0, 2.0) == 0);       // ahead
        CHECK(turn_to_face_deg(-1.0, 0.0) == 90);     // to the left
        CHECK(turn_to_face_deg(1.0, 0.0) == -90);     // to the right
        CHECK(turn_to_face_deg(0.0, -1.0) == 180);    // behind: one way round, never -180
        CHECK(turn_to_face_deg(0.6, -1.6) == -159);   // behind-right
    }

    // ── view_steps / remembered_goto_text: the recipe, built in code; it passes the same check a planner's does ────
    {
        rc::chat::RememberedObject here{.name = "chair_34", .subtype = "chair", .room = "room_1", .x = -1.72, .y = -1.34};
        here.turn_deg = 145;
        CHECK((view_steps(here, 3.0) == std::vector<std::string>{"move turn_deg=145deg", "wait s=3s"}));
        const std::string t = remembered_goto_text(here, {}, 3.0);
        CHECK(t == "mission \"go to where chair_34 was last seen\"\n"
                   "  move turn_deg=145deg\n"
                   "  wait s=3s\n"
                   "  approach chair_34 on_fail=skip\n"
                   "  report \"whether chair_34 is still here\"\n");
        const auto names = [](std::string_view n) { return n == "door_1" or n == "room_1" or n == "room_2" or n == "chair_34"
                                                           or n == "chair_39"; };
        const auto place_of = [&](const rc::chat::RememberedObject& o)
        { return RememberedPlace{.pose = {.room = o.room, .x = o.x, .y = o.y}, .view_steps = view_steps(o, 3.0)}; };
        rc::chat::RememberedObject far{.name = "chair_39", .subtype = "chair", .room = "room_2", .x = -2.49, .y = -0.66};
        const auto remembered = [&](std::string_view n) -> std::optional<RememberedPlace>
        {
            if (n == "chair_34") return place_of(here);
            if (n == "chair_39") return place_of(far);
            return std::nullopt;
        };
        CHECK(check_planned(t, names, remembered).has_value());
        // in another room (no turn from here): the legs first, each with retry=2, then go_to its last pose
        CHECK((view_steps(far, 3.0) == std::vector<std::string>{"go_to pose:room_2(-2.49, -0.66) on_fail=skip"}));
        const std::string f = remembered_goto_text(far, {"cross door_1 expect=room_2"}, 3.0);
        CHECK(f == "mission \"go to where chair_39 was last seen\"\n"
                   "  cross door_1 expect=room_2 retry=2\n"
                   "  go_to pose:room_2(-2.49, -0.66) on_fail=skip\n"
                   "  approach chair_39 on_fail=skip\n"
                   "  report \"whether chair_39 is still here\"\n");
        CHECK(check_planned(f, names, remembered).has_value());
    }

    // ── report_wake: the steps SKIPPED before the report are named (chat_e2e2: the go_to was skipped and the model said
    //    "I have arrived at the location where I last saw chair_34") ─────────────────────────────────────────────────
    {
        const std::string none = report_wake("whether chair_34 is still here", "room_1", "No object perceived");
        CHECK(not none.contains("SKIPPED"));
        const std::string r = report_wake("whether chair_34 is still here", "room_1", "No object perceived",
                                          {"1:go_to pose:room_1(-1.71, -2.61, 0, 0) (no_route)"});
        CHECK(r.contains("SKIPPED") and r.contains("1:go_to pose:room_1(-1.71, -2.61, 0, 0) (no_route)"));
        CHECK(r.contains("did not happen"));
    }

    std::println("test_mission_plan: {} checks passed{}", checks - failures, failures ? " — FAILURES" : "");
    return failures ? 1 : 0;
}
