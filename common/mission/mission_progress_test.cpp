// g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC common/mission/mission_progress_test.cpp
//     $(pkg-config --cflags --libs Qt6Core) -o /tmp/mission_progress_test && /tmp/mission_progress_test
//
// The controller → reader form of a running mission (mission_progress.h): the per-step snapshot the chat's plan view
// animates. Encoded from a real Interpreter run, decoded back, compared.
#include "mission_progress.h"
#include "mission_text.h"

#include <clocale>
#include <print>
#include <string>

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

using namespace rc::mission;
using SS = Interpreter::StepState;

/// Every leaf runs until the test says otherwise.
struct RunHost final : rc::mission::Host
{
    std::map<std::string, LeafStatus> leaf;
    std::optional<std::string> start(const Step& s) override
    {
        leaf.erase(s.id);
        return std::nullopt;
    }
    LeafStatus poll(const Step& s) override
    {
        const auto it = leaf.find(s.id);
        return it == leaf.end() ? LeafStatus{} : it->second;
    }
    void cancel(const Step&) override {}
    void say(const std::string&) override {}
    void halt() override {}
    std::optional<Pose> pose() override { return Pose{.room = "room_1", .x = 1.25, .y = -0.5, .yaw = 0.5}; }
    std::optional<double> eval(const Pred&) override { return std::nullopt; }
};
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");   // es_ES here: a decimal comma must not reach the JSON (CLAUDE.md)

    // ── state words round-trip ───────────────────────────────────────────────────────────────────────────────────
    for (const SS s : {SS::Pending, SS::Running, SS::Satisfied, SS::Failed, SS::Skipped, SS::Cancelled})
        CHECK(step_state_from(step_state_word(s)) == s);
    CHECK(not step_state_from("bogus").has_value());

    // ── the snapshot: mission identity, overall state, every step ───────────────────────────────────────────────
    {
        RunHost h;
        Interpreter in(h);
        CHECK(in.serial() == 0);
        in.start(parse_text("mission \"round\"\n  anchor here\n  repeat 2\n    cross door_1 retry=1\n  go_to anchor:here\n").value(),
                 1000);
        CHECK(in.serial() == 1);
        in.tick(1000);
        h.leaf["2.1"] = {.state = LeafState::Failed, .detail = "unreachable"};
        in.tick(1100);   // attempt 2 of 2.1
        h.leaf["2.1"] = {.state = LeafState::Succeeded};
        in.tick(1200);   // iteration 2 begins

        const std::string text = encode_progress(in, "remote");
        const auto p = decode_progress(text);
        CHECK(p.has_value());
        CHECK(p->serial == 1 and p->source == "remote" and p->title == "round");
        CHECK(p->status == "in_progress" and p->t0 == 1000 and p->t1 == 0);
        CHECK(p->line == in.progress());
        CHECK(p->steps.size() == 4);   // 1 anchor · 2 repeat · 2.1 cross · 3 go_to
        CHECK(p->steps.at(0).id == "1" and p->steps.at(0).state == SS::Satisfied);
        CHECK(p->steps.at(1).id == "2" and p->steps.at(1).op == "repeat" and p->steps.at(1).iteration == 1);
        CHECK(p->steps.at(2).id == "2.1" and p->steps.at(2).state == SS::Running and p->steps.at(2).attempt == 1);
        CHECK(p->steps.at(3).state == SS::Pending);

        in.cancel("stopped_by_user", 1300);
        const auto q = decode_progress(encode_progress(in, "remote"));
        CHECK(q->status == "aborted|stopped_by_user" and q->t1 == 1300);
        CHECK(q->steps.at(2).state == SS::Cancelled and q->steps.at(2).outcome == "stopped_by_user");
    }

    // ── an idle interpreter encodes as nothing running; garbage is refused, not half-read ──────────────────────
    {
        RunHost h;
        Interpreter in(h);
        const auto p = decode_progress(encode_progress(in, ""));
        CHECK(p.has_value() and p->serial == 0 and p->steps.empty() and p->status.empty());
        CHECK(not decode_progress("{\"steps\":[{\"id\":\"1\",\"state\":\"flying\"}]}").has_value());
        CHECK(not decode_progress("not json").has_value());
        CHECK(not decode_progress("[]").has_value());   // an object is expected
    }

    std::println("mission_progress_test: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
