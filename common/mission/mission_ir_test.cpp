// g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC common/mission/mission_ir_test.cpp -o /tmp/mission_ir_test $(pkg-config --cflags --libs Qt6Core) && /tmp/mission_ir_test
//
// The grounded mission IR (mission_ir.h) and its wire codec (mission_json.h), pinned without a graph or an LLM.
// What is pinned is the CONTRACT between chat_agent and the controller (docs/mission_language_2026-10-13.md §7):
//   · a mission round-trips through JSON unchanged;
//   · compatibility: an unknown op / ref kind / predicate rejects the WHOLE mission, an unknown optional field is
//     ignored, a field named in `requires` that this build does not know rejects, `v` must match;
//   · v1 reservations: step ids, `bind`, `{"var":…}`, `prefs` are in the format but rejected by v1 validation;
//   · termination is checkable before running: every loop (repeat-until, explore) carries a budget;
//   · probabilistic predicates carry their decision boundary, fact predicates do not.
#include "mission_json.h"

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

/// The error code a JSON text is rejected with, or "" when it decodes.
std::string rejected(std::string_view json)
{
    const auto m = rc::mission::decode_json(json);
    return m.has_value() ? std::string{} : m.error().status();
}

/// A minimal valid mission around one step body (the step's JSON without braces).
std::string one_step(std::string_view step_body)
{
    return std::format(R"({{"v":1,"steps":[{{"id":"1",{}}}]}})", step_body);
}

const char* kExample = R"({
  "v": 1,
  "title": "explore room_2 and return",
  "anchors": { "start": {"room":"room_1", "x":0.42, "y":-1.10, "yaw":1.54, "sigma_xy":0.05} },
  "steps": [
    {"id":"1", "op":"cross",   "door":{"node":"door_27"}, "expect":"room_2", "on_fail":{"retry":2}},
    {"id":"2", "op":"explore", "room":{"node":"room_2"},
                               "until":{"p":"explored", "room":{"node":"room_2"}, "ge":0.9}, "budget_s":300},
    {"id":"3", "op":"cross",   "door":{"passage":"passage_0", "from":"room_2"}, "expect":"room_1",
                               "on_fail":{"retry":2}},
    {"id":"4", "op":"go_to",   "to":{"anchor":"start"}}
  ] })";
}   // namespace

int main()
{
    using namespace rc::mission;
    std::setlocale(LC_ALL, "");   // the agents run under es_ES (decimal comma): the codec must not care (CLAUDE.md)

    // ── The §7.3 example decodes into the expected tree ──────────────────────────────────────────────────────────
    const auto ex = decode_json(kExample);
    CHECK(ex.has_value());
    if (ex.has_value())
    {
        const Mission& m = ex.value();
        CHECK(m.v == 1 and m.title == "explore room_2 and return");
        CHECK(m.anchors.contains("start") and m.anchors.at("start").room == "room_1");
        CHECK(m.anchors.at("start").x == 0.42 and m.anchors.at("start").y == -1.10);   // es_ES must not truncate
        CHECK(m.steps.size() == 4);
        const Step& s1 = m.steps[0];
        CHECK(s1.id == "1" and s1.op == "cross");
        CHECK(s1.ref("door").has_value() and s1.ref("door")->kind == Ref::Kind::Node and s1.ref("door")->name == "door_27");
        CHECK(s1.string("expect") == "room_2");
        CHECK(s1.on_fail.kind == OnFail::Kind::Retry and s1.on_fail.retries == 2);
        const Step& s2 = m.steps[1];
        CHECK(s2.number("budget_s") == 300.0);
        CHECK(s2.pred("until").has_value() and s2.pred("until")->name == "explored");
        CHECK(s2.pred("until")->cmp.has_value() and s2.pred("until")->cmp->op == Comparison::Op::Ge
              and s2.pred("until")->cmp->value == 0.9);
        const auto door3 = m.steps[2].ref("door");
        CHECK(door3.has_value() and door3->kind == Ref::Kind::Passage and door3->name == "passage_0"
              and door3->from == "room_2");
        CHECK(m.steps[3].ref("to")->kind == Ref::Kind::Anchor and m.steps[3].ref("to")->name == "start");
        CHECK(m.steps[3].on_fail.kind == OnFail::Kind::Abort);   // the default

        // ── Round trip: encode → decode gives the same mission ──────────────────────────────────────────────────
        const auto again = decode_json(encode_json(m));
        CHECK(again.has_value() and again.value() == m);
    }

    // ── Resources come from the registry as a SET (one owner per resource; §7.4 point 3, §7.6) ─────────────────
    CHECK(op_spec("cross") != nullptr and op_spec("cross")->claims == kBase and not op_spec("cross")->instant);
    CHECK(op_spec("wait") != nullptr and op_spec("wait")->claims == kBase);     // holding still is owning the base
    CHECK(op_spec("say") != nullptr and op_spec("say")->claims == kNone and op_spec("say")->instant);
    CHECK(op_spec("repeat") != nullptr and op_spec("repeat")->claims == kNone and op_spec("repeat")->has_body);
    CHECK(op_spec("fly") == nullptr);
    CHECK((kBase & kArm) == 0 and (kArm & kGripper) == 0 and (kBase & kGripper) == 0);   // distinct resources
    {
        Step body{.op = "seq"};
        body.steps = {Step{.op = "say"}, Step{.op = "cross"}};
        Step rep{.op = "repeat"};
        rep.steps = {body};
        CHECK(claims_of(rep) == kBase);                       // a composite claims what its body claims
        CHECK(claims_of(Step{.op = "say"}) == kNone);
        CHECK(claims_of(Step{.op = "fly"}) == kNone);          // unknown op: nothing (validate rejects it anyway)
    }

    // ── Compatibility rules (§7.1) ──────────────────────────────────────────────────────────────────────────────
    CHECK(rejected(R"({"v":2,"steps":[{"id":"1","op":"stop"}]})") == "unsupported_version");
    CHECK(rejected(R"({"steps":[{"id":"1","op":"stop"}]})") == "malformed");                  // v is mandatory
    CHECK(rejected(one_step(R"("op":"fly")")) == "unsupported:fly");                          // unknown op
    CHECK(rejected(one_step(R"("op":"go_to","to":{"region":"corridor"})")) == "unsupported:ref:region");
    CHECK(rejected(one_step(R"("op":"explore","room":{"node":"room_2"},"budget_s":60,)"
                            R"("until":{"p":"battery_low"})")) == "unsupported:pred:battery_low");
    CHECK(rejected(one_step(R"("op":"stop","speed":"slow")")).empty());                       // unknown optional: ignored
    CHECK(rejected(R"({"v":1,"requires":["speed"],"steps":[{"id":"1","op":"stop","speed":"slow"}]})")
          == "unsupported:speed");                                                            // …unless required
    CHECK(rejected(R"({"v":1,"requires":["standoff_m"],"steps":[{"id":"1","op":"stop"}]})").empty());   // known
    // a rejection is WHOLE-mission: one unknown op after valid steps still rejects
    CHECK(rejected(R"({"v":1,"steps":[{"id":"1","op":"stop"},{"id":"2","op":"fly"}]})") == "unsupported:fly");

    // ── Structure ───────────────────────────────────────────────────────────────────────────────────────────────
    CHECK(rejected("{not json") == "malformed");
    CHECK(rejected(R"({"v":1,"steps":[]})") == "bad_value");                                  // nothing to do
    CHECK(rejected(R"({"v":1,"steps":[{"op":"stop"}]})") == "missing_field");                 // id on every step
    CHECK(rejected(R"({"v":1,"steps":[{"id":"1","op":"stop"},{"id":"1","op":"stop"}]})") == "duplicate_id");
    CHECK(rejected(one_step(R"("op":"cross")")) == "missing_field");                          // door is required
    CHECK(rejected(one_step(R"("op":"say","text":3)")) == "bad_type");
    CHECK(rejected(one_step(R"("op":"cross","door":{"anchor":"start"})")) == "bad_value");    // wrong ref kind
    CHECK(rejected(one_step(R"("op":"cross","door":{"passage":"passage_0"})")) == "missing_field");   // needs from
    CHECK(rejected(one_step(R"("op":"go_to","to":{"node":"a","anchor":"b"})")) == "malformed");      // two kinds
    CHECK(rejected(one_step(R"("op":"stop","steps":[{"id":"1.1","op":"stop"}])")) == "bad_value");   // no body
    CHECK(rejected(one_step(R"("op":"seq","steps":[])")) == "bad_value");                    // empty body
    CHECK(rejected(one_step(R"("op":"seq","steps":[{"id":"1.1","op":"stop"}])")).empty());
    CHECK(rejected(one_step(R"("op":"seq","steps":[{"id":"1","op":"stop"}])")) == "duplicate_id");   // across depth
    CHECK(rejected(one_step(R"("op":"stop","on_fail":"skip")")).empty());
    // approach (2026-10-18): the object's own approach affordance, optionally from a side it offers
    CHECK(rejected(one_step(R"("op":"approach","object":{"node":"table_1"})")).empty());
    CHECK(rejected(one_step(R"("op":"approach","object":{"node":"table_1"},"side":"yp")")).empty());
    CHECK(rejected(one_step(R"("op":"approach")")) == "missing_field");
    CHECK(rejected(one_step(R"("op":"approach","object":{"anchor":"start"})")) == "bad_value");   // a pose is go_to's
    CHECK(op_spec("approach") != nullptr and op_spec("approach")->claims == kBase and not op_spec("approach")->instant);
    CHECK(rejected(one_step(R"("op":"stop","on_fail":{"retry":0})")) == "bad_value");
    CHECK(rejected(one_step(R"("op":"stop","on_fail":"pray")")) == "bad_value");

    // ── v1 reservations (§7.4): in the format, refused by v1 validation ─────────────────────────────────────────
    CHECK(rejected(one_step(R"("op":"anchor","name":"here","bind":"h")")) == "unsupported:bind");
    CHECK(rejected(one_step(R"("op":"go_to","to":{"var":"h"})")) == "unsupported:var");
    CHECK(rejected(R"({"v":1,"prefs":{"avoid:room_3":-2},"steps":[{"id":"1","op":"stop"}]})") == "unsupported:prefs");
    CHECK(rejected(R"({"v":1,"prefs":{},"steps":[{"id":"1","op":"stop"}]})").empty());

    // ── Termination is checkable before running ─────────────────────────────────────────────────────────────────
    CHECK(rejected(one_step(R"("op":"explore","room":{"node":"room_2"})")) == "missing_field");   // budget_s
    CHECK(rejected(one_step(R"("op":"explore","room":{"node":"room_2"},"budget_s":0)")) == "bad_value");
    CHECK(rejected(one_step(R"("op":"repeat","steps":[{"id":"1.1","op":"stop"}])")) == "bad_value");   // times|until
    CHECK(rejected(one_step(R"("op":"repeat","times":2,"until":{"p":"in_room","room":{"node":"r"}},)"
                            R"("budget_s":9,"steps":[{"id":"1.1","op":"stop"}])")) == "bad_value");   // not both
    CHECK(rejected(one_step(R"("op":"repeat","until":{"p":"in_room","room":{"node":"r"}},)"
                            R"("steps":[{"id":"1.1","op":"stop"}])")) == "missing_field");    // until needs budget
    CHECK(rejected(one_step(R"("op":"repeat","times":2.5,"steps":[{"id":"1.1","op":"stop"}])")) == "bad_type");
    CHECK(rejected(one_step(R"("op":"repeat","times":2,"steps":[{"id":"1.1","op":"stop"}])")).empty());
    CHECK(rejected(one_step(R"("op":"wait")")) == "bad_value");                               // wait for what?
    CHECK(rejected(one_step(R"("op":"wait","s":5)")).empty());
    CHECK(rejected(one_step(R"("op":"move")")) == "bad_value");
    CHECK(rejected(one_step(R"("op":"move","distance_m":-0.5)")).empty());                    // backwards is fine

    // ── Predicates: probabilistic ones carry the boundary, facts do not ─────────────────────────────────────────
    const auto with_until = [](std::string_view pred)
    { return one_step(std::format(R"("op":"wait","s":30,"until":{})", pred)); };
    CHECK(rejected(with_until(R"({"p":"door_open","door":{"node":"door_19"}})")) == "missing_field");
    CHECK(rejected(with_until(R"({"p":"door_open","door":{"node":"door_19"},"lt":0.5})")).empty());
    CHECK(rejected(with_until(R"({"p":"door_open","door":{"node":"door_19"},"lt":0.5,"ge":0.2})")) == "bad_value");
    CHECK(rejected(with_until(R"({"p":"door_open","door":{"node":"door_19"},"ge":1.5})")) == "bad_value");
    CHECK(rejected(with_until(R"({"p":"in_room","room":{"node":"room_2"},"ge":0.5})")) == "bad_value");
    CHECK(rejected(with_until(R"({"p":"in_room","room":{"node":"room_2"}})")).empty());

    // ── Anchors must be defined before use (references resolve before anything moves) ───────────────────────────
    CHECK(rejected(one_step(R"("op":"go_to","to":{"anchor":"nowhere"})")) == "bad_value");
    CHECK(rejected(R"({"v":1,"steps":[{"id":"1","op":"anchor","name":"here"},)"
                   R"({"id":"2","op":"go_to","to":{"anchor":"here"}}]})").empty());
    CHECK(rejected(R"({"v":1,"steps":[{"id":"1","op":"go_to","to":{"anchor":"here"}},)"
                   R"({"id":"2","op":"anchor","name":"here"}]})") == "bad_value");             // used before defined

    // ── assign_ids: hierarchical ids for a resolver-built tree ──────────────────────────────────────────────────
    {
        Mission m;
        Step seq{.op = "seq"};
        seq.steps = {Step{.op = "stop"}, Step{.op = "stop"}};
        m.steps = {Step{.op = "stop"}, seq};
        assign_ids(m);
        CHECK(m.steps[0].id == "1" and m.steps[1].id == "2");
        CHECK(m.steps[1].steps[0].id == "2.1" and m.steps[1].steps[1].id == "2.2");
        CHECK(not validate(m).has_value());
    }

    std::println("mission_ir_test: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
