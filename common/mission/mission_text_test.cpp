// g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC common/mission/mission_text_test.cpp -o /tmp/mission_text_test $(pkg-config --cflags --libs Qt6Core) && /tmp/mission_text_test
//
// The DSL (mission_text.h): the people-facing syntax of the grounded mission — confirmation card, logs,
// remote_cmd.txt mission scripts. Pinned here:
//   · the §7.3 example in text and in JSON decode to the SAME mission (one IR, two syntaxes);
//   · print → parse is the identity, for every op in the registry (the DSL is generated, not hand-written);
//   · the printer's canonical form (golden text), so the card does not drift;
//   · strictness: an unknown keyword is an error (people type this), units must match the field;
//   · the same unsupported:<name> codes as the JSON codec, so the controller reports the same refusal either way.
#include "mission_json.h"
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

std::string rejected(std::string_view text)
{
    const auto m = rc::mission::parse_text(text);
    return m.has_value() ? std::string{} : m.error().status();
}

const char* kExampleJson = R"({
  "v": 1, "title": "explore room_2 and return",
  "anchors": { "start": {"room":"room_1", "x":0.42, "y":-1.10, "yaw":1.54, "sigma_xy":0.05} },
  "steps": [
    {"id":"1", "op":"cross",   "door":{"node":"door_27"}, "expect":"room_2", "on_fail":{"retry":2}},
    {"id":"2", "op":"explore", "room":{"node":"room_2"},
                               "until":{"p":"explored", "room":{"node":"room_2"}, "ge":0.9}, "budget_s":300},
    {"id":"3", "op":"cross",   "door":{"passage":"passage_0", "from":"room_2"}, "expect":"room_1",
                               "on_fail":{"retry":2}},
    {"id":"4", "op":"go_to",   "to":{"anchor":"start"}}
  ] })";

// Hand-written: comments, blank lines, units, a keyword spelled for a positional field.
const char* kExampleText = R"(mission "explore room_2 and return"
anchor start = pose:room_1(0.42, -1.10, 1.54, 0.05)

  # out through the door we know
  cross door_27 expect=room_2 retry=2
  explore room_2 until=explored(room_2)>=0.9 budget_s=300s
  cross door=passage:passage_0@room_2 expect=room_1 retry=2   # the face is found on arrival
  go_to anchor:start
)";

// What print_text must produce for that mission, byte for byte.
const char* kExampleCanonical = R"(mission "explore room_2 and return"
anchor start = pose:room_1(0.42, -1.1, 1.54, 0.05)
  cross door_27 expect=room_2 retry=2
  explore room_2 until=explored(room_2)>=0.9 budget_s=300s
  cross passage:passage_0@room_2 expect=room_1 retry=2
  go_to anchor:start
)";
}   // namespace

int main()
{
    using namespace rc::mission;
    std::setlocale(LC_ALL, "");   // es_ES: from_chars must not care (CLAUDE.md)

    // ── One IR, two syntaxes ────────────────────────────────────────────────────────────────────────────────────
    const auto from_json = decode_json(kExampleJson);
    const auto from_text = parse_text(kExampleText);
    CHECK(from_json.has_value());
    CHECK(from_text.has_value());
    if (not from_text.has_value()) std::println("  parse_text: {}", from_text.error().message());
    if (from_json.has_value() and from_text.has_value())
    {
        CHECK(from_text.value() == from_json.value());
        CHECK(print_text(from_json.value()) == kExampleCanonical);
        if (print_text(from_json.value()) != kExampleCanonical) std::print("  printed:\n{}", print_text(from_json.value()));
        const auto again = parse_text(print_text(from_json.value()));
        CHECK(again.has_value() and again.value() == from_json.value());
    }

    // ── Every op in the registry survives print → parse and JSON, and the two agree ─────────────────────────────
    {
        const char* all = R"(mission "every op"
anchor a = pose:room_1(1, 2)
requires standoff_m
  repeat 2
    seq
      say "it's \"open\""
      stop
    move distance_m=-0.5 turn_deg=90deg
  repeat until=in_room(room_2) budget_s=60s on_fail=skip
    open door_19
  anchor here
  go_to anchor:here standoff_m=0.8m
  go_to pose:room_1(0.5, -0.25, 3.14, 0.1)
  wait s=5s until=door_open(passage:passage_0@room_1)<0.5
  explore room_1 budget_s=120s
  cross door_19
  close door_19
  approach table_1 side=yp
  report "objects here"
)";
        const auto m = parse_text(all);
        CHECK(m.has_value());
        if (not m.has_value()) std::println("  every-op: {}", m.error().message());
        if (m.has_value())
        {
            std::set<std::string> seen;
            const auto collect = [&](this const auto& self, const std::vector<Step>& list) -> void
            {
                for (const auto& s : list) { seen.insert(s.op); self(s.steps); }
            };
            collect(m->steps);
            for (const auto& op : ops()) CHECK(seen.contains(std::string(op.name)));   // the fixture covers the registry
            CHECK(m->steps[0].steps[0].steps[0].string("text") == R"(it's "open")");
            CHECK(m->steps[0].steps[1].number("distance_m") == -0.5);
            CHECK(m->steps[1].on_fail.kind == OnFail::Kind::Skip);
            CHECK(m->steps[1].steps[0].id == "2.1");                                    // hierarchical ids
            const auto text_again = parse_text(print_text(*m));
            CHECK(text_again.has_value() and text_again.value() == *m);
            const auto json_again = decode_json(encode_json(*m));
            CHECK(json_again.has_value() and json_again.value() == *m);
        }
    }

    // ── Registry hygiene: no op field collides with a step-level keyword ────────────────────────────────────────
    for (const auto& op : ops())
        for (const auto& f : op.fields)
            CHECK(f.name != "retry" and f.name != "on_fail" and f.name != "bind" and f.name != "id" and f.name != "op");

    // ── Strictness and the shared refusal codes ─────────────────────────────────────────────────────────────────
    CHECK(rejected("mission\n  fly room_2\n") == "unsupported:fly");
    CHECK(rejected("mission\n  go_to region:corridor\n") == "unsupported:ref:region");
    CHECK(rejected("mission\n  wait s=5 until=battery_low(robot)\n") == "unsupported:pred:battery_low");
    CHECK(rejected("mission\n  go_to $found\n") == "unsupported:var");
    CHECK(rejected("mission\n  anchor here bind=h\n") == "unsupported:bind");
    CHECK(rejected("mission\n  cross door_19 standof_m=0.8\n") == "bad_value");        // typo: not silently dropped
    CHECK(rejected("mission\n  explore room_2 budget_s=5m\n") == "bad_value");         // wrong unit
    CHECK(rejected("mission\n  move turn_deg=90s\n") == "bad_value");
    CHECK(rejected("mission\n  repeat 2.5\n    stop\n") == "bad_type");
    CHECK(rejected("mission\n  cross expect=room_2 door_19\n") == "malformed");        // positional after keyword
    CHECK(rejected("mission\n  cross door_19 door_20\n") == "malformed");              // too many positionals
    CHECK(rejected("mission\n  stop\n    stop\n") == "bad_value");                     // stop takes no body
    CHECK(rejected("mission\n  seq\n      stop\n    stop\n") == "malformed");          // dedent to no open level
    CHECK(rejected("mission\n\tstop\n") == "malformed");                               // tabs
    CHECK(rejected("  stop\n") == "malformed");                                         // no mission line
    CHECK(rejected("mission\n  stop\nrequires speed\n") == "malformed");               // header after steps
    CHECK(rejected("mission\n") == "bad_value");                                        // nothing to do
    CHECK(rejected("mission\n  say \"unterminated\n") == "malformed");
    CHECK(rejected("mission\n  cross passage:passage_0\n") == "missing_field");         // the face needs its room
    CHECK(rejected("mission\n  wait s=5 until=door_open(door_19)\n") == "missing_field");   // state the decision
    CHECK(rejected("mission\n  wait s=5 until=in_room(room_2)>=0.5\n") == "bad_value");     // a fact has none
    CHECK(rejected("mission\n  stop retry=0\n") == "bad_value");
    CHECK(rejected("mission\n  stop\n").empty());
    CHECK(rejected("mission \"t\"\n  stop  # trailing comment\n\n").empty());

    // ── close (2026-11-01, user: the PLAN triggers door_concept's close affordance, not the door passage) ─────────────────
    CHECK(rejected("mission\n  cross room_2\n  close room_1\n").empty());   // the door of this room back to room_1
    CHECK(rejected("mission\n  close door_7\n").empty());
    CHECK(rejected("mission\n  close\n") == "missing_field");
    if (const auto m = parse_text("mission\n  close room_1\n"); m.has_value())
        CHECK(m->steps[0].op == "close" and m->steps[0].ref("door").has_value() and m->steps[0].ref("door")->name == "room_1");
    else
        CHECK(false);

    // ── bind / $NAME: a cross names the room it lands in; a later step may use it ───────────────────────────────
    {
        const std::string bound = "mission \"t\"\n  cross door_1 bind=next\n  explore $next budget_s=10s\n  report \"objects here\"\n";
        CHECK(rejected(bound).empty());
        const auto m = parse_text(bound);
        CHECK(m.has_value() and print_text(m.value()) == bound);
        CHECK(rejected("mission\n  explore $next budget_s=10s\n") == "unbound");
        const auto u = parse_text("mission\n  explore $next budget_s=10s\n");
        CHECK(not u.has_value() and u.error().text.find("next") != std::string::npos);
        CHECK(rejected("mission\n  explore room_1 budget_s=10s bind=x\n") == "unsupported:bind");
        CHECK(rejected("mission\n  report\n") == "missing_field");
    }

    // ── Locale: decimals parse the same under es_ES ─────────────────────────────────────────────────────────────
    if (const auto m = parse_text("mission\n  move distance_m=-1.25 turn_deg=1e1\n"); m.has_value())
        CHECK(m->steps[0].number("distance_m") == -1.25 and m->steps[0].number("turn_deg") == 10.0);
    else
        CHECK(false);

    // ── wait at= (tense spec 2026-11-14 §1): an absolute local date-time with offset; round trip; a bare clock rejected ──
    {
        const auto m = parse_text("mission\n  wait at=\"2026-11-14T18:00:00+01:00\"\n  cross door_1 retry=2\n");
        CHECK(m.has_value());
        if (m.has_value())
        {
            CHECK(m->steps[0].string("at") == "2026-11-14T18:00:00+01:00");
            CHECK(print_text(*m).contains("wait at=\"2026-11-14T18:00:00+01:00\""));
            const auto again = parse_text(print_text(*m));
            CHECK(again.has_value() and *again == *m);
            const auto j = decode_json(encode_json(*m));
            CHECK(j.has_value() and *j == *m);
        }
        const auto bare = parse_text("mission\n  wait at=\"18:00\"\n");
        CHECK(not bare.has_value() and bare.error().code == "bad_value");
        CHECK(not parse_text("mission\n  wait at=\"2026-11-14T25:00:00+01:00\"\n").has_value());
        CHECK(not parse_text("mission\n  wait at=\"2026-11-14T18:00:00\"\n").has_value());   // no offset
        CHECK(parse_text("mission\n  wait at=\"2026-11-14T18:00:00Z\" until=door_open(door_1)>=0.8\n").has_value());
    }
    // ── mission_time.h ──
    {
        CHECK(parse_iso("1970-01-01T00:00:00Z") == 0);
        CHECK(parse_iso("2026-11-14T18:00:00+01:00") == parse_iso("2026-11-14T17:00:00Z"));
        CHECK(parse_iso("2024-02-29T12:00:00-03:30").has_value() and not parse_iso("18:00").has_value());
        CHECK(*parse_iso("2026-11-15T00:00:00Z") - *parse_iso("2026-11-14T00:00:00Z") == 86'400'000);
        CHECK(parse_clock("18:00") == 64800 and parse_clock("07:05:09") == 25509);
        CHECK(not parse_clock("24:00").has_value() and not parse_clock("18h") .has_value() and not parse_clock("8:00").has_value());
        CHECK(iso_hhmm("2026-11-14T18:05:00+01:00") == "18:05" and iso_hhmm("x").empty());
        const std::int64_t now = *parse_iso("2026-11-14T00:31:00Z");
        const auto g = ground_clock("18:00", now);   // in this machine's zone, AHEAD of now and within a day
        CHECK(g.has_value() and parse_iso(*g).has_value() and *parse_iso(*g) > now and *parse_iso(*g) - now <= 86'400'000);
        CHECK(g.has_value() and g->substr(11, 5) == "18:00");
        const auto past = ground_clock(format_iso_local(now).substr(11, 5), now + 60'000);   // a minute ago: tomorrow
        CHECK(past.has_value() and *parse_iso(*past) > now + 60'000 and *parse_iso(*past) - now > 86'000'000);
        CHECK(ground_clock("2026-11-14T18:00:00+01:00", now) == "2026-11-14T18:00:00+01:00");   // ISO kept
        CHECK(not ground_clock("soon", now).has_value());
        CHECK(parse_iso(format_iso_local(now)) == now);
    }

    std::println("mission_text_test: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
