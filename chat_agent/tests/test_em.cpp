// The episodic-memory tools' answers (src/em_answers.h): pure formatting over a fixed rc::em::Store, the unavailable
// case, the HH:MM / epoch-ms time arguments and the STATE's "today:" line. No DSR, no ollama.
#include "../src/em_answers.h"

#include <chrono>
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

bool has(const std::string& s, std::string_view part) { return s.find(part) != std::string::npos; }
void show(std::string_view label, const std::string& s) { std::println("--- {} ---\n{}", label, s); }

using namespace rc::em;
using namespace rc::chat;

// Wall-clock ms of today's local HH:MM (what the tool argument "HH:MM" means).
std::uint64_t at(int h, int m)
{
    using namespace std::chrono;
    const auto* z = current_zone();
    const auto today = floor<days>(z->to_local(system_clock::now()));
    return std::uint64_t(duration_cast<milliseconds>(
        z->to_sys(today + hours{h} + minutes{m}, choose::earliest).time_since_epoch()).count());
}

Record rec(std::string id, std::string parent, Level lv, Author au, std::string kind, std::string target,
           std::uint64_t t0, std::optional<std::uint64_t> t1 = std::nullopt, std::string outcome = {})
{
    Record r;
    r.id = std::move(id); r.parent = std::move(parent); r.level = lv; r.author = au;
    r.kind = std::move(kind); r.target = std::move(target); r.t0 = t0; r.t1 = t1; r.outcome = std::move(outcome);
    return r;
}

Store fixed()
{
    Store s;
    Record m1 = rec("20261017-1", "", Level::Mission, Author::Human, "mission", "", at(10, 0), at(10, 3), "succeeded");
    m1.action_json = R"({"title":"cross door_27"})"; m1.room = "room_2"; m1.source = "chat";
    Record s1 = rec("20261017-2", "20261017-1", Level::Step, Author::Human, "cross", "door_27", at(10, 0), at(10, 3),
                    "succeeded");
    s1.target_ltsm = "passage_0"; s1.attempt = 2;
    s1.pre.push_back(Fact{.node = "door_27", .type = "door", .ltsm_id = "passage_0",
                          .values = {{"door_open_prob", 0.12}, {"x", 1.5}}});
    s1.post.push_back(Fact{.node = "door_27", .type = "door", .ltsm_id = "passage_0",
                           .values = {{"door_open_prob", 0.93}, {"x", 1.5}}});
    s1.events.push_back(Event{.dt_ms = 12000, .kind = "current_moved", .node = "room_3"});
    Record c1 = rec("20261017-3", "20261017-2", Level::Claim, Author::Self, "open", "door_27", at(10, 1), at(10, 2), "kept");
    c1.promised = 0.5; c1.delivered = 0.25;
    Record m2 = rec("20261017-4", "", Level::Mission, Author::Human, "mission", "", at(11, 0), at(11, 5), "failed");
    m2.action_json = "go to the kitchen"; m2.room = "room_3"; m2.source = "chat";
    // A self mission as em_builder.h opens it (ruling R10): kind "mission", no action, no target; room at t0.
    Record m3 = rec("20261017-5", "", Level::Mission, Author::Self, "mission", "", at(12, 0));
    m3.room = "room_5"; m3.source = "self";
    s.today = {m1, s1, c1, m2, m3};

    Stats cross{.key_kind = "cross", .key_identity = "", .outcomes = {{"succeeded", 9}, {"failed", 3}}, .n = 12,
                .dur_mean_s = 20.0, .dur_m2 = 44.0, .n_gain = 0, .promised_sum = 0, .delivered_sum = 0,
                .first_t0 = at(9, 0), .last_t0 = at(10, 0)};
    Stats cross27 = cross;
    cross27.key_identity = "passage_0"; cross27.outcomes = {{"succeeded", 4}, {"lost", 1}}; cross27.n = 5;
    Stats open{.key_kind = "open", .key_identity = "", .outcomes = {{"kept", 2}}, .n = 2, .dur_mean_s = 4.0,
               .dur_m2 = 0.0, .n_gain = 2, .promised_sum = 1.0, .delivered_sum = 0.5, .first_t0 = 0, .last_t0 = 0};
    s.stats = {cross, cross27, open};
    return s;
}
}   // namespace

int main()
{
    // These run inside a Qt program under es_ES (decimal COMMA); the parsing must not care (CLAUDE.md).
    if (not std::setlocale(LC_ALL, "es_ES.UTF-8")) std::setlocale(LC_ALL, "");

    const std::optional<Store> st = fixed();
    const std::optional<Store> absent;
    const std::optional<Store> empty = Store{};

    // ── unavailable: no EM graph joined, or joined but holding nothing ──
    for (const auto* s : {&absent, &empty})
    {
        CHECK(answer_episodes(*s, {}, {}, "", "", "") == kEmUnavailable);
        CHECK(answer_episode(*s, "20261017-1") == kEmUnavailable);
        CHECK(answer_outcome_stats(*s, "cross", "") == kEmUnavailable);
        CHECK(answer_last_episode(*s, "door_27") == kEmUnavailable);
    }
    CHECK(std::string(kEmUnavailable) == "episodic memory is not available");

    // ── episodes ──
    const std::string all = answer_episodes(st, {}, {}, "", "", "");
    show("episodes()", all);
    CHECK(has(all, "5 episodes today"));
    CHECK(has(all, "20261017-1 10:00-10:03 mission \"cross door_27\" in room_2, asked by a person (chat): succeeded"));
    CHECK(has(all, "20261017-2 10:00-10:03 step cross door_27, attempt 2 (part of 20261017-1): succeeded"));
    CHECK(has(all, "20261017-3 10:01-10:02 claim open door_27, promised 0.50 delivered 0.25 nats (part of 20261017-2): kept"));
    CHECK(has(all, "20261017-5 12:00-now mission in room_5, on its own: still open"));
    CHECK(not has(all, "mission mission"));
    CHECK(all.find("20261017-1 ") < all.find("20261017-4 "));   // by start time

    const std::string missions = answer_episodes(st, at(10, 30), {}, "human", "mission", "");
    show("episodes(since 10:30, human, kind mission)", missions);
    CHECK(has(missions, "1 episode today"));
    CHECK(has(missions, "go to the kitchen"));
    const std::string self_missions = answer_episodes(st, {}, {}, "self", "mission", "");
    CHECK(has(self_missions, "1 episode today") and has(self_missions, "20261017-5"));

    const std::string selfs = answer_episodes(st, {}, at(12, 0), "self", "", "");
    CHECK(has(selfs, "1 episode today") and has(selfs, "claim open door_27"));   // until exclusive: 12:00 not in
    CHECK(answer_episodes(st, {}, {}, "", "", "door_99") == "no episode recorded today matches that");
    CHECK(has(answer_episodes(st, {}, {}, "robot", "", ""), "author must be"));

    // ── episode(id): the tree, facts on beliefs (pose omitted), events ──
    const std::string tree = answer_episode(st, "20261017-1");
    show("episode(20261017-1)", tree);
    CHECK(has(tree, "20261017-1 10:00-10:03 mission"));
    CHECK(has(tree, "\n  20261017-2 10:00-10:03 step cross door_27"));
    CHECK(has(tree, "\n    20261017-3 10:01-10:02 claim open door_27"));
    CHECK(has(tree, "door_27: door_open_prob 0.12 -> 0.93"));
    CHECK(not has(tree, "x 1.50"));
    CHECK(has(tree, "events: +12 s current_moved room_3"));
    CHECK(answer_episode(st, "nope") == "no episode with id 'nope' today");

    // ── outcome_stats ──
    const std::string sc = answer_outcome_stats(st, "cross", "");
    show("outcome_stats(cross)", sc);
    CHECK(has(sc, "cross (all time): 12 times: succeeded 9, failed 3; mean duration 20.0 s (sd 2.0 s)"));
    CHECK(not has(sc, "passage_0"));   // only the per-kind row when no target is given
    const std::string s27 = answer_outcome_stats(st, "cross", "door_27");   // live name -> memory identity passage_0
    show("outcome_stats(cross, door_27)", s27);
    CHECK(has(s27, "cross door_27 (all time): 5 times: succeeded 4, lost 1"));
    CHECK(has(answer_outcome_stats(st, "cross", "passage_0"), "5 times"));   // the identity itself works too
    const std::string so = answer_outcome_stats(st, "open", "");
    CHECK(has(so, "information promised 0.50, delivered 0.25 nats on average over 2"));
    CHECK(not has(so, "first"));   // no first/last time recorded: omitted, not 1970
    const std::string s99 = answer_outcome_stats(st, "cross", "door_99");
    show("outcome_stats(cross, door_99)", s99);
    CHECK(has(s99, "no statistics for cross on door_99"));
    CHECK(has(s99, "cross (all time): 12 times"));   // falls back to the per-kind row, said as such
    CHECK(answer_outcome_stats(st, "fly", "") == "no statistics for fly");

    // ── last_episode ──
    const std::string l27 = answer_last_episode(st, "door_27");
    show("last_episode(door_27)", l27);
    CHECK(has(l27, "20261017-3 10:01-10:02 claim open door_27"));   // newest t0 touching door_27
    CHECK(answer_last_episode(st, "fridge_1") == "nothing recorded today about 'fridge_1'");

    // ── time arguments ──
    CHECK(parse_em_time("10:30") == at(10, 30));
    CHECK(parse_em_time("9:05") == at(9, 5));
    CHECK(parse_em_time("1760000000000") == 1760000000000ULL);
    CHECK(not parse_em_time("25:00").has_value());
    CHECK(not parse_em_time("10:3x").has_value());
    CHECK(not parse_em_time("yesterday").has_value());

    // ── the STATE line ──
    CHECK(not today_digest_line(absent).has_value());
    CHECK(not today_digest_line(empty).has_value());   // the tools say "not available" here: the STATE agrees
    CHECK(today_digest_line(st) == "today: 2 missions (1 succeeded); last: go to the kitchen → room_3");

    std::println("test_em: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
