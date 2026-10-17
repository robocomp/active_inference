// The query library (em_query.h): filters, episode tree walk, last(), stats, today_line.
#include "../em_query.h"

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

using namespace rc::em;

Record rec(std::string id, std::string parent, Level lv, Author au, std::string kind, std::string target,
           std::uint64_t t0, std::optional<std::uint64_t> t1 = std::nullopt, std::string outcome = {})
{
    Record r;
    r.id = std::move(id); r.parent = std::move(parent); r.level = lv; r.author = au;
    r.kind = std::move(kind); r.target = std::move(target); r.t0 = t0; r.t1 = t1; r.outcome = std::move(outcome);
    return r;
}

std::vector<std::string> ids(const std::vector<Record>& v)
{
    std::vector<std::string> o;
    for (const Record& r : v) o.push_back(r.id);
    return o;
}
}   // namespace

int main()
{
    Store s;
    Record m1 = rec("m1", "", Level::Mission, Author::Human, "mission", "", 1000, 2000, "succeeded");
    m1.action_json = R"({"title":"cross door_27"})"; m1.room = "room_2";
    Record st1 = rec("s1", "m1", Level::Step, Author::Human, "cross", "door_27", 1100, 1900, "succeeded");
    Record c1 = rec("c1", "s1", Level::Claim, Author::Self, "open", "door_27", 1200, 1300, "kept");
    Record m2 = rec("m2", "", Level::Mission, Author::Human, "mission", "", 3000, 4000, "completed");
    m2.action_json = "go somewhere in the kitchen please and wait for me there ok thanks a lot truly"; m2.room = "";
    Record m3 = rec("m3", "", Level::Mission, Author::Human, "mission", "", 5000, 6000, "failed");
    m3.action_json = "plain"; m3.room = "room_9";
    Record sm = rec("sm", "", Level::Mission, Author::Self, "explore", "room_5", 7000);
    // a child still open after its parent closed (m3 closed at 6000)
    Record c3 = rec("c3", "m3", Level::Claim, Author::Self, "open", "door_3", 5500);
    Record f = rec("f1", "m2", Level::Step, Author::Human, "look", "", 3100, 3200, "succeeded");
    f.post.push_back(Fact{.node = "fridge_1", .type = "refrigerator", .ltsm_id = {}, .values = {}});
    s.today = {m1, st1, c1, m2, f, m3, c3, sm};

    // filters
    CHECK(ids(episodes(s, std::nullopt, std::nullopt, std::nullopt)).size() == 8);
    CHECK((ids(episodes(s, std::nullopt, std::nullopt, Author::Self)) == std::vector<std::string>{"c1", "c3", "sm"}));
    CHECK((ids(episodes(s, 3000, 5000, std::nullopt)) == std::vector<std::string>{"m2", "f1"}));   // since incl, until excl
    CHECK((ids(episodes(s, std::nullopt, std::nullopt, std::nullopt, "open")) == std::vector<std::string>{"c1", "c3"}));
    CHECK((ids(episodes(s, std::nullopt, std::nullopt, std::nullopt, "open", "door_3")) == std::vector<std::string>{"c3"}));
    CHECK((ids(episodes(s, std::nullopt, std::nullopt, Author::Human, "cross", "door_27")) == std::vector<std::string>{"s1"}));
    CHECK(episodes(s, 9000, std::nullopt, std::nullopt).empty());

    // episode: record + descendants, pre-order, by parent id (c3 outlives m3)
    CHECK((ids(episode(s, "m1")) == std::vector<std::string>{"m1", "s1", "c1"}));
    CHECK((ids(episode(s, "m3")) == std::vector<std::string>{"m3", "c3"}));
    CHECK((ids(episode(s, "s1")) == std::vector<std::string>{"s1", "c1"}));
    CHECK(episode(s, "nope").empty());

    // last: target or a Fact node, newest t0
    CHECK(last(s, "door_27") and last(s, "door_27")->id == "c1");
    CHECK(last(s, "fridge_1") and last(s, "fridge_1")->id == "f1");     // only in a Fact
    CHECK(last(s, "door_3") and last(s, "door_3")->id == "c3");
    CHECK(not last(s, "ghost"));

    // stats
    Stats a; a.key_kind = "cross"; a.key_identity = "ltsm_a"; a.n = 1;
    Stats b; b.key_kind = "cross"; b.key_identity = "ltsm_b"; b.n = 2;
    Stats c; c.key_kind = "open"; c.key_identity = "ltsm_a"; c.n = 3;
    s.stats = {a, b, c};
    CHECK(stats(s, "cross").size() == 2);
    CHECK(stats(s, "cross", "ltsm_b").size() == 1 and stats(s, "cross", "ltsm_b")[0].n == 2);
    CHECK(stats(s, "").size() == 3);
    CHECK(stats(s, "", "ltsm_a").size() == 2);
    CHECK(stats(s, "none").empty());

    // today_line: 3 human missions, 2 succeeded (succeeded + completed); last closed = m3 "plain → room_9"
    CHECK(today_line(s) == "3 missions (2 succeeded); last: plain → room_9");
    // drop m3: last = m2, raw text truncated to 60 chars, no room
    s.today = {m1, m2};
    CHECK(today_line(s) == "2 missions (2 succeeded); last: " + m2.action_json.substr(0, 60));
    s.today = {m1};
    CHECK(today_line(s) == "1 missions (1 succeeded); last: cross door_27 → room_2");
    // empty / no human missions
    CHECK(today_line(Store{}) == "no missions today");
    s.today = {sm, st1};
    CHECK(today_line(s) == "no missions today");

    std::println("{} checks, {} failures", checks, failures);
    return failures ? 1 : 0;
}
