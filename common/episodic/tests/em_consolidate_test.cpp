// Consolidation (em_consolidate.h): Welford over durations, gain only when both numbers exist, keys, rollover.
#include "../em_consolidate.h"

#include <chrono>
#include <cmath>
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

// Noon UTC on 2026-10-17 + day offset: the same calendar date in every zone from UTC-11 to UTC+11.
std::uint64_t noon_ms(int day_offset = 0)
{
    using namespace std::chrono;
    const sys_time<milliseconds> t = sys_days{year{2026} / 10 / 17} + days{day_offset} + hours{12};
    return std::uint64_t(t.time_since_epoch().count());
}

Record closed(const std::string& id, const std::string& outcome, int dur_s, std::uint64_t t0 = noon_ms())
{
    Record r;
    r.id = id;
    r.level = Level::Claim;
    r.kind = "cross_door";
    r.target_ltsm = "passage_0";
    r.t0 = t0;
    r.t1 = t0 + std::uint64_t(dur_s) * 1000;
    r.outcome = outcome;
    return r;
}

void test_fold_counts_and_duration()
{
    Stats st;
    fold(st, closed("a", "satisfied", 2));
    fold(st, closed("b", "satisfied", 4));
    fold(st, closed("c", "timeout", 6));
    CHECK(st.n == 3);
    CHECK(st.outcomes.at("satisfied") == 2);
    CHECK(st.outcomes.at("timeout") == 1);
    CHECK(std::abs(st.dur_mean_s - 4.0) < 1e-9);
    CHECK(std::abs(st.dur_m2 - 8.0) < 1e-9);
    // dur_m2 is the sum of squared deviations (8); the SAMPLE variance dur_m2/(n-1) is 4 (population would be 8/3).
    CHECK(std::abs(st.dur_m2 / (st.n - 1) - 4.0) < 1e-9);
    CHECK(st.first_t0 == noon_ms() and st.last_t0 == noon_ms());
}

void test_gain_only_when_both()
{
    Stats st;
    Record a = closed("a", "satisfied", 1);
    a.promised = 2.0;
    a.delivered = 1.5;
    Record b = closed("b", "satisfied", 1);
    b.promised = 3.0;                       // no delivered
    fold(st, a);
    fold(st, b);
    CHECK(st.n == 2);
    CHECK(st.n_gain == 1);
    CHECK(std::abs(st.promised_sum - 2.0) < 1e-9 and std::abs(st.delivered_sum - 1.5) < 1e-9);
}

void test_key_without_ltsm()
{
    Record r = closed("a", "satisfied", 1);
    r.target_ltsm.clear();
    CHECK((stats_key(r) == StatsKey{"claim", "cross_door", "", ""}));
    CHECK((stats_key(closed("b", "x", 1)) == StatsKey{"claim", "cross_door", "", "passage_0"}));
}

// A step op and a claim kind of the same name are different things; so are a person's and the robot's own missions.
void test_key_separates_level_and_author()
{
    Record step = closed("s", "satisfied", 1);
    step.level = Level::Step;
    step.kind = "cross";
    Record claim = closed("c", "success", 1);
    claim.kind = "cross";
    CHECK(stats_key(step) != stats_key(claim));
    claim.author = Author::Self;                                   // author is part of the key for missions only
    CHECK((stats_key(claim) == StatsKey{"claim", "cross", "", "passage_0"}));
    Record hm = closed("h", "completed", 1);
    hm.level = Level::Mission; hm.kind = "mission"; hm.target_ltsm.clear();
    Record sm = hm;
    sm.author = Author::Self;
    CHECK((stats_key(hm) == StatsKey{"mission", "mission", "human", ""}));
    CHECK((stats_key(sm) == StatsKey{"mission", "mission", "self", ""}));
    Stats st;
    fold(st, sm);
    CHECK(st.key_level == "mission" and st.key_author == "self" and st.key_kind == "mission");
}

// lost / interrupted_by_em_shutdown durations are EM's stop time, not the commitment's: counted, never timed.
void test_duration_skips_em_closed()
{
    Stats st;
    fold(st, closed("a", "satisfied", 2));
    fold(st, closed("b", "lost", 500));
    fold(st, closed("c", "interrupted_by_em_shutdown", 900));
    fold(st, closed("d", "satisfied", 4));
    CHECK(st.n == 4 and st.n_dur == 2);
    CHECK(st.outcomes.at("lost") == 1 and st.outcomes.at("interrupted_by_em_shutdown") == 1);
    CHECK(std::abs(st.dur_mean_s - 3.0) < 1e-9 and std::abs(st.dur_m2 - 2.0) < 1e-9);
}

void test_rollover_skips_open()
{
    std::map<StatsKey, Stats> stats;
    Record open = closed("open", "", 1, noon_ms(-1));
    open.t1.reset();
    const Record done = closed("done", "satisfied", 1, noon_ms(-1));
    const Record today = closed("today", "satisfied", 1, noon_ms(0));
    const std::vector<Record> rs{open, done, today};
    const auto ids = rollover(stats, rs, day_of(noon_ms(0)));
    CHECK((ids == std::vector<std::string>{"done"}));
    CHECK(stats.size() == 1);
    CHECK(stats.at(StatsKey{"claim", "cross_door", "", "passage_0"}).n == 1);
}

void test_stats_json_roundtrip()
{
    Stats st;
    Record a = closed("a", "satisfied", 3);
    a.promised = 2.0;
    a.delivered = 1.0;
    fold(st, a);
    const auto back = stats_from_json(to_json(st));
    CHECK(back.has_value() and *back == st);
    CHECK(to_json(st)["key_level"].toString() == "claim" and to_json(st)["n_dur"].toInt() == 1);
    CHECK(not stats_from_json(QJsonObject{}).has_value());
}
}   // namespace

int main()
{
    test_fold_counts_and_duration();
    test_gain_only_when_both();
    test_key_without_ltsm();
    test_key_separates_level_and_author();
    test_duration_skips_em_closed();
    test_rollover_skips_open();
    test_stats_json_roundtrip();
    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
