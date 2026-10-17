// Offline checks for episodic_agent's Recorder (src/em_recorder.h): startup stats recomputed from scratch over all
// past days, today's records reseeded, ids continuing, and each record folded EXACTLY once across a day change.
// Qt Core only (no DSR): the graph is a recording fake sink.
#include "../src/em_recorder.h"

#include <clocale>
#include <cstdio>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <unistd.h>

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (not(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

namespace
{
struct FakeSink : rc::em::RecordSink
{
    std::map<std::string, rc::em::Record> records;
    std::map<std::string, rc::em::Stats> stats;   // by kind|identity
    int stats_writes = 0;
    void upsert(const rc::em::Record& r) override { records[r.id] = r; }
    void upsert(const rc::em::Stats& s) override { stats[s.key_kind + "|" + s.key_identity] = s; ++stats_writes; }
    void remove(const std::string& id) override { records.erase(id); }
};

// Local noon of a given date, as ms since epoch (day_of() is local-time, so tests must be too).
std::uint64_t noon(int y, unsigned m, unsigned d)
{
    using namespace std::chrono;
    const local_days ld{year{y} / month{m} / day{d}};
    const auto t = current_zone()->to_sys(ld + hours{12});
    return std::uint64_t(duration_cast<milliseconds>(t.time_since_epoch()).count());
}

rc::em::Record rec(const std::string& id, std::uint64_t t0, std::optional<std::uint64_t> t1, const std::string& kind = "cross",
                   const std::string& outcome = "succeeded")
{
    rc::em::Record r;
    r.id = id;
    r.level = rc::em::Level::Claim;
    r.author = rc::em::Author::Self;
    r.kind = kind;
    r.target_ltsm = "passage_0";
    r.t0 = t0;
    r.t1 = t1;
    r.outcome = t1 ? outcome : "";
    return r;
}
}   // namespace

int main()
{
    if (not std::setlocale(LC_ALL, "es_ES.UTF-8")) std::setlocale(LC_ALL, "");
    const auto dir = std::filesystem::temp_directory_path() / ("em_recorder_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);

    const auto d15 = noon(2026, 10, 15), d16 = noon(2026, 10, 16), d17 = noon(2026, 10, 17), d18 = noon(2026, 10, 18);
    {
        rc::em::Archive a(dir);
        // Two past days: 3 closed + 1 left open by a crash (closed `lost` at startup, R21). Open+close lines of the same id count once.
        a.append(rec("20261015-1", d15, std::nullopt));
        a.append(rec("20261015-1", d15, d15 + 2000));
        a.append(rec("20261015-2", d15 + 10, d15 + 4010));
        a.append(rec("20261016-1", d16, d16 + 6000));
        a.append(rec("20261016-2", d16 + 50, std::nullopt));
        // Today: one closed, one open.
        a.append(rec("20261017-1", d17, d17 + 1000));
        a.append(rec("20261017-4", d17 + 100, std::nullopt, "mission", ""));
    }

    // ── Startup (R21): records a crashed run left OPEN, on any day, close `lost` at now and are archived ───────
    rc::em::Archive archive(dir);
    FakeSink sink;
    rc::em::Recorder rc(archive, sink, "20261017");
    const std::uint64_t boot = d17 + 200;
    rc.start(boot);
    CHECK(sink.records.size() == 2);                            // today's records reseeded, nothing else
    CHECK(sink.records.contains("20261017-1") and sink.records.contains("20261017-4"));
    CHECK(sink.records["20261017-4"].outcome == "lost" and sink.records["20261017-4"].t1 == boot);
    const auto key = std::pair<std::string, std::string>{"cross", "passage_0"};
    const auto mkey = std::pair<std::string, std::string>{"mission", "passage_0"};
    CHECK(rc.stats().size() == 1);
    CHECK(rc.stats().contains(key) and rc.stats().at(key).n == 4);   // 15-1, 15-2, 16-1 + 16-2 closed lost
    CHECK(rc.stats().at(key).outcomes.at("lost") == 1);
    CHECK(sink.stats.size() == 1 and sink.stats["cross|passage_0"].n == 4);
    {
        const auto d16r = archive.load_day("20261016");
        const auto it = std::ranges::find(d16r, std::string("20261016-2"), &rc::em::Record::id);
        CHECK(it != d16r.end() and it->outcome == "lost" and it->t1 == boot);   // the closing line is in the archive
    }

    // A second start must not double-count (rollover is not idempotent: start() recomputes from scratch) and finds
    // nothing left open to close.
    const auto lines = [&](const char* day)
    {
        std::ifstream f(dir / (std::string(day) + ".jsonl"));
        return std::ranges::count(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>(), '\n');
    };
    const auto l16 = lines("20261016"), l17 = lines("20261017");
    rc.start(boot + 1);
    CHECK(rc.stats().at(key).n == 4);
    CHECK(lines("20261016") == l16 and lines("20261017") == l17);

    // Ids continue after the largest seq of the day, and per day.
    CHECK(rc.next_id(d17 + 500) == "20261017-5");
    CHECK(rc.next_id(d17 + 600) == "20261017-6");
    CHECK(rc.next_id(d16 + 600) == "20261016-3");

    // ── Same day ─────────────────────────────────────────────────────────────────────────────────────────────
    rc.write({rec("20261017-5", d17 + 500, std::nullopt)});
    rc.write({rec("20261017-5", d17 + 500, d17 + 900)});
    rc.write({rec("20261017-6", d17 + 600, std::nullopt, "mission", "")});   // still open at midnight
    CHECK(sink.records.contains("20261017-5") and sink.records["20261017-5"].t1.has_value());
    CHECK(rc.stats().at(key).n == 4);                           // today's records are not folded yet

    // ── Day change: the first record of the 18th folds the 17th's CLOSED records once, deletes their nodes ───
    rc.write({rec("20261018-1", d18, std::nullopt)});
    CHECK(rc.today() == "20261018");
    CHECK(rc.stats().at(key).n == 6);                           // + 17-1, 17-5
    CHECK(rc.stats().contains(mkey) and rc.stats().at(mkey).n == 1);   // 17-4 (closed lost at boot)
    CHECK(not sink.records.contains("20261017-1") and not sink.records.contains("20261017-5"));
    CHECK(not sink.records.contains("20261017-4"));
    CHECK(sink.records.contains("20261017-6"));                 // still open: stays as a node
    CHECK(sink.records.contains("20261018-1"));
    CHECK(sink.stats["cross|passage_0"].n == 6);

    // A second record of the new day folds nothing again.
    rc.write({rec("20261018-1", d18, d18 + 100)});
    CHECK(rc.stats().at(key).n == 6);

    // The 17th's open mission closes after the rollover: folded at its close, exactly once, node removed.
    rc.write({rec("20261017-6", d17 + 600, d18 + 50, "mission", "succeeded")});
    CHECK(rc.stats().at(mkey).n == 2);
    CHECK(not sink.records.contains("20261017-6"));
    CHECK(rc.stats().at(key).n == 6);

    // Everything went to the archive (last line per id wins).
    const auto day17 = archive.load_day("20261017");
    CHECK(day17.size() == 4);
    for (const auto& r : day17) CHECK(r.t1.has_value());

    // A restart on the 18th recomputes the same totals from the archive.
    FakeSink sink2;
    rc::em::Recorder again(archive, sink2, "20261018");
    again.start(d18 + 1000);
    CHECK(again.stats().at(key).n == 6);
    CHECK(again.stats().at(mkey).n == 2);
    CHECK(sink2.records.size() == 1 and sink2.records.contains("20261018-1"));
    CHECK(again.next_id(d18 + 1) == "20261018-2");

    // The shutdown path (R22): closed records reach the archive AND the graph.
    again.write({rec("20261018-2", d18 + 1, d18 + 2, "cross", "interrupted_by_em_shutdown")});
    CHECK(sink2.records.contains("20261018-2"));
    CHECK(archive.load_day("20261018").size() == 2);

    std::filesystem::remove_all(dir);
    if (failures) { std::fprintf(stderr, "recorder_test: %d failure(s)\n", failures); return 1; }
    std::puts("recorder_test: all passed");
    return 0;
}
