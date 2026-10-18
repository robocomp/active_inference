// The append-only archive (em_archive.h): last line per id wins, next_seq, damaged lines skipped, locale-proof.
#include "../em_archive.h"

#include <chrono>
#include <clocale>
#include <filesystem>
#include <format>
#include <fstream>
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
namespace fs = std::filesystem;

// Noon UTC on 2026-10-17: the same calendar date in every zone from UTC-11 to UTC+11.
std::uint64_t noon_ms(int extra_s = 0)
{
    using namespace std::chrono;
    const sys_time<milliseconds> t = sys_days{year{2026} / 10 / 17} + hours{12} + seconds{extra_s};
    return std::uint64_t(t.time_since_epoch().count());
}

struct TempDir
{
    fs::path p;
    explicit TempDir(std::string_view tag)
        : p(fs::temp_directory_path() / std::format("em_archive_test_{}_{}", tag, std::chrono::steady_clock::now().time_since_epoch().count()))
    {}
    ~TempDir() { std::error_code ec; fs::remove_all(p, ec); }
};

Record rec(std::string id, std::uint64_t t0)
{
    Record r;
    r.id = std::move(id);
    r.level = Level::Step;
    r.author = Author::Self;
    r.kind = "cross";
    r.t0 = t0;
    return r;
}

void test_reload_last_line_wins()
{
    TempDir d("last");
    Archive a(d.p / "arch");
    Record open = rec("20261017-1", noon_ms());
    a.append(open);
    Record other = rec("20261017-2", noon_ms(5));
    a.append(other);
    Record closed = open;
    closed.t1 = noon_ms(10);
    closed.outcome = "success";
    a.append(closed);

    const auto got = Archive(d.p / "arch").load_day("20261017");     // a fresh instance reads what is on disk
    CHECK(got.size() == 2);
    CHECK(got.size() == 2 and got[0] == closed);                      // ordered by t0, closed version won
    CHECK(got.size() == 2 and got[1] == other);
    CHECK(a.days() == std::vector<std::string>{"20261017"});
    CHECK(a.load_day("19990101").empty());
}

void test_next_seq()
{
    TempDir d("seq");
    Archive a(d.p);
    CHECK(a.next_seq("20261017") == 1);                               // empty day starts at 1
    a.append(rec("20261017-1", noon_ms()));
    a.append(rec("20261017-7", noon_ms(1)));
    CHECK(a.next_seq("20261017") == 8);
    CHECK(a.next_seq("20261016") == 1);
}

void test_truncated_tail_skipped()
{
    TempDir d("trunc");
    Archive a(d.p);
    a.append(rec("20261017-1", noon_ms()));
    a.append(rec("20261017-2", noon_ms(1)));
    {
        std::ofstream f(d.p / "20261017.jsonl", std::ios::app);
        f << R"({"id":"20261017-3","level":"step","aut)" << '\n';     // crash mid-write
        Record noid = rec("", noon_ms(2));                              // parses, but has no identity
        f << QJsonDocument(to_json(noid)).toJson(QJsonDocument::Compact).toStdString() << '\n';
        f << R"({"id":"20261017-4","level":"step","aut)";               // truncated, no newline
    }
    const auto got = a.load_day("20261017");
    CHECK(got.size() == 2);
    CHECK(a.next_seq("20261017") == 3);
}

void test_append_heals_torn_tail()
{
    TempDir d("heal");
    Archive a(d.p);
    CHECK(a.append(rec("20261017-1", noon_ms())));
    {
        std::ofstream f(d.p / "20261017.jsonl", std::ios::app);
        f << R"({"id":"20261017-2","level":"step","aut)";               // crash mid-write, no newline
    }
    CHECK(a.append(rec("20261017-3", noon_ms(2))));
    const auto got = a.load_day("20261017");
    CHECK(got.size() == 2);                                             // the post-crash record survives
    CHECK(got.size() == 2 and got[1].id == "20261017-3");
    CHECK(a.next_seq("20261017") == 4);
}

void test_append_reports_failure()
{
    TempDir d("fail");
    Archive a(d.p / "arch");
    fs::remove_all(d.p / "arch");
    { std::ofstream f(d.p / "arch"); f << "not a directory"; }          // the archive dir is now a regular file
    CHECK(not a.append(rec("20261017-1", noon_ms())));
}

void test_next_seq_rejects_garbage_suffix()
{
    TempDir d("garb");
    Archive a(d.p);
    a.append(rec("20261017-2", noon_ms()));
    a.append(rec("20261017-12abc", noon_ms(1)));
    CHECK(a.next_seq("20261017") == 3);
}

void test_locale_roundtrip()
{
    TempDir d("loc");
    Archive a(d.p);
    Record r = rec("20261017-1", noon_ms());
    r.level = Level::Claim;
    r.promised = 0.260417;
    r.delivered = -9.2e-05;
    a.append(r);
    const auto got = a.load_day("20261017");
    CHECK(got.size() == 1);
    CHECK(got.size() == 1 and got[0].promised == std::optional<double>(0.260417));
    CHECK(got.size() == 1 and got[0].delivered == std::optional<double>(-9.2e-05));
}
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "es_ES.UTF-8") or std::setlocale(LC_ALL, "");
    test_reload_last_line_wins();
    test_next_seq();
    test_truncated_tail_skipped();
    test_append_heals_torn_tail();
    test_append_reports_failure();
    test_next_seq_rejects_garbage_suffix();
    test_locale_roundtrip();
    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
