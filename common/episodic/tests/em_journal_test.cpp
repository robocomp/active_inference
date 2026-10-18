// The controller journal (em_journal.h): serials, bounded capacity, wire round trip, dirty flag.
#include "../em_journal.h"

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

using namespace rc::em;

JournalEntry entry(JournalEntry::Kind k) { JournalEntry e; e.kind = k; return e; }

void test_serials_monotonic()
{
    Journal j(1000);
    CHECK(j.push(entry(JournalEntry::Kind::MissionStart)) == 1);
    CHECK(j.push(entry(JournalEntry::Kind::StepStart)) == 2);
    CHECK(j.push(entry(JournalEntry::Kind::StepEnd)) == 3);
}

void test_capacity_drops_oldest()
{
    Journal j(1000, 3);
    for (int i = 0; i < 5; ++i) j.push(entry(JournalEntry::Kind::DriveMode));
    const auto w = journal_from_wire(j.to_wire());
    CHECK(w.has_value());
    CHECK(w->entries.size() == 3);
    CHECK(w->entries[0].serial == 3 and w->entries[1].serial == 4 and w->entries[2].serial == 5);
}

void test_wire_roundtrip()
{
    Journal j(1760000000123ull);
    JournalEntry a = entry(JournalEntry::Kind::ClaimStart);
    a.t_ms = 1760000001000ull; a.mission = "m12"; a.step_id = "2.1"; a.op = "open_door";
    a.source = "chat"; a.text = R"({"affordance":"aff_x","target":"door_3"})"; a.outcome = "";
    a.attempt = 2; a.iteration = 7; a.promised = 0.25;
    a.candidates = R"([{"name":"a","gain":0.1}])";
    JournalEntry b = entry(JournalEntry::Kind::MissionEnd);   // promised stays absent
    b.mission = "m12"; b.outcome = "done";
    j.push(a); j.push(b);
    const auto w = journal_from_wire(j.to_wire());
    CHECK(w.has_value());
    CHECK(w->epoch == 1760000000123ull);
    CHECK(w->entries.size() == 2);
    const JournalEntry& x = w->entries[0];
    CHECK(x.serial == 1 and x.t_ms == a.t_ms and x.kind == a.kind and x.mission == "m12");
    CHECK(x.step_id == "2.1" and x.op == "open_door" and x.source == "chat");
    CHECK(x.text == a.text and x.candidates == a.candidates);
    CHECK(x.attempt == 2 and x.iteration == 7);
    CHECK(x.promised.has_value() and *x.promised == 0.25);
    const JournalEntry& y = w->entries[1];
    CHECK(y.serial == 2 and y.kind == JournalEntry::Kind::MissionEnd and y.outcome == "done");
    CHECK(not y.promised.has_value());
    // every kind survives
    Journal k(1);
    using K = JournalEntry::Kind;
    for (K kd : {K::MissionStart, K::MissionEnd, K::StepStart, K::StepEnd, K::ClaimStart, K::ClaimEnd, K::DriveMode})
        k.push(entry(kd));
    const auto kw = journal_from_wire(k.to_wire());
    CHECK(kw.has_value() and kw->entries.size() == 7 and kw->entries[6].kind == K::DriveMode
          and kw->entries[4].kind == K::ClaimStart);
}

void test_wire_rejects_garbage()
{
    CHECK(not journal_from_wire("not json").has_value());
    CHECK(not journal_from_wire(R"({"epoch":1,"entries":[{"serial":1,"kind":"bogus"}]})").has_value());
    CHECK(not journal_from_wire(R"({"entries":[]})").has_value());
}

void test_dirty_flag()
{
    Journal j(1);
    CHECK(not j.dirty());
    j.push(entry(JournalEntry::Kind::StepStart));
    CHECK(j.dirty());
    j.clear_dirty();
    CHECK(not j.dirty());
}
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "es_ES.UTF-8") or std::setlocale(LC_ALL, "");
    test_serials_monotonic();
    test_capacity_drops_oldest();
    test_wire_roundtrip();
    test_wire_rejects_garbage();
    test_dirty_flag();
    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
