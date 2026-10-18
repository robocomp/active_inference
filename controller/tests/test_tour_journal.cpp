// TourJournal (tour_journal.h): panel tours as human missions in the episodic journal (source "panel").
// Build+run (Qt Core + std):
//   g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -Wno-missing-field-initializers -o /tmp/ttj test_tour_journal.cpp $(pkg-config --cflags --libs Qt6Core) && /tmp/ttj
#include "../src/tour_journal.h"

#include <print>

namespace
{
int failures = 0;
void check(bool ok, int line, const char* what)
{
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)
using K = rc::em::JournalEntry::Kind;
}   // namespace

int main()
{
    rc::TourJournal tj;
    using V = rc::TourJournal::View;
    CHECK(tj.observe(V{.running = false, .name = {}, .loops = 0, .mode = "mission_only", .stop_reason = {}}).empty());

    auto e = tj.observe(V{.running = true, .name = "kitchen loop", .loops = 2, .mode = "mission_only", .stop_reason = {}});
    CHECK(e.size() == 1);
    if (e.size() == 1)
    {
        CHECK(e[0].kind == K::MissionStart and e[0].mission == "p1" and e[0].source == "panel");
        const auto d = QJsonDocument::fromJson(QByteArray::fromStdString(e[0].text)).object();
        CHECK(d["title"].toString() == "tour kitchen loop" and d["tour"].toString() == "kitchen loop");
        CHECK(d["loops"].toInt() == 2 and d["mode"].toString() == "mission_only");
    }
    CHECK(tj.observe(V{.running = true, .name = "kitchen loop", .loops = 1, .mode = "mission_only", .stop_reason = {}}).empty());

    e = tj.observe(V{.running = false, .name = {}, .loops = 0, .mode = "mission_only", .stop_reason = "completed"});
    CHECK(e.size() == 1 and e[0].kind == K::MissionEnd and e[0].mission == "p1" and e[0].outcome == "completed");

    tj.observe(V{.running = true, .name = "b", .loops = 1, .mode = "mission_only", .stop_reason = "completed"});
    e = tj.observe(V{.running = false, .name = {}, .loops = 0, .mode = "target", .stop_reason = "superseded by click"});
    CHECK(e.size() == 1 and e[0].mission == "p2" and e[0].outcome == "aborted|superseded by click");

    tj.observe(V{.running = true, .name = "c", .loops = 1, .mode = "mission_only", .stop_reason = {}});
    e = tj.observe(V{.running = false, .name = {}, .loops = 0, .mode = "mission_only", .stop_reason = "exhausted"});
    CHECK(e.size() == 1 and e[0].mission == "p3" and e[0].outcome == "completed");

    std::println("test_tour_journal: {}", failures == 0 ? "all passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
