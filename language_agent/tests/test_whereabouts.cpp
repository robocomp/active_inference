// whereabouts.h offline checks (tense spec 2026-11-14 §3): sampling, at(t), visits, the STATE line, where_was.
// Build+run: tests/run_tests.sh
#include "../src/whereabouts.h"

#include <clocale>
#include <print>

static int checks = 0, failures = 0;
static void check(bool ok, int line, const char* what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)

int main()
{
    using namespace rc::chat;
    std::setlocale(LC_ALL, "");
    constexpr std::int64_t T0 = 1'794'612'000'000;   // 2026-11-14 ~00:20 local
    constexpr std::int64_t MIN = 60'000;
    const auto at = [](std::string room, std::int64_t t, double x = 0, double y = 0)
    { return WhereSample{.t_ms = t, .room = std::move(room), .has_pose = true, .x = x, .y = y, .yaw_deg = 90}; };

    // ── sampling: a room change is always stored; otherwise one per period; the latest refresh is still known ──
    {
        Whereabouts w(5'000);
        CHECK(w.add(at("room_1", T0, 1, 1)));
        CHECK(not w.add(at("room_1", T0 + 500, 1.1, 1)));          // inside the period: not stored
        CHECK(w.add(at("room_3", T0 + 1'000, 2, 2)));               // room changed: stored at once
        CHECK(not w.add(at("room_3", T0 + 3'000, 2.5, 2)));
        CHECK(w.add(at("room_3", T0 + 6'000, 3, 2)));               // a period after the last stored one
        CHECK(w.size() == 3);
        CHECK(not w.add(at("room_3", T0 + 5'000)));                  // the clock went back: ignored
        const auto s = w.at(T0 + 900);
        CHECK(s.has_value() and s->room == "room_1" and s->x == 1);     // the stored sample: resolution is one per period
        CHECK(w.at(T0 + 1'000)->room == "room_3" and w.at(T0 + 1'000)->x == 2);   // the room-entry sample is intact
        CHECK(not w.at(T0 - 1).has_value());
        CHECK(w.at(T0 + 999'999)->x == 3);
    }
    // ── the 24 h cap ──
    {
        Whereabouts w(5'000, 10 * MIN);
        for (int i = 0; i <= 30; ++i) w.add(at("room_1", T0 + i * MIN));
        CHECK(w.first_ms() == T0 + 20 * MIN);
    }
    // ── visits: the latest one per room, latest first; the current room marked now ──
    Whereabouts w(5'000);
    w.add(at("room_1", T0, -1.09, -0.97));
    w.add(at("room_1", T0 + 2 * MIN));
    w.add(at("room_3", T0 + 3 * MIN, 1.2, -0.4));
    w.add(at("room_3", T0 + 8 * MIN, 1.2, -0.4));
    w.add(at("room_2", T0 + 9 * MIN));
    w.add(at("room_1", T0 + 15 * MIN));
    w.add(at("room_1", T0 + 20 * MIN, -1.0, -1.0));
    {
        const auto v = w.visits();
        CHECK(v.size() == 3 and v[0].room == "room_1" and v[0].now and v[0].from_ms == T0 + 15 * MIN);
        CHECK(v[1].room == "room_2" and not v[1].now and v[2].room == "room_3");
        CHECK(v[2].from_ms == T0 + 3 * MIN and v[2].to_ms == T0 + 8 * MIN);
        const std::string line = w.state_line(T0 + 20 * MIN);
        CHECK(line.starts_with("where you have been (since " + clock_hhmm(T0)));
        CHECK(line.contains("room_1 now, since " + clock_hhmm(T0 + 15 * MIN)));
        CHECK(line.contains("room_3 " + clock_hhmm(T0 + 3 * MIN) + "-" + clock_hhmm(T0 + 8 * MIN) + " (12 min ago)"));
        CHECK(line.find("room_2") < line.find("room_3"));             // latest first
        CHECK(Whereabouts{}.state_line(T0).empty());
    }
    // ── where_was: same room, another room (with the way), too far back, the future ──
    {
        const std::int64_t now = T0 + 20 * MIN;
        const std::string other = w.where_was(T0 + 5 * MIN, now, "room_1", "cross door_22 expect=room_3");
        CHECK(other.contains("15 min ago") and other.contains("in room_3 at pose:room_3(1.20, -0.40, 90deg)"));
        CHECK(other.contains("it is in room_1 now") and other.contains("way there: cross door_22 expect=room_3"));
        CHECK(other.contains("plan_mission") and other.contains("pose:room_3(1.20, -0.40)"));
        const std::string same = w.where_was(T0 + MIN, now, "room_1");
        CHECK(same.contains("in room_1 at pose:room_1(-1.09, -0.97") and same.contains("the room it is in now"));
        CHECK(w.where_was(T0 - MIN, now, "room_1").starts_with("no record before"));
        // "13:38" names the whole minute: a record that starts inside it answers (the chat started a few s after)
        CHECK(w.where_was(T0 - 5'000, now, "room_1").starts_with("no record before"));
        CHECK(w.where_was(T0 - 5'000, now, "room_1", {}, 60'000).contains("in room_1 at pose:room_1(-1.09, -0.97"));
        CHECK(w.where_was(T0 - 61'000, now, "room_1", {}, 60'000).starts_with("no record before"));
        CHECK(w.where_was(now + MIN, now, "room_1") == "that time is in the future");
        CHECK(Whereabouts{}.where_was(T0, now, "room_1").starts_with("no record yet"));
    }
    // ── ago_text ──
    CHECK(ago_text(40'000) == "40 s ago" and ago_text(12 * MIN) == "12 min ago" and ago_text(125 * MIN) == "2 h 5 min ago");

    std::println("test_whereabouts: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
