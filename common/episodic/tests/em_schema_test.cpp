// The episodic record (em_schema.h): JSON round trip, absent stays absent, unknown level refused, day_of.
#include "../em_schema.h"

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

using namespace rc::em;

Record full_claim()
{
    Record r;
    r.id = "20261017-3";
    r.parent = "20261017-1";
    r.level = Level::Claim;
    r.author = Author::Self;
    r.kind = "cross";
    r.target = "door_27";
    r.target_ltsm = "passage_0";
    r.action_json = R"({"op":"cross","door":{"node":"door_27"}})";
    r.source = "self";
    r.t0 = 1792238400123ULL;
    r.t1 = 1792238460456ULL;
    r.room = "room_1";
    r.pre = {Fact{.node = "door_27", .type = "door", .ltsm_id = "passage_0",
                  .values = {{"door_open_prob", 0.25}, {"x", 1.5}, {"cov_xx", 0.0004}}}};
    r.post = {Fact{.node = "door_27", .type = "door", .ltsm_id = "", .values = {{"door_open_prob", 0.9}}}};
    r.outcome = "ok";
    r.attempt = 2;
    r.iteration = 7;
    r.promised = 1.25;
    r.delivered = 0.0;      // a real zero: must survive, unlike an absent value
    r.alternatives_json = R"([{"kind":"cross","score":0.4}])";
    r.events = {Event{.dt_ms = 1500, .kind = "current_moved", .node = "room_2"}};
    r.mission_key = "m12";
    r.step_id = "2.1";
    r.serial_start = 41;
    r.serial_end = 57;
    r.room_after = "room_2";
    r.t_post = 1792319746401;
    r.settled = true;
    r.scene_pre = Scene{.things = {"door_3", "room_1"}, .offered = {"aff_door_3_open", "afford_room"}};
    r.scene_post = Scene{.things = {}, .offered = {}};      // empty is a reading, not absence
    return r;
}

void test_roundtrip()
{
    const Record r = full_claim();
    const auto back = record_from_json(to_json(r));
    CHECK(back.has_value());
    CHECK(back.has_value() and *back == r);
    CHECK(back.has_value() and back->delivered.has_value() and *back->delivered == 0.0);
}

void test_absent_stays_absent()
{
    Record r = full_claim();
    r.promised.reset();
    r.delivered.reset();
    r.t1.reset();
    r.serial_start.reset();
    r.serial_end.reset();
    r.room_after.clear();
    r.t_post.reset();
    r.settled = false;
    r.scene_pre.reset();
    r.scene_post.reset();
    r.pre = {Fact{.node = "door_27", .type = "door", .ltsm_id = "", .values = {{"x", 1.0}}}};
    const QJsonObject j = to_json(r);
    CHECK(not j.contains("promised"));
    CHECK(not j.contains("delivered"));
    CHECK(not j.contains("t1"));
    CHECK(not j.contains("serial_start") and not j.contains("serial_end"));
    CHECK(not j.contains("room_after") and not j.contains("t_post") and not j.contains("settled"));
    CHECK(not j.contains("scene_pre") and not j.contains("scene_post"));
    CHECK(j["mission_key"].toString() == "m12");
    const QJsonObject f = j["pre"].toArray().at(0).toObject();
    CHECK(not f["values"].toObject().contains("door_open_prob"));
    const auto back = record_from_json(j);
    CHECK(back.has_value());
    CHECK(back.has_value() and not back->promised.has_value());
    CHECK(back.has_value() and not back->delivered.has_value());
    CHECK(back.has_value() and not back->t1.has_value());
    CHECK(back.has_value() and not back->serial_start and not back->serial_end and back->mission_key == "m12");
    CHECK(back.has_value() and not back->pre.at(0).values.contains("door_open_prob"));
    CHECK(back.has_value() and back->room_after.empty() and not back->t_post and not back->settled);
    CHECK(back.has_value() and not back->scene_pre and not back->scene_post);
}

void test_unknown_level_rejected()
{
    QJsonObject j = to_json(full_claim());
    j["level"] = "fly";
    CHECK(not record_from_json(j).has_value());
    j = to_json(full_claim());
    j["author"] = "alien";
    CHECK(not record_from_json(j).has_value());
}

void test_day_of()
{
    using namespace std::chrono;
    // Noon UTC is the same calendar date in every zone from UTC-11 to UTC+11.
    const sys_time<milliseconds> noon = sys_days{year{2026} / 10 / 17} + hours{12};
    CHECK(day_of(std::uint64_t(noon.time_since_epoch().count())) == "20261017");
    // Same instant, computed through the local zone the way the contract defines it.
    const auto t = sys_days{year{2026} / 1 / 2} + hours{23} + minutes{30};
    const year_month_day ymd{floor<days>(zoned_time{current_zone(), floor<milliseconds>(t)}.get_local_time())};
    const std::string want = std::format("{:04}{:02}{:02}", int(ymd.year()), unsigned(ymd.month()), unsigned(ymd.day()));
    CHECK(day_of(std::uint64_t(duration_cast<milliseconds>(t.time_since_epoch()).count())) == want);
}
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "es_ES.UTF-8") or std::setlocale(LC_ALL, "");
    test_roundtrip();
    test_absent_stays_absent();
    test_unknown_level_rejected();
    test_day_of();
    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
