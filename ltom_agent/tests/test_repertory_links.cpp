// Offline test of the identity link in the repertory (spec 2026-11-15-changed-object, owner's choice B): a live object
// believed to be a remembered one that moved; the owner's "no" cuts it and marks the memory gone from that room.
#include "../src/recall_select.h"
#include "../src/repertory.h"
#include "../src/repertory_io.h"

#include <clocale>
#include <filesystem>
#include <fstream>
#include <print>
#include <unistd.h>

using namespace ltom;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

constexpr std::uint64_t kMem = (1ull << 40) | 50;    // chair_50, remembered in room 2
constexpr std::uint64_t kLive = (1ull << 40) | 57;   // chair_57, born new this run
static const std::vector<std::string> kChairs{"chair"};

static Keyframe kf(std::int64_t t, std::int64_t room, double x, double y)
{
    Keyframe k; k.t_ms = t; k.room_key = room; k.pose.x = x; k.pose.y = y; k.pose.vx = k.pose.vy = 0.01; k.pose.vyaw = 0.1;
    return k;
}
// chair_50 died at t=100 in room 2 at (-2.57, -0.61), magenta.
static Repertory with_memory()
{
    Repertory r(1000);
    Entry m; m.uid = kMem; m.cls = "chair"; m.name = "chair_50"; m.color_name = "magenta"; m.color_prob = 0.74f;
    m.lives.push_back(Life{.born_ms = 10, .first_seen_ms = 10, .last_seen_ms = 100, .end_ms = 100, .end = End::death});
    m.keyframes = {kf(50, 2, -2.57, -0.61)};
    r.insert_loaded(m);
    return r;
}
static Sighting live(float same_prob, float left_prob)
{
    Sighting s; s.uid = kLive; s.cls = "chair"; s.name = "chair_57"; s.room_key = 2; s.born_ms = 1000;
    s.pose.x = 1.2; s.pose.y = 1.5; s.pose.vx = s.pose.vy = 0.01; s.pose.vyaw = 0.1;
    s.link = LinkObs{kMem, same_prob, left_prob};
    return s;
}
static void see(Repertory& r, const Sighting& s, std::int64_t now)
{
    const std::vector<Sighting> v{s};
    r.observe(v, {1}, {s.uid}, now);
}

static void record_link_keeps_old_keyframe()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.2f), 1000);
    const Entry& m = r.entries().at(kMem);
    CHECK(m.continued_by == kLive);
    CHECK(r.entries().at(kLive).continues == kMem);
    CHECK(m.moves.size() == 1);
    CHECK(m.moves.back().from.pose.x == -2.57 and m.moves.back().from.room_key == 2);
    CHECK(m.moves.back().to.pose.x == 1.2);
    CHECK(not m.moves.back().confirmed);
    CHECK(m.keyframes.size() == 1);   // the remembered pose stays in the history
    see(r, live(0.95f, 0.8f), 1100);
    CHECK(r.entries().at(kMem).moves.size() == 1);   // one link, one move: not re-recorded each sighting
}
static void below_map_no_link()
{
    auto r = with_memory();
    see(r, live(0.4f, 0.2f), 1000);
    CHECK(not r.entries().at(kMem).continued_by.has_value());
}
static void link_dropped_below_map_cuts_without_gone()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.2f), 1000);
    see(r, live(0.3f, 0.2f), 1100);
    const Entry& m = r.entries().at(kMem);
    CHECK(not m.continued_by.has_value());
    CHECK(not r.entries().at(kLive).continues.has_value());
    CHECK(m.moves.empty());
    CHECK(m.gone.empty());
}
static void confirmed_link_survives_a_low_sighting()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.2f), 1000);
    r.confirm_link(kMem);
    CHECK(r.entries().at(kMem).moves.back().confirmed);
    see(r, live(0.3f, 0.2f), 1100);
    CHECK(r.entries().at(kMem).continued_by == kLive);   // the owner said yes: evidence no longer undoes it
}
static void gone_from_room_not_recalled()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    r.cut_link(kMem, 1200, "no, I didn't");
    const Entry& m = r.entries().at(kMem);
    CHECK(not m.continued_by.has_value());
    CHECK(m.gone.size() == 1 and m.gone.back().room_key == 2 and m.gone.back().words == "no, I didn't");
    CHECK(m.moves.empty());
    CHECK(select_recalls(r, 2, {kLive}, kChairs).empty());
    CHECK(select_recalls(r, 2, {}, kChairs).empty());
}
static void gone_memory_seen_again_is_recalled_there()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    r.cut_link(kMem, 1200, "no");
    Sighting back; back.uid = kMem; back.cls = "chair"; back.name = "chair_50"; back.room_key = 2; back.born_ms = 1300;
    back.pose.x = -2.5; back.pose.y = -0.6; back.pose.vx = back.pose.vy = 0.01; back.pose.vyaw = 0.1;
    see(r, back, 1300);
    Repertory& rr = r;
    const std::vector<Sighting> none;
    rr.observe(none, {1}, {}, 1400);
    rr.observe(none, {1}, {}, 3000);   // past the grace: dead again
    CHECK(select_recalls(rr, 2, {}, kChairs).size() == 1);   // a later sighting there outdates the "gone" record
}
static void continued_offered_only_while_live_present()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    auto v = select_recalls(r, 2, {kLive}, kChairs);
    CHECK(v.size() == 1 and v[0].uid == kMem and v[0].same_as_uid == kLive);
    CHECK(v[0].same_as_prob > 0.85f and v[0].same_as_prob < 0.95f);
    CHECK(v[0].color_name == "magenta");
    r.confirm_link(kMem);
    CHECK(select_recalls(r, 2, {kLive}, kChairs).at(0).same_as_prob == 1.f);
    CHECK(select_recalls(r, 2, {}, kChairs).empty());
}
static void repertory_io_roundtrips_links_and_questions()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    r.confirm_link(kMem);
    Entry other; other.uid = (1ull << 40) | 44; other.cls = "chair"; other.name = "chair_44";
    other.lives.push_back(Life{.end = End::death});
    other.keyframes = {kf(40, 2, -2.5, -0.6)};
    other.gone.push_back(GoneFromRoom{900, 2, "no"});
    r.insert_loaded(other);
    const fs::path f = fs::temp_directory_path() / ("ltom_links_" + std::to_string(::getpid()) + ".json");
    CHECK(save_repertory(f, r));
    Repertory got(1000);
    CHECK(load_repertory(f, got) == LoadResult::loaded);
    const Entry& m = got.entries().at(kMem);
    CHECK(m.continued_by == kLive);
    CHECK(got.entries().at(kLive).continues == kMem);
    CHECK(m.moves.size() == 1 and m.moves[0].confirmed and m.moves[0].live_uid == kLive);
    CHECK(m.moves[0].from.pose.x == -2.57 and m.moves[0].to.pose.x == 1.2 and m.moves[0].t_ms == 1000);
    const Entry& o = got.entries().at((1ull << 40) | 44);
    CHECK(o.gone.size() == 1 and o.gone[0].t_ms == 900 and o.gone[0].room_key == 2 and o.gone[0].words == "no");
    fs::remove(f);
}
static void older_file_loads_without_links()
{
    const fs::path f = fs::temp_directory_path() / ("ltom_links_old_" + std::to_string(::getpid()) + ".json");
    {
        std::ofstream o(f);
        o << R"({"version": 1, "entries": [{"uid": "1099511627826", "cls": "chair", "name": "chair_50", "w": 0.5, "d": 0.5, "h": 1.0,
              "n_sightings": 3, "lives": [{"born_ms": 1, "first_seen_ms": 1, "last_seen_ms": 2, "end_ms": 2, "end": "death"}],
              "keyframes": [{"t_ms": 1, "room_key": 2, "x": 0.5, "y": 0.5, "yaw": 0, "vx": 0.01, "vy": 0.01, "vyaw": 0.1}]}]})";
    }
    Repertory got(1000);
    CHECK(load_repertory(f, got) == LoadResult::loaded);
    const Entry& e = got.entries().at(1099511627826ull);
    CHECK(not e.continued_by.has_value() and not e.continues.has_value() and e.moves.empty() and e.gone.empty());
    fs::remove(f);
}

int main()
{
    std::setlocale(LC_ALL, "");   // the agents run under es_ES (CLAUDE.md)
    record_link_keeps_old_keyframe();
    below_map_no_link();
    link_dropped_below_map_cuts_without_gone();
    confirmed_link_survives_a_low_sighting();
    gone_from_room_not_recalled();
    gone_memory_seen_again_is_recalled_there();
    continued_offered_only_while_live_present();
    repertory_io_roundtrips_links_and_questions();
    older_file_loads_without_links();
    if (failures == 0) std::println("test_repertory_links: all passed");
    return failures == 0 ? 0 : 1;
}
