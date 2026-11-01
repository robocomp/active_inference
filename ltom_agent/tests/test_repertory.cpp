// Offline test of src/repertory.h (pure model).
#include "../src/repertory.h"

#include <cstdio>
#include <numbers>
#include <print>

using namespace ltom;

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

constexpr std::uint64_t kUid = (20ull << 40) | 7;
constexpr std::uint32_t kOwner = 20;
constexpr std::int64_t kGrace = 3000;

static Sighting mk(std::uint64_t uid = kUid, double x = 1.0, std::int64_t room = 1)
{
    Sighting s;
    s.uid = uid; s.cls = "chair"; s.name = "chair_1"; s.room_key = room;
    s.pose = Pose2{x, 2.0, 0.3, 0.03 * 0.03, 0.03 * 0.03, 0.01};
    s.w = 0.5f; s.d = 0.5f; s.h = 1.0f; s.born_ms = 100;
    return s;
}
static Changes obs(Repertory& r, std::vector<Sighting> v, std::set<std::uint32_t> owners, std::int64_t t,
                   std::set<std::uint64_t> extra_present = {})
{
    std::set<std::uint64_t> present = std::move(extra_present);
    for (const auto& s : v) present.insert(s.uid);
    return r.observe(v, owners, present, t);
}
static const std::set<std::uint32_t> kHere{kOwner};
static const std::set<std::uint32_t> kGone{};

static void test_new_uid_creates_entry_and_keyframe()
{
    Repertory r(kGrace);
    const auto ch = obs(r, {mk()}, kHere, 1000);
    CHECK(ch.born.size() == 1 and ch.born[0] == kUid);
    const Entry& e = r.entries().at(kUid);
    CHECK(e.lives.size() == 1 and e.lives[0].born_ms == 100 and e.lives[0].first_seen_ms == 1000);
    CHECK(e.keyframes.size() == 1 and e.n_sightings == 1);
    CHECK(r.dirty());
    r.clear_dirty(); CHECK(not r.dirty());
}
static void test_resighting_counts_without_keyframe_when_still()
{
    Repertory r(kGrace);
    for (int i = 0; i < 10; ++i) obs(r, {mk()}, kHere, 1000 + 100 * i);
    const Entry& e = r.entries().at(kUid);
    CHECK(e.n_sightings == 10 and e.keyframes.size() == 1 and e.lives[0].last_seen_ms == 1900);
}
static void test_moved_object_keyframes()
{
    Repertory r(kGrace);
    obs(r, {mk()}, kHere, 1000);
    obs(r, {mk(kUid, 1.5)}, kHere, 1100);
    CHECK(r.entries().at(kUid).keyframes.size() == 2);
}
static void test_room_change_keyframes()
{
    Repertory r(kGrace);
    obs(r, {mk()}, kHere, 1000);
    obs(r, {mk(kUid, 1.0, 2)}, kHere, 1100);
    CHECK(r.entries().at(kUid).keyframes.size() == 2);
}
static void test_death_after_grace()
{
    Repertory r(kGrace);
    obs(r, {mk()}, kHere, 1000);
    CHECK(obs(r, {}, kHere, 2000).died.empty());
    CHECK(obs(r, {}, kHere, 4900).died.empty());
    const auto ch = obs(r, {}, kHere, 5000);   // now - vanish >= grace
    CHECK(ch.died.size() == 1 and ch.owner_exited.empty());
    const Life& l = r.entries().at(kUid).lives[0];
    CHECK(l.end == End::death and l.end_ms == 2000);
}
static void test_owner_exit_after_graceful_cleanup_order()
{
    Repertory r(kGrace);
    obs(r, {mk()}, kHere, 1000);
    CHECK(obs(r, {}, kHere, 2000).owner_exited.empty());
    const auto ch = obs(r, {}, kGone, 2500);
    CHECK(ch.owner_exited.size() == 1 and ch.died.empty());
    CHECK(r.entries().at(kUid).lives[0].end == End::owner_exit and r.entries().at(kUid).lives[0].end_ms == 2000);
    CHECK(obs(r, {}, kGone, 9000).died.empty());
}
static void test_owner_absent_at_vanish_is_owner_exit()
{
    Repertory r(kGrace);
    obs(r, {mk()}, kHere, 1000);
    const auto ch = obs(r, {}, kGone, 2000);
    CHECK(ch.owner_exited.size() == 1 and r.entries().at(kUid).lives[0].end_ms == 2000);
}
static void test_flap_within_grace_is_not_an_end()
{
    Repertory r(kGrace);
    obs(r, {mk()}, kHere, 1000);
    obs(r, {}, kHere, 2000);
    const auto ch = obs(r, {mk()}, kHere, 3000);
    CHECK(ch.born.empty() and ch.reopened.empty());
    CHECK(obs(r, {}, kHere, 9000).died.empty());      // pending restarts from the new vanish
    const Entry& e = r.entries().at(kUid);
    CHECK(e.lives.size() == 1 and e.lives[0].end == End::alive and e.lives[0].end_ms == -1);
}
static void test_gap_then_same_uid_reopens()
{
    Repertory r(kGrace);
    Entry e; e.uid = kUid; e.lives.push_back(Life{100, 1000, 1500, -1, End::alive});
    e.keyframes.push_back(Keyframe{1000, 1, mk().pose});
    r.insert_loaded(e);
    r.close_open_lives_as_gap();
    const Life& g = r.entries().at(kUid).lives[0];
    CHECK(g.end == End::gap and g.end_ms == 1500);
    const auto ch = obs(r, {mk()}, kHere, 8000);
    CHECK(ch.reopened.size() == 1 and ch.born.empty());
    const Entry& f = r.entries().at(kUid);
    CHECK(f.lives.size() == 1 and f.lives[0].end == End::alive and f.lives[0].end_ms == -1);
}
static void test_death_then_same_uid_is_new_life()
{
    Repertory r(kGrace);
    obs(r, {mk()}, kHere, 1000);
    obs(r, {}, kHere, 2000);
    obs(r, {}, kHere, 6000);
    CHECK(r.entries().at(kUid).lives[0].end == End::death);
    const auto ch = obs(r, {mk()}, kHere, 7000);
    CHECK(ch.born.size() == 1 and r.entries().at(kUid).lives.size() == 2);
}
static void test_present_but_unresolved_is_held()
{
    Repertory r(kGrace);
    obs(r, {mk()}, kHere, 1000);
    for (int i = 0; i < 10; ++i)
    {
        const auto ch = obs(r, {}, kHere, 2000 + 600 * i, {kUid});   // still on the graph, just not resolvable
        CHECK(ch.died.empty() and ch.owner_exited.empty());
    }
    const Entry& e = r.entries().at(kUid);
    CHECK(e.lives.size() == 1 and e.lives[0].end == End::alive and e.lives[0].end_ms == -1 and e.n_sightings == 1);
    // owner gone while merely unresolvable is still not an end: the node is there
    CHECK(obs(r, {}, kGone, 9000, {kUid}).owner_exited.empty());
}
static void test_pending_cleared_when_present_again()
{
    Repertory r(kGrace);
    obs(r, {mk()}, kHere, 1000);
    obs(r, {}, kHere, 2000);                       // vanishes: pending
    obs(r, {}, kHere, 3000, {kUid});               // back on the graph (unresolved): flap, pending cleared
    CHECK(obs(r, {}, kHere, 5100).died.empty());   // would be death had the pending survived (5100-2000 >= 3000)
    CHECK(r.entries().at(kUid).lives[0].end == End::alive);
}
static void test_yaw_wrap_no_keyframe()
{
    Repertory r(kGrace);
    Sighting a = mk(); a.pose.yaw = 0.1; a.pose.vyaw = 0.05 * 0.05;
    Sighting b = a;    b.pose.yaw = 2 * std::numbers::pi - 0.1;
    obs(r, {a}, kHere, 1000);
    obs(r, {b}, kHere, 1100);
    CHECK(r.entries().at(kUid).keyframes.size() == 1);
}
static void test_owner_of() { CHECK(owner_of((20ull << 40) | 7) == 20); }

int main()
{
    test_new_uid_creates_entry_and_keyframe();
    test_resighting_counts_without_keyframe_when_still();
    test_moved_object_keyframes();
    test_room_change_keyframes();
    test_death_after_grace();
    test_owner_exit_after_graceful_cleanup_order();
    test_owner_absent_at_vanish_is_owner_exit();
    test_flap_within_grace_is_not_an_end();
    test_gap_then_same_uid_reopens();
    test_death_then_same_uid_is_new_life();
    test_present_but_unresolved_is_held();
    test_pending_cleared_when_present_again();
    test_yaw_wrap_no_keyframe();
    test_owner_of();
    if (failures) return 1;
    std::println("test_repertory: ok");
}
