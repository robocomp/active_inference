#include "../src/recall_select.h"
#include <algorithm>
#include <cstdio>

using namespace ltom;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static Repertory make(std::initializer_list<Entry> es)
{
    Repertory r(1000);
    for (const auto& e : es) r.insert_loaded(e);
    return r;
}
static Entry ent(std::uint64_t uid, const char* cls, End end, std::vector<Keyframe> kfs)
{
    Entry e; e.uid = uid; e.cls = cls; e.lives.push_back(Life{.end = end}); e.keyframes = std::move(kfs);
    return e;
}
static Keyframe kf(std::int64_t t, std::int64_t room, double x) { Keyframe k; k.t_ms = t; k.room_key = room; k.pose.x = x; return k; }
static const std::vector<std::string> chairs{"chair"};

static void test_dead_chair_in_room_recalled()
{
    auto r = make({ent(5, "chair", End::owner_exit, {kf(10, 1, 2.0)})});
    auto v = select_recalls(r, 1, {}, chairs);
    CHECK(v.size() == 1); CHECK(v[0].uid == 5); CHECK(v[0].cls == "chair"); CHECK(v[0].kf.pose.x == 2.0);
}
static void test_other_room_not_recalled()
{
    auto r = make({ent(5, "chair", End::death, {kf(10, 1, 2.0)})});
    CHECK(select_recalls(r, 2, {}, chairs).empty());
}
static void test_alive_uid_not_recalled()
{
    auto r = make({ent(5, "chair", End::alive, {kf(10, 1, 2.0)}), ent(6, "chair", End::gap, {kf(10, 1, 3.0)})});
    auto v = select_recalls(r, 1, {6}, chairs);
    CHECK(v.empty());
    v = select_recalls(r, 1, {}, chairs);
    CHECK(v.size() == 1 and v[0].uid == 6);
}
static void test_class_filter()
{
    auto r = make({ent(5, "table", End::death, {kf(10, 1, 2.0)}), ent(6, "chair", End::death, {kf(10, 1, 3.0)})});
    auto v = select_recalls(r, 1, {}, chairs);
    CHECK(v.size() == 1 and v[0].uid == 6);
}
static void test_last_keyframe_of_that_room()
{
    auto r = make({ent(5, "chair", End::death, {kf(10, 1, 1.0), kf(20, 2, 7.0)})});
    CHECK(select_recalls(r, 1, {}, chairs).empty());
    auto v = select_recalls(r, 2, {}, chairs);
    CHECK(v.size() == 1 and v[0].kf.pose.x == 7.0);
}
static void test_ordered_by_uid_and_edge_cases()
{
    auto r = make({ent(9, "chair", End::death, {kf(1, 1, 0)}), ent(3, "chair", End::death, {kf(1, 1, 0)}),
                   ent(4, "chair", End::death, {})});
    Entry nolife; nolife.uid = 7; nolife.cls = "chair"; nolife.keyframes = {kf(1, 1, 0)};
    r.insert_loaded(nolife);
    auto v = select_recalls(r, 1, {}, chairs);
    CHECK(v.size() == 2 and v[0].uid == 3 and v[1].uid == 9);
}

int main()
{
    test_dead_chair_in_room_recalled(); test_other_room_not_recalled(); test_alive_uid_not_recalled();
    test_class_filter(); test_last_keyframe_of_that_room(); test_ordered_by_uid_and_edge_cases();
    std::puts(fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
