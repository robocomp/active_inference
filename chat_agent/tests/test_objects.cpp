// object_facts.h: the STATE's OBJECTS section, the query tools and the change tracking, on synthetic facts.
// Build+run: tests/run_tests.sh
#include "../src/object_facts.h"
#include <cassert>
#include <print>

using namespace rc::chat;

namespace
{
ObjectFact obj(std::uint64_t id, std::string name, std::string sub, bool here, double dist, double bearing,
               std::string room = "room_1")
{
    ObjectFact o{.id = id, .name = std::move(name), .subtype = std::move(sub), .room = std::move(room), .in_current_room = here};
    if (dist >= 0) { o.dist_m = dist; o.bearing_deg = bearing; }
    return o;
}
}   // namespace

int main()
{
    int checks = 0;
    std::vector<ObjectFact> all{
        obj(1, "chair_1", "chair", true, 2.1, 35),
        obj(2, "chair_2", "chair", true, 3.4, -100),
        obj(3, "table_1", "table", true, 2.8, 10),
        obj(4, "bottle_1", "bottle", true, 2.9, 12),
        obj(5, "refrigerator_1", "refrigerator", false, -1, 0, "room_2"),
        obj(6, "person_1", "person", true, 1.2, 170),
        obj(7, "cabinet_1", "cabinet", true, 4.0, 80),
        obj(8, "chair_3", "chair", true, 5.0, 0),
        obj(9, "hood_1", "hood", true, -1, 0),
    };
    all[2].width_m = 1.2f; all[2].depth_m = 0.8f; all[2].height_m = 0.75f;

    const std::string sec = objects_section(all, 6, 2);
    std::println("{}", sec);
    assert(sec.find("1 bottle, 1 cabinet, 3 chair, 1 hood, 1 person, 1 table") != std::string::npos); ++checks;
    assert(sec.find("refrigerator") == std::string::npos); ++checks;                  // other room: not listed
    assert(sec.find("- person_1 1.2 m, +170 deg behind") != std::string::npos); ++checks;   // nearest first
    assert(sec.find("table_1 2.8 m, +10 deg ahead, 1.20x0.80x0.75 m") != std::string::npos); ++checks;
    assert(sec.find("chair_3") == std::string::npos and sec.find("hood_1") == std::string::npos); ++checks;   // > N
    assert(sec.find("find_objects lists the rest") != std::string::npos); ++checks;
    assert(sec.find("unidentified obstacles") != std::string::npos); ++checks;
    assert(objects_section({}, 6, 0).find(": none") != std::string::npos); ++checks;

    auto f = answer_find_objects(all, "chairs", "room_1");
    assert(f.starts_with("3 chairs perceived in room_1")); ++checks;
    assert(f.find("chair_1") < f.find("chair_2") and f.find("chair_2") < f.find("chair_3")); ++checks;
    f = answer_find_objects(all, "sofa", "room_1");
    assert(f.starts_with("No sofa perceived") and f.find("does not mean there is none") != std::string::npos); ++checks;
    f = answer_find_objects(all, "fridge", "room_1");
    assert(f.starts_with("No refrigerator perceived")); ++checks;                    // it is in another room
    f = answer_find_objects(all, "", "room_1");
    assert(f.starts_with("8 objects")); ++checks;
    f = answer_find_objects(all, "silla", "room_1");
    assert(f.starts_with("3 chairs")); ++checks;

    auto w = answer_where_is(all, "fridge", "room_1");
    assert(w.find("refrigerator_1 — in room_2") != std::string::npos); ++checks;
    w = answer_where_is(all, "chair_2", "room_1");
    assert(w.find("chair_2 3.4 m, -100 deg to the right — in the current room") != std::string::npos); ++checks;
    w = answer_where_is(all, "the table", "room_1");
    assert(w.find("table_1 2.8 m") != std::string::npos); ++checks;
    w = answer_where_is(all, "sofa", "room_1");
    assert(w.find("is not perceived") != std::string::npos); ++checks;

    std::map<std::uint64_t, ObjectFact> a, b;
    for (const auto& o : all) if (o.in_current_room) a.emplace(o.id, o);
    b = a;
    b.erase(4);
    b.emplace(10, obj(10, "chair_4", "chair", true, 2.1, 40));
    const auto c = diff_objects(a, b);
    assert(c.born.size() == 1 and c.gone.size() == 1); ++checks;
    assert(changes_line(c) == "new: chair_4 (2.1 m, ahead-left); gone: bottle_1"); ++checks;
    assert(diff_objects(a, a).empty() and changes_line(diff_objects(a, a)).empty()); ++checks;

    // Token budget: the section with N=6 nearest stays a few hundred characters (~4 chars/token).
    std::println("OBJECTS section: {} chars (~{} tokens)", sec.size(), sec.size() / 4);
    assert(sec.size() < 800); ++checks;
    std::println("test_objects: {} checks passed", checks);
    return 0;
}
