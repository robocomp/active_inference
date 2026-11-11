// object_facts.h: the STATE's OBJECTS section, the query tools and the change tracking, on synthetic facts.
// Build+run: tests/run_tests.sh
#include "../src/object_facts.h"
#include <cassert>
#include <print>

using namespace rc::chat;

namespace
{
Remembered undated_room_only() { return {.objects = {{.name = "chair_1", .subtype = "chair"}}, .now_ms = 1}; }

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
    all[2].shape = "round"; all[2].shape_prob = 0.87f;

    const std::string sec = objects_section(all, 6, 2);
    std::println("{}", sec);
    assert(sec.find("1 bottle, 1 cabinet, 3 chair, 1 hood, 1 person, 1 table") != std::string::npos); ++checks;
    assert(sec.find("refrigerator") == std::string::npos); ++checks;                  // other room: not listed
    assert(sec.find("- person_1 1.2 m, +170 deg behind") != std::string::npos); ++checks;   // nearest first
    assert(sec.find("table_1 2.8 m, +10 deg ahead, 1.20x0.80x0.75 m, round top (p 0.87)") != std::string::npos); ++checks;
    assert(sec.find("chair_3") == std::string::npos and sec.find("hood_1") == std::string::npos); ++checks;   // > N
    assert(sec.find("find_objects lists the rest") != std::string::npos); ++checks;
    assert(sec.find("unidentified obstacles") != std::string::npos); ++checks;
    assert(objects_section({}, 6, 0).find(": none") != std::string::npos); ++checks;
    // an object that offers its approach: its sides, labelled as seen from here (approach.h sides_line)
    assert(object_line(ObjectFact{.name = "chair_1", .subtype = "chair", .approach_sides = "front (its own front; far side as seen from here)"})
               .ends_with(", approach sides: front (its own front; far side as seen from here)")); ++checks;
    assert(object_line(ObjectFact{.name = "chair_2", .subtype = "chair"}).find("approach") == std::string::npos); ++checks;

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

    // Long-term memory: what the tools add about objects that are NOT on the live graph right now.
    {
        constexpr std::int64_t now = 1'800'000'000'000;
        const Remembered mem{.objects = {
            {.name = "chair_34", .subtype = "chair", .room = "room_1", .last_seen_ms = now - 25 * 60'000},
            {.name = "table_9", .subtype = "table", .room = "", .last_seen_ms = now - 3 * 3'600'000},
            {.name = "chair_35", .subtype = "chair", .room = "room_3", .last_seen_ms = now - 30'000}},
            .now_ms = now};
        const auto contains = [](const std::string& s, std::string_view p) { return s.find(p) != std::string::npos; };

        // find_objects: the current room as before, then what is remembered, most recent first, marked as not perceived.
        auto r = answer_find_objects(all, "chair", "room_1", mem);
        assert(contains(r, "perceived in room_1")); ++checks;
        assert(contains(r, "NOT perceived now")); ++checks;
        assert(contains(r, "chair_34 (last seen in room_1, 25 min ago)")); ++checks;
        assert(contains(r, "chair_35 (last seen in room_3, less than a minute ago)")); ++checks;
        assert(r.find("chair_35") < r.find("chair_34")); ++checks;
        assert(not contains(r, "table_9")); ++checks;

        // A class with nothing live: still the honest "not perceived" sentence, then the memory.
        r = answer_find_objects({}, "table", "room_2", mem);
        assert(contains(r, "No table perceived")); ++checks;
        assert(contains(r, "table_9 (last seen in an unknown room, 3 h ago)")); ++checks;

        // No memory joined, or none of that class: exactly the old answer.
        assert(answer_find_objects(all, "chair", "room_1", Remembered{}) == answer_find_objects(all, "chair", "room_1")); ++checks;
        assert(answer_find_objects({}, "sofa", "room_2", mem) == answer_find_objects({}, "sofa", "room_2")); ++checks;

        // where_is by name: only memory knows it, and it never claims the object is there now.
        r = answer_where_is({}, "chair_34", "room_2", mem);
        assert(contains(r, "chair_34 (last seen in room_1, 25 min ago)")); ++checks;
        assert(contains(r, "NOT perceived now")); ++checks;
        assert(not contains(r, "is not perceived. The robot has not seen it")); ++checks;

        // where_is by name: the live object wins over a remembered one of the same name.
        std::vector<ObjectFact> live_named{{.id = 1, .name = "chair_34", .subtype = "chair", .room = "room_2"}};
        r = answer_where_is(live_named, "chair_34", "room_1", mem);
        assert(contains(r, "in room_2")); ++checks;
        assert(not contains(r, "last seen")); ++checks;

        // where_is by class: every live one, then the remembered ones.
        r = answer_where_is({}, "chair", "room_2", mem);
        assert(contains(r, "chair_34 (last seen")); ++checks;
        assert(contains(r, "chair_35 (last seen")); ++checks;
        assert(not contains(r, "table_9")); ++checks;

        // Neither live nor remembered: the old sentence.
        assert(answer_where_is({}, "sofa", "room_2", mem) == answer_where_is({}, "sofa", "room_2")); ++checks;

        // The planner's line: where to drive, only for objects whose room is known, marked as a memory.
        const Remembered places{.objects = {
            {.name = "chair_39", .subtype = "chair", .room = "room_2", .last_seen_ms = now, .x = 1.2, .y = -0.5},
            {.name = "table_9", .subtype = "table", .room = "", .last_seen_ms = now}},
            .now_ms = now};
        const auto line = remembered_places_line(places);
        assert(contains(line, "chair_39 in room_2 at pose:room_2(1.20, -0.50)")); ++checks;
        assert(contains(line, "NOT seen now")); ++checks;
        assert(not contains(line, "table_9")); ++checks;
        assert(remembered_places_line(Remembered{}).empty()); ++checks;
        assert(is_remembered(places, "chair_39")); ++checks;
        assert(not is_remembered(places, "chair_40")); ++checks;
        assert(remembered_places_line(undated_room_only()).empty()); ++checks;
        // in the robot's own room the line also gives the turn that faces it (computed in code: no model trigonometry)
        Remembered turned = places;
        turned.objects[0].turn_deg = 145;
        assert(contains(remembered_places_line(turned),
                        "chair_39 in room_2 at pose:room_2(1.20, -0.50), from here: move turn_deg=145deg")); ++checks;

        // The age in words, and a stamp that is missing or in the future.
        assert(age_words(90 * 60'000) == "1 h ago"); ++checks;
        assert(age_words(3 * 86'400'000LL) == "3 days ago"); ++checks;
        assert(age_words(-5) == "less than a minute ago"); ++checks;
        const Remembered undated{.objects = {{.name = "chair_1", .subtype = "chair", .room = "room_1"}}, .now_ms = now};
        assert(contains(answer_where_is({}, "chair_1", "room_2", undated), "chair_1 (last seen in room_1, time unknown)")); ++checks;

        // from_memory: set when the answer carries remembered objects, so the chat says it AS IT IS (live 2026-11-10:
        // gemma4 reworded the block, moved chair_39 to room_1 and dropped "it may have moved").
        bool fm = false;
        (void)answer_where_is({}, "chair_34", "room_2", mem, &fm); assert(fm); ++checks;              // by name, memory only
        fm = false; (void)answer_where_is({}, "chair", "room_2", mem, &fm); assert(fm); ++checks;     // by class, memory only
        fm = true; (void)answer_where_is(live_named, "chair_34", "room_1", mem, &fm); assert(not fm); ++checks;   // live wins
        fm = true; (void)answer_where_is({}, "sofa", "room_2", mem, &fm); assert(not fm); ++checks;   // nothing at all
        fm = false; (void)answer_find_objects({}, "chair", "room_2", mem, &fm); assert(fm); ++checks;
        fm = true; (void)answer_find_objects({}, "sofa", "room_2", mem, &fm); assert(not fm); ++checks;
        fm = true; (void)answer_find_objects(all, "chair", "room_1", Remembered{}, &fm); assert(not fm); ++checks;
    }

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
    {
        // The robot's MEASURED motion (live 2026-10-18: "you ARE moving" — the model said it was not, since no
        // command of its own was running; the STATE carried no velocity at all).
        const std::string m = motion_line(0.42f, 0.0f, 0.21f, 120);
        assert(m == "robot motion now (odometry, 120 ms old): forward 0.42 m/s, sideways 0.00 m/s, turning left 12 deg/s"); ++checks;
        assert(motion_line(-0.30f, 0.05f, -0.5f, 40) ==
               "robot motion now (odometry, 40 ms old): backward 0.30 m/s, sideways 0.05 m/s, turning right 29 deg/s"); ++checks;
        assert(motion_line(0.f, 0.f, 0.f, 3000).find("3000 ms old") != std::string::npos); ++checks;   // stale: said, not hidden
    }
    std::println("test_objects: {} checks passed", checks);
    return 0;
}
