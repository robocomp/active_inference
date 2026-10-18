// approach_object's pure side (src/approach.h): which object, where to stand, and the one-step go_to mission.
#include "../src/approach.h"

#include <clocale>
#include <cmath>
#include <print>

#include "../../common/mission/mission_json.h"

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
bool near(double a, double b) { return std::abs(a - b) < 1e-6; }

using namespace rc::chat;
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");   // es_ES here: the mission JSON must still carry decimal points

    // ── which object: a name, else the nearest of a class, in the CURRENT room only ─────────────────────────────
    const std::vector<ObjectFact> objs{
        {.id = 1, .name = "chair_1", .subtype = "chair", .room = "room_1", .in_current_room = true, .dist_m = 2.2},
        {.id = 2, .name = "chair_2", .subtype = "chair", .room = "room_1", .in_current_room = true, .dist_m = 1.4},
        {.id = 3, .name = "table_1", .subtype = "table", .room = "room_1", .in_current_room = true, .dist_m = 2.3,
         .width_m = 0.71f, .depth_m = 0.78f},
        {.id = 4, .name = "fridge_1", .subtype = "refrigerator", .room = "room_2", .in_current_room = false},
    };
    {
        const auto a = pick_object(objs, "chair_1");
        CHECK(a.has_value() and a->name == "chair_1");          // by name, even when not the nearest chair
        const auto b = pick_object(objs, "the chair");
        CHECK(b.has_value() and b->name == "chair_2");          // by class: the nearest
        const auto c = pick_object(objs, "Table");
        CHECK(c.has_value() and c->name == "table_1");
        const auto d = pick_object(objs, "fridge");
        CHECK(not d.has_value() and d.error().find("room_2") != std::string::npos);   // seen, but not here
        const auto e = pick_object(objs, "sofa");
        CHECK(not e.has_value() and e.error().find("not perceived") != std::string::npos);
    }

    // ── where to stand: on the line object→robot, footprint radius + standoff from the object's centre ────────────
    {
        // object at (2,0), robot at (0,0), footprint radius 0.4, standoff 0.65 → stand at x = 2 - 1.05 = 0.95
        const auto p = approach_point({2.0, 0.0}, {0.0, 0.0}, 0.4, 0.65);
        CHECK(p.has_value() and near(p->x, 0.95) and near(p->y, 0.0));
        CHECK(p.has_value() and near(p->yaw, 0.0));                     // facing the object (+x)
        const auto q = approach_point({0.0, 3.0}, {0.0, 0.0}, 0.5, 0.5);
        CHECK(q.has_value() and near(q->x, 0.0) and near(q->y, 2.0) and near(q->yaw, std::numbers::pi / 2));
        CHECK(not approach_point({1.0, 0.0}, {0.0, 0.0}, 0.4, 0.65).has_value());   // already that close
        CHECK(near(footprint_radius(ObjectFact{.width_m = 0.6f, .depth_m = 0.8f}), 0.5));   // half the diagonal
        CHECK(near(footprint_radius(ObjectFact{}), 0.0));                                     // size unknown: centre
    }

    // ── the mission: one go_to to that pose, in the current room ─────────────────────────────────────────────────
    {
        const rc::mission::Mission m = approach_mission("table_1", "room_1", ApproachPoint{.x = 0.95, .y = -0.25, .yaw = 0.5});
        CHECK(m.title == "approach table_1");
        CHECK(m.steps.size() == 1 and m.steps[0].op == "go_to");
        const auto to = m.steps[0].ref("to");
        CHECK(to.has_value() and to->kind == rc::mission::Ref::Kind::Pose and to->pose.room == "room_1"
              and near(to->pose.x, 0.95) and near(to->pose.y, -0.25));
        const std::string json = rc::mission::encode_json(m);
        CHECK(json.starts_with("{"));                                 // the controller's inbox reads '{' as a mission
        CHECK(json.find("0.95") != std::string::npos and json.find("0,95") == std::string::npos);
        const auto back = rc::mission::decode_json(json);
        CHECK(back.has_value() and *back == m);                       // and its own validator accepts it
        CHECK(display_command(json) == "mission \"approach table_1\"");   // the STATE / status lines, not the JSON
        CHECK(display_command("cross door_1") == "cross door_1");
    }

    // ── an object's approach SIDES (its aff_<object>_approach options): where each is, as seen from the robot ──────
    {
        // looking from (0,0) at an object at (2,0): +y is the robot's left
        CHECK(view_of({2, 0}, {0, 0}, {1.3, 0}) == "near");
        CHECK(view_of({2, 0}, {0, 0}, {2.7, 0}) == "far");
        CHECK(view_of({2, 0}, {0, 0}, {2, 0.7}) == "left");
        CHECK(view_of({2, 0}, {0, 0}, {2, -0.7}) == "right");

        const auto bad = parse_sides("xn,yp", {1.3}, {0, 0.7}, {0, 0});
        CHECK(not bad.has_value());                                   // parallel lists of different length
        const auto t = parse_sides("xn,yp,yn,xp", {1.3, 2, 2, 2.7}, {0, 0.7, -0.7, 0}, {0, -1.57, 1.57, 3.14});
        CHECK(t.has_value() and t->size() == 4 and t->front().name == "xn");
        auto table = *t;
        label_sides(table, {2, 0}, {0, 0});
        CHECK(table[0].view == "near" and table[1].view == "left" and table[2].view == "right" and table[3].view == "far");

        // a frontless object: the person's words are taken as seen from the robot
        CHECK((resolve_side(table, "") == std::expected<std::string, std::string>("")));   // the object's default
        CHECK((resolve_side(table, "right") == std::expected<std::string, std::string>("yn")));
        CHECK((resolve_side(table, "the right side") == std::expected<std::string, std::string>("yn")));
        CHECK((resolve_side(table, "Derecha") == std::expected<std::string, std::string>("yn")));
        CHECK((resolve_side(table, "por la izquierda") == std::expected<std::string, std::string>("yp")));
        CHECK((resolve_side(table, "the far side") == std::expected<std::string, std::string>("xp")));
        CHECK((resolve_side(table, "behind it") == std::expected<std::string, std::string>("xp")));
        CHECK((resolve_side(table, "xp") == std::expected<std::string, std::string>("xp")));   // the option itself
        auto against_wall = table;
        against_wall.pop_back();                                      // xp not offered
        const auto r = resolve_side(against_wall, "far side");
        CHECK(not r.has_value() and r.error().find("xn") != std::string::npos and r.error().find("near") != std::string::npos);
        CHECK(not resolve_side(table, "upside").has_value());

        // a fronted object: left/right/front/back are the OBJECT's own; near/far are still as seen from here
        auto chair = *parse_sides("front,left,back,right", {2, 2, 2, 2.7}, {0.7, -0.7, -0.7, 0}, {0, 0, 0, 0});
        label_sides(chair, {2, 0}, {0, 0});
        CHECK((resolve_side(chair, "its right") == std::expected<std::string, std::string>("right")));   // not the view
        CHECK((resolve_side(chair, "from behind") == std::expected<std::string, std::string>("back")));
        CHECK((resolve_side(chair, "de frente") == std::expected<std::string, std::string>("front")));
        CHECK((resolve_side(chair, "the far side") == std::expected<std::string, std::string>("right")));   // its view

        // the STATE line
        CHECK(sides_line(table) == "xn (near side, as seen from here), yp (left side, as seen from here), "
                                   "yn (right side, as seen from here), xp (far side, as seen from here)");
        CHECK(sides_line(chair).starts_with("front (its own front; left side as seen from here), "));
    }

    std::println("test_approach: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
