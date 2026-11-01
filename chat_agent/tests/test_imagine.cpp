// imagined_world.h: opening words, interventions on a copy, the "there" resolver.
#include "../src/imagine_fit.h"

#include <cassert>
#include <print>

using namespace rc::chat;
using namespace rc::chat::imagine;

namespace
{
constexpr float kPi = std::numbers::pi_v<float>;
bool near(float a, float b, float e = 1e-4f) { return std::abs(a - b) < e; }

ImaginedWorld world()
{
    ImaginedWorld w;
    w.scene.room = "room_1";
    SeeDoor d1; d1.name = "door_1"; d1.xy = {2, 0}; d1.width_m = 0.9f; d1.p_open = 0.3f;
    SeeDoor d2; d2.name = "door_2"; d2.xy = {0, 3}; d2.width_m = 0.9f;
    w.scene.doors = {d1, d2};
    SeeObject t; t.name = "table_1"; t.subtype = "table"; t.x = 1; t.y = 1; t.w = 0.6f; t.d = 0.8f;
    SeeObject n; n.name = "chair_1"; n.subtype = "chair"; n.x = -1; n.y = 1;
    w.scene.objects = {t, n};
    w.scene.robot_xy = Eigen::Vector2f(4, 1);
    w.scene.robot_yaw = 0.2f;
    return w;
}
}

int main()
{
    // parse_opening
    assert(near(*parse_opening("shut")->phi_rad, 0));
    assert(near(*parse_opening(" Closed ")->phi_rad, 0));
    assert(near(*parse_opening("ajar")->phi_rad, kPi / 12));
    const auto half = parse_opening("half");
    assert(near(*half->phi_rad, kPi / 4) and half->words.contains("45"));
    assert(near(*parse_opening("half open")->phi_rad, kPi / 4));
    assert(near(*parse_opening("wide")->phi_rad, kPi / 2));
    assert(near(*parse_opening("fully open")->phi_rad, kPi / 2));
    assert(near(*parse_opening("30")->phi_rad, kPi / 6));
    assert(near(*parse_opening("30 deg")->phi_rad, kPi / 6));
    assert(near(*parse_opening("30°")->phi_rad, kPi / 6));
    assert(near(*parse_opening("120")->phi_rad, kPi / 2));
    assert(near(*parse_opening("-5")->phi_rad, 0));
    assert(not parse_opening("as_is")->phi_rad.has_value());
    assert(not parse_opening("")->phi_rad.has_value());
    const auto bad = parse_opening("banana");
    assert(not bad.has_value() and bad.error().starts_with("I don't know how open 'banana' is"));

    // interventions: value semantics
    const ImaginedWorld w0 = world();
    const auto w1 = with_door_phi(w0, "door_1", kPi / 4);
    assert(w1.has_value());
    assert(near(*w1->scene.doors[0].phi_rad, kPi / 4) and not w1->scene.doors[0].p_open.has_value());
    assert(not w1->scene.doors[1].phi_rad.has_value());
    assert(not w0.scene.doors[0].phi_rad.has_value() and w0.scene.doors[0].p_open.has_value() and w0.notes.empty());
    assert(w1->notes.size() == 1);
    const auto nodoor = with_door_phi(w0, "door_9", 0);
    assert(not nodoor.has_value() and nodoor.error() == "I don't know a door called door_9");

    const auto wm = with_object_moved(w0, "table_1", 0.5f, -0.25f);
    assert(wm.has_value() and near(wm->scene.objects[0].x, 1.5f) and near(wm->scene.objects[0].y, 0.75f));
    assert(near(w0.scene.objects[0].x, 1.0f));
    assert(not with_object_moved(w0, "sofa_1", 1, 1).has_value());

    const auto wr = with_robot_at(w0, {1, 2}, 0.5f);
    assert(wr.has_value() and near(wr->scene.robot_xy->x(), 1) and near(wr->scene.robot_yaw, 0.5f));

    // resolve_standpoint
    const auto here = resolve_standpoint(w0, "");
    assert(here.has_value() and near(here->xy.x(), 4) and here->yaw.has_value());
    assert(resolve_standpoint(w0, "room_1").has_value());
    ImaginedWorld lost = w0; lost.scene.robot_xy.reset();
    assert(resolve_standpoint(lost, "here").error() == "I don't know where I am");
    const auto dp = resolve_standpoint(w0, "door_2");
    assert(dp.has_value() and near(dp->xy.y(), 3) and not dp->yaw.has_value());
    const auto op = resolve_standpoint(w0, "table_1");
    assert(op.has_value());
    const float r = 0.5f * std::hypot(0.6f, 0.8f) + kRobotRadiusM;
    assert(near((op->xy - Eigen::Vector2f(1, 1)).norm(), r));
    assert(op->xy.x() > 1);                       // the robot's side (robot at x = 4)
    assert(near(std::cos(*op->yaw), -1.0f));      // facing the object, back toward -x
    const auto byclass = resolve_standpoint(w0, "the table");
    assert(byclass.has_value() and near(byclass->xy.x(), op->xy.x()));   // article stripped
    assert(resolve_standpoint(w0, "table").has_value());
    const auto nosize = resolve_standpoint(w0, "chair_1");
    assert(nosize.has_value() and nosize->words.contains("size is unknown"));
    assert(near((nosize->xy - Eigen::Vector2f(-1, 1)).norm(), kRobotRadiusM));
    assert(resolve_standpoint(w0, "door 2").has_value() and resolve_standpoint(w0, "the door_2").has_value());
    assert(with_door_phi(w0, "the Door-2", 0.1f).has_value() and with_object_moved(w0, "the table_1", 1, 0).has_value());
    assert(near(*parse_opening("half-open")->phi_rad, kPi / 4) and near(*parse_opening("half_open")->phi_rad, kPi / 4));
    assert(near(*parse_opening("halfway")->phi_rad, kPi / 4) and parse_opening("half-open")->words.contains("45"));
    assert(near(*parse_opening("medio abierta")->phi_rad, kPi / 4) and near(*parse_opening("cracked")->phi_rad, kPi / 12));
    assert(near(*parse_opening("de par en par")->phi_rad, kPi / 2) and near(*parse_opening("cerrada")->phi_rad, 0));
    assert(near(*parse_opening("fully_open")->phi_rad, kPi / 2) and near(*parse_opening("all the way")->phi_rad, kPi / 2));
    assert(not parse_opening("as_is")->phi_rad.has_value());
    assert(resolve_standpoint(w0, "the moon").error() == "ASK: which place do you mean by 'there'?");

    // ── Q1: through a door ──
    {
        ImaginedWorld f = world();
        f.body_half_width = 0.31f;
        f.robot_sigma_m = 0.02f; f.door_sigma_m["door_1"] = 0.02f;
        const auto leaves = rc::chat::imagine::fit_detail::leaves_at(0.9f, kPi / 4);
        const auto g = rc::anchor_clear_gap(0.9f, &leaves[0]);
        assert(g.width > 0.2f and g.width < 0.3f);
        const auto a45 = answer_fit_door(f, "door_1", *parse_opening("45"));
        assert(a45.p and *a45.p < 0.05f and near(a45.slack_m, g.width - 0.62f, 1e-3f));
        assert(a45.text.contains("P(fit) = ") and a45.text.contains("cm") and a45.text.contains("45") and a45.text.contains("about 7")
               and a45.text.contains("depth across the doorway") and not a45.text.contains("passable"));
        const auto a90 = answer_fit_door(f, "door_1", *parse_opening("wide"));
        assert(a90.p and *a90.p > 0.9f);
        // phi* round trip
        assert(a45.phi_needed_rad.has_value());
        const auto rt = answer_fit_door(f, "door_1", Opening{*a45.phi_needed_rad, "phi*"});
        assert(rt.p and std::abs(*rt.p - 0.5f) < 0.02f);
        // sigma pulls P toward 0.5, from below and from above
        ImaginedWorld big = f; big.door_sigma_m["door_1"] = 0.3f;
        assert(*answer_fit_door(big, "door_1", *parse_opening("45")).p > *a45.p);
        assert(*answer_fit_door(big, "door_1", *parse_opening("wide")).p < *a90.p);
        // no body model, unknown door, unknown opening, exact geometry, never fits
        ImaginedWorld nb = f; nb.body_half_width.reset();
        assert(not answer_fit_door(nb, "door_1", *parse_opening("45")).p and answer_fit_door(nb, "door_1", *parse_opening("45")).text == "I have no model of my body width");
        assert(answer_fit_door(f, "door_7", *parse_opening("45")).text == "I don't know a door called door_7");
        const auto unk = answer_fit_door(f, "door_1", *parse_opening("as_is"));
        assert(not unk.p and unk.text.contains("shut") and unk.text.contains("wide open"));
        ImaginedWorld ex = f; ex.robot_sigma_m.reset(); ex.door_sigma_m.clear();
        assert(answer_fit_door(ex, "door_1", *parse_opening("wide")).text.contains("treating the geometry as exact"));
        ImaginedWorld fat = f; fat.body_half_width = 0.6f;
        assert(answer_fit_door(fat, "door_1", *parse_opening("wide")).text.contains("even fully open I would not fit"));
        ImaginedWorld thin = f; thin.body_half_width = 0.0f;
        assert(answer_fit_door(thin, "door_1", *parse_opening("wide")).text.contains("I fit at any opening"));
    }

    // ── geometry ──
    {
        using namespace rc::chat::imagine::fit_detail;
        const auto A = box_corners(0, 0, 0, 1, 1), B = box_corners(2, 0, 0, 1, 1);
        assert(near(footprint_distance(A, B), 1.0f));
        const auto C = box_corners(2, 2, 0, 1, 1);
        assert(near(footprint_distance(A, C), std::sqrt(2.0f)));                     // corner to corner
        const auto R = box_corners(2, 0, std::numbers::pi_v<float> / 4, 1, 1);                              // diamond, near vertex at x = 2 - 0.7071
        assert(near(footprint_distance(A, R), 2.0f - std::sqrt(0.5f) - 0.5f));
        assert(near(footprint_distance(A, box_corners(0.5f, 0, 0, 1, 1)), 0.0f));     // overlap
        assert(near(footprint_distance(A, box_corners(0, 0, 0, 0.2f, 0.2f)), 0.0f));  // containment
        assert(near(footprint_distance(A, {Eigen::Vector2f(3, -5), Eigen::Vector2f(3, 5)}), 2.5f));   // box-segment
        assert(near(footprint_distance(A, {Eigen::Vector2f(-5, 2), Eigen::Vector2f(5, 2)}), 1.5f));
        assert(near(footprint_distance(A, {Eigen::Vector2f(-5, 0), Eigen::Vector2f(5, 0)}), 0.0f));   // crosses it
    }

    // ── Q1b: between two referents ──
    {
        ImaginedWorld f = world();
        f.body_half_width = 0.31f;
        f.scene.polygon = {{-5, -5}, {5, -5}, {5, 5}, {-5, 5}};
        f.scene.objects.clear();
        SeeObject c; c.name = "chair_1"; c.subtype = "chair"; c.x = 4.45f - 0.2f; c.y = 0; c.w = 0.4f; c.d = 0.4f; c.h = 0.9f; c.z_origin = 0.f;
        // wall at x = 5; chair's far face at 4.45 ⇒ gap 0.55
        c.x = 4.25f;
        f.scene.objects = {c};
        const auto lo = answer_fit_between(f, "chair_1", "wall");
        assert(lo.p and *lo.p < 0.1f and near(lo.slack_m, 0.55f - 0.62f, 1e-3f) and lo.text.contains("chair_1")
               and lo.text.contains("would have to move") and lo.text.contains("7 cm") and lo.text.contains("live LiDAR"));
        assert(answer_fit_between(f, "the chair", "wall").p.has_value());              // class with one instance
        const auto mv = with_object_moved(f, "chair_1", -0.2f, 0);
        assert(mv.has_value());
        const auto hi = answer_fit_between(*mv, "chair_1", "wall");
        assert(hi.p and *hi.p > *lo.p and near(hi.slack_m, 0.75f - 0.62f, 1e-3f));
        ImaginedWorld wide = f; wide.scene.objects[0].x = 4.1f - 0.2f + 0.2f;          // gap 0.80 - sizes: far face 4.3, gap 0.70
        wide.scene.objects[0].x = 4.0f;                                                 // far face 4.2 ⇒ gap 0.80
        assert(*answer_fit_between(wide, "chair_1", "wall").p > 0.9f);
        // the displacement needed round-trips to P = 0.5
        const float need = -lo.slack_m;
        ImaginedWorld fs = f; fs.robot_sigma_m = 0.05f;
        const auto mv2 = with_object_moved(fs, "chair_1", -need, 0);
        assert(std::abs(*answer_fit_between(*mv2, "chair_1", "wall").p - 0.5f) < 0.02f);
        // a wall shelf above the band does not block; at base 0 it does
        ImaginedWorld sh = f;
        sh.scene.objects[0].z_origin = 1.4f; sh.scene.objects[0].h = 0.3f;
        const auto up = answer_fit_between(sh, "chair_1", "wall");
        assert(up.p and near(*up.p, 1.0f) and up.text.contains("does not block"));
        sh.scene.objects[0].z_origin = 0.f;
        assert(*answer_fit_between(sh, "chair_1", "wall").p < 0.1f);
        // errors
        assert(answer_fit_between(f, "wall", "the wall").text == "between two walls is the room, ask me about a door");
        assert(answer_fit_between(f, "sofa", "wall").text == "I don't know an object called sofa");
        // two objects: the larger sigma moves
        SeeObject t; t.name = "table_1"; t.subtype = "table"; t.x = 4.25f - 1.0f; t.y = 0; t.w = 0.4f; t.d = 0.4f; t.h = 0.7f;
        ImaginedWorld two = f; two.scene.objects.push_back(t);
        two.object_sigma_m["chair_1"] = 0.01f; two.object_sigma_m["table_1"] = 0.1f;
        const auto tw = answer_fit_between(two, "chair_1", "table_1");
        assert(tw.p and near(tw.slack_m, 0.6f - 0.62f, 1e-3f) and tw.text.contains("table_1 would have to move"));
        ImaginedWorld nb = f; nb.body_half_width.reset();
        assert(not answer_fit_between(nb, "chair_1", "wall").p);
    }

    std::println("test_imagine: ok");
    return 0;
}
