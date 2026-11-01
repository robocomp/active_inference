// imagined_world.h: opening words, interventions on a copy, the "there" resolver.
#include "../src/imagined_world.h"

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
    assert(not byclass.has_value());              // "the table" is not a class word: ASK
    assert(resolve_standpoint(w0, "table").has_value());
    const auto nosize = resolve_standpoint(w0, "chair_1");
    assert(nosize.has_value() and nosize->words.contains("size is unknown"));
    assert(near((nosize->xy - Eigen::Vector2f(-1, 1)).norm(), kRobotRadiusM));
    assert(resolve_standpoint(w0, "the moon").error() == "ASK: which place do you mean by 'there'?");

    std::println("test_imagine: ok");
    return 0;
}
