// can_see pure side (src/can_see.h): name resolution, the World from a SeeScene, and the answer wording on fixed Results.
#include "../src/can_see.h"

#include <cassert>
#include <print>

using namespace rc::chat;
using rc::card::Card;
using rc::card::ChannelResult;
using rc::card::Limit;
using rc::card::Result;

namespace
{
Card fridge_card(const char* calibration = "fitted")
{
    Card c;
    c.concept_ = "refrigerator";
    c.calibration = calibration;
    return c;
}

Result result(float pn, float pr, Limit lim, std::string occ = "")
{
    Result r;
    r.p_notice = pn; r.p_recognise = pr; r.limit = lim; r.occluder = occ;
    ChannelResult z; z.sensor = "zed"; z.p = pr; z.limit = lim; z.occluder = occ;
    r.channels.push_back(z);
    return r;
}

SeeScene scene()
{
    SeeScene s;
    s.room = "room_1";
    s.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
    s.robot_xy = Eigen::Vector2f(1, 1);
    SeeDoor d; d.name = "door_1"; d.xy = {3, 0}; d.width_m = 1.0f; d.phi_rad = 0.0f;
    s.doors.push_back(d);
    SeeObject f; f.name = "refrigerator_1"; f.subtype = "refrigerator"; f.x = 5; f.y = 3; f.w = 0.7f; f.d = 0.7f; f.h = 1.8f;
    f.standoff = Eigen::Vector2f(4.2f, 3.0f);
    SeeObject c1; c1.name = "chair_1"; c1.subtype = "chair"; c1.x = 2; c1.y = 2; c1.w = 0.4f; c1.d = 0.4f; c1.h = 0.9f;
    SeeObject c2 = c1; c2.name = "chair_2"; c2.x = 3;
    s.objects = {f, c1, c2};
    return s;
}
bool has(const std::string& s, const char* w) { return s.find(w) != std::string::npos; }
}   // namespace

int main()
{
    // door states: flush = shut, a full swing = the whole width, P(open) = a two-point mixture, none = uniform phi
    {
        SeeDoor d; d.width_m = 1.0f; d.phi_rad = 0.0f;
        assert(door_half_gap(d) == 0.0f and door_states(d).size() == 1);
        d.phi_rad = std::numbers::pi_v<float> / 2;
        assert(std::abs(door_half_gap(d) - 0.5f) < 1e-5f);
        d.phi_rad = -1.0f; d.p_open = 0.3f;
        const auto st = door_states(d);
        assert(st.size() == 2 and std::abs(st[0].weight - 0.7f) < 1e-6f and st[0].half_gap == 0 and std::abs(st[1].half_gap - 0.5f) < 1e-6f);
        d.p_open = -1.0f;                       // both unmeasured: 8 nodes, weights sum to 1, mean inside (0, 0.5)
        const auto u = door_states(d);
        float sw = 0; for (const auto& x : u) sw += x.weight;
        assert(u.size() == 8 and std::abs(sw - 1.0f) < 1e-5f and door_half_gap(d) > 0 and door_half_gap(d) < 0.5f);
        d.width_m = 0;                           // no width: open at the nominal width, never a shut wall
        assert(door_states(d).size() == 1 and door_half_gap(d) > 0);
    }
    // LiDAR mount is body <- sensor: the evaluator's origin is robot pose x mount, the pose applied ONCE
    {
        rc::media::SensorModel m;
        Eigen::Isometry3f mount = Eigen::Isometry3f::Identity();
        mount.translation() = Eigen::Vector3f(0.10f, 0.0f, 0.50f);
        const auto rig = lidar_rig_from_body_mount(m, mount);
        assert(rig.mount.translation().isApprox(mount.translation()));          // no robot pose inside the mount
        Eigen::Isometry3f robot = Eigen::Isometry3f::Identity();                // what evaluate_lidar does
        robot.linear() = Eigen::AngleAxisf(0.7f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
        robot.translation() = Eigen::Vector3f(2, 1, 0);
        const Eigen::Vector3f o = (robot * rig.mount).translation();
        assert(std::abs(o.x() - (2 + 0.10f * std::cos(0.7f))) < 1e-5f and std::abs(o.y() - (1 + 0.10f * std::sin(0.7f))) < 1e-5f);
        assert(std::abs(o.z() - 0.50f) < 1e-6f);
    }
    // doors as a mixture (R10): a wall at x=6, door in it, target behind, seen only through the door
    {
        Card c;
        c.concept_ = "fridge"; c.level = rc::card::Level::Countable;
        c.channels = {{"zed", rc::card::Role::Identity, true}};
        c.geometry = {0.7f, 0.7f, 1.8f, 0.0f, 1.8f};
        rc::card::SubtypeParams sp; sp.env = {0.0f, 1.32f, 0.26f};
        c.subtypes[""] = sp;
        SeeScene s;
        s.room = "r"; s.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        rc::nbv::Sensor z; constexpr float f = 448.0f;
        z.hfov_rad = 2.0f * std::atan(640.0f / f); z.vfov_rad = 2.0f * std::atan(360.0f / f); z.height_m = 0.945f;
        s.zed = z; s.zed_mount_z = 0.945f;
        SeeDoor d; d.name = "door_1"; d.xy = {6, 2}; d.width_m = 0.9f;
        s.doors = {d};
        Target tg; tg.name = ""; tg.label = "fridge"; tg.h.concept_ = "fridge"; tg.h.x = 7.5f; tg.h.y = 2.0f;
        const rc::card::Viewpoint vp{Eigen::Vector2f(3, 2), std::nullopt};
        const auto at_phi = [&](float phi) { auto s2 = s; s2.doors[0].phi_rad = phi; s2.doors[0].p_open.reset(); return evaluate_scene(c, s2, tg, vp); };
        const auto shut = at_phi(0.0f), open = at_phi(std::numbers::pi_v<float> / 2);
        assert(open.p_recognise > shut.p_recognise);
        auto s3 = s; s3.doors[0].phi_rad = -1.0f; s3.doors[0].p_open = 0.3f;
        const auto m = evaluate_scene(c, s3, tg, vp);
        assert(std::abs(m.p_recognise - (0.3f * open.p_recognise + 0.7f * shut.p_recognise)) < 1e-5f);
        assert(std::abs(m.p_notice - (0.3f * open.p_notice + 0.7f * shut.p_notice)) < 1e-5f);
        assert(std::ranges::any_of(m.notes, [](const std::string& n) { return n.starts_with("assumption: door_1"); }));
        assert(has(answer_can_see(std::optional<Card>(c), "fridge", "", m), "Assumed: door_1"));
        s3.doors[0].p_open.reset();               // both unmeasured: strictly between shut and open
        const auto u = evaluate_scene(c, s3, tg, vp);
        assert(u.p_recognise > shut.p_recognise and u.p_recognise < open.p_recognise);
        // a door not on the sight line is not mixed (expected state), and unmeasured width never cuts a shut wall
        auto s4 = s3; s4.doors[0].xy = {6, 0.2f}; s4.doors[0].p_open = 0.3f;
        assert(not see_detail::door_on_sight_line(s4, s4.doors[0], vp.xy, Eigen::Vector2f(7.5f, 2.0f)));
        assert(see_detail::door_on_sight_line(s3, s3.doors[0], vp.xy, Eigen::Vector2f(7.5f, 2.0f)));
    }
    // the World excludes the target and keeps the others as NAMED obstacles
    {
        const auto w = world_for(scene(), "refrigerator_1");
        assert(w.obstacles.size() == 2 and w.obstacles[0].first == "chair_1");
        assert(w.polygon.size() == 4 and w.gaps.empty());   // the door is shut (phi = 0)
        assert(not w.zed.has_value());
    }
    // names: fridge = refrigerator, plural stripped
    assert(canonical_concept("Fridge") == "refrigerator" and canonical_concept("chairs") == "chair");
    assert(canonical_concept("glass") == "glass" and canonical_concept("door_1") == "door_1");

    // target resolution: unique instance, several = ASK, none = not perceived, named instance
    {
        const auto s = scene();
        const auto card = std::optional<Card>(fridge_card());
        const auto t = resolve_target(s, "fridge", "", card);
        assert(t.has_value() and t->name == "refrigerator_1" and t->h.x == 5 and t->h.size.has_value() and t->h.size->w == 0.7f);
        const auto a = resolve_target(s, "chair", "", std::nullopt);
        assert(not a.has_value() and a.error().starts_with("ASK: which chair") and has(a.error(), "chair_1, chair_2"));
        const auto n = resolve_target(s, "chair", "chair_2", std::nullopt);
        assert(n.has_value() and n->name == "chair_2" and n->h.x == 3);
        const auto none = resolve_target(s, "table", "", std::nullopt);
        assert(not none.has_value() and has(none.error(), "not perceived"));
    }
    // from: robot, a door, an object's standoff, anything else = ASK
    {
        const auto s = scene();
        assert(resolve_from(s, "").value().xy == Eigen::Vector2f(1, 1));
        assert(resolve_from(s, "door_1").value().xy == Eigen::Vector2f(3, 0));
        assert(resolve_from(s, "refrigerator_1").value().xy == Eigen::Vector2f(4.2f, 3.0f));
        const auto bad = resolve_from(s, "the kitchen");
        assert(not bad.has_value() and bad.error() == "ASK: which place do you mean?");
        auto lost = s; lost.robot_xy.reset();
        assert(not resolve_from(lost, "").has_value());
    }

    // wording
    {
        const auto card = std::optional<Card>(fridge_card());
        assert(answer_can_see(std::nullopt, "x", "", result(1, 1, Limit::None)) == "I have no detector for that.");
        // notice and recognise within 0.1: ONE probability
        auto a = answer_can_see(card, "refrigerator_1", "", result(0.75f, 0.75f, Limit::Occluded, "chair_1"));
        assert(has(a, "Yes") and has(a, "0.75") and not has(a, "0.80") and has(a, "my camera (the zed)"));
        assert(has(a, "hidden behind chair_1") and has(a, "from here"));
        // differing by more than 0.1: both
        auto b = answer_can_see(card, "refrigerator_1", "door_1", result(0.90f, 0.40f, Limit::TooSmall));
        assert(has(b, "0.90") and has(b, "0.40") and has(b, "from door_1") and has(b, "too small"));
        // no separation threshold: they are shown apart whenever the printed decimals differ, together when they agree
        auto c = answer_can_see(card, "x", "", result(0.71f, 0.70f, Limit::None));
        assert(has(c, "0.71") and has(c, "0.70"));
        auto c2 = answer_can_see(card, "x", "", result(0.701f, 0.699f, Limit::None));
        assert(not has(c2, "0.70 ") and has(c2, "(P 0.70)") and not has(c2, "notice"));
        // the two "no card" messages
        assert(answer_no_card(true, "/r") == "I have no detector for that.");
        assert(answer_no_card(false, "/r") == "I couldn't find any perception cards (looked in /r).");
        // a wall
        assert(has(answer_can_see(card, "x", "", result(0.1f, 0.1f, Limit::BehindWall)), "a wall is in the way"));
        // the ricoh is never claimed, whatever it reports
        Result r = result(0.5f, 0.5f, Limit::None);
        ChannelResult ric; ric.sensor = "ricoh"; ric.p = 0.99f; ric.limit = Limit::None;
        r.channels.push_back(ric);
        assert(not has(answer_can_see(card, "x", "", r), "ricoh"));
        Result only; only.channels.push_back(ric);
        assert(has(answer_can_see(card, "x", "", only), "no model") and not has(answer_can_see(card, "x", "", only), "ricoh"));
        // a lidar channel carries it
        Result l = result(0.6f, 0.6f, Limit::None);
        l.channels[0].p = 0.1f;
        ChannelResult hel; hel.sensor = "helios"; hel.p = 0.6f; hel.limit = Limit::BetweenRings;
        l.channels.push_back(hel);
        assert(has(answer_can_see(card, "x", "", l), "helios") and has(answer_can_see(card, "x", "", l), "between the lidar's beams"));
        // an infeasible viewpoint
        Result in = result(0, 0, Limit::Inside);
        in.notes.push_back("viewpoint inside chair_1");
        assert(has(answer_can_see(card, "x", "chair_1", in), "viewpoint inside chair_1"));
        // calibration "prior" says so, "fitted" does not
        assert(has(answer_can_see(std::optional<Card>(fridge_card("prior")), "x", "", result(0.8f, 0.8f, Limit::None)),
                   "My detector for refrigerators isn't calibrated yet"));
        assert(not has(a, "calibrated"));
    }
    // where to stand: relative to the nearest landmark that is not the target
    {
        const auto card = std::optional<Card>(fridge_card());
        const auto s = scene();
        std::vector<rc::card::Stand> st;
        st.push_back({Eigen::Vector2f(4.0f, 0.5f), 1.0f, result(0.9f, 0.85f, Limit::None)});
        st.push_back({Eigen::Vector2f(1.0f, 3.0f), 0.0f, result(0.8f, 0.6f, Limit::Occluded, "chair_1")});
        const auto a = answer_where_to_stand(card, "refrigerator_1", st, landmarks_of(s));
        assert(has(a, "1.1 m from door_1") and has(a, "0.85") and has(a, "Other spots") and has(a, "from chair_1"));
        assert(not has(a, "from refrigerator_1"));
        assert(has(answer_where_to_stand(card, "x", {}, {}), "cannot find a spot"));
        assert(answer_where_to_stand(std::nullopt, "x", st, {}) == "I have no detector for that.");
    }
    std::println("test_can_see: ok");
}
