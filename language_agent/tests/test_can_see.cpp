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
    // door states: flush = shut, a full swing = the whole width, P(open) = a two-point mixture, none = uniform phi;
    // and the MOST PROBABLE state (R12) where one door state must be chosen: never the mean slit
    {
        SeeDoor d; d.width_m = 1.0f; d.phi_rad = 0.0f;
        assert(door_probable_half_gap(d) == 0.0f and door_states(d).size() == 1);
        d.phi_rad = std::numbers::pi_v<float> / 2;
        assert(std::abs(door_probable_half_gap(d) - 0.5f) < 1e-5f);
        d.phi_rad = -1.0f; d.p_open = 0.3f;
        const auto st = door_states(d);
        assert(st.size() == 2 and std::abs(st[0].weight - 0.7f) < 1e-6f and st[0].half_gap == 0 and std::abs(st[1].half_gap - 0.5f) < 1e-6f);
        assert(door_probable_half_gap(d) == 0.0f and has(door_probable_note(d), "most probable state, shut"));
        d.p_open = 0.7f;
        assert(door_probable_half_gap(d) == 0.5f and has(door_probable_note(d), "most probable state, open"));
        d.p_open = -1.0f;                       // both unmeasured: 8 nodes, weights sum to 1; most probable = shut
        const auto u = door_states(d);
        float sw = 0; for (const auto& x : u) sw += x.weight;
        assert(u.size() == 8 and std::abs(sw - 1.0f) < 1e-5f);
        assert(door_probable_half_gap(d) == 0.0f and has(door_probable_note(d), "taken as shut"));
        d.width_m = 0;                           // no width: open at the nominal width, never a shut wall
        assert(door_states(d).size() == 1 and door_probable_half_gap(d) > 0 and has(door_probable_note(d), "width unknown"));
    }
    // LiDAR mount is body <- sensor; the evaluator places it at Rz(heading - pi/2) * mount (Shadow faces body +y), the
    // pose applied ONCE. A mount 0.10 m to the body's RIGHT (+x body), robot at (2, 1) heading 0.7 rad: body +x points
    // at heading - pi/2, so the sensor sits at (2, 1) + 0.10 (cos(0.7 - pi/2), sin(0.7 - pi/2)).
    {
        rc::media::SensorModel m;
        Eigen::Isometry3f mount = Eigen::Isometry3f::Identity();
        mount.translation() = Eigen::Vector3f(0.10f, 0.0f, 0.50f);
        const auto rig = lidar_rig_from_body_mount(m, mount);
        assert(rig.mount.translation().isApprox(mount.translation()));          // no robot pose inside the mount
        rc::card::Hypothesis h; h.x = 5; h.y = 5;
        const Eigen::Vector3f o = rc::card::rig_in_room(rig, rc::card::Viewpoint{Eigen::Vector2f(2, 1), 0.7f}, h).mount.translation();
        const float a = 0.7f - std::numbers::pi_v<float> / 2;
        assert(std::abs(o.x() - (2 + 0.10f * std::cos(a))) < 1e-5f and std::abs(o.y() - (1 + 0.10f * std::sin(a))) < 1e-5f);
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
        // a door not on the sight line is not mixed (most probable state) and gets NO "Assumed" note
        auto s4 = s3; s4.doors[0].xy = {6, 0.2f}; s4.doors[0].p_open = 0.3f;
        assert(not see_detail::door_on_sight_line(s4, s4.doors[0], vp.xy, Eigen::Vector2f(7.5f, 2.0f)));
        assert(see_detail::door_on_sight_line(s3, s3.doors[0], vp.xy, Eigen::Vector2f(7.5f, 2.0f)));
        const auto off = evaluate_scene(c, s4, tg, vp);
        assert(std::ranges::none_of(off.notes, [](const std::string& n) { return n.starts_with("assumption: door_1"); }));
        assert(not has(answer_can_see(std::optional<Card>(c), "fridge", "", off), "Assumed"));
        // ... but a width-unknown door is always named, on the sight line or not
        auto s5 = s4; s5.doors[0].width_m = 0;
        assert(has(answer_can_see(std::optional<Card>(c), "fridge", "", evaluate_scene(c, s5, tg, vp)), "width unknown"));
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
        assert(resolve_from(s, "door_1").value().xy.isApprox(Eigen::Vector2f(3, kRobotRadiusM)));   // one robot radius inside the room
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
    // ── end to end through the evaluator ──
    const auto zed_scene = [](SeeScene s)
    {
        rc::nbv::Sensor z; constexpr float f = 448.0f;
        z.hfov_rad = 2.0f * std::atan(640.0f / f); z.vfov_rad = 2.0f * std::atan(360.0f / f); z.height_m = 0.945f;
        s.zed = z; s.zed_mount_z = 0.945f;
        rc::card::LidarRig hel;
        hel.model.version = 1; hel.model.rings = 32; hel.model.fov_start_deg = -55.0f; hel.model.fov_end_deg = 15.0f;
        hel.model.azimuth_step_deg = 0.4f; hel.model.range_min_m = 0.1f; hel.model.range_max_m = 80.0f;
        hel.mount = Eigen::Isometry3f::Identity();
        hel.mount.linear() = Eigen::AngleAxisf(std::numbers::pi_v<float>, Eigen::Vector3f::UnitX()).toRotationMatrix();
        hel.mount.translation() = Eigen::Vector3f(0.0f, -0.155f, 1.075f);
        s.helios = hel;
        return s;
    };
    // the refrigerator card as refrigerator_card.h builds it (common/perception_card/tests/evaluator_combine_test.cpp real_fridge)
    const auto real_fridge = []
    {
        Card c; c.concept_ = "refrigerator"; c.graph_type = "object"; c.level = rc::card::Level::Countable; c.calibration = "fitted";
        c.channels = {{"zed", rc::card::Role::Identity, true}, {"helios", rc::card::Role::Occupancy, true}, {"ricoh", rc::card::Role::Identity, false}};
        c.geometry = {0.60f, 0.60f, 1.90f, 0.0f, 1.90f};
        c.walls = rc::card::WallOcclusion{0.70f, -1.0f, 0.0f};
        rc::card::SubtypeParams d; d.env = {0.0f, 1.32f, 0.26f}; d.q_return = 0.85f;
        d.terms = {{"envelope", {{"min_fill", 0.0f}, {"max_fill", 1.32f}, {"soft", 0.26f}, {"axes", 1.0f}}}, {"central", {}}};
        c.subtypes[""] = d;
        return c;
    }();
    // I1: a bottle ON a table is where it is, not flattened onto the floor under the table top
    {
        Card b; b.concept_ = "bottle"; b.graph_type = "object"; b.level = rc::card::Level::Countable; b.calibration = "fitted";
        b.channels = {{"zed", rc::card::Role::Identity, true}};
        b.geometry = {0.07f, 0.07f, 0.25f, 0.0f, 0.25f};
        rc::card::SubtypeParams sp; sp.env = {0.10f, 0.60f, 0.06f};
        b.subtypes[""] = sp;
        b.walls = rc::card::WallOcclusion{0.0f, -1.0f, 0.10f};
        SeeScene s = zed_scene({});
        s.room = "room_1"; s.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        s.robot_xy = Eigen::Vector2f(1.8f, 2.0f); s.robot_yaw = -std::numbers::pi_v<float> / 2;   // heading +x, at the table
        SeeObject t; t.name = "table_1"; t.subtype = "table"; t.x = 3; t.y = 2; t.z_origin = 0.0f; t.w = 0.8f; t.d = 0.8f; t.h = 0.75f;
        SeeObject bo; bo.name = "bottle_1"; bo.subtype = "bottle"; bo.x = 3; bo.y = 2; bo.w = 0.07f; bo.d = 0.07f; bo.h = 0.25f;
        bo.z_origin = 0.75f + 0.125f; bo.origin_is_centre = true;          // the bottle's node origin is its CENTRE
        s.objects = {t, bo};
        const auto tg = resolve_target(s, "bottle", "", std::optional<Card>(b));
        assert(tg.has_value() and tg->h.size.has_value());
        assert(std::abs(tg->h.size->z0 - 0.75f) < 1e-5f and std::abs(tg->h.size->z1 - 1.0f) < 1e-5f);
        const auto w = world_for(s, "bottle_1");
        assert(w.obstacles.size() == 1 and w.obstacles[0].second.z0 == 0.0f and std::abs(w.obstacles[0].second.z1 - 0.75f) < 1e-6f);
        const rc::card::Viewpoint vp{*s.robot_xy, 0.0f};
        rc::card::Hypothesis truth; truth.concept_ = "bottle"; truth.x = 3; truth.y = 2;
        truth.size = rc::card::GeometryPrior{0.07f, 0.07f, 0.25f, 0.75f, 1.0f};
        const auto direct = rc::card::evaluate(b, w, truth, vp);
        const auto r = evaluate_scene(b, s, *tg, vp);
        const std::string reply = can_see_reply(b, s, *tg, "");
        std::println("I1 bottle on table: P {:.3f} (direct {:.3f}) limit {} occ '{}' | {}", r.p_recognise, direct.p_recognise,
                     int(r.limit), r.occluder, reply);
        assert(std::abs(r.p_recognise - direct.p_recognise) < 1e-6f and r.p_recognise > 0.0f);
        // (the table may still be NAMED as costing a sliver -- the bottle's bottom ring sits exactly on the top -- but
        // it is not what limits the look, and the answer does not say it is hidden behind it)
        assert(r.limit != Limit::Occluded and not has(reply, "hidden behind table_1"));
        // the old floor-flattened bottle (z 0..0.25 inside the table footprint) WAS hidden behind table_1
        rc::card::Hypothesis floor = truth; floor.size = rc::card::GeometryPrior{0.07f, 0.07f, 0.25f, 0.0f, 0.25f};
        const auto flat = rc::card::evaluate(b, w, floor, vp);
        std::println("I1 flattened (old): P {:.3f} limit {} occ '{}' | {}", flat.p_recognise, int(flat.limit), flat.occluder,
                     answer_can_see(std::optional<Card>(b), "bottle_1", "", flat));
        assert(flat.occluder == "table_1" and flat.limit == Limit::Occluded and flat.p_recognise < r.p_recognise);
        assert(has(answer_can_see(std::optional<Card>(b), "bottle_1", "", flat), "hidden behind table_1"));
        // no published height: the card's span is kept, on the object's base
        auto s2 = s; s2.objects[1].h.reset();
        const auto t2 = resolve_target(s2, "bottle", "", std::optional<Card>(b));
        assert(t2.has_value() and std::abs(t2->h.size->z1 - t2->h.size->z0 - 0.25f) < 1e-6f and t2->h.size->z0 > 0.7f);
        // a hood keeps its hanging span (origin = its body's underside)
        Card hc = b; hc.concept_ = "hood"; hc.geometry = {0.9f, 0.5f, 0.5f, 1.55f, 2.05f};
        SeeObject hd; hd.name = "hood_1"; hd.subtype = "hood"; hd.x = 5; hd.y = 3.7f; hd.z_origin = 1.55f; hd.w = 0.9f; hd.d = 0.5f; hd.h = 0.5f;
        auto s3 = s; s3.objects = {hd};
        const auto th = resolve_target(s3, "hood", "", std::optional<Card>(hc));
        assert(th.has_value() and std::abs(th->h.size->z0 - 1.55f) < 1e-6f and std::abs(th->h.size->z1 - 2.05f) < 1e-6f);
        hd.h.reset(); s3.objects = {hd};
        const auto th2 = resolve_target(s3, "hood", "", std::optional<Card>(hc));
        assert(th2.has_value() and std::abs(th2->h.size->z0 - 1.55f) < 1e-6f and std::abs(th2->h.size->z1 - 2.05f) < 1e-6f);
    }
    // I2: "from door_X" for doors on OPPOSITE walls (max-x and min-x): both evaluated, neither refused as outside
    {
        for (const bool ccw : {true, false})
        {
            SeeScene s = zed_scene({});
            s.room = "room_1";
            s.polygon = ccw ? std::vector<Eigen::Vector2f>{{0, 0}, {6, 0}, {6, 4}, {0, 4}}
                            : std::vector<Eigen::Vector2f>{{0, 4}, {6, 4}, {6, 0}, {0, 0}};
            SeeDoor e; e.name = "door_east"; e.xy = {6, 2}; e.width_m = 0.9f; e.phi_rad = 0.0f;
            SeeDoor wd; wd.name = "door_west"; wd.xy = {0, 2}; wd.width_m = 0.9f; wd.phi_rad = 0.0f;
            s.doors = {e, wd};
            SeeObject f; f.name = "refrigerator_1"; f.subtype = "refrigerator"; f.x = 3; f.y = 3.5f; f.z_origin = 0.0f;
            f.w = 0.6f; f.d = 0.6f; f.h = 1.9f;
            s.objects = {f};
            const auto tg = resolve_target(s, "fridge", "", std::optional<Card>(real_fridge));
            assert(tg.has_value());
            for (const char* dn : {"door_east", "door_west"})
            {
                const auto vp = resolve_from(s, dn);
                assert(vp.has_value() and not vp->yaw.has_value());     // a named viewpoint keeps best heading
                const auto r = evaluate_scene(real_fridge, s, *tg, *vp);
                std::println("I2 {} (ccw {}): vp ({:.2f}, {:.2f}) P {:.3f} limit {}", dn, ccw, vp->xy.x(), vp->xy.y(), r.p_recognise, int(r.limit));
                assert(r.limit != Limit::Inside and r.p_recognise > 0.0f and not r.channels.empty());
                assert(std::abs(std::abs(vp->xy.x() - 3.0f) - (3.0f - kRobotRadiusM)) < 1e-5f and std::abs(vp->xy.y() - 2.0f) < 1e-5f);
                assert(not has(can_see_reply(real_fridge, s, *tg, dn), "cannot judge"));
            }
        }
    }
    // R11: "from here" uses the heading the robot HAS; a fridge directly behind it is not recognised, and the answer
    // says what turning to face it would give
    {
        SeeScene s = zed_scene({});
        s.room = "room_1"; s.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        s.robot_xy = Eigen::Vector2f(3.5f, 2.0f); s.robot_yaw = -std::numbers::pi_v<float> / 2;   // body yaw -pi/2 = heading +x
        SeeObject f; f.name = "refrigerator_1"; f.subtype = "refrigerator"; f.x = 0.6f; f.y = 2.0f; f.z_origin = 0.0f;
        f.w = 0.6f; f.d = 0.6f; f.h = 1.9f;
        s.objects = {f};
        const auto tg = resolve_target(s, "fridge", "", std::optional<Card>(real_fridge));
        assert(tg.has_value());
        const auto vp = resolve_from(s, "");
        assert(vp.has_value() and vp->yaw.has_value() and std::abs(*vp->yaw) < 1e-6f);
        const auto now = evaluate_scene(real_fridge, s, *tg, *vp);
        const auto turned = evaluate_scene(real_fridge, s, *tg, rc::card::Viewpoint{vp->xy, std::nullopt});
        const std::string reply = can_see_reply(real_fridge, s, *tg, "");
        std::println("R11 behind: now P {:.3f} turned P {:.3f} | {}", now.p_recognise, turned.p_recognise, reply);
        assert(now.p_recognise < 0.05f and turned.p_recognise > 0.5f);
        assert(has(reply, std::format("(P {})", see_detail::p2(now.p_recognise)).c_str()) or has(reply, see_detail::p2(now.p_recognise).c_str()));
        assert(has(reply, "If I turn to face it") and has(reply, see_detail::p2(turned.p_recognise).c_str()));
        // facing it already: no turn clause
        auto s2 = s; s2.robot_yaw = std::numbers::pi_v<float> / 2;                  // heading pi: facing the fridge
        assert(not has(can_see_reply(real_fridge, s2, *tg, ""), "If I turn"));
    }
    // R12: where_to_stand takes an uncertain door at its MOST PROBABLE state, never the average slit
    {
        Card c = real_fridge; c.channels = {{"zed", rc::card::Role::Identity, true}};
        SeeScene s = zed_scene({});
        s.room = "r"; s.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        SeeDoor d; d.name = "door_1"; d.xy = {6, 2}; d.width_m = 0.9f; d.phi_rad = -1.0f; d.p_open = 0.3f;
        s.doors = {d};
        Target tg; tg.label = "fridge"; tg.h.concept_ = "refrigerator"; tg.h.x = 7.5f; tg.h.y = 2.0f;
        const auto expect_with = [&](float half_gap, const std::vector<std::string>& notes)
        {
            std::vector<float> g{half_gap};
            return answer_where_to_stand(std::optional<Card>(c), tg.label, rc::card::where_to_stand(c, world_for(s, "", &g), tg.h, 3),
                                         landmarks_of(s), notes);
        };
        const std::string shut = where_to_stand_reply(c, s, tg);
        std::println("R12 p_open 0.3: {}", shut);
        assert(shut == expect_with(0.0f, {door_probable_note(s.doors[0])}) and has(shut, "most probable state, shut"));
        assert(shut != expect_with(door_states(s.doors[0])[1].weight * 0.45f, {door_probable_note(s.doors[0])}));   // not the mean slit
        s.doors[0].p_open = 0.7f;
        const std::string open = where_to_stand_reply(c, s, tg);
        std::println("R12 p_open 0.7: {}", open);
        assert(open == expect_with(0.45f, {door_probable_note(s.doors[0])}) and has(open, "most probable state, open"));
        s.doors[0].p_open = -1.0f;                // both unmeasured: shut
        assert(where_to_stand_reply(c, s, tg) == expect_with(0.0f, {door_probable_note(s.doors[0])}));
    }
    // subtype: the instance's card subtype key (a table's obj_shape) reaches the evaluator
    {
        Card c = real_fridge; c.concept_ = "table"; c.channels = {{"zed", rc::card::Role::Identity, true}};
        c.geometry = {1.0f, 0.6f, 0.75f, 0.0f, 0.75f};
        rc::card::SubtypeParams round = c.subtypes[""]; round.env = {0.30f, 0.60f, 0.06f};
        round.terms[0].params["min_fill"] = 0.30f;
        c.subtypes["round"] = round;
        SeeScene s = zed_scene({});
        s.room = "r"; s.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        s.robot_xy = Eigen::Vector2f(1.0f, 2.0f); s.robot_yaw = -std::numbers::pi_v<float> / 2;
        SeeObject t; t.name = "table_1"; t.subtype = "table"; t.shape = "round"; t.x = 4; t.y = 2; t.z_origin = 0.0f;
        t.w = 1.0f; t.d = 1.0f; t.h = 0.75f;
        s.objects = {t};
        const auto tg = resolve_target(s, "table", "", std::optional<Card>(c));
        assert(tg.has_value() and tg->h.subtype == "round");
        const rc::card::Viewpoint vp{*s.robot_xy, 0.0f};
        Target dflt = *tg; dflt.h.subtype = "";
        const float pr = evaluate_scene(c, s, *tg, vp).p_recognise, pd = evaluate_scene(c, s, dflt, vp).p_recognise;
        std::println("subtype: round {:.3f} default {:.3f}", pr, pd);
        assert(pr != pd);
        s.objects[0].shape = "oval";              // a shape the card has no entry for: its default
        assert(resolve_target(s, "table", "", std::optional<Card>(c))->h.subtype.empty());
    }
    // Inside wording: the evaluator's reason, not whichever note comes first. The helios (Role::Both) sits 0.155 m
    // behind the robot centre INSIDE a box the robot backs onto: Inside comes from that channel, its notes ("range
    // sigma ... assumed") come first, and there is no feasibility note.
    {
        Card c = real_fridge; c.channels = {{"helios", rc::card::Role::Both, true}};
        SeeScene s = zed_scene({});
        s.zed.reset();
        s.room = "r"; s.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        s.robot_xy = Eigen::Vector2f(3.0f, 2.0f); s.robot_yaw = -std::numbers::pi_v<float> / 2;   // heading +x: helios at (2.845, 2)
        SeeObject box; box.name = "box_1"; box.subtype = "box"; box.x = 2.75f; box.y = 2.0f; box.z_origin = 0.0f;
        box.w = 0.4f; box.d = 0.4f; box.h = 1.2f;          // spans x 2.55..2.95: holds the helios, not the robot centre
        SeeObject f; f.name = "refrigerator_1"; f.subtype = "refrigerator"; f.x = 5.0f; f.y = 2.0f; f.z_origin = 0.0f;
        f.w = 0.6f; f.d = 0.6f; f.h = 1.9f;
        s.objects = {box, f};
        const auto tg = resolve_target(s, "fridge", "", std::optional<Card>(c));
        const auto vp = resolve_from(s, "");
        const auto r = evaluate_scene(c, s, *tg, *vp);
        assert(r.limit == Limit::Inside and not r.notes.empty() and not r.notes.front().starts_with("viewpoint"));
        const std::string a = answer_can_see(std::optional<Card>(c), "refrigerator_1", "", r);
        std::println("Inside (channel): {}", a);
        assert(has(a, "not somewhere I can stand") and not has(a, r.notes.front().c_str()));
        // the feasibility reason when evaluate() gives one, even behind an earlier note
        Result in = result(0, 0, Limit::Inside);
        in.notes = {"walls unknown", "viewpoint inside chair_1"};
        assert(has(answer_can_see(std::optional<Card>(c), "x", "chair_1", in), "viewpoint inside chair_1"));
    }
    std::println("test_can_see: ok");
}
