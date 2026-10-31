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
    // door aperture: flush = shut, a full swing = the whole width, unmeasured = width x P(open)
    {
        SeeDoor d; d.width_m = 1.0f; d.phi_rad = 0.0f;
        assert(door_half_gap(d) == 0.0f);
        d.phi_rad = std::numbers::pi_v<float> / 2;
        assert(std::abs(door_half_gap(d) - 0.5f) < 1e-5f);
        d.phi_rad = -1.0f; d.p_open = 0.5f;
        assert(std::abs(door_half_gap(d) - 0.25f) < 1e-6f);
        d.p_open.reset();
        assert(door_half_gap(d) == 0.0f);
        d.width_m = 0;
        d.p_open = 1.0f;
        assert(door_half_gap(d) == 0.0f);
    }
    // the World excludes the target and keeps the others as NAMED obstacles
    {
        const auto w = world_for(scene(), "refrigerator_1");
        assert(w.obstacles.size() == 2 and w.obstacles[0].first == "chair_1");
        assert(w.polygon.size() == 4 and w.gaps.empty());   // the door is shut
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
        auto a = answer_can_see(card, "refrigerator_1", "", result(0.80f, 0.75f, Limit::Occluded, "chair_1"));
        assert(has(a, "Yes") and has(a, "0.75") and not has(a, "0.80") and has(a, "my camera (the zed)"));
        assert(has(a, "hidden behind chair_1") and has(a, "from here"));
        // differing by more than 0.1: both
        auto b = answer_can_see(card, "refrigerator_1", "door_1", result(0.90f, 0.40f, Limit::TooSmall));
        assert(has(b, "0.90") and has(b, "0.40") and has(b, "from door_1") and has(b, "too small"));
        // within 0.1 is NOT "more than"
        auto c = answer_can_see(card, "x", "", result(0.79f, 0.70f, Limit::None));
        assert(not has(c, "0.79"));
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
