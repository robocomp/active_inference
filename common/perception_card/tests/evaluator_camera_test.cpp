// Camera half of the card evaluator. 6x4 m room, ZED 1280x720 f=448 mounted 0.945 m, fridge 0.7x0.7x1.8.
//   LANG=es_ES.UTF-8 ./evaluator_camera_test
#include "perception_card/evaluator.h"
#include <clocale>
#include <cstdio>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
using namespace rc::card;

static Card fridge()
{
    Card c; c.concept_ = "fridge"; c.graph_type = "object"; c.level = Level::Countable;
    c.channels = {{"zed", Role::Identity, true}};
    c.geometry = {0.7f, 0.7f, 1.8f, 0.0f, 1.8f};
    SubtypeParams d; d.env = {0.0f, 1.32f, 0.26f};   // the refrigerator agent's REAL config: refrigerator_concept/etc/config.toml:166-168
                                    // (DetectMinFill = 0.0 -- 'inactive over the operating range', DetectMaxFill = 1.32, DetectSoft = 0.26)
    c.subtypes[""] = d;
    return c;
}
static World room()
{
    World w;
    w.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
    rc::nbv::Sensor s;
    constexpr float f = 448.0f;
    s.hfov_rad = 2.0f * std::atan(640.0f / f); s.vfov_rad = 2.0f * std::atan(360.0f / f); s.height_m = 0.945f;
    w.zed = s; w.zed_mount_z = 0.945f;
    return w;
}
static Hypothesis at(float x, float y) { Hypothesis h; h.concept_ = "fridge"; h.x = x; h.y = y; return h; }

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const Card c = fridge();

    { // (a) clear view at 3 m
        const auto r = evaluate_camera(c, room(), at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        std::printf("a: p=%.3f limit=%d\n", r.p, int(r.limit));
        // p is NOT near 1 even in a clear view: MinFill = 0 leaves the lower shoulder sig(fill_min/0.26) active at a
        // short-axis fill of ~0.14 (about 0.63), so the model's own answer is ~0.57 and the limit it names is the size.
        CHECK(std::abs(r.p - 0.570f) < 0.01f and r.limit == Limit::TooSmall and r.limit_gain > 0.0f);
        // the explicit heading at the target gives the same answer as "best heading"
        const auto r2 = evaluate_camera(c, room(), at(5.0f, 2.0f), {{2.0f, 2.0f}, 0.0f});
        CHECK(std::abs(r.p - r2.p) < 1e-6f);
    }
    { // (b) a 1x1x1.2 m box halfway
        World w = room();
        w.obstacles.push_back({"box_1", rc::nbv::Obstacle{3.5f, 2.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.2f}});
        const auto r = evaluate_camera(c, w, at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        std::printf("b: p=%.3f limit=%d occ=%s\n", r.p, int(r.limit), r.occluder.c_str());
        CHECK(r.limit == Limit::Occluded and r.occluder == "box_1");
        CHECK(r.p < evaluate_camera(c, room(), at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt}).p);
    }
    { // (c) 0.3 m from the fridge
        const auto r = evaluate_camera(c, room(), at(5.0f, 2.0f), {{4.35f, 2.0f}, std::nullopt});
        std::printf("c: p=%.3f limit=%d\n", r.p, int(r.limit));
        CHECK(r.limit == Limit::TooClose);
    }
    { // (d) next room behind a wall at x=6, door gap at (6, 2)
        const auto run = [&](float half_w)
        {
            World w = room();
            w.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
            w.gaps = {{{6.0f, 2.0f}, half_w}};
            return evaluate_camera(c, w, at(7.5f, 2.0f), {{3.0f, 2.0f}, std::nullopt});
        };
        const auto shut = run(0.0f), open = run(0.45f);
        std::printf("d: shut p=%.3f limit=%d  open p=%.3f limit=%d\n", shut.p, int(shut.limit), open.p, int(open.limit));
        CHECK(shut.limit == Limit::BehindWall);
        CHECK(open.p > shut.p);
    }
    { // (e) no polygon: answers, says so
        World w = room(); w.polygon.clear();
        const auto r = evaluate_camera(c, w, at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        CHECK(std::ranges::find(r.notes, std::string("walls unknown")) != r.notes.end());
        CHECK(r.p > 0.5f);
    }
    { // (f) animate: the longer unseen, the lower p
        Card a = fridge(); a.level = Level::Animate;
        a.subtypes[""].terms = {{"motion", {{"sigma0", 0.05f}, {"k", 0.02f}}}};
        Hypothesis h0 = at(5.0f, 2.0f), h30 = h0; h30.seconds_unseen = 30.0f;
        const Viewpoint vp{{2.0f, 2.0f}, 0.0f};   // fixed heading: the robot cannot follow a target it has lost
        const float p0 = evaluate_camera(a, room(), h0, vp).p, p30 = evaluate_camera(a, room(), h30, vp).p;
        std::printf("f: p(0)=%.3f p(30)=%.3f\n", p0, p30);
        CHECK(p30 < p0);
    }
    { // articulated: an edge-on door is worse than a face-on one
        Card d = fridge(); d.level = Level::Articulated;
        d.subtypes[""].terms = {{"leaf_obliquity", {}}};
        Hypothesis closed = at(5.0f, 2.0f), edge = closed; closed.phi = 0.0f; edge.phi = 1.5707963f;
        // the closed leaf's normal is the object's y axis; look along y, so phi=0 is face-on and phi=pi/2 edge-on
        const Viewpoint vp{{5.0f, 0.5f}, std::nullopt};
        const auto rc_ = evaluate_camera(d, room(), closed, vp), re = evaluate_camera(d, room(), edge, vp);
        std::printf("art: closed p=%.3f edge p=%.3f limit=%d\n", rc_.p, re.p, int(re.limit));
        CHECK(re.p < rc_.p and re.limit == Limit::EdgeOn);
    }
    { // (i) two obstacles between viewpoint and target: the one whose removal helps more is named
        World w = room();
        w.obstacles.push_back({"small_box", rc::nbv::Obstacle{3.5f, 1.7f, 0.2f, 0.2f, 0.0f, 0.0f, 1.8f}});
        w.obstacles.push_back({"wide_box",  rc::nbv::Obstacle{3.5f, 2.3f, 0.2f, 0.6f, 0.0f, 0.0f, 1.8f}});   // disjoint shadows, so leave-one-out is not masked by the other
        const auto r = evaluate_camera(c, w, at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        std::printf("i: p=%.3f occ=%s\n", r.p, r.occluder.c_str());
        CHECK(r.occluder == "wide_box");
    }
    { // (ii) viewpoint inside an obstacle
        World w = room();
        w.obstacles.push_back({"box_1", rc::nbv::Obstacle{2.0f, 2.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.2f}});
        const auto r = evaluate_camera(c, w, at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        CHECK(r.limit == Limit::Inside and r.p == 0.0f);
    }
    { // (iii) mass: coverage only -- p is exactly visible_fraction, whatever the envelope says
        Card m = fridge(); m.level = Level::Mass;
        m.subtypes[""].env = {0.9f, 0.95f, 0.01f};      // an envelope that would zero any detector level
        World w = room();
        w.obstacles.push_back({"box_1", rc::nbv::Obstacle{3.5f, 2.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.2f}});
        const auto r = evaluate_camera(m, w, at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        rc::nbv::Sensor s = *w.zed; s.env = m.subtypes[""].env;
        std::vector<rc::nbv::Obstacle> all = rc::nbv::wall_obstacles(w.polygon, 0.12f, w.gaps);
        all.push_back(w.obstacles[0].second);
        rc::nbv::Target t; t.cx = 5; t.cy = 2; t.w = 0.7f; t.h = 0.7f; t.z0 = 0; t.z1 = 1.8f;
        const float v = rc::nbv::visible_fraction(t, {2.0f, 2.0f}, 0.0f, s, all);
        std::printf("iii: p=%.4f v=%.4f\n", r.p, v);
        CHECK(v > 0.0f and std::abs(r.p - v) < 1e-6f);
    }
    { // (iv) partly occluded, p in the 0.6 region: still names its occluder
        Card k = fridge(); k.subtypes[""].env = {0.0f, 1.32f, 0.26f};
        World w = room();
        // a thin low post right on the line of sight: costs part of the silhouette only
        w.obstacles.push_back({"post", rc::nbv::Obstacle{3.5f, 2.0f, 0.2f, 0.2f, 0.0f, 0.0f, 0.5f}});
        const auto clear = evaluate_camera(k, room(), at(5.0f, 2.0f), {{1.5f, 2.0f}, std::nullopt});
        const auto r = evaluate_camera(k, w, at(5.0f, 2.0f), {{1.5f, 2.0f}, std::nullopt});
        std::printf("iv: clear=%.3f p=%.3f occ=%s\n", clear.p, r.p, r.occluder.c_str());
        CHECK(r.p < clear.p and r.p > 0.3f and r.occluder == "post");
    }
    { // (v) articulated, phi unset: says so
        Card d = fridge(); d.level = Level::Articulated; d.subtypes[""].terms = {{"leaf_obliquity", {}}};
        const auto r = evaluate_camera(d, room(), at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        CHECK(std::ranges::find(r.notes, std::string("phi unknown: leaf taken closed")) != r.notes.end());
    }
    { // NaN safety: standing at the target centre (no obstacle) is finite, p = 0, too close
        const auto r = evaluate_camera(c, room(), at(3.0f, 2.0f), {{3.0f, 2.0f}, std::nullopt});
        CHECK(std::isfinite(r.p) and r.p == 0.0f and r.limit == Limit::TooClose);
    }
    { // not modelled
        Card n = fridge(); n.channels[0].modelled = false;
        CHECK(evaluate_camera(n, room(), at(5, 2), {{2, 2}, std::nullopt}).limit == Limit::NotModelled);
    }
    if (bad) { std::printf("evaluator_camera_test: %d FAILURES\n", bad); return 1; }
    std::printf("evaluator_camera_test: ok\n");
    return 0;
}
