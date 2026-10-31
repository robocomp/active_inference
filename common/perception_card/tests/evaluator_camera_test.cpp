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
    SubtypeParams d; d.env = {0.0f, 1.32f, 0.26f};   // the refrigerator agent's measured envelope
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
        CHECK(r.p > 0.5f and r.limit == Limit::None);
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
    { // not modelled
        Card n = fridge(); n.channels[0].modelled = false;
        CHECK(evaluate_camera(n, room(), at(5, 2), {{2, 2}, std::nullopt}).limit == Limit::NotModelled);
    }
    if (bad) { std::printf("evaluator_camera_test: %d FAILURES\n", bad); return 1; }
    std::printf("evaluator_camera_test: ok\n");
    return 0;
}
