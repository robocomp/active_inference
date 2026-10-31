// LiDAR half of the card evaluator. Helios inverted at 1.075 m, bpearl inverted at 0.67 m (as in Webots).
//   LANG=es_ES.UTF-8 ./evaluator_lidar_test
#include "perception_card/evaluator.h"
#include <clocale>
#include <cstdio>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
using namespace rc::card;

static Card thing(float q = 1.0f)
{
    Card c; c.concept_ = "fridge"; c.graph_type = "object"; c.level = Level::Countable;
    c.channels = {{"helios", Role::Occupancy, true}, {"bpearl", Role::Occupancy, true}};
    c.geometry = {0.7f, 0.7f, 1.8f, 0.0f, 1.8f};
    SubtypeParams d; d.q_return = q;
    c.subtypes[""] = d;
    return c;
}
static LidarRig helios()
{
    LidarRig r;
    r.model.version = 1; r.model.rings = 32; r.model.fov_start_deg = -55.0f; r.model.fov_end_deg = 15.0f;
    r.model.azimuth_step_deg = 0.4f; r.model.range_min_m = 0.1f; r.model.range_max_m = 80.0f;
    r.mount = Eigen::Isometry3f::Identity();
    r.mount.linear() = Eigen::AngleAxisf(std::numbers::pi_v<float>, Eigen::Vector3f::UnitX()).toRotationMatrix();
    r.mount.translation() = Eigen::Vector3f(0, 0, 1.075f);
    return r;
}
static LidarRig bpearl()
{
    LidarRig r;
    r.model.version = 1; r.model.rings = 32; r.model.fov_start_deg = 0.0f; r.model.fov_end_deg = 90.0f;
    r.model.azimuth_step_deg = 0.4f; r.model.range_min_m = 0.1f; r.model.range_max_m = 30.0f;
    r.mount = Eigen::Isometry3f::Identity();
    r.mount.linear() = Eigen::AngleAxisf(std::numbers::pi_v<float>, Eigen::Vector3f(-1, -1, 0).normalized()).toRotationMatrix();
    r.mount.translation() = Eigen::Vector3f(0, 0, 0.67f);
    return r;
}
static World room()
{
    World w;
    w.polygon = {{-8, -8}, {8, -8}, {8, 8}, {-8, 8}};
    w.helios = helios(); w.bpearl = bpearl();
    return w;
}
static Hypothesis at(float x, float y, float w, float d, float z0, float z1)
{
    Hypothesis h; h.concept_ = "fridge"; h.x = x; h.y = y; h.size = GeometryPrior{w, d, z1 - z0, z0, z1};
    return h;
}
static const Viewpoint kO{{0.0f, 0.0f}, 0.0f};

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const Card c = thing();
    { // (a) fridge at 3 m
        const auto r = evaluate_lidar(c, room(), at(3, 0, 0.7f, 0.7f, 0, 1.8f), kO, "helios");
        std::printf("a: p=%.4f limit=%d\n", r.p, int(r.limit));
        CHECK(r.p > 0.99f and r.sensor == "helios");
        const float lam = expected_returns(*room().helios, rc::nbv::Target{3, 0, 0, 0.7f, 0.7f, 0, 1.8f}, {0, 0}, {});
        std::printf("a: lambda=%.1f\n", lam);
        CHECK(lam > 50.0f);
    }
    { // (b) 2 cm slab at a height midway between two helios rings, 6 m away
        // inverted: room elevation = -device elevation. Rings -55..15 over 32: spacing 70/31 deg.
        const float step = 70.0f / 31.0f, e = (-55.0f + 10 * step + 0.5f * step) * std::numbers::pi_v<float> / 180.0f;
        const float z = 1.075f - 6.0f * std::tan(e);
        const auto r = evaluate_lidar(c, room(), at(6, 0, 1.0f, 1.0f, z - 0.01f, z + 0.01f), kO, "helios");
        std::printf("b: p=%.4f limit=%d gain=%.3f\n", r.p, int(r.limit), r.limit_gain);
        CHECK(r.limit == Limit::BetweenRings and r.p < 0.1f and r.limit_gain > 0.0f);
    }
    { // (c) 5 cm object on the floor at 0.5 m
        const Hypothesis h = at(0.5f, 0, 0.05f, 0.05f, 0.0f, 0.05f);
        const auto rh = evaluate_lidar(c, room(), h, kO, "helios");
        const auto rb = evaluate_lidar(c, room(), h, kO, "bpearl");
        std::printf("c: helios p=%.4f limit=%d  bpearl p=%.4f\n", rh.p, int(rh.limit), rb.p);
        CHECK(rh.limit == Limit::BelowFan);
        CHECK(rb.p > 0.9f);
    }
    { // (d) 0.05 m from the bpearl
        const auto r = evaluate_lidar(c, room(), at(0.05f, 0, 0.02f, 0.02f, 0.0f, 0.6f), kO, "bpearl");
        std::printf("d: p=%.4f limit=%d\n", r.p, int(r.limit));
        CHECK(r.limit == Limit::DeadZone);
    }
    { // (e) no helios in the world
        World w = room(); w.helios.reset();
        const auto r = evaluate_lidar(c, w, at(3, 0, 0.7f, 0.7f, 0, 1.8f), kO, "helios");
        CHECK(r.limit == Limit::NotModelled and r.p == 0.0f);
    }
    { // (f) q_return = 0.1 lowers p. A distant small target keeps lambda moderate.
        const Hypothesis h = at(7.4f, 0, 0.1f, 0.1f, 0.0f, 1.0f);
        const float p1 = evaluate_lidar(thing(1.0f), room(), h, kO, "helios").p;
        const float p01 = evaluate_lidar(thing(0.1f), room(), h, kO, "helios").p;
        std::printf("f: p(q=1)=%.4f p(q=.1)=%.4f\n", p1, p01);
        CHECK(p01 < p1);
    }
    { // (g) a named obstacle between the sensor and the target is named
        World w = room();
        w.obstacles.push_back({"box_1", rc::nbv::Obstacle{1.5f, 0.0f, 0.5f, 2.0f, 0.0f, 0.0f, 2.5f}});
        const auto r = evaluate_lidar(c, w, at(3, 0, 0.7f, 0.7f, 0, 1.8f), kO, "helios");
        std::printf("g: p=%.4f limit=%d occ=%s\n", r.p, int(r.limit), r.occluder.c_str());
        CHECK(r.occluder == "box_1" and r.p < 0.99f);
    }
    std::printf(bad ? "FAILED %d\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
