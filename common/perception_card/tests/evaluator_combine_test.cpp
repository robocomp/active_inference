// Channel combination (notice vs recognise), Inside handling, where_to_stand.
//   LANG=es_ES.UTF-8 ./evaluator_combine_test
#include "perception_card/evaluator.h"
#include <chrono>
#include <clocale>
#include <cstdio>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
using namespace rc::card;

static Card fridge()
{
    Card c; c.concept_ = "fridge"; c.graph_type = "object"; c.level = Level::Countable;
    c.channels = {{"zed", Role::Identity, true}, {"helios", Role::Occupancy, true}};
    c.geometry = {0.7f, 0.7f, 1.8f, 0.0f, 1.8f};
    SubtypeParams d; d.env = {0.02f, 1.32f, 0.05f}; d.q_return = 1.0f;
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
static World room()
{
    World w;
    w.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
    rc::nbv::Sensor s;
    constexpr float f = 448.0f;
    s.hfov_rad = 2.0f * std::atan(640.0f / f); s.vfov_rad = 2.0f * std::atan(360.0f / f); s.height_m = 0.945f;
    w.zed = s; w.zed_mount_z = 0.945f;
    w.helios = helios();
    return w;
}
static Hypothesis at(float x, float y) { Hypothesis h; h.concept_ = "fridge"; h.x = x; h.y = y; return h; }

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const Card c = fridge();

    { // (a) low box: ZED (0.945 m) is blocked, the LiDAR (1.075 m) sees over it
        World w = room();
        w.obstacles.push_back({"low_box", rc::nbv::Obstacle{3.5f, 2.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f}});
        const auto r = evaluate(c, w, at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        for (const auto& ch : r.channels) std::printf("a: %s p=%.3f limit=%d\n", ch.sensor.c_str(), ch.p, int(ch.limit));
        std::printf("a: notice=%.3f recognise=%.3f limit=%d occ=%s\n", r.p_notice, r.p_recognise, int(r.limit), r.occluder.c_str());
        CHECK(r.p_notice > 0.9f and r.p_recognise < 0.3f);
        CHECK(r.limit == Limit::Occluded and r.occluder == "low_box" and r.limit_gain > 0.0f);
    }
    { // NotModelled channels are left out of both products
        World w = room(); w.helios.reset();
        const auto r = evaluate(c, w, at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        CHECK(r.p_notice == 0.0f and r.p_recognise > 0.0f);
    }
    { // (b) viewpoint inside the box
        World w = room();
        w.obstacles.push_back({"box_1", rc::nbv::Obstacle{2.0f, 2.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.2f}});
        const auto r = evaluate(c, w, at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        CHECK(r.limit == Limit::Inside and r.p_notice == 0.0f and r.p_recognise == 0.0f and r.occluder == "box_1");
        CHECK(std::isfinite(r.p_notice) and std::isfinite(r.p_recognise));
        // outside the polygon, and the target inside a wall
        const auto o = evaluate(c, room(), at(5.0f, 2.0f), {{-2.0f, 2.0f}, std::nullopt});
        CHECK(o.limit == Limit::Inside and o.p_recognise == 0.0f);
        const auto tw = evaluate(c, room(), at(6.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        CHECK(tw.limit == Limit::Inside and tw.p_recognise == 0.0f);
    }
    { // (c)+(d) where_to_stand for the occluded fridge
        World w = room();
        w.obstacles.push_back({"box_1", rc::nbv::Obstacle{3.5f, 2.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.2f}});
        const Hypothesis h = at(5.0f, 2.0f);
        const auto t0 = std::chrono::steady_clock::now();
        const auto st = where_to_stand(c, w, h);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("where_to_stand: %zu stands, %.1f ms\n", st.size(), ms);
        for (const auto& s : st) std::printf("  (%.2f, %.2f) yaw=%.2f rec=%.3f\n", s.xy.x(), s.xy.y(), s.yaw, s.r.p_recognise);
        CHECK(not st.empty() and st.front().r.p_recognise > 0.8f);
        // not behind the box: the straight line stand->fridge must not cross it
        if (not st.empty())
        {
            const Eigen::Vector2f s = st.front().xy;
            const rc::nbv::Obstacle box = w.obstacles[0].second;
            bool crosses = false;
            for (int i = 0; i <= 100; ++i)
            {
                const Eigen::Vector2f p = s + (Eigen::Vector2f(h.x, h.y) - s) * (static_cast<float>(i) / 100.0f);
                crosses = crosses or (std::abs(p.x() - box.cx) <= 0.5f * box.w and std::abs(p.y() - box.cy) <= 0.5f * box.h);
            }
            CHECK(not crosses);
        }
        CHECK(ms < 200.0);
    }
    if (bad) { std::printf("evaluator_combine_test: %d FAILURES\n", bad); return 1; }
    std::printf("evaluator_combine_test: ok\n");
    return 0;
}
