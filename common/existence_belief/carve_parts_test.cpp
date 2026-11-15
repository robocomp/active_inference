// carve_parts_test — rc::exist::carve_parts, the 3-D carve against a union of part boxes (plan 2026-11-15 task 1).
// Run: g++ -std=c++23 -O2 -I/usr/include/eigen3 carve_parts_test.cpp -o /tmp/carve_parts_test && /tmp/carve_parts_test
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include "existence_belief.h"

using namespace rc::exist;
using V3 = Eigen::Vector3f;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static SensorModel model()
{
    SensorModel m;
    m.sensor_sigma_m = 0.03f; m.detection_prob = 0.85f; m.clutter_prob = 0.05f;
    return m;
}

// first hit of the ray O -> T against axis-aligned boxes (BoxPart, world frame) and the floor z = 0
static V3 first_hit(const V3& O, const V3& T, const std::vector<BoxPart>& scene)
{
    const V3 d = T - O;
    float best = std::numeric_limits<float>::max();
    if (d.z() < 0.0f) best = std::min(best, -O.z() / d.z());
    for (const auto& b : scene)
    {
        const float lo[3] = {b.cx - 0.5f * b.sx, b.cy - 0.5f * b.sy, b.z0};
        const float hi[3] = {b.cx + 0.5f * b.sx, b.cy + 0.5f * b.sy, b.z1};
        float tn = 0.0f, tf = std::numeric_limits<float>::max();
        bool miss = false;
        for (int a = 0; a < 3 and not miss; ++a)
        {
            if (std::abs(d(a)) < 1e-9f) { if (O(a) < lo[a] or O(a) > hi[a]) miss = true; }
            else
            {
                float t1 = (lo[a] - O(a)) / d(a), t2 = (hi[a] - O(a)) / d(a);
                if (t1 > t2) std::swap(t1, t2);
                tn = std::max(tn, t1); tf = std::min(tf, t2);
            }
        }
        if (not miss and tn <= tf and tn > 0.0f) best = std::min(best, tn);
    }
    return O + best * d;
}

static std::vector<V3> sweep_to(const V3& O, const std::vector<V3>& targets, const std::vector<BoxPart>& scene)
{
    std::vector<V3> s;
    for (const auto& t : targets) s.push_back(first_hit(O, t, scene));
    return s;
}

int main()
{
    const SensorModel sm = model();
    const V3 O(-2.0f, 0.0f, 1.1075f);
    const BoxPart box{0, 0, 0.5f, 0.5f, 0, 0.62f};
    const std::vector<BoxPart> box_scene{box};

    // top returns are SURFACE, not interior (and carve_box documents the bug)
    {
        std::vector<V3> tg;
        for (int i = 0; i < 200; ++i)
            tg.emplace_back(-0.2f + 0.4f * (i % 20) / 19.0f, -0.2f + 0.4f * (i / 20) / 9.0f, 0.62f);
        const auto sw = sweep_to(O, tg, box_scene);
        const auto ev = carve_parts(O, sw, 0, 0, 0, std::span<const BoxPart>(&box, 1), 0.02f, sm);
        CHECK(ev.e_occ > 80.0f);   // a surface return is half-reached (same convention as carve_box)
        CHECK(ev.e_interior + ev.e_free < 5.0f);
        const auto old = carve_box(O, sw, 0, 0, 0, 0.5f, 0.5f, 0.0f, 0.62f, 0.02f, sm);
        CHECK(old.e_interior > 50.0f);
    }
    // beams passing above the box add nothing
    {
        std::vector<V3> tg;
        for (int i = 0; i < 50; ++i) tg.emplace_back(6.0f, -0.2f + 0.4f * i / 49.0f, 0.0f);
        const auto sw = sweep_to(O, tg, {});
        const auto ev = carve_parts(O, sw, 0, 0, 0, std::span<const BoxPart>(&box, 1), 0.02f, sm);
        CHECK(ev.n_reached == 0);
    }
    // a return on the front face of a thin panel is not "through"
    {
        const BoxPart panel{0, 0, 1.75f, 0.10f, 0, 1.25f};
        const BoxPart pan_world{0, 0, 0.10f, 1.75f, 0, 1.25f};   // panel's long side along world y
        // panel long axis is local x; with yaw = pi/2 its face normal points along world x toward the sensor
        std::vector<V3> tg;
        for (int i = 0; i < 100; ++i)
            tg.emplace_back(0.0f, -0.8f + 1.6f * (i % 10) / 9.0f, 0.1f + 1.1f * (i / 10) / 9.0f);
        const auto sw = sweep_to(O, tg, {pan_world});
        const auto ev = carve_parts(O, sw, 0, 0, 1.5707963f, std::span<const BoxPart>(&panel, 1), 0.05f, sm);
        CHECK(ev.e_free < 5.0f);
        CHECK(ev.e_occ > 40.0f);
    }
    // box over empty floor: wall behind it, returns come out the back
    {
        // (a wall at x = 4 cannot be reached through the box from a 1.1 m sensor: the beam is still at 0.74 m over
        // the box. The beams land on the floor at x = 1.5, which they reach after crossing the box's side face.)
        std::vector<V3> tg;
        for (int i = 0; i < 50; ++i) tg.emplace_back(1.5f, -0.2f + 0.4f * i / 49.0f, 0.0f);
        const auto sw = sweep_to(O, tg, {});
        const auto ev = carve_parts(O, sw, 0, 0, 0, std::span<const BoxPart>(&box, 1), 0.02f, sm);
        CHECK(ev.e_free + ev.e_interior > ev.e_occ);
        CHECK(solid_delta(ev, sm) < 0.0f);
    }
    // box drawn over a wall: wall returns lie deeper inside than sigma from the entry
    {
        const BoxPart wall{2.0f, 0, 0.2f, 6, 0, 3};
        std::vector<V3> tg;
        for (int i = 0; i < 50; ++i) tg.emplace_back(2.0f, -0.2f + 0.4f * i / 49.0f, 0.3f);
        const auto sw = sweep_to(O, tg, {wall});
        const BoxPart b2{0, 0, 0.5f, 0.5f, 0, 0.62f};
        const auto ev = carve_parts(O, sw, 2.1f, 0, 0, std::span<const BoxPart>(&b2, 1), 0.02f, sm);
        CHECK(solid_delta(ev, sm) < 0.0f);
    }
    // no top returns: one whole-box part agrees with carve_box on e_occ
    {
        std::vector<V3> tg;
        for (int i = 0; i < 100; ++i)
            tg.emplace_back(-0.25f, -0.2f + 0.4f * (i % 10) / 9.0f, 0.2f + 0.2f * (i / 10) / 9.0f);
        const auto sw = sweep_to(O, tg, box_scene);
        const auto a = carve_parts(O, sw, 0, 0, 0, std::span<const BoxPart>(&box, 1), 0.02f, sm);
        const auto b = carve_box(O, sw, 0, 0, 0, 0.5f, 0.5f, 0.0f, 0.62f, 0.02f, sm);
        CHECK(a.e_occ > 0.0f);
        CHECK(std::abs(a.e_occ - b.e_occ) <= 0.05f * std::max(a.e_occ, b.e_occ));
    }
    // origin inside the part: contributes nothing
    {
        const V3 Oin(0.0f, 0.0f, 0.3f);
        std::vector<V3> sw;
        for (int i = 0; i < 40; ++i) sw.emplace_back(3.0f, -1.0f + 2.0f * i / 39.0f, 0.5f);
        const auto ev = carve_parts(Oin, sw, 0, 0, 0, std::span<const BoxPart>(&box, 1), 0.02f, sm);
        CHECK(ev.e_occ == 0.0f and ev.e_free == 0.0f and ev.e_interior == 0.0f and ev.n_reached == 0);
    }
    // parts follow the object's yaw
    {
        const BoxPart part{0, 0.2f, 0.5f, 0.1f, 0, 1.0f};
        const std::vector<V3> sw{V3(-0.25f, 0.0f, 0.9f)};
        const auto a = carve_parts(O, sw, 0, 0, 1.5707963f, std::span<const BoxPart>(&part, 1), 0.02f, sm);
        const auto b = carve_parts(O, sw, 0, 0, 0.0f,       std::span<const BoxPart>(&part, 1), 0.02f, sm);
        CHECK(a.e_occ > 0.3f);
        CHECK(b.e_occ < 1e-3f);
    }
    std::printf(bad ? "carve_parts_test: %d FAILED\n" : "carve_parts_test: all passed\n", bad);
    return bad ? 1 : 0;
}
