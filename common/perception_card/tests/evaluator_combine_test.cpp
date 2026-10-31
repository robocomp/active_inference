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
    // The refrigerator agent's REAL config: refrigerator_concept/etc/config.toml:166-168 (DetectMinFill 0.0, MaxFill 1.32, Soft 0.26)
    SubtypeParams d; d.env = {0.0f, 1.32f, 0.26f}; d.q_return = 1.0f;
    c.subtypes[""] = d;
    return c;
}
// The refrigerator agent's card as refrigerator_card.h builds it from etc/config.toml (refrigerator_card_test pins
// those numbers): 0.60 x 0.60 x 1.90 m, envelope 0.0/1.32/0.26 single-fill (axes 1) x central, q_return 0.85, and
// level 1 = refrigerator_projection.cpp's own-wall skip 0.70 m against walls_block's LINE walls.
static Card real_fridge()
{
    Card c; c.concept_ = "refrigerator"; c.graph_type = "object"; c.level = Level::Countable;
    c.channels = {{"zed", Role::Identity, true}, {"helios", Role::Occupancy, true}, {"ricoh", Role::Identity, false}};
    c.geometry = {0.60f, 0.60f, 1.90f, 0.0f, 1.90f};
    c.walls = WallOcclusion{0.70f, -1.0f, 0.0f};
    SubtypeParams d; d.env = {0.0f, 1.32f, 0.26f}; d.q_return = 0.85f;
    d.terms = {{"envelope", {{"min_fill", 0.0f}, {"max_fill", 1.32f}, {"soft", 0.26f}, {"axes", 1.0f}}}, {"central", {}}};
    c.subtypes[""] = d;
    return c;
}
// SYNTHETIC: not a real card. A high-envelope fridge (MinFill 0.02, Soft 0.05) whose clear view reaches ~0.96, so the
// absolute 'recognised' level (> 0.8) can be exercised; the real envelope caps a clear view near 0.57.
static Card synthetic_high_envelope()
{
    Card c = fridge(); c.concept_ = "synthetic_high_envelope";
    c.subtypes[""].env = {0.02f, 1.32f, 0.05f};
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

    { // (a) wide box 1.2 m high right in front of the ZED line of sight; LiDAR (1.075 m) is blocked too, so lower the
      // box to 1.0 m: the ZED (0.945) is behind it, the LiDAR looks over it. 2.5 m wide so no edge leaks.
        World w = room();
        w.obstacles.push_back({"low_box", rc::nbv::Obstacle{3.0f, 2.0f, 1.0f, 2.5f, 0.0f, 0.0f, 1.0f}});
        const auto r = evaluate(c, w, at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        for (const auto& ch : r.channels) std::printf("a: %s p=%.3f limit=%d\n", ch.sensor.c_str(), ch.p, int(ch.limit));
        std::printf("a: notice=%.3f recognise=%.3f limit=%d occ=%s\n", r.p_notice, r.p_recognise, int(r.limit), r.occluder.c_str());
        CHECK(r.p_notice > 0.9f and r.p_recognise < 0.5f * r.p_notice);
        CHECK(r.limit == Limit::Occluded and r.occluder == "low_box" and r.limit_gain > 0.0f);
    }
    { // I3, a ZED-only look: NotModelled channels are left out of both products, and the identifying ZED look also
      // NOTICES (recognising implies noticing), so with no Occupancy channel P(notice) = P(recognise), not 0.
        World w = room(); w.helios.reset();
        const auto r = evaluate(c, w, at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        std::printf("I3 zed-only: notice=%.3f recognise=%.3f\n", r.p_notice, r.p_recognise);
        CHECK(r.p_recognise > 0.0f and r.p_notice >= r.p_recognise and std::abs(r.p_notice - r.p_recognise) < 1e-6f);
        // both channels: p_notice = 1 - (1 - p_occ)(1 - p_rec), and >= p_recognise
        const auto b = evaluate(c, room(), at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        float p_occ = 0, p_rec = 0;
        for (const auto& ch : b.channels) (ch.sensor == "zed" ? p_rec : p_occ) = ch.p;
        CHECK(std::abs(b.p_notice - (1.0f - (1.0f - p_occ) * (1.0f - p_rec))) < 1e-6f and b.p_notice >= b.p_recognise);
        // an occluded fridge with the real card too
        World wo = room(); wo.helios.reset();
        wo.obstacles.push_back({"box_1", rc::nbv::Obstacle{3.5f, 2.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.2f}});
        const auto ro = evaluate(real_fridge(), wo, at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        CHECK(ro.p_notice >= ro.p_recognise);
    }
    { // I4: own-wall occlusion, the REAL fridge card. Back 0.35 m from the x = 6 wall (centre 6 - 0.35 - 0.30 = 5.35),
      // seen from a 3.15 m stand-off. P must be the same with that wall and with it moved 3 m away.
        const Card rf = real_fridge();
        const Hypothesis h = at(5.35f, 2.0f);
        const Viewpoint vp{{5.35f - 3.15f, 2.0f}, std::nullopt};
        World far = room(); far.polygon = {{0, 0}, {9, 0}, {9, 4}, {0, 4}};
        const auto rw = evaluate(rf, room(), h, vp), rn = evaluate(rf, far, h, vp);
        std::printf("I4 back 0.35 m: with wall rec=%.4f notice=%.4f limit=%d | wall far rec=%.4f notice=%.4f\n",
                    rw.p_recognise, rw.p_notice, int(rw.limit), rn.p_recognise, rn.p_notice);
        CHECK(std::abs(rw.p_recognise - rn.p_recognise) < 1e-6f and std::abs(rw.p_notice - rn.p_notice) < 1e-6f);
        CHECK(rw.limit != Limit::BehindWall);
        for (const auto& ch : rw.channels) CHECK(ch.limit != Limit::BehindWall);
        // The case the skip exists for: the estimated wall line cuts the carcass (x = 5.5, 0.15 m in front of the back
        // face). Behind the 0.70 m skip it is the fridge's OWN wall and hides nothing; an undeclared card (the old
        // evaluator: 0.12 m boxes, no skip) reads the back half as behind a wall.
        World cut = room(); cut.polygon = {{0, 0}, {5.5f, 0}, {5.5f, 4}, {0, 4}};
        const auto rc_ = evaluate(rf, cut, h, vp);
        std::printf("I4 wall through the carcass: rec=%.4f notice=%.4f limit=%d\n", rc_.p_recognise, rc_.p_notice, int(rc_.limit));
        CHECK(std::abs(rc_.p_recognise - rn.p_recognise) < 1e-6f and std::abs(rc_.p_notice - rn.p_notice) < 1e-6f);
        CHECK(rc_.limit != Limit::BehindWall);
        Card legacy = rf; legacy.walls.reset();
        const auto rl = evaluate(legacy, cut, h, vp);
        std::printf("I4 same, no level-1 settings: rec=%.4f notice=%.4f limit=%d\n", rl.p_recognise, rl.p_notice, int(rl.limit));
        CHECK(rl.p_recognise < rn.p_recognise - 0.1f);
        CHECK(std::ranges::any_of(rl.channels, [](const ChannelResult& ch) { return ch.limit == Limit::BehindWall; }));
        // A wall that is NOT its own still hides it: a partition between the robot and the fridge.
        World part = room(); part.polygon = {{0, 0}, {6, 0}, {6, 4}, {3.5f, 4}, {3.5f, 0.5f}, {3.4f, 0.5f}, {3.4f, 4}, {0, 4}};
        const auto rp = evaluate(rf, part, h, {{2.2f, 3.0f}, std::nullopt});
        std::printf("I4 behind a partition: rec=%.4f limit=%d\n", rp.p_recognise, int(rp.limit));
        CHECK(rp.p_recognise < 0.05f and rp.limit == Limit::BehindWall);
        // the card's thickness is used: the same polygon as 1 mm lines vs 0.12 m boxes lands on the card's own walls
        CHECK(detail::wall_thickness_for(rf) == detail::kLineWallM and detail::wall_thickness_for(legacy) == 0.12f);
    }
    { // central from the projected offset when the heading is GIVEN: 20 deg off the optical axis, the factor is
      // 1 - tan(20 deg) / tan(hfov / 2), exactly (compare against the same card without the central term)
        const Card rf = real_fridge();
        Card nc = rf; nc.subtypes[""].terms.pop_back();
        const Hypothesis h = at(5.0f, 2.0f);
        const float yaw = -20.0f * std::numbers::pi_v<float> / 180.0f;   // target bearing 0 from (2, 2)
        const Viewpoint off{{2.0f, 2.0f}, yaw};
        World w = room(); w.helios.reset();
        const float pc = evaluate(rf, w, h, off).p_recognise, pn = evaluate(nc, w, h, off).p_recognise;
        const float expect = 1.0f - std::tan(20.0f * std::numbers::pi_v<float> / 180.0f) / (640.0f / 448.0f);
        std::printf("central: with %.4f without %.4f ratio %.4f expect %.4f\n", pc, pn, pc / pn, expect);
        CHECK(pn > 0.0f and std::abs(pc / pn - expect) < 1e-3f);
        // best heading: central is 1, the two cards agree
        const float bc = evaluate(rf, w, h, {{2.0f, 2.0f}, std::nullopt}).p_recognise;
        const float bn = evaluate(nc, w, h, {{2.0f, 2.0f}, std::nullopt}).p_recognise;
        CHECK(std::abs(bc - bn) < 1e-6f);
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
        CHECK(o.occluder.empty());   // empty: outside the polygon is not caused by any named obstacle
        CHECK(o.channels.empty());
        const auto tw = evaluate(c, room(), at(6.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        CHECK(tw.limit == Limit::Inside and tw.p_recognise == 0.0f and tw.occluder == "wall");
    }
    { // (c)+(d) where_to_stand for the occluded fridge, REAL envelope: compare with its own clear-view ceiling
        World w = room();
        w.obstacles.push_back({"box_1", rc::nbv::Obstacle{3.5f, 2.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.2f}});
        const Hypothesis h = at(5.0f, 2.0f);
        const Viewpoint occluded{{2.0f, 2.0f}, std::nullopt};
        const float p_occ = evaluate(c, w, h, occluded).p_recognise;
        const auto t0 = std::chrono::steady_clock::now();
        const auto st = where_to_stand(c, w, h);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("where_to_stand: %zu stands, %.1f ms\n", st.size(), ms);
        for (const auto& s : st) std::printf("  (%.2f, %.2f) yaw=%.2f rec=%.3f\n", s.xy.x(), s.xy.y(), s.yaw, s.r.p_recognise);
        CHECK(not st.empty());
        if (not st.empty())
        {
            const float p_clear = evaluate(c, room(), h, {st.front().xy, st.front().yaw}).p_recognise;   // same stand, no box
            std::printf("c: p_clear=%.3f best_stand=%.3f occluded_view=%.3f\n", p_clear, st.front().r.p_recognise, p_occ);
            // The REAL fridge card's clear-view ceiling (R6), with its own-wall settings and without them.
            const Card rf = real_fridge();
            Card rl = rf; rl.walls.reset();
            const auto st_r = where_to_stand(rf, w, h);
            if (not st_r.empty())
            {
                const Viewpoint v{st_r.front().xy, st_r.front().yaw};
                std::printf("c(real card): p_clear=%.3f (no level-1: %.3f) best_stand=%.3f occluded_view=%.3f\n",
                            evaluate(rf, room(), h, v).p_recognise, evaluate(rl, room(), h, v).p_recognise,
                            st_r.front().r.p_recognise, evaluate(rf, w, h, occluded).p_recognise);
            }
            CHECK(st.front().r.p_recognise >= 0.95f * p_clear);
            CHECK(st.front().r.p_recognise > p_occ);
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
        CHECK(where_to_stand(c, w, h, 1).size() == 1);
    }
    { // synthetic high envelope: the absolute level is reachable
        const Card s = synthetic_high_envelope();
        World w = room();
        w.obstacles.push_back({"box_1", rc::nbv::Obstacle{3.5f, 2.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.2f}});
        const auto st = where_to_stand(s, w, at(5.0f, 2.0f));
        CHECK(not st.empty() and st.front().r.p_recognise > 0.8f);
    }
    { // Role::Both contributes to both products; no identity channel => NotModelled
        Card b = fridge(); b.channels = {{"helios", Role::Both, true}};
        const auto r = evaluate(b, room(), at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        CHECK(r.p_notice > 0.9f and r.p_recognise > 0.9f);
        Card o = fridge(); o.channels = {{"helios", Role::Occupancy, true}};
        const auto ro = evaluate(o, room(), at(5.0f, 2.0f), {{2.0f, 2.0f}, std::nullopt});
        CHECK(ro.p_notice > 0.9f and ro.p_recognise == 0.0f and ro.limit == Limit::NotModelled);
    }
    { // an infeasible stand is dropped: a big box swallowing the ring the fridge's best stands lie on
        World w = room();
        const Hypothesis h = at(3.0f, 2.0f);
        const auto free_st = where_to_stand(c, w, h, 1);
        CHECK(not free_st.empty());
        if (not free_st.empty())
        {
            const Eigen::Vector2f b = free_st.front().xy;
            w.obstacles.push_back({"blocker", rc::nbv::Obstacle{b.x(), b.y(), 1.0f, 1.0f, 0.0f, 0.0f, 0.5f}});
            const auto st = where_to_stand(c, w, h, 36);
            CHECK(not st.empty());
            for (const auto& s : st)
                CHECK(not (std::abs(s.xy.x() - b.x()) <= 0.5f and std::abs(s.xy.y() - b.y()) <= 0.5f));
        }
    }
    if (bad) { std::printf("evaluator_combine_test: %d FAILURES\n", bad); return 1; }
    std::printf("evaluator_combine_test: ok\n");
    return 0;
}
