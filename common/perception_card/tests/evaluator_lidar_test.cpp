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
        CHECK(r.limit == Limit::DeadZone and r.p < 0.05f);   // range factor is common-mode: ~Phi(-2.5)=0.006
    }
    { // (d3) range limit named: a target the bpearl rings DO cross, beyond 30 m. Bpearl room elevation reaches only
      // -90..0, so a z span at the sensor height (0.67) crosses the 0 deg ring.
        World w = room(); w.polygon = {{-40, -40}, {40, -40}, {40, 40}, {-40, 40}};
        const auto r = evaluate_lidar(c, w, at(32.0f, 0, 3.0f, 3.0f, 0.5f, 0.9f), kO, "bpearl");
        std::printf("d3: p=%.4f limit=%d\n", r.p, int(r.limit));
        CHECK(r.limit == Limit::Range and r.p < 0.05f);
    }
    { // AboveFan: helios room fan tops out at +55 deg; a 2.5 m high object at 0.5 m is above it
        const auto r = evaluate_lidar(c, room(), at(0.5f, 0, 0.3f, 0.3f, 2.5f, 2.7f), kO, "helios");
        std::printf("above: p=%.4f limit=%d\n", r.p, int(r.limit));
        CHECK(r.limit == Limit::AboveFan);
    }
    { // (e) no helios in the world
        World w = room(); w.helios.reset();
        const auto r = evaluate_lidar(c, w, at(3, 0, 0.7f, 0.7f, 0, 1.8f), kO, "helios");
        // Not-modelled is expressed by the limit alone (ChannelResult has no other flag), so a combiner drops the
        // channel on `limit == NotModelled`. p is deliberately not asserted.
        CHECK(r.limit == Limit::NotModelled);
    }
    { // (e2) bpearl absent
        World w = room(); w.bpearl.reset();
        CHECK(evaluate_lidar(c, w, at(3, 0, 0.7f, 0.7f, 0, 1.8f), kO, "bpearl").limit == Limit::NotModelled);
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
        CHECK(r.limit == Limit::Occluded and r.occluder == "box_1" and r.p < 0.99f);
    }
    { // uniform-ring fallback == the same rings given explicitly
        LidarRig a = helios(), b = helios();
        for (int i = 0; i < 32; ++i) b.model.ring_elev_deg.push_back(-55.0f + 70.0f * i / 31.0f);
        const rc::nbv::Target t{3, 0, 0, 0.7f, 0.7f, 0, 1.8f};
        const float la = expected_returns(a, t, {0, 0}, {}), lb = expected_returns(b, t, {0, 0}, {});
        std::printf("uniform: %.2f vs table %.2f\n", la, lb);
        CHECK(std::abs(la - lb) < 1e-3f and la > 0.0f);
        World w = room();
        const auto r = evaluate_lidar(c, w, at(3, 0, 0.7f, 0.7f, 0, 1.8f), kO, "helios");
        CHECK(std::ranges::any_of(r.notes, [](const std::string& n) { return n.find("uniformly") != std::string::npos; }));
    }
    { // upright (non-flipped) mount: same ring set mirrored about the sensor height
        LidarRig up = helios(); up.mount.linear() = Eigen::Matrix3f::Identity();
        const float oz = 1.075f;
        const rc::nbv::Target inv{3, 0, 0, 0.7f, 0.7f, 0.35f, 1.85f};
        const rc::nbv::Target mir{3, 0, 0, 0.7f, 0.7f, 2 * oz - 1.85f, 2 * oz - 0.35f};
        const float l1 = expected_returns(helios(), inv, {0, 0}, {}), l2 = expected_returns(up, mir, {0, 0}, {});
        std::printf("mirror: inverted %.2f upright-mirrored %.2f\n", l1, l2);
        CHECK(std::abs(l1 - l2) < 1e-3f and l1 > 0.0f);
        // and the fan really is flipped: an object at +40 deg elevation is above the upright fan, inside the inverted one
        World w = room(); w.helios = up;
        const Hypothesis hi = at(0.5f, 0, 0.3f, 0.3f, 1.5f, 1.6f);
        CHECK(evaluate_lidar(c, w, hi, kO, "helios").limit == Limit::AboveFan);
        CHECK(evaluate_lidar(c, room(), hi, kO, "helios").limit != Limit::AboveFan);
    }
    { // I5: the mount is body_T_sensor and Shadow's forward is body +y, so room_T_sensor = Rz(heading - pi/2) * mount.
      // Helios at body (0, -0.155) (ROBOT_GEOMETRY.md: 15.5 cm BEHIND the body origin), robot at (0, 0):
      //   heading pi/2 (facing room +y): Rz(0) * (0, -0.155)          = (0, -0.155)   -- behind, toward -y
      //   heading 0    (facing room +x): Rz(-pi/2) * (0, -0.155)
      //                                  = (0*cos(-pi/2) - (-0.155)*sin(-pi/2), 0*sin(-pi/2) + (-0.155)*cos(-pi/2))
      //                                  = (-0.155, 0)                                 -- behind, toward -x
      // (The old x-forward lever Rz(heading) put it at (0.155, 0) for heading pi/2: beside the robot, not behind.)
        LidarRig hb = helios(); hb.mount.translation() = Eigen::Vector3f(0.0f, -0.155f, 1.075f);
        const Hypothesis far = at(0.0f, 2.9f, 0.3f, 0.3f, 0.0f, 1.8f);
        const Eigen::Vector3f o_up = rig_in_room(hb, {{0.0f, 0.0f}, std::numbers::pi_v<float> / 2.0f}, far).mount.translation();
        const Eigen::Vector3f o_x  = rig_in_room(hb, {{0.0f, 0.0f}, 0.0f}, far).mount.translation();
        std::printf("I5: heading pi/2 -> (%.4f, %.4f, %.3f); heading 0 -> (%.4f, %.4f)\n", o_up.x(), o_up.y(), o_up.z(), o_x.x(), o_x.y());
        CHECK(std::abs(o_up.x()) < 1e-5f and std::abs(o_up.y() + 0.155f) < 1e-5f and std::abs(o_up.z() - 1.075f) < 1e-5f);
        CHECK(std::abs(o_x.x() + 0.155f) < 1e-5f and std::abs(o_x.y()) < 1e-5f);
        // best heading (vp.yaw unset) = the bearing to the target: a target on +x gives the heading-0 origin
        const Eigen::Vector3f o_best = rig_in_room(hb, {{0.0f, 0.0f}, std::nullopt}, at(2.9f, 0.0f, 0.3f, 0.3f, 0.0f, 1.8f)).mount.translation();
        CHECK(std::abs(o_best.x() + 0.155f) < 1e-5f and std::abs(o_best.y()) < 1e-5f);
        // End to end through evaluate(): a 3.0 m far limit (sigma 1 cm) and the target 2.9 m from the ROBOT. Behind
        // the robot by 0.155 the sensor is 3.055 m away (past the limit, P ~ 0); the x-forward lever would have put it
        // 2.904 m away (inside, P ~ 1).
        LidarRig sh = hb; sh.model.range_max_m = 3.0f; sh.model.range_sigma_floor_m = 0.01f;
        Card hc = thing(); hc.channels = {{"helios", Role::Both, true}};
        World w = room(); w.helios = sh; w.bpearl.reset();
        const auto r_up = evaluate(hc, w, far, {{0.0f, 0.0f}, std::numbers::pi_v<float> / 2.0f});
        const auto r_best = evaluate(hc, w, at(2.9f, 0.0f, 0.3f, 0.3f, 0.0f, 1.8f), {{0.0f, 0.0f}, std::nullopt});
        std::printf("I5 evaluate: heading pi/2 p=%.4f limit=%d; best heading p=%.4f limit=%d\n",
                    r_up.p_notice, int(r_up.limit), r_best.p_notice, int(r_best.limit));
        CHECK(r_up.p_notice < 1e-3f and r_up.limit == Limit::Range);
        CHECK(r_best.p_notice < 1e-3f and r_best.limit == Limit::Range);
        // facing AWAY from it (heading -pi/2) the helios is 0.155 m closer: 2.745 m, inside the limit
        const auto r_away = evaluate(hc, w, far, {{0.0f, 0.0f}, -std::numbers::pi_v<float> / 2.0f});
        CHECK(r_away.p_notice > 0.99f);
    }
    { // beyond range with NO ring crossing: bpearl (room elevations -90..0 deg) and a target ABOVE it, 32 m away.
      // Neither fixing the rings nor fixing the range alone raises P; the limit must still say why, never None at p 0.
        World w = room(); w.polygon = {{-40, -40}, {40, -40}, {40, 40}, {-40, 40}};
        const auto r = evaluate_lidar(c, w, at(32.0f, 0, 3.0f, 3.0f, 1.0f, 1.2f), kO, "bpearl");
        std::printf("beyond+nofan: p=%.4f limit=%d gain=%.3f\n", r.p, int(r.limit), r.limit_gain);
        CHECK(r.p < 1e-3f and r.limit == Limit::Range and r.limit_gain > 0.5f);
        // the same height in range: the fan is what is missing
        const auto rn = evaluate_lidar(c, w, at(3.0f, 0, 0.3f, 0.3f, 1.0f, 1.2f), kO, "bpearl");
        CHECK(rn.limit == Limit::AboveFan);
        // and in the dead zone with no ring: DeadZone, not None
        const auto rd = evaluate_lidar(c, w, at(0.05f, 0, 0.02f, 0.02f, 1.0f, 1.2f), kO, "bpearl");
        std::printf("dead+nofan: p=%.4f limit=%d\n", rd.p, int(rd.limit));
        CHECK(rd.limit != Limit::None);
    }
    std::printf(bad ? "FAILED %d\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
