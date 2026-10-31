// render_lidar vs the evaluator's returns model: ring ray-cast through the real mount.
//   LANG=es_ES.UTF-8 ./render_lidar_test
#include "perception_card/render.h"
#include <chrono>
#include <clocale>
#include <cstdio>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
using namespace rc::card;

static Card chair_card(float q = 1.0f)
{
    Card c; c.concept_ = "chair"; c.graph_type = "object"; c.level = Level::Countable;
    c.channels = {{"helios", Role::Occupancy, true}, {"bpearl", Role::Occupancy, true}};
    c.geometry = {0.45f, 0.45f, 0.9f, 0.0f, 0.9f};
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
// seat + back + four legs at (x, y), heading yaw
static Placed chair_at(float x, float y, float yaw)
{
    Placed p; const float c = std::cos(yaw), s = std::sin(yaw);
    const auto put = [&](float lx, float ly, float w, float d, float z0, float z1)
    { p.boxes.push_back(rc::nbv::Obstacle{x + lx * c - ly * s, y + lx * s + ly * c, w, d, yaw, z0, z1}); };
    put(0, 0, 0.45f, 0.45f, 0.40f, 0.48f);            // 8 cm seat cushion
    put(0, -0.2f, 0.45f, 0.06f, 0.48f, 0.9f);        // back
    for (const float sx : {-1.0f, 1.0f}) for (const float sy : {-1.0f, 1.0f}) put(sx * 0.19f, sy * 0.19f, 0.05f, 0.05f, 0.0f, 0.40f);
    return p;
}
static const Viewpoint kO{{0.0f, 0.0f}, 0.0f};

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const Card card = chair_card();
    const World w = room();
    const float yaw0 = std::numbers::pi_v<float> / 2.0f;

    { // (a) lambda0 against the evaluator's expected_returns on the bounding box (controller ruling P2: [0.3, 1.05] x box).
      // The chair's back is local -y: yaw pi/2 puts it on the far side, square to the viewer (the usual case); yaw 0 shows it
      // edge-on, so only the thin back's edge faces the rays. The edge-on ratio is printed, and bound above only.
        for (const float yaw : {std::numbers::pi_v<float> / 2.0f, 0.0f})
        {
            const bool front = yaw > 0.1f;
            const Placed ch = chair_at(3.0f, 0.0f, yaw);
            const rc::nbv::Target t = detail::target_of(ch);
            for (const char* name : {"helios", "bpearl"})
            {
                const LidarRig& rig = std::string(name) == "helios" ? *w.helios : *w.bpearl;
                Hypothesis h; h.x = 3.0f; h.y = 0.0f;
                const LidarRig rr = rig_in_room(rig, kO, h);
                const float box = expected_returns(rr, t, rr.mount.translation().head<2>(), {});
                const auto sc = render_lidar(w, kO, &rig, ch, 1, card);
                std::printf("a %s %s: render lambda0=%.1f  box lambda0=%.1f  ratio=%.3f  on_target=%zu on_other=%zu\n", name,
                            front ? "front-on" : "edge-on", sc.lambda0, box, box > 0 ? sc.lambda0 / box : 0.0f, sc.on_target.size(), sc.on_other.size());
                CHECK(sc.modelled and box > 20.0f);
                CHECK(sc.lambda0 <= 1.05f * box);
                if (front) CHECK(sc.lambda0 >= 0.3f * box);
                CHECK(sc.ring_elev_room_deg.size() == 32);
            }
        }
    }
    { // (b) a thin seat midway between two helios rings at 6 m: no returns on the seat's z band
        World wb = w; wb.helios->model.range_sigma_floor_m = 0.001f;
        const float step = 70.0f / 31.0f, e = (-55.0f + 10 * step + 0.5f * step) * std::numbers::pi_v<float> / 180.0f;
        const float z = 1.075f - 6.0f * std::tan(e);
        Placed p;
        p.boxes.push_back(rc::nbv::Obstacle{6.0f, 0.0f, 0.45f, 0.45f, 0.0f, z - 0.01f, z + 0.01f});
        p.boxes.push_back(rc::nbv::Obstacle{6.0f, 0.0f, 0.04f, 0.04f, 0.0f, 0.0f, z - 0.05f});
        const auto sc = render_lidar(wb, kO, &*wb.helios, p, 1, card);
        int on_seat = 0;
        for (const auto& q : sc.on_target) if (std::abs(q.z() - z) < 0.02f) ++on_seat;
        std::printf("b: seat z=%.3f on_target=%zu on_seat_band=%d lambda0=%.0f\n", z, sc.on_target.size(), on_seat, sc.lambda0);
        CHECK(on_seat == 0);
    }
    { // (c) determinism, and a different seed differs (q < 1 so the draw matters)
        const Card c5 = chair_card(0.5f);
        const Placed ch = chair_at(3.0f, 0.0f, yaw0);
        const auto a = render_lidar(w, kO, &*w.helios, ch, 7, c5), b = render_lidar(w, kO, &*w.helios, ch, 7, c5);
        const auto d = render_lidar(w, kO, &*w.helios, ch, 8, c5);
        CHECK(a.on_target.size() == b.on_target.size() and not a.on_target.empty());
        bool same = a.on_target.size() == b.on_target.size();
        for (std::size_t i = 0; same and i < a.on_target.size(); ++i) same = a.on_target[i] == b.on_target[i];
        CHECK(same);
        CHECK(a.on_target.size() != d.on_target.size() or a.on_target.front() != d.on_target.front());
        CHECK(a.on_target.size() < a.lambda0);   // q = 0.5 thins the kept hits
    }
    { // (d) not modelled
        const Placed ch = chair_at(3.0f, 0.0f, yaw0);
        CHECK(not render_lidar(w, kO, nullptr, ch, 1, card).modelled);
        LidarRig empty; empty.mount = Eigen::Isometry3f::Identity();
        CHECK(not render_lidar(w, kO, &empty, ch, 1, card).modelled);
    }
    { // (e) an obstacle in front hides the chair: lambda0 drops, hits land on the obstacle (on_other)
        World wo = w; wo.obstacles.push_back({"box_1", rc::nbv::Obstacle{1.5f, 0.0f, 0.5f, 2.0f, 0.0f, 0.0f, 2.5f}});
        const Placed ch = chair_at(3.0f, 0.0f, yaw0);
        const auto free = render_lidar(w, kO, &*w.helios, ch, 1, card), occ = render_lidar(wo, kO, &*wo.helios, ch, 1, card);
        std::printf("e: free lambda0=%.0f occluded lambda0=%.0f on_other=%zu\n", free.lambda0, occ.lambda0, occ.on_other.size());
        CHECK(occ.lambda0 < 0.1f * free.lambda0 and not occ.on_other.empty());
    }
    { // timing
        const Placed ch = chair_at(3.0f, 0.0f, yaw0);
        const int N = 200; std::size_t sink = 0;
        for (const LidarRig* rig : {&*w.helios, &*w.bpearl})
        {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i) sink += render_lidar(w, kO, rig, ch, i, card).on_target.size();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N;
            std::printf("timing %s: %.3f ms per render_lidar (sink %zu)\n", rig == &*w.helios ? "helios" : "bpearl", ms, sink);
        }
    }
    std::printf(bad ? "FAILED (%d)\n" : "OK\n", bad);
    return bad ? 1 : 0;
}
