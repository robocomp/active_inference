// test_door_ricoh_lifecycle.cpp — the 360 camera's door lifecycle end to end on the REAL existence slice (task Q, fix round 1):
//   A. a real ricoh-only door is born exactly once and SURVIVES 400 frames of continued ricoh detections (and the ZED never
//      probes it - no ZED mask pixels at all): no birth -> remove -> rebirth churn.
//   B. a ricoh PHANTOM (the 360 stops seeing it; the ZED cannot probe it either) is REMOVED by the ricoh's own absence channel
//      - the same P(detect | ricoh, viewpoint) that weighted its birth - exactly once, and not reborn.
//   C. weak looks are weak evidence: from far away (range weight small) the same absence takes longer than from near.
// Build + run (from door_concept/): see the g++ line of test_door_core_existence.cpp, with this file in its place.
#include <charconv>
#include <clocale>
#include <cmath>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>
#include "../src/core/door_core.h"
#include "../src/core/door_existence.h"

using namespace rc::door::core;
using K = rc::door::DoorDecisionKind;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

// The fit test's world (test_door_core_fit.cpp) plus the existence slice's two removal callbacks.
struct ExistWorld final : TrackerWorld
{
    DoorCore* core = nullptr;
    rc::DoorConfig cfg;
    std::uint64_t room = 7;
    std::vector<Vec2f> polygon{{-5.f, -5.f}, {5.f, -5.f}, {5.f, 5.f}, {-5.f, 5.f}};
    mutable std::unordered_map<std::uint64_t, rc::DoorFitState> st;
    std::uint64_t next_id = 1000;
    std::vector<std::string> events;   // the order of before_remove / remove_door

    std::uint64_t room_id() const override { return room; }
    std::vector<Vec2f> room_polygon() const override { return polygon; }
    std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> fit_states() const override
    {
        std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> v;
        for (auto& [id, s] : st) v.emplace_back(id, &s);
        return v;
    }
    std::optional<GhostMatch> match_ghost(const Vec2f&) const override { return std::nullopt; }
    std::optional<IdentityMatch> match_identity(const Vec2f&) const override { return std::nullopt; }
    std::vector<std::string> reserved_names() const override { return {}; }
    void retire(std::uint64_t id) override { st.erase(id); }
    void set_assigned_mask(std::uint64_t id, int slice) override
    { if (auto it = st.find(id); it != st.end()) it->second.assigned_mask_idx = slice; }
    void after_association(const TrackerCycleInfo&) override {}
    std::uint64_t create_door(const BirthRequest&) override
    {
        const std::uint64_t id = next_id++;
        rc::DoorState s;
        s.w = cfg.door_prior_w_m; s.h = cfg.door_prior_h_m; s.thickness = cfg.door_thickness_m;
        rc::DoorFitState f;
        core->fit().init_instance(f, id, "door_" + std::to_string(st.size() + 1), s);
        st.emplace(id, std::move(f));
        return id;
    }
    void reacquire(std::uint64_t, const GhostMatch&) override {}
    void forget_ghost(const Vec2f&) override {}
    std::optional<std::string> adopt_birth(std::uint64_t id, int slice, const Vec3f&) override
    {
        const auto it = st.find(id);
        if (it == st.end()) return std::nullopt;
        it->second.assigned_mask_idx = slice;
        return it->second.node_name;
    }
    std::vector<std::uint64_t> fit_targets() override
    {
        std::vector<std::uint64_t> v;
        for (const auto& [id, s] : st) v.push_back(id);
        return v;
    }
    rc::DoorFitState* begin_fit(std::uint64_t id) override
    { const auto it = st.find(id); return it == st.end() ? nullptr : &it->second; }
    // ── the existence slice's removal: the world's half (live: remember_ghost + teardown, then the graph delete) ──
    void before_remove(std::uint64_t id) override { events.push_back("before " + std::to_string(id)); }
    void remove_door(std::uint64_t id, float) override { events.push_back("remove " + std::to_string(id)); st.erase(id); }
};


constexpr float kDoorX = 1.0f, kWallY = 4.99f;

// One ricoh frame: the robot at `cam`; the door slice (source 1, LiDAR-filled depth) when `door`; a depth-less ricoh slice of
// something else always (so the packet says "the 360 ran"). No ZED mask pixels.
static DoorCycleInput ricoh_cycle(int k, Vec2f cam, bool door, float conf = 0.9f, float dvar = 0.02f)
{
    DoorCycleInput in;
    in.cycle = static_cast<std::uint64_t>(k);
    in.t_steady_ns = static_cast<std::int64_t>(k) * 100'000'000;
    in.t_wall_ms = 1'700'000'000'000 + static_cast<std::int64_t>(k) * 100;
    in.robot.present = true;
    in.robot.room_T_body = Mat4d::Identity();
    in.robot.room_T_body->coeffRef(0, 3) = cam.x(); in.robot.room_T_body->coeffRef(1, 3) = cam.y();
    in.robot.body_T_zed = Mat4d::Identity();
    in.camera.present = true; in.camera.width = 1280; in.camera.height = 720; in.camera.focal_x = 700.f; in.camera.focal_y = 700.f;
    MaskPacket p;
    p.valid = true; p.frame_id = k; p.timestamp_ms = 1000 + static_cast<std::uint64_t>(k);
    if (door)
    {
        for (int i = 0; i <= 20; ++i)
            for (int j = 0; j <= 21; ++j)
                p.support_points.emplace_back(kDoorX - 0.5f + 0.05f * static_cast<float>(i), kWallY, 0.1f * static_cast<float>(j));
        MaskSlice s;
        s.label = "door"; s.class_id = 1000.f; s.confidence = conf;
        s.centroid = {kDoorX, kWallY, 1.05f}; s.bbox_min = {kDoorX - 0.5f, kWallY - 0.01f, 0.0f}; s.bbox_max = {kDoorX + 0.5f, kWallY + 0.01f, 2.1f};
        s.support_begin = 0; s.support_end = p.support_points.size();
        s.range = (Vec2f(kDoorX, kWallY) - cam).norm(); s.source = 1; s.has_depth = true; s.depth_var = dvar;
        p.slices.push_back(s);
    }
    MaskSlice o;
    o.label = "plant"; o.class_id = 5.f; o.confidence = 0.8f; o.source = 1; o.has_depth = false;
    p.slices.push_back(o);
    in.masks = std::move(p);
    return in;
}
static Vec2f viewpoint(int k, float r = 1.5f) { return Vec2f(r * std::cos(0.45f * k), -1.0f + r * std::sin(0.45f * k)); }

struct Tally { int births = 0, removes = 0, first_birth = -1, first_remove = -1; };

int main()
{
    if (std::setlocale(LC_ALL, "es_ES.UTF-8") == nullptr) std::setlocale(LC_ALL, "");
    const auto drive = [](Tally& t, ExistWorld& w, DoorCore& core, int from, int to, bool door, float r)
    {
        for (int k = from; k <= to; ++k)
            for (const auto& d : core.step(ricoh_cycle(k, viewpoint(k, r), door)).decisions)
            {
                if (d.kind == static_cast<std::uint16_t>(K::Birth))  { ++t.births;  if (t.first_birth < 0) t.first_birth = k; }
                if (d.kind == static_cast<std::uint16_t>(K::Remove)) { ++t.removes; if (t.first_remove < 0) t.first_remove = k; }
            }
        (void) w;
    };
    {   // A. born once, survives
        DoorCoreConfig cfg; ExistWorld w; w.cfg = cfg.door; DoorCore core(cfg); w.core = &core; core.connect(&w);
        Tally t;
        drive(t, w, core, 1, 700, true, 1.5f);
        std::print("A: born at {}, births {}, removes {}, instances {}\n", t.first_birth, t.births, t.removes, w.st.size());
        CHECK(t.births == 1 and t.removes == 0 and w.st.size() == 1);
        w.core = nullptr;
    }
    int removal_after[2] = {-1, -1};
    for (int near = 0; near < 2; ++near)
    {   // B/C. born, then the door vanishes (the 360 keeps running and sees nothing there)
        DoorCoreConfig cfg; ExistWorld w; w.cfg = cfg.door; DoorCore core(cfg); w.core = &core; core.connect(&w);
        const float r = near ? 1.0f : 3.0f;   // orbit radius: 3 m reaches y = 2 (door ~3 m away), 1 m stays ~5 m away
        Tally t;
        drive(t, w, core, 1, 300, true, r);
        const int born = t.first_birth;
        Tally t2;
        drive(t2, w, core, 301, 1500, false, r);
        std::print("B{}: born {}, removed {} (after {} frames), births after {} removes {}, instances {}\n", near, born,
                   t2.first_remove, t2.first_remove > 0 ? t2.first_remove - 300 : -1, t2.births, t2.removes, w.st.size());
        CHECK(t.births == 1 and born > 0);
        CHECK(t2.removes == 1 and t2.births == 0 and w.st.empty());   // removed once, not reborn
        removal_after[near] = t2.first_remove;
        w.core = nullptr;
    }
    CHECK(removal_after[0] > 0 and removal_after[1] > 0);
    CHECK(removal_after[0] < removal_after[1]);   // C. the nearer (better-resolving) looks condemn it sooner; the far, weak looks take longer
    {   // D. the hit-rate Beta moves per VIEWPOINT, not per frame: 100 parked frames = 1 trial; K distinct viewpoints = K trials
        const Vec2f door(1.0f, 4.99f);
        RicohHitPool parked;
        float h = 0, t = 0;
        for (int k = 0; k < 100; ++k) { const auto [dh, dt] = parked.add(Vec2f(0.0f, 0.0f), door, 0.6f, k % 5 == 0); h += dh; t += dt; }
        std::print("D: 100 parked frames -> {:.2f} trials, {:.2f} hits (20% detected)\n", t, h);
        CHECK(std::abs(t - 1.0f) < 0.05f);
        CHECK(h > 0.0f and h < 0.5f);                      // the view's detected FRACTION, once
        RicohHitPool moving;
        float t2 = 0;
        const int K = 8;
        for (int v = 0; v < K; ++v)
            for (int k = 0; k < 20; ++k)                   // 20 frames at each of K well-separated viewpoints
                t2 += moving.add(Vec2f(-9.0f + 2.5f * v, 0.0f), door, 0.6f, true).second;
        std::print("D: {} viewpoints x 20 frames -> {:.2f} trials\n", K, t2);
        CHECK(std::abs(t2 - static_cast<float>(K)) < 0.5f);
    }
    if (fails == 0) std::print("test_door_ricoh_lifecycle: ALL PASS\n");
    else            std::print("test_door_ricoh_lifecycle: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
