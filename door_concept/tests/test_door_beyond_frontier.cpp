// test_door_beyond_frontier.cpp — clean51 door_3 (2026-11-11): a doorway whose mask is SEEN THROUGH it.
//
// DOOR_0 is open in room_2's wall y = 0.13 (room_2's frame). Its "door" mask labels the doorway, but the depth under the mask is the
// bathroom's far wall 2.2 m beyond (y ≈ −2.05). The fit reads that correctly — associate_wall seats the door in the wall its
// sightline LEAVES the room by, y ≈ 0.22 (live: (3.44, 0.22)) — but the tracker associated detections by the mask CENTROID,
// still 2.2 m beyond. Live: door_3's birth frame was its only fresh frame; matched_frames stayed 1 < kFitWithoutFreshDataFrames,
// so the fit never ran again (one row in ai2_log_phi.csv in 28 min), phi stayed at its prior, no open / cross was offered.
// Pre-existing (the HEAD build fails this test the same way; clean50 escaped on 5 lucky frames).
// THE FIX (door_core.cpp, sightline_exit): in a promoted room a ZED detection whose centroid lies beyond the room's walls is
// placed where its sightline leaves the room — the same reading of the mask the fit makes.
//   Phase 1, proto (the ring with the through-doorway spur, as at the birth): door_3 is born at the detection and seated on the wall.
//   Phase 2, promoted, the robot at the approach vantage: every later mask must associate (matched_frames grows), and the leaf
//   angle is measured. HEAD: matched_frames stays 1.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_beyond_frontier.cpp src/core/door_core.cpp src/core/door_fit.cpp \
//       src/core/door_existence.cpp src/core/door_visibility.cpp src/core/existence_update.cpp src/core/door_cycle_io.cpp src/door_belief.cpp src/door_model.cpp src/door_frame_evidence.cpp \
//       ../common/contour_edge/contour_edge_check.cpp ../common/contour_edge/contour_edge_project.cpp \
//       ../common/contour_edge/contour_depth_check.cpp -lzstd -lopencv_core -lopencv_imgproc -o /tmp/test_door_beyond_frontier && /tmp/test_door_beyond_frontier
#include <clocale>
#include <cstdlib>
#include <cmath>
#include <format>
#include <numbers>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>
#include "../src/core/door_core.h"

using namespace rc::door::core;

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

// A world with no DSR: the fit states in an unordered_map, as the agent's DoorFitter holds them.
struct FitWorld final : TrackerWorld
{
    DoorCore* core = nullptr;
    rc::DoorConfig cfg;
    std::uint64_t room = 7;
    std::vector<Vec2f> polygon{{-5.f, -5.f}, {5.f, -5.f}, {5.f, 5.f}, {-5.f, 5.f}};
    mutable std::unordered_map<std::uint64_t, rc::DoorFitState> st;
    std::uint64_t next_id = 1000;
    int after_fit_calls = 0;

    std::uint64_t room_id() const override { return room; }
    std::vector<Vec2f> room_polygon() const override { return polygon; }
    std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> fit_states() const override
    {
        std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> v;
        for (auto& [id, s] : st) v.emplace_back(id, &s);
        return v;
    }
    std::optional<IdentityMatch> match_identity(const Vec2f&) const override { return std::nullopt; }
    void retire(std::uint64_t id) override { st.erase(id); }
    void set_assigned_mask(std::uint64_t id, int slice) override
    { if (auto it = st.find(id); it != st.end()) it->second.assigned_mask_idx = slice; }
    void after_association(const TrackerCycleInfo&) override {}
    // ★As live: the node insert fires modify_node_slot SYNCHRONOUSLY (same thread), so ensure_instance builds the
    // instance inside create_door — before the RT edge exists and before the core notes the birth seed: the newborn's
    // pre-belief box is at (0, 0), yaw 0, the prior size (task-4b-report.md; a pre-existing defect kept as is).
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
    std::optional<std::string> adopt_birth(std::uint64_t id, int slice, const Vec3f&) override
    {
        const auto it = st.find(id);   // ensure_instance: it exists already (create_door) — it takes its birth slice
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
    void after_fit(std::uint64_t, const DoorObservation&, float) override { ++after_fit_calls; }
    void remove_door(std::uint64_t id, float) override { st.erase(id); }   // (the existence slice's removal)
};


// room_2 in its own frame (clean51 map_layout.csv). While PROVISIONAL (t 574042) its free-space ring had a spur down to y = −2.3
// for x in [2.21, 3.51] (the view through the open doorway); once PROMOTED it is the 6.2 x 3.4 m rectangle y in [0.13, 3.54].
static const std::vector<Vec2f> kProtoRing{{3.51f, -2.29f}, {3.44f, 3.53f}, {-2.75f, 3.45f}, {-2.71f, 0.148f}, {2.21f, 0.207f}, {2.24f, -2.31f}};
static const std::vector<Vec2f> kRoom{{3.48f, 0.13f}, {3.44f, 3.53f}, {-2.75f, 3.45f}, {-2.71f, 0.13f}};
constexpr float kWallY = 0.13f, kFarY = -2.05f;   // the wall that holds DOOR_0, and the far wall seen through it
constexpr float kApX = 2.93f;                      // the aperture centre: 0.55 m from the corner (DOOR_0, 5 cm of wall beside its post)

static Mat4d pose(double x, double y, double z, double yaw)
{
    Mat4d T = Mat4d::Identity();
    T(0, 0) = std::cos(yaw); T(0, 1) = -std::sin(yaw); T(1, 0) = std::sin(yaw); T(1, 1) = std::cos(yaw);
    T(0, 3) = x; T(1, 3) = y; T(2, 3) = z;
    return T;
}

// Phase 1 (proto, the robot at (−0.10, 2.79) as at the birth): the mask's depth lands on the far wall at x in [kApX ± 0.5].
// Phase 2 (promoted, the robot at the approach vantage (2.83, 1.84)): the mask labels the open doorway [kApX ± 0.5] on the
// wall y = 0.13; every depth point is the far wall SEEN THROUGH it (the ray from the camera through the doorway, on to y = −2.05).
static DoorCycleInput cycle_input(std::uint64_t cycle, bool proto)
{
    const Vec2f cam = proto ? Vec2f(-0.10f, 2.79f) : Vec2f(2.83f, 1.84f);
    const Vec2f look = proto ? Vec2f(kApX, kFarY) : Vec2f(kApX, kWallY);
    const double yaw = std::atan2(look.y() - cam.y(), look.x() - cam.x()) - 0.5 * std::numbers::pi;   // zed +y = forward
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = static_cast<std::int64_t>(cycle) * 100'000'000;
    in.t_wall_ms = 1'700'000'000'000 + static_cast<std::int64_t>(cycle) * 100;
    in.robot.present = true;
    in.robot.room_T_body = pose(cam.x(), cam.y(), 0.0, yaw);
    in.robot.body_T_zed = pose(0.0, 0.0, 1.2, 0.0);
    in.camera.present = true;
    in.camera.width = 1280; in.camera.height = 720; in.camera.focal_x = 700.f; in.camera.focal_y = 700.f;
    RoomInfo r;
    r.id = 7; r.name = "room_2"; r.is_proto = proto; r.polygon = proto ? kProtoRing : kRoom;
    in.rooms.rooms.push_back(r);
    MaskPacket p;
    p.valid = true;
    p.frame_id = static_cast<int>(cycle);
    p.timestamp_ms = 1000 + cycle;
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    for (int i = 0; i <= 20; ++i)
        for (int j = 0; j <= 21; ++j)
        {
            const float z = 0.1f * static_cast<float>(j);
            Eigen::Vector3f q;
            if (proto)
                q = {kApX - 0.5f + 0.05f * static_cast<float>(i), kFarY, z};
            else
            {   // through the doorway point (xa, kWallY, z) from the camera at height 1.2, on to the far wall
                const Eigen::Vector3f c(cam.x(), cam.y(), 1.2f), a(kApX - 0.5f + 0.05f * static_cast<float>(i), kWallY, z);
                const float t = (kFarY - c.y()) / (a.y() - c.y());
                q = c + t * (a - c);
            }
            p.support_points.push_back(q);
            sum += q;
        }
    const Eigen::Vector3f cen = sum / static_cast<float>(p.support_points.size());
    MaskSlice s;
    s.label = "door"; s.class_id = 1000.f; s.confidence = 0.9f;
    s.centroid = cen; s.bbox_min = cen - Eigen::Vector3f(0.5f, 0.01f, 1.05f); s.bbox_max = cen + Eigen::Vector3f(0.5f, 0.01f, 1.05f);
    s.support_begin = 0; s.support_end = p.support_points.size();
    s.trunc_frac = 0.0f; s.range = (cen.head<2>() - cam).norm(); s.centroid_radius = 0.1f; s.source = 0; s.has_depth = true;
    p.slices.push_back(s);
    in.masks = std::move(p);
    BpearlSweep b;   // the low LiDAR: through the open doorway, on the wall either side of it
    b.room_fresh = true;
    b.origin_room = {cam.x(), cam.y(), 0.35f};
    for (int k = 0; k <= 80; ++k)
    {
        const float x = kApX - 2.0f + 0.05f * static_cast<float>(k);
        const bool through = std::abs(x - kApX) < 0.45f;
        for (const float z : {0.4f, 0.8f})
            if (through) { const float t = (kFarY - cam.y()) / (kWallY - cam.y()); b.pts_room.emplace_back(cam.x() + t * (x - cam.x()), kFarY, z); }
            else b.pts_room.emplace_back(x, kWallY - 0.02f, z);
    }
    in.bpearl = std::move(b);
    return in;
}

int main()
{
    if (std::setlocale(LC_ALL, "es_ES.UTF-8") == nullptr) std::setlocale(LC_ALL, "");
    DoorCoreConfig cfg;
    cfg.door.exist_enabled = false;   // this is about association and the fit; live, door_3's existence stood (L up to +5.3)
    FitWorld w;
    w.cfg = cfg.door;
    DoorCore core(cfg);
    w.core = &core;
    core.connect(&w, {.decision = {}, .diag = [](const DiagRow& d) { if (d.stream == "stdout" and std::getenv("BF_TRACE")) std::print("{}", d.line); }});
    // door_3 as the tracker BORN it (clean51: "BIRTH 'door_3' at room (3.65,-2.05) parent=(room)"): a node at the detection, the
    // prior size, no belief yet. (A synthetic single view never accrues the existence evidence a birth needs, so the birth is
    // stated, not replayed; everything after it — association, the belief's first fit, every later cycle — is the core's.)
    {
        rc::DoorState s0;
        s0.cx = kApX; s0.cy = kFarY; s0.yaw = 0.0f;
        s0.w = cfg.door.door_prior_w_m; s0.h = cfg.door.door_prior_h_m; s0.thickness = cfg.door.door_thickness_m;
        rc::DoorFitState f;
        core.fit().init_instance(f, 3003, "door_3", s0);
        w.st.emplace(3003, std::move(f));
    }
    constexpr int kProtoCycles = 10, kCycles = 60;
    int frames_at_promotion = -1;
    for (int c = 1; c <= kCycles; ++c)
    {
        const bool proto = c <= kProtoCycles;
        w.polygon = proto ? kProtoRing : kRoom;
        (void) core.step(cycle_input(static_cast<std::uint64_t>(c), proto));
        if (c == kProtoCycles and w.st.size() == 1) frames_at_promotion = w.st.begin()->second.matched_frames;
    }
    std::print("instances {}, matched frames when promoted {}\n", w.st.size(), frames_at_promotion);
    CHECK(w.st.size() == 1);
    for (const auto& [id, f] : w.st)
    {
        const auto c = f.ai2_initialized ? f.ai2_belief.center_xy() : Eigen::Vector2f(NAN, NAN);
        std::print("{}: aperture ({:.2f}, {:.2f}) matched_frames {} processed {} phi {:.2f} measured {}\n",
                   f.node_name, c.x(), c.y(), f.matched_frames, f.processed_cycles, f.phi_est, f.phi_measured);
        CHECK(f.ai2_initialized);
        CHECK(std::abs(c.y() - kWallY) < 0.25f);                          // seated in the wall the doorway is cut in
        CHECK(f.matched_frames >= frames_at_promotion + 40);              // its masks, seen THROUGH it, keep associating
        CHECK(f.phi_measured);
    }
    w.core = nullptr;
    std::print("{}\n", fails == 0 ? "test_door_beyond_frontier: ALL PASS" : std::format("test_door_beyond_frontier: {} FAILURE(S)", fails));
    return fails == 0 ? 0 : 1;
}
