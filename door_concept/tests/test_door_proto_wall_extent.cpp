// test_door_proto_wall_extent.cpp — the entry mirror of a freshly entered PROVISIONAL room stays on the hole it was crossed
// through (pingpong1, 2026-10-30).
//
// THE DEFECT. The along-wall channel's own-wall term (Task D-C round 2, rc::door::along_wall_update's Φ factors) asks
// the room polygon for "the extent of the wall this doorway is cut in" (rc::door::door_wall_extent). In a provisional room
// (rc::room::is_proto) that polygon is room_concept's free-space ring, and at the doorway the robot came through the ring
// runs ACROSS the hole, as wide as the free space seen in it: pingpong1's room_2 had an edge (−0.33, 0.12)→(0.27, 0.12),
// 0.60 m, for a 1.00 m door. "The aperture lies in a 0.60 m wall" cannot hold, and the Φ factors pick whichever span sits
// furthest along that edge — a one-bin through-sliver at x ≈ 0.30 over the true 0.53 m span. The carried mirror (σ_s 0.5,
// the birth seed's) slid +0.28 m in ~10 cycles; door_concept's own replay of the tape reproduces it to the cm.
// THE FIX (door_fit.cpp, estimate_phi): a frontier is no evidence of where a wall ends, so in a proto room the term is
// absent, as with no polygon. DoorCore::step reads is_proto from the cycle's RoomInfo.
//
// This drives the CORE (DoorCore::step → fit_doors → run_inference → estimate_phi → along_wall_update) with that geometry:
// a carried door (no camera masks, matched_frames = kFitWithoutFreshDataFrames, its leaf open 90° out of the room) at the
// hole (0, 0) in a polygon whose mouth edge is (−0.33, 0)→(0.27, 0), and a low-LiDAR sweep from (0, 0.4) — inside the
// room, in the doorway as the robot was — whose rays go THROUGH the hole for x ∈ [−0.27, 0.25] and in two 2 cm slivers
// either side of it (pingpong1's spans at the mirror's first cycle), and stop on the wall line everywhere else.
//   (a) RED reproduction: the same polygon read as WALLS (is_proto = false) slides the aperture > 0.15 m toward +x
//       (+0.25 m here; +0.28 m live).
//   (b) the fix: as a PROTO room the aperture stays within 5 cm of the hole.
//   (c) a room whose wall runs through the doorway (the room_1 case): the flag moves nothing (< 1 cm) — the term only ever
//       mattered where the polygon's edge at the door is a frontier.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_proto_wall_extent.cpp \
//       src/core/door_core.cpp src/core/door_fit.cpp src/core/door_existence.cpp src/core/door_visibility.cpp src/core/door_cycle_io.cpp \
//       src/door_belief.cpp src/door_model.cpp src/door_frame_evidence.cpp ../common/contour_edge/contour_edge_check.cpp \
//       ../common/contour_edge/contour_edge_project.cpp ../common/contour_edge/contour_depth_check.cpp \
//       -lzstd -lopencv_core -lopencv_imgproc -o /tmp/test_door_proto_wall_extent && /tmp/test_door_proto_wall_extent
#include <clocale>
#include <cmath>
#include <numbers>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/core/door_core.h"
#include "../src/door_identity_table.h"   // kFitWithoutFreshDataFrames

using namespace rc::door::core;

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

namespace
{
constexpr std::uint64_t kRoom = 7, kDoor = 500;

// room_2 as room_concept's free-space ring had it a second after the crossing (pingpong1, cycle 645), the wall line moved
// to y = 0: the mouth edge across the hole, then the ring round the room (+y is into the room).
const std::vector<Vec2f> kProtoRing{{-0.33f, 0.0f}, {0.27f, 0.0f}, {0.37f, 0.15f}, {2.82f, 1.35f}, {2.77f, 4.05f},
                                    {-3.58f, 3.9f}, {-3.53f, 1.9f}, {-0.48f, 0.3f}};
// A room whose wall runs THROUGH the doorway (room_1's polygon closes on the wall line across the hole).
const std::vector<Vec2f> kWalledRoom{{-3.0f, 0.0f}, {3.0f, 0.0f}, {3.0f, 4.0f}, {-3.0f, 4.0f}};

struct World final : TrackerWorld
{
    DoorCore* core = nullptr;
    std::vector<Vec2f> polygon;
    mutable std::unordered_map<std::uint64_t, rc::DoorFitState> st;

    std::uint64_t room_id() const override { return kRoom; }
    std::vector<Vec2f> room_polygon() const override { return polygon; }
    std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> fit_states() const override
    {
        std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> v;
        for (auto& [id, s] : st) v.emplace_back(id, &s);
        return v;
    }
    std::optional<GhostMatch> match_ghost(const Vec2f&) const override { return std::nullopt; }
    std::optional<IdentityMatch> match_identity(const Vec2f&) const override { return std::nullopt; }
    void retire(std::uint64_t id) override { st.erase(id); }
    void set_assigned_mask(std::uint64_t id, int slice) override
    { if (auto it = st.find(id); it != st.end()) it->second.assigned_mask_idx = slice; }
    void after_association(const TrackerCycleInfo&) override {}
    std::uint64_t create_door(const BirthRequest&) override { return 0; }   // no camera ⇒ no births here
    void reacquire(std::uint64_t, const GhostMatch&) override {}
    void forget_ghost(const Vec2f&) override {}
    std::optional<std::string> adopt_birth(std::uint64_t, int, const Vec3f&) override { return std::nullopt; }
    std::vector<std::uint64_t> fit_targets() override
    {
        std::vector<std::uint64_t> v;
        for (const auto& [id, s] : st) v.push_back(id);
        return v;
    }
    rc::DoorFitState* begin_fit(std::uint64_t id) override
    { const auto it = st.find(id); return it == st.end() ? nullptr : &it->second; }
    void remove_door(std::uint64_t id, float) override { st.erase(id); }
};

Mat4d translation(double x, double y, double z)
{
    Mat4d T = Mat4d::Identity();
    T(0, 3) = x; T(1, 3) = y; T(2, 3) = z;
    return T;
}

// One cycle: the robot parked in the doorway at (0, 0.4); the low LiDAR's sweep (see the header); no camera masks.
DoorCycleInput cycle_input(std::uint64_t cycle, bool is_proto, const std::vector<Vec2f>& polygon)
{
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = static_cast<std::int64_t>(cycle) * 100'000'000;   // 10 Hz
    in.t_wall_ms = 1'700'000'000'000 + static_cast<std::int64_t>(cycle) * 100;
    in.robot.present = true;
    in.robot.room_T_body = translation(0.0, 0.4, 0.0);
    in.robot.body_T_zed = translation(0.0, 0.0, 1.2);
    RoomInfo r;
    r.id = kRoom; r.name = "room_2"; r.is_proto = is_proto; r.polygon = polygon;
    in.rooms.rooms.push_back(r);

    BpearlSweep b;
    b.room_fresh = true;
    b.origin_room = {0.0f, 0.4f, 0.35f};
    // pingpong1's spans at the mirror's first cycle (room_2 x): two 2 cm slivers either side of the 0.52 m main span.
    const auto through = [](float x)
    { return (x >= -0.31f and x <= -0.29f) or (x >= -0.27f and x <= 0.25f) or (x >= 0.27f and x <= 0.29f); };
    for (int k = 0; k <= 1500; ++k)
    {
        const float x = -1.5f + 0.002f * static_cast<float>(k);   // where the ray meets the wall line y = 0
        for (const float z : {0.4f, 0.8f})
        {
            if (through(x))
                b.pts_room.emplace_back(6.0f * x, -2.0f, z);   // the ray from (0, 0.4) through (x, 0) on to y = −2
            else
                b.pts_room.emplace_back(x, 0.0f, z);           // stopped on the wall / a jamb / the open leaf's edge
        }
    }
    in.bpearl = std::move(b);
    return in;
}

struct Result { float ap_x = NAN, ap_y = NAN, s_sd = NAN; };

// The carried mirror, as install_carried_door leaves it: the node's RT (0, 0) yaw π and the source's size; its leaf memory
// (open 90°, hinged at the −x jamb, opening OUT of this room); fitted although unseen.
Result run(bool is_proto, const std::vector<Vec2f>& polygon, int cycles)
{
    DoorCoreConfig cfg;
    World w;
    w.polygon = polygon;
    DoorCore core(cfg);
    w.core = &core;
    core.connect(&w, {});
    rc::DoorState s;
    s.cx = 0.0f; s.cy = 0.0f; s.yaw = std::numbers::pi_v<float>;
    s.w = 1.0f; s.h = 2.1f; s.thickness = cfg.door.door_thickness_m;
    rc::DoorFitState f;
    core.fit().init_instance(f, kDoor, "door_2", s);
    f.leaf_side_hint = std::make_pair(Eigen::Vector2f(-0.5f, 0.0f), Eigen::Vector2f(0.0f, -1.0f));
    f.phi_est = 1.5708f; f.phi_sigma = 0.01f;
    f.matched_frames = rc::door::kFitWithoutFreshDataFrames;
    w.st.emplace(kDoor, std::move(f));
    for (int c = 1; c <= cycles; ++c)
        (void)core.step(cycle_input(static_cast<std::uint64_t>(c), is_proto, polygon));
    Result r;
    if (const auto it = w.st.find(kDoor); it != w.st.end())
    {
        const auto& m = it->second.model.state();
        r.ap_x = m.ap_cx; r.ap_y = m.ap_cy;
        r.s_sd = std::sqrt(std::max(0.0f, it->second.ai2_belief.covariance()(0, 0)));
    }
    w.core = nullptr;
    return r;
}
}  // namespace

int main()
{
    std::setlocale(LC_ALL, "");   // as the agent runs (Qt): the core must not parse anything through the C locale
    constexpr int N = 40;

    // (a) the proto ring read as WALLS — the defect
    const Result walls = run(false, kProtoRing, N);
    std::print("(a) ring read as walls : aperture ({:+.3f}, {:+.3f}) σ_s {:.3f}\n", walls.ap_x, walls.ap_y, walls.s_sd);
    CHECK(std::isfinite(walls.ap_x) and walls.ap_x > 0.15f);

    // (b) the same ring as what it is: a provisional room's free-space frontier
    const Result proto = run(true, kProtoRing, N);
    std::print("(b) proto ring          : aperture ({:+.3f}, {:+.3f}) σ_s {:.3f}\n", proto.ap_x, proto.ap_y, proto.s_sd);
    CHECK(std::isfinite(proto.ap_x) and std::abs(proto.ap_x) < 0.05f);

    // (c) a wall through the doorway: the flag changes nothing that the extent did not already allow
    const Result room_walled = run(false, kWalledRoom, N);
    const Result room_proto  = run(true, kWalledRoom, N);
    std::print("(c) walled room         : aperture x {:+.4f} (walls) vs {:+.4f} (proto)\n", room_walled.ap_x, room_proto.ap_x);
    CHECK(std::isfinite(room_walled.ap_x) and std::abs(room_walled.ap_x) < 0.05f);
    CHECK(std::abs(room_walled.ap_x - room_proto.ap_x) < 0.01f);

    if (fails == 0) std::print("test_door_proto_wall_extent: ALL PASS\n");
    else            std::print("test_door_proto_wall_extent: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
