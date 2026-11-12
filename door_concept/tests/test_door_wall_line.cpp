// test_door_wall_line.cpp — a door in a promoted room lies IN its wall: its plane on the wall's line, its yaw parallel to it
// (clean58 door_3, 2026-11-12; owner: "door_3 is misplaced again, it is out of the wall" / "doors have to be aligned with walls").
//
// clean58: door_3 (the bathroom's DOOR_0) was born while the bathroom was provisional. Its wall frame was authored on the proto
// outline's FRONTIER edge across the mouth of the unseen west alcove — perpendicular to the south wall the doorway is cut in — and
// nothing re-authored it after promotion: the published door stood ACROSS the wall (centre 4 cm from the line, yaw 89° off it).
// THE FIX (door_fit.cpp refresh_geometry, DoorBelief::reframe_onto_line, door_phi_posterior.h door_carrying_run): in a promoted room
// the carrying wall's line is the aperture's plane, recomputed every cycle from that cycle's outline.
//
// Drives the CORE (DoorCore::step → fit → refresh_geometry) with a carried door (no camera masks, fitted although unseen):
//   (a) PROMOTED, then room_concept moves the wall: the door was authored on an outline whose wall ran 0.15 m off and 10° skewed
//       from where the next outline puts it (a 6 m wall: its ends are 0.52 m off the door's plane, so the old in-plane test could not
//       see it). The door must lie on the new line within σ_e and turn parallel within σ_ψ — and follow it back when it moves again.
//   (b) PROTO-BORN, the door_3 geometry: frame on a frontier edge perpendicular to the wall that holds it. While proto: no term, the
//       frame is untouched. First cycle after promotion: the door is on the south wall's line and parallel to it.
//   (d) BORN IN A PROVISIONAL ROOM next to a frontier edge (clean58 door_3's birth): the detection 0.13 m beyond the south wall,
//       2.33 m from the frontier across the unseen alcove. The frame and the centre come from the detection, not from the frontier
//       (before: associate_wall framed it on an outline edge and projected the centre onto it). After promotion it is on the south
//       wall at the detection's along-wall place.
//   (e) frame_from_evidence alone: a slanted surface's orientation from its depth gradient; one point ⇒ facing the camera with a
//       uniform half-turn yaw σ; noisy depth ⇒ a wider σ.
//   (c) NO WALL CARRIES IT: the door's centre 0.45 m from every wall (outside every wall's own-wall band): no term, frame untouched.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_wall_line.cpp src/core/door_core.cpp src/core/door_fit.cpp \
//       src/core/door_existence.cpp src/core/door_visibility.cpp src/core/existence_update.cpp src/core/door_cycle_io.cpp src/door_belief.cpp src/door_model.cpp src/door_frame_evidence.cpp \
//       ../common/contour_edge/contour_edge_check.cpp ../common/contour_edge/contour_edge_project.cpp \
//       ../common/contour_edge/contour_depth_check.cpp -lzstd -lopencv_core -lopencv_imgproc -o /tmp/test_door_wall_line && /tmp/test_door_wall_line
#include <clocale>
#include <cmath>
#include <limits>
#include <numbers>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/core/door_core.h"
#include "../src/door_identity_table.h"   // kFitWithoutFreshDataFrames
#include "../src/door_standpoint.h"       // frame_from_evidence

using namespace rc::door::core;

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

namespace
{
constexpr std::uint64_t kRoom = 7, kDoor = 500;
constexpr float kPi = std::numbers::pi_v<float>;

struct World final : TrackerWorld
{
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
    std::optional<IdentityMatch> match_identity(const Vec2f&) const override { return std::nullopt; }
    void retire(std::uint64_t id) override { st.erase(id); }
    void set_assigned_mask(std::uint64_t id, int slice) override
    { if (auto it = st.find(id); it != st.end()) it->second.assigned_mask_idx = slice; }
    void after_association(const TrackerCycleInfo&) override {}
    std::uint64_t create_door(const BirthRequest&) override { return 0; }
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

Mat4d pose(double x, double y, double z)
{
    Mat4d T = Mat4d::Identity();
    T(0, 3) = x; T(1, 3) = y; T(2, 3) = z;
    return T;
}

DoorCycleInput cycle_input(std::uint64_t cycle, bool is_proto, const std::vector<Vec2f>& polygon, const Vec2f& robot)
{
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = static_cast<std::int64_t>(cycle) * 100'000'000;   // 10 Hz
    in.t_wall_ms = 1'700'000'000'000 + static_cast<std::int64_t>(cycle) * 100;
    in.robot.present = true;
    in.robot.room_T_body = pose(robot.x(), robot.y(), 0.0);
    in.robot.body_T_zed = pose(0.0, 0.0, 1.2);
    RoomInfo r;
    r.id = kRoom; r.name = "room_2"; r.is_proto = is_proto; r.polygon = polygon;
    in.rooms.rooms.push_back(r);
    return in;
}

// A line through p along direction angle th.
float dist_to_line(const Vec2f& c, const Vec2f& p, float th) { return std::abs(-std::sin(th) * (c.x() - p.x()) + std::cos(th) * (c.y() - p.y())); }
// |a − b| mod π, in [0, π/2]
float yaw_err(float a, float b) { const float d = std::remainder(a - b, kPi); return std::abs(d); }

struct Harness
{
    DoorCoreConfig cfg;
    World w;
    DoorCore core{cfg};
    std::uint64_t cyc = 0;
    Vec2f robot{0.0f, 1.5f};

    Harness() { core.connect(&w, {}); }
    void add_door(const Vec2f& c, float yaw, bool yaw_known = true)
    {
        rc::DoorState s;
        s.cx = c.x(); s.cy = c.y(); s.yaw = yaw;
        s.ap_cx = c.x(); s.ap_cy = c.y(); s.ap_yaw = yaw_known ? yaw : std::numeric_limits<float>::quiet_NaN();
        s.w = 1.0f; s.h = 2.1f; s.thickness = cfg.door.door_thickness_m;
        rc::DoorFitState f;
        core.fit().init_instance(f, kDoor, "door_3", s);
        f.matched_frames = rc::door::kFitWithoutFreshDataFrames;
        w.st.emplace(kDoor, std::move(f));
    }
    void run(bool proto, const std::vector<Vec2f>& poly, int n)
    {
        w.polygon = poly;
        for (int k = 0; k < n; ++k) (void)core.step(cycle_input(++cyc, proto, poly, robot));
    }
    const rc::DoorFitState* door() const { const auto it = w.st.find(kDoor); return it == w.st.end() ? nullptr : &it->second; }
    Vec2f centre() const { return door()->ai2_belief.center_xy(); }
    float yaw() const { return door()->ai2_belief.yaw(); }
};
}  // namespace

int main()
{
    std::setlocale(LC_ALL, "");   // as the agent runs (Qt)
    const float sb = DoorCoreConfig{}.door.exist_boundary_sigma_m;

    // ── (a) promoted; room_concept moves the wall 0.15 m and turns it 10° ─────────────────────────────────────────────────────
    {
        Harness h;
        const float th = 10.0f * kPi / 180.0f;
        // outline A: a 6 x 4 room whose bottom wall is the line through (0, 0.15) at −10° (the door was authored on it)
        const Vec2f p0(0.0f, 0.15f), dir(std::cos(-th), std::sin(-th)), nrm(-dir.y(), dir.x());
        const std::vector<Vec2f> A{p0 - 3.0f * dir, p0 + 3.0f * dir, p0 + 3.0f * dir + 4.0f * nrm, p0 - 3.0f * dir + 4.0f * nrm};
        // outline B: the same room with the bottom wall where it really is, y = 0, 6 m long
        const std::vector<Vec2f> B{{-3.0f, 0.0f}, {3.0f, 0.0f}, {3.0f, 4.0f}, {-3.0f, 4.0f}};
        h.add_door(p0, -th);
        h.run(false, A, 5);
        CHECK(h.door() != nullptr);
        if (h.door())
        {
            const Vec2f c0 = h.centre();
            std::print("(a) on outline A : centre ({:+.3f}, {:+.3f}) yaw {:+.2f} deg — {:.3f} m from y=0, {:.1f} deg off it\n",
                       c0.x(), c0.y(), h.yaw() * 180 / kPi, dist_to_line(c0, {0, 0}, 0), yaw_err(h.yaw(), 0) * 180 / kPi);
            CHECK(dist_to_line(c0, p0, -th) < 0.02f and yaw_err(h.yaw(), -th) < 0.01f);   // A's frame: on A's wall
            h.run(false, B, 3);
            const Vec2f c1 = h.centre();
            const float sd_yaw = std::numbers::sqrt2_v<float> * sb / 6.0f;
            std::print("(a) on outline B : centre ({:+.3f}, {:+.3f}) yaw {:+.3f} deg — {:.4f} m from y=0 (σ_e {:.3f}), {:.3f} deg off "
                       "(σ_ψ {:.2f} deg); carried {}\n", c1.x(), c1.y(), h.yaw() * 180 / kPi, dist_to_line(c1, {0, 0}, 0), sb,
                       yaw_err(h.yaw(), 0) * 180 / kPi, sd_yaw * 180 / kPi, h.door()->wall_line_carried);
            CHECK(h.door()->wall_line_carried == 1);
            CHECK(dist_to_line(c1, {0, 0}, 0) < sb);
            CHECK(yaw_err(h.yaw(), 0) < sd_yaw);
            CHECK(std::abs(c1.x() - c0.x()) < 0.10f);   // it moved ACROSS the wall (and turned), not along it
            // the outline moves again (back to A): the door follows — the factor is replaced, not accumulated
            h.run(false, A, 2);
            const Vec2f c2 = h.centre();
            std::print("(a) back on A    : centre ({:+.3f}, {:+.3f}) — {:.4f} m from A's line, {:.3f} deg off it\n", c2.x(), c2.y(),
                       dist_to_line(c2, p0, -th), yaw_err(h.yaw(), -th) * 180 / kPi);
            CHECK(dist_to_line(c2, p0, -th) < sb and yaw_err(h.yaw(), -th) < sd_yaw);
        }
    }

    // ── (b) proto-born on a frontier perpendicular to its wall (clean58 door_3), then the room is promoted ────────────────────────
    {
        Harness h;
        h.robot = {2.0f, 2.0f};
        // proto ring: the robot has seen x ∈ [0, 3.3] only; the alcove x ∈ [−2.3, 0], y ∈ [0, 1.9] is unseen, so the ring closes on the
        // frontier x = 0 from y = 0 up to y = 6.2. The door (DOOR_0 seen through) got its frame on that frontier: centre (0, 0.15),
        // yaw +90° — across the south wall y = 0.
        const std::vector<Vec2f> proto{{0.0f, 0.0f}, {3.3f, 0.0f}, {3.3f, 6.2f}, {0.0f, 6.2f}};
        // promoted outline: the bathroom's L (GT clean58 room_2, south wall y = 0 from x = −2.4 to 3.3)
        const std::vector<Vec2f> promoted{{-2.4f, 0.0f}, {3.3f, 0.0f}, {3.3f, 6.2f}, {0.0f, 6.2f}, {0.0f, 1.9f}, {-2.4f, 1.9f}};
        h.add_door({0.0f, 0.15f}, 0.5f * kPi);
        h.run(true, proto, 5);
        CHECK(h.door() != nullptr);
        if (h.door())
        {
            const Vec2f c0 = h.centre();
            const float y0 = h.yaw();
            std::print("(b) proto        : centre ({:+.3f}, {:+.3f}) yaw {:+.1f} deg; carried {} (expect 0: a frontier is no wall)\n",
                       c0.x(), c0.y(), y0 * 180 / kPi, h.door()->wall_line_carried);
            CHECK(h.door()->wall_line_carried == 0);
            CHECK(yaw_err(y0, 0.5f * kPi) < 0.01f);   // untouched while proto
            h.run(false, promoted, 1);   // the FIRST cycle after promotion
            const Vec2f c1 = h.centre();
            const float sd_yaw = std::numbers::sqrt2_v<float> * sb / 5.7f;
            std::print("(b) promoted (1st cycle): centre ({:+.3f}, {:+.3f}) yaw {:+.3f} deg — {:.4f} m from y=0, {:.3f} deg off "
                       "(σ_ψ {:.2f} deg); carried {}\n", c1.x(), c1.y(), h.yaw() * 180 / kPi, std::abs(c1.y()),
                       yaw_err(h.yaw(), 0) * 180 / kPi, sd_yaw * 180 / kPi, h.door()->wall_line_carried);
            CHECK(h.door()->wall_line_carried == 1);
            CHECK(std::abs(c1.y()) < sb);
            CHECK(yaw_err(h.yaw(), 0) < sd_yaw);
            // the published read-back (what write_rt_pose publishes) is the wall-aligned aperture
            const auto& m = h.door()->model.state();
            CHECK(std::abs(m.ap_cy) < sb and yaw_err(m.ap_yaw, 0) < sd_yaw);
        }
    }

    // ── (c) no wall carries it ────────────────────────────────────────────────────────────────────────────────────────────────
    {
        Harness h;
        const std::vector<Vec2f> A{{-3.0f, 0.45f}, {3.0f, 0.45f}, {3.0f, 4.0f}, {-3.0f, 4.0f}};   // authored on a wall at y = 0.45
        const std::vector<Vec2f> B{{-3.0f, 0.0f}, {3.0f, 0.0f}, {3.0f, 4.0f}, {-3.0f, 4.0f}};      // the next outline: 0.45 m away
        h.add_door({0.0f, 0.45f}, 0.0f);
        h.run(false, A, 3);
        CHECK(h.door() != nullptr);
        if (h.door())
        {
            const Vec2f c0 = h.centre();
            h.run(false, B, 3);
            const Vec2f c1 = h.centre();
            std::print("(c) 0.45 m off   : centre ({:+.3f}, {:+.3f}) -> ({:+.3f}, {:+.3f}); carried {} (expect 0)\n", c0.x(), c0.y(),
                       c1.x(), c1.y(), h.door()->wall_line_carried);
            CHECK(h.door()->wall_line_carried == 0);
            CHECK((c1 - c0).norm() < 1e-3f);
        }
    }

    // ── (d) born in a provisional room next to a frontier edge: keeps the detection's position ───────────────────────────────────
    {
        Harness h;
        h.robot = {2.0f, 2.0f};
        const std::vector<Vec2f> proto{{0.0f, 0.0f}, {3.3f, 0.0f}, {3.3f, 6.2f}, {0.0f, 6.2f}};
        const std::vector<Vec2f> promoted{{-2.4f, 0.0f}, {3.3f, 0.0f}, {3.3f, 6.2f}, {0.0f, 6.2f}, {0.0f, 1.9f}, {-2.4f, 1.9f}};
        const Vec2f det(-2.33f, -0.13f);   // the detection (DOOR_0 seen from the east part), no orientation in it
        h.add_door(det, 0.0f, false);
        h.run(true, proto, 5);
        CHECK(h.door() != nullptr);
        if (h.door())
        {
            const Vec2f c0 = h.centre();
            std::print("(d) proto birth  : centre ({:+.3f}, {:+.3f}) (detection ({:+.2f}, {:+.2f})), birth yaw σ {:.1f} deg\n", c0.x(), c0.y(),
                       det.x(), det.y(), h.door()->dbg_birth_yaw_sd * 180 / kPi);
            CHECK((c0 - det).norm() < 0.01f);   // the detection's place, not projected onto the frontier x = 0 (2.33 m away)
            h.run(false, promoted, 1);
            const Vec2f c1 = h.centre();
            std::print("(d) promoted     : centre ({:+.3f}, {:+.3f}) yaw {:+.2f} deg; along-wall error {:.3f} m; carried {}\n", c1.x(), c1.y(),
                       h.yaw() * 180 / kPi, std::abs(c1.x() - det.x()), h.door()->wall_line_carried);
            CHECK(h.door()->wall_line_carried == 1);
            CHECK(std::abs(c1.y()) < sb and yaw_err(h.yaw(), 0) < 0.01f);
            // along the wall: the detection's place (−2.33), its west jamb then pulled inside the wall's end at −2.4 by the along-wall
            // support (door_wall_support.h) — DOOR_0's GT centre in this frame is −1.81. Before: x = 0, on the frontier.
            CHECK(std::abs(c1.x() - det.x()) < 0.6f and c1.x() - 0.5f > -2.4f - sb);
            CHECK(std::abs(c1.x() - (-1.81f)) < 0.2f);
        }
    }

    // ── (e) frame_from_evidence: the orientation a detection carries, and the width of what it does not ─────────────────────────
    {
        const Vec2f cam(0.0f, 0.0f), centre(0.0f, 3.0f);
        // a door turned 30° from facing the camera: points along (cos 30°, sin 30°) through the centre
        std::vector<Vec2f> pts;
        const float a30 = 30.0f * kPi / 180.0f;
        for (int i = -10; i <= 10; ++i) pts.push_back(centre + 0.05f * static_cast<float>(i) * Vec2f(std::cos(a30), std::sin(a30)));
        const auto f1 = rc::door::frame_from_evidence(pts, centre, cam, cam, std::nullopt);
        std::print("(e) slanted points: yaw {:+.2f} deg σ {:.2f} deg (expect ±30 mod 180, σ ≈ 0)\n",
                   std::atan2(f1.u.y(), f1.u.x()) * 180 / kPi, f1.yaw_sd * 180 / kPi);
        CHECK(yaw_err(std::atan2(f1.u.y(), f1.u.x()), a30) < 0.01f and f1.yaw_sd < 0.01f);
        CHECK((-f1.u.y()) * (cam.x() - centre.x()) + f1.u.x() * (cam.y() - centre.y()) > 0.0f);   // across_u points at the camera
        // one point: no orientation in it — faces the camera, uniform yaw over a half-turn
        const std::vector<Vec2f> one{centre};
        const auto f2 = rc::door::frame_from_evidence(one, centre, cam, cam, std::nullopt);
        std::print("(e) one point    : yaw {:+.2f} deg σ {:.1f} deg (expect 0 / 180 facing the camera, σ = 180/√12 = 52.0)\n",
                   std::atan2(f2.u.y(), f2.u.x()) * 180 / kPi, f2.yaw_sd * 180 / kPi);
        CHECK(yaw_err(std::atan2(f2.u.y(), f2.u.x()), 0.0f) < 1e-4f and std::abs(f2.yaw_sd - kPi / std::sqrt(12.0f)) < 1e-4f);
        // noisy depth: the yaw σ widens with the residual
        std::vector<Vec2f> noisy;
        for (int i = -10; i <= 10; ++i) noisy.push_back(centre + Vec2f(0.05f * static_cast<float>(i), (i % 2 == 0 ? 0.1f : -0.1f)));
        const auto f3 = rc::door::frame_from_evidence(noisy, centre, cam, cam, std::nullopt);
        std::print("(e) noisy depth  : yaw {:+.2f} deg σ {:.1f} deg\n", std::atan2(f3.u.y(), f3.u.x()) * 180 / kPi, f3.yaw_sd * 180 / kPi);
        CHECK(f3.yaw_sd > 2.0f * kPi / 180.0f and f3.yaw_sd > f1.yaw_sd);   // analytic: 0.1 m / √Σa² ≈ 4.1°
    }

    if (fails == 0) std::print("test_door_wall_line: ALL PASS\n");
    else            std::print("test_door_wall_line: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
