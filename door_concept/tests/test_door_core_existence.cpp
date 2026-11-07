// test_door_core_existence.cpp — Task 8 of the door-existence model: DoorCore::update_existence on ONE belief per door place
// (prior + carried + crossing + view ledger, costed decision). Rewritten; no case of the old file survives except the
// determinism and classic-locale checks and the ImageSource tape check. The scene is the old one: a door in the top wall
// (y = 5) of a 10 x 10 m room, the ZED 1280 x 720, f = 700, 1.2 m up, the ricoh (equirect 3840 x 1920) 1.0 m up.
//
// Core-level cases drive DoorCore::step with a TrackerWorld whose door is born through the tracker from one ZED detection
// (birth by comparison: one good hit view). Helper-level cases drive existence_update.h's observe_place directly, where a
// case needs views from separate visits (the ledger never merges views of different visits) or a camera the core input
// cannot stage simply (the ricoh with known strips).
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_core_existence.cpp \
//       src/core/door_core.cpp src/core/door_fit.cpp src/core/door_existence.cpp src/core/existence_update.cpp \
//       src/core/door_visibility.cpp src/core/door_cycle_io.cpp src/door_belief.cpp src/door_model.cpp \
//       src/door_frame_evidence.cpp ../common/contour_edge/contour_edge_check.cpp ../common/contour_edge/contour_edge_project.cpp \
//       ../common/contour_edge/contour_depth_check.cpp -lzstd -lopencv_core -lopencv_imgproc \
//       -o /tmp/test_door_core_existence && /tmp/test_door_core_existence
#include <clocale>
#include <filesystem>
#include <cmath>
#include <numbers>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "../src/core/door_core.h"
#include "../src/core/door_existence.h"
#include "../src/core/existence_update.h"
#include "../../common/contour_edge/contour_edge_check.h"

using namespace rc::door::core;
using K = rc::door::DoorDecisionKind;

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

struct ExistWorld final : TrackerWorld
{
    DoorCore* core = nullptr;
    rc::DoorConfig cfg;
    std::uint64_t room = 7;
    std::vector<Vec2f> polygon{{-5.f, -5.f}, {5.f, -5.f}, {5.f, 5.f}, {-5.f, 5.f}};
    mutable std::unordered_map<std::uint64_t, rc::DoorFitState> st;
    std::uint64_t next_id = 1000;
    std::vector<std::string> events;

    std::uint64_t room_id() const override { return room; }
    std::vector<Vec2f> room_polygon() const override { return polygon; }
    std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> fit_states() const override
    {
        std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> v;
        for (auto& [id, s] : st) v.emplace_back(id, &s);
        std::ranges::sort(v, {}, &std::pair<std::uint64_t, rc::DoorFitState*>::first);   // a deterministic order
        return v;
    }
    std::optional<IdentityMatch> match_identity(const Vec2f&) const override { return std::nullopt; }
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
        core->fit().init_instance(f, id, "door_" + std::to_string(id - 999), s);
        st.emplace(id, std::move(f));
        return id;
    }
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
        std::ranges::sort(v);
        return v;
    }
    rc::DoorFitState* begin_fit(std::uint64_t id) override
    { const auto it = st.find(id); return it == st.end() ? nullptr : &it->second; }
    void before_remove(std::uint64_t id) override { events.push_back("before " + std::to_string(id)); }
    void remove_door(std::uint64_t id, float) override { events.push_back("remove " + std::to_string(id)); st.erase(id); }
};

constexpr float kWallY = 4.99f;
constexpr int kW = 1280, kH = 720;
constexpr float kF = 700.f;
const std::vector<Vec2f> kRoom{{-5.f, -5.f}, {5.f, -5.f}, {5.f, 5.f}, {-5.f, 5.f}};

// The body at `at`, yawed so the ZED's forward (+y of the camera frame) points at `target`.
static Mat4d body_pose(const Vec2f& at, const Vec2f& target)
{
    const Vec2f d = (target - at).normalized();
    const double th = std::atan2(d.y(), d.x()) - std::numbers::pi / 2.0;
    Mat4d T = Mat4d::Identity();
    T(0, 0) = std::cos(th); T(0, 1) = -std::sin(th);
    T(1, 0) = std::sin(th); T(1, 1) = std::cos(th);
    T(0, 3) = at.x(); T(1, 3) = at.y();
    return T;
}
static Mat4d lift(double z) { Mat4d T = Mat4d::Identity(); T(2, 3) = z; return T; }

static CameraIntrinsics zed_cam()
{
    CameraIntrinsics c; c.present = true; c.width = kW; c.height = kH; c.focal_x = kF; c.focal_y = kF; return c;
}
static CameraIntrinsics ricoh_cam()
{
    CameraIntrinsics c; c.present = true; c.width = 3840; c.height = 1920; c.projection_model = 1; return c;
}

struct DoorDet { float x = 1.0f, y = kWallY, w = 1.0f; };

// One cycle: the robot at `at` looking at `target`, these ZED door detections (an empty list = a frame with no door mask: a
// small plant in the corner, so the packet is a real frame).
static DoorCycleInput input(std::uint64_t cycle, int frame, const Vec2f& at, const Vec2f& target, const std::vector<DoorDet>& dets,
                            double hz = 10.0, std::optional<Mat6d> cov = std::nullopt)
{
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = static_cast<std::int64_t>(static_cast<double>(cycle) * 1e9 / hz);
    in.t_wall_ms = 1'700'000'000'000 + static_cast<std::int64_t>(static_cast<double>(cycle) * 1000.0 / hz);
    in.robot.present = true;
    in.robot.room_T_body = body_pose(at, target);
    in.robot.body_T_zed = lift(1.2);
    in.robot.rt_cov = cov;
    in.camera = zed_cam();
    RoomInfo r; r.id = 7; r.name = "room_7"; in.rooms.rooms.push_back(r);
    MaskPacket p;
    p.valid = true;
    p.frame_id = frame;
    p.timestamp_ms = 1000 + static_cast<std::uint64_t>(frame);
    const Mat4d zed_T_room = (*in.robot.room_T_body * lift(1.2)).inverse();
    for (const auto& d : dets)
    {
        MaskSlice s;
        s.support_begin = p.support_points.size();
        for (int i = 0; i <= 20; ++i)
            for (int j = 0; j <= 21; ++j)
            {
                const float jitter = 0.004f * static_cast<float>(((i * 7 + j * 13) % 5) - 2);
                p.support_points.emplace_back(d.x - 0.5f * d.w + d.w * static_cast<float>(i) / 20.0f, d.y + jitter, 0.1f * static_cast<float>(j));
            }
        s.support_end = p.support_points.size();
        s.label = "door"; s.class_id = 1000.f; s.confidence = 0.9f;
        s.centroid = {d.x, d.y, 1.05f}; s.bbox_min = {d.x - 0.5f * d.w, d.y - 0.01f, 0.0f}; s.bbox_max = {d.x + 0.5f * d.w, d.y + 0.01f, 2.1f};
        s.trunc_frac = 0.0f; s.range = (Vec2f(d.x, d.y) - at).norm(); s.centroid_radius = 0.1f; s.source = 0; s.has_depth = true;
        // the mask's pixels: the projected box (a pinhole, the cycle's pose)
        float c0 = 1e9f, c1 = -1e9f, r0 = 1e9f, r1 = -1e9f;
        for (const float xx : {d.x - 0.5f * d.w, d.x + 0.5f * d.w})
            for (const float zz : {0.0f, 2.1f})
            {
                const Eigen::Vector4d P = zed_T_room * Eigen::Vector4d(xx, d.y, zz, 1.0);
                if (P.y() <= 0.2) continue;
                const Eigen::Vector2d uv = camera_project(in.camera, P.head<3>());
                c0 = std::min(c0, float(uv.x())); c1 = std::max(c1, float(uv.x())); r0 = std::min(r0, float(uv.y())); r1 = std::max(r1, float(uv.y()));
            }
        s.pixel_begin = p.mask_pixels.size();
        for (float r = std::max(0.f, r0); r < std::min(float(kH), r1); r += 3.0f)
            for (float c = std::max(0.f, c0); c < std::min(float(kW), c1); c += 3.0f) p.mask_pixels.emplace_back(c, r);
        s.pixel_end = p.mask_pixels.size();
        p.slices.push_back(s);
    }
    {
        MaskSlice s;
        s.label = "plant"; s.class_id = 5.f; s.confidence = 0.8f; s.source = 0; s.has_depth = true;
        s.pixel_begin = p.mask_pixels.size();
        for (int r = 0; r < 30; r += 3) for (int c = 0; c < 30; c += 3) p.mask_pixels.emplace_back(c, r);
        s.pixel_end = p.mask_pixels.size();
        p.slices.push_back(s);
    }
    in.masks = std::move(p);
    return in;
}

struct Rig
{
    DoorCoreConfig cfg;
    ExistWorld w;
    DoorCore core;
    std::vector<rc::door::DoorDecision> removes, births;
    std::vector<std::string> csv, cost_rows;
    std::uint64_t cycle = 0;
    int frame = 0;
    Rig() : core(cfg)
    {
        w.cfg = cfg.door;
        w.core = &core;
        // These cases exercise the decision's mechanics at C_keep = C_delete (the delete line is exactly L = 0, `default_costs()`); the
        // agent never runs on equal costs (CostEstimator::logged_default() is the core's own default) - cases that need the real
        // line set costs themselves.
        core.set_costs(CostEstimator{});
        core.connect(&w, {.decision = [this](const rc::door::DoorDecision& d)
                          {
                              if (d.kind == static_cast<std::uint16_t>(K::Remove)) { removes.push_back(d); w.events.push_back("REMOVE"); }
                              if (d.kind == static_cast<std::uint16_t>(K::Birth)) births.push_back(d);
                          },
                          .diag = [this](const DiagRow& d)
                          {
                              if (d.stream == "door_existence_log.csv") csv.push_back(d.line);
                              if (d.stream == "door_cost_events.csv") cost_rows.push_back(d.line);
                          }});
    }
    DoorCycleOutput step(DoorCycleInput in) { return core.step(std::move(in)); }
    DoorCycleOutput look(const Vec2f& at, const Vec2f& target, const std::vector<DoorDet>& dets = {}, double hz = 10.0,
                         std::optional<Mat6d> cov = std::nullopt)
    { ++cycle; ++frame; return step(input(cycle, frame, at, target, dets, hz, cov)); }
    rc::DoorFitState* door() { return w.st.empty() ? nullptr : &w.st.begin()->second; }
    /// One cycle looking AWAY from the door (nothing observed): the newborn's prior is replaced by its fitted size's.
    void settle(const Vec2f& at) { look(at, {at.x(), at.y() - 5.0f}); }
};

// The standpoint at `range` from the door (1, 5) along bearing `deg` from its inward normal (−y).
static Vec2f stand(float range, float deg, float door_x = 1.0f)
{
    const float a = deg * std::numbers::pi_v<float> / 180.0f;
    return {door_x + range * std::sin(a), kWallY - range * std::cos(a)};
}
static const Vec2f kDoor{1.0f, kWallY + 0.01f};

// ── helper level: one place, frames composed by hand ──────────────────────────────────────────────────────────────────
struct Bench
{
    rc::DoorConfig cfg;
    MaskPacket pkt;
    std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> doors;
    DetectorTable det = default_detectors();
    Bench() { pkt.valid = true; }
    ExistenceTrace frame(PlaceBelief& b, const rc::DoorFitState& th, const Vec2f& at, const Vec2f& target, bool zed, bool ricoh,
                         std::vector<FrameDetection> dets, int visit, std::optional<Mat6d> cov = std::nullopt)
    {
        const Mat4d body = body_pose(at, target);
        const std::optional<Mat4d> room_T_zed = body * lift(1.2);
        const CameraIntrinsics cam = zed_cam(), ric = ricoh_cam();
        const ExistenceScene scene{cfg, room_T_zed, cam, pkt, kRoom, rc::door::RoomBand{cfg.exist_wall_thickness_m, cfg.exist_boundary_sigma_m}, doors};
        ExistenceFrame fr{scene, cfg, det};
        fr.zed_ran = zed; fr.zed_spec = camera_spec(cam);
        if (ricoh) { fr.ricoh_ran = true; fr.ricoh_spec = camera_spec(ric); fr.room_T_ricoh = body * lift(1.0); }
        fr.rt_cov = cov;
        fr.dets = dets;
        return observe_place(b, th, fr, visit, 0.0f, nullptr);
    }
};
static rc::DoorFitState shut_door(float width = 0.85f) { return typical_door_at({1.0f, kWallY}, width, rc::DoorConfig{}.door_prior_h_m, kRoom); }
static SizePrior fit_size_prior() { const rc::DoorConfig c; return {c.door_prior_w_m, c.door_prior_w_std, c.door_prior_h_m, c.door_prior_h_std}; }
static Costs default_costs() { return CostEstimator{}.costs(rc::DoorConfig{}.exist_max_room_s); }

int main()
{
    if (std::setlocale(LC_ALL, "es_ES.UTF-8") == nullptr) std::setlocale(LC_ALL, "");
    const Vec2f A0 = stand(4.0f, 0.0f);

    // ── birth_by_comparison (spec test 8): 0.80 / 1.00 m candidates born after one good hit view ────────────────────────
    for (const float w : {0.80f, 1.00f})
    {
        Rig r;
        r.look(A0, kDoor, {{1.0f, kWallY, w}});
        std::print("birth_by_comparison: {:.2f} m candidate after one hit view -> {} birth(s), L={:.3f}\n", w, r.births.size(),
                   r.door() ? r.door()->belief.L() : NAN);
        CHECK(r.births.size() == 1);
        CHECK(r.door() != nullptr and r.door()->belief.L() > 0.0f);
    }
    {   // a 3 m candidate: not after one, born after enough (separate passes: views of different visits never merge)
        Bench b;
        const rc::DoorFitState th = typical_door_at({1.0f, kWallY}, 3.0f, rc::DoorConfig{}.door_prior_h_m, kRoom);
        PlaceBelief pb;
        pb.prior = prior_of(th, kRoom, false, {0.2f, 0.1f}, 0, b.cfg.tracker_detection_noise_m, 0.0f, true, fit_size_prior(), 0.0f);
        int n = 0;
        bool born_after_one = false, born = false;
        while (not born and n < 200)
        {
            ++n;
            b.frame(pb, th, A0, kDoor, true, false, {{{1.0f, kWallY}, -1.0f, 0}}, n);
            born = decide(pb.L(), Role::Ordinary, default_costs()).exists;
            if (n == 1) born_after_one = born;
        }
        std::print("birth_by_comparison: 3 m candidate (size term {:.2f}) born after {} hit views\n", pb.prior.size, n);
        CHECK(not born_after_one);
        CHECK(born and n > 1);
    }
    {   // off any wall: a door detection in the middle of the room is never born, however long it is seen
        Rig r;
        for (int k = 0; k < 300; ++k)
            r.look(stand(4.0f, static_cast<float>(k % 7) * 20.0f - 60.0f), {1.0f, 0.5f}, {{1.0f, 0.5f, 0.9f}});
        std::print("birth_by_comparison: off-wall candidate after 300 frames -> {} birth(s)\n", r.births.size());
        CHECK(r.births.empty());
    }
    {   // a second door on the same wall needs more views, and is still born under strong evidence
        Rig r;
        r.look(stand(4.0f, 0.0f, -2.0f), {-2.0f, kWallY}, {{-2.0f, kWallY, 0.9f}});
        CHECK(r.births.size() == 1);
        const Vec2f d2{2.5f, kWallY};
        r.look(stand(4.0f, 0.0f, 2.5f), d2, {{2.5f, kWallY, 0.9f}});
        const bool after_one = r.births.size() == 2;
        int views = 1;
        for (const float deg : {-45.0f, 45.0f, -70.0f, 70.0f})
        {
            if (r.births.size() == 2) break;
            ++views;
            r.look(stand(3.0f, deg, 2.5f), d2, {{2.5f, kWallY, 0.9f}});
        }
        std::print("birth_by_comparison: second door on the wall born after {} views (not after one: {})\n", views, not after_one);
        CHECK(not after_one);
        CHECK(r.births.size() == 2);
    }

    // ── unseen_door_unchanged (spec test 2): behind the camera, around a wall, overflowing the ZED at 0.5 m — both cameras ─
    {
        Rig r;
        // an L-shaped room: the lower-right quadrant is cut away, so a robot in the lower-left leg cannot see the top wall's right
        r.w.polygon = {{-5.f, -5.f}, {0.f, -5.f}, {0.f, 2.f}, {5.f, 2.f}, {5.f, 5.f}, {-5.f, 5.f}};
        const Vec2f door{3.0f, kWallY};
        r.look({3.0f, 2.5f}, door, {{3.0f, kWallY, 0.9f}});
        CHECK(r.births.size() == 1);
        r.settle({3.0f, 2.5f});
        const float L0 = r.door() ? r.door()->belief.L() : NAN;
        const std::size_t v0 = r.door() ? r.door()->belief.ledger.views().size() : 0;
        const auto with_ricoh = [&](DoorCycleInput in, float win_from, float win_to)
        {
            in.ricoh = ricoh_cam();
            in.robot.body_T_ricoh = lift(1.0);
            if (win_to > win_from) in.masks->ricoh_windows = {Vec2f(win_from, win_to)};   // the ricoh's processed strip
            return in;
        };
        const float pi = std::numbers::pi_v<float>;
        const auto phase = [&](const char* what, const Vec2f& at, const Vec2f& tgt, float wf, float wt)
        {
            for (int k = 0; k < 600; ++k)
            { ++r.cycle; ++r.frame; r.step(with_ricoh(input(r.cycle, r.frame, at, tgt, {}), wf, wt)); }
            std::print("unseen_door_unchanged: after 1 min {}: L {:.6f}, {} views; last row {}", what, r.door() ? r.door()->belief.L() : NAN,
                       r.door() ? r.door()->belief.ledger.views().size() : 0, r.csv.empty() ? std::string("\n") : r.csv.back());
        };
        phase("behind the camera", {3.0f, 2.5f}, {3.0f, -5.0f}, -0.75f * pi, -0.25f * pi);   // the ricoh's strip looks away too
        phase("around a wall", {-2.0f, -4.0f}, door, 0.0f, 2.0f * pi);                       // the other leg, looking at it
        // 0.5 m from the aperture the fitted leaf is all around the 360 camera: this phase is the ZED's (its case, tour18's), so
        // the packet carries no fresh ricoh frame (no strips, no ricoh slice ⇒ the ricoh looked at nothing).
        phase("overflowing the ZED at 0.5 m", {3.0f, 4.4f}, door, 0.0f, 0.0f);
        const float L1 = r.door() ? r.door()->belief.L() : NAN;
        std::print("unseen_door_unchanged: L {:.6f} -> {:.6f} after 3 min unseen ({} -> {} views), {} REMOVE\n", L0, L1, v0,
                   r.door() ? r.door()->belief.ledger.views().size() : 0, r.removes.size());
        CHECK(r.door() != nullptr);
        CHECK(std::bit_cast<std::uint32_t>(L0) == std::bit_cast<std::uint32_t>(L1));
        CHECK(r.removes.empty());
    }

    // ── phantom_removed_three_views (spec test 3, fix round 1): no mask, no edge, from the NATURAL post-birth L ─────────────
    // The phantom is born from one false detection (6 m, 75° off), then seen face-on at the range where the compiled ZED envelope
    // expects a door most (q is largest: the vertical extent fits, the horizontal one clears the 0.10 shoulder), from standpoints
    // 40° apart, then — if it is still alive — from further distinct standpoints. Reported: at which view it is removed, and why.
    {
        // the range of largest q, face-on (the compiled envelope: min_fill 0.10, max_fill 0.60, soft 0.06)
        float best_q = 0.0f, best_r = 0.0f;
        {
            Bench b;
            const rc::DoorFitState th = shut_door();
            for (float rr = 1.5f; rr <= 6.01f; rr += 0.1f)
            {
                PlaceBelief pb;
                const auto t = b.frame(pb, th, stand(rr, 0.0f), kDoor, true, false, {}, 1);
                if (std::isfinite(t.zed_q) and t.zed_q > best_q) { best_q = t.zed_q; best_r = rr; }
            }
            std::print("phantom_removed_three_views: face-on q peaks at {:.3f} at {:.1f} m (at 2.0 m the door overflows vertically)\n",
                       best_q, best_r);
        }
        std::vector<Vec2f> S{stand(best_r, -40.0f), stand(best_r, 0.0f), stand(best_r, 40.0f)};
        for (const float rr : {best_r, best_r - 1.5f, best_r + 1.5f})   // further DISTINCT standpoints (kernel ~17° and ~1.3 m at 4 m)
            for (const float deg : {-20.0f, 20.0f, -60.0f, 60.0f, 0.0f, -40.0f, 40.0f})
                if (rr != best_r or (deg != 0.0f and deg != -40.0f and deg != 40.0f)) S.push_back(stand(rr, deg));
        // fix round 4 (ONE size prior): the phantom's fit holds w at the fit's own prior mean, so its size term is now ~0 (it was
        // -0.76 under the second, existence-only prior); its post-birth L is that much higher and the weak edge-on views need more
        // of themselves to bring it to the line. More DISTINCT standpoints (a fourth ring, and the fan's edges): same process.
        for (const float deg : {-20.0f, 20.0f, -60.0f, 60.0f, 0.0f, -40.0f, 40.0f})
            S.push_back(stand(best_r + 3.0f, deg));
        for (const float rr : {best_r, best_r - 1.5f, best_r + 1.5f})
            for (const float deg : {-80.0f, 80.0f})
                S.push_back(stand(rr, deg));
        {
        Rig r;
        r.look(stand(6.0f, -75.0f), kDoor, {{1.0f, kWallY, 0.9f}});
        r.settle(stand(6.0f, -75.0f));
        const float L0 = r.door() ? r.door()->belief.L() : NAN;
        if (r.door()) std::print("phantom_removed_three_views: the phantom's one-frame fit: phi {:.0f} deg, w {:.2f}, curve {}\n",
                                 r.door()->phi_est * 57.2958f, r.door()->aperture.w, r.door()->phi_curve.size());
        int removed_at = -1;
        std::vector<float> ell;
        for (std::size_t s = 0; s < S.size() and r.door() != nullptr; ++s)
        {
            for (int k = 0; k < 20 and r.door() != nullptr; ++k) r.look(S[s], kDoor);
            if (r.door()) ell.push_back(r.door()->belief.ledger.views().back().llr);
            if (not r.removes.empty()) { removed_at = static_cast<int>(s) + 1; break; }
        }
        std::string ls;
        for (const float e : ell) ls += std::format(" {:+.3f}", e);
        std::print("phantom_removed_three_views: post-birth L {:+.3f}; {} distinct views, llrs{}; removed at standpoint {} (-1 = not "
                   "within these {})\n", L0, r.door() ? r.door()->belief.ledger.views().size() : 0, ls, removed_at, S.size());
        // ★FIX ROUND 1, REPORTED (not forced): from the natural post-birth L the phantom is NOT removed by these views. Its one-frame
        //  fit puts the leaf at ~60° (a flat φ curve), so the closed-panel envelope sees it nearly edge-on (q ≈ 0.1-0.25, ℓ ≈ −0.1 to
        //  −0.3 per view) against +3.4 from its one false hit; the compiled ZED envelope stays until refit from logs (ruling).
        CHECK(removed_at >= 3);   // removed by the natural process (fixed rates, fix round 3), never before the third view
        }
    }

    // ── detector rates stay FIXED within a session (fix round 3, ruling): a phantom's many misses leave them at the CSV values ──
    {
        Rig r;
        DetectorTable csv = default_detectors();
        csv.zed.hit_rate = 0.85f; csv.zed.false_alarm = 0.03f; csv.zed.n_hit_trials = 12.0f;   // as loaded from door_detectors.csv
        r.core.set_detectors(csv);
        r.look(stand(6.0f, -75.0f), kDoor, {{1.0f, kWallY, 0.9f}});
        r.settle(stand(6.0f, -75.0f));
        int frames = 0;
        for (const float rr : {4.2f, 2.7f, 5.7f})
            for (const float deg : {-60.0f, -40.0f, -20.0f, 0.0f, 20.0f, 40.0f, 60.0f})
                for (int k = 0; k < 20; ++k, ++frames) r.look(stand(rr, deg), kDoor);
        const DetectorTable& d = r.core.detectors();
        std::print("detector_rates_fixed: after {} miss frames on a phantom ({} REMOVE): zed hit {:.3f} fa {:.3f} n {:.0f}; ricoh hit {:.3f}\n",
                   frames, r.removes.size(), d.zed.hit_rate, d.zed.false_alarm, d.zed.n_hit_trials, d.ricoh.hit_rate);
        CHECK(d.zed.hit_rate == csv.zed.hit_rate and d.zed.false_alarm == csv.zed.false_alarm and d.zed.n_hit_trials == csv.zed.n_hit_trials);
        CHECK(d.ricoh.hit_rate == csv.ricoh.hit_rate and d.ricoh.false_alarm == csv.ricoh.false_alarm);
    }

    // ── birth respects cross-concept exclusion and frame admissibility (fix round 1, review I-1) ────────────────────────────
    {
        Rig claimed;
        auto in = input(++claimed.cycle, ++claimed.frame, A0, kDoor, {{1.0f, kWallY, 0.9f}});
        ObjectInfo fridge;
        fridge.id = 77; fridge.name = "fridge_1"; fridge.type = "object";
        fridge.width_m = 0.9f; fridge.depth_m = 0.6f; fridge.height_m = 2.0f;
        Mat4d T = Mat4d::Identity(); T(0, 3) = 1.0; T(1, 3) = kWallY; fridge.room_T_node = T;
        in.objects.push_back(fridge);
        claimed.step(std::move(in));
        Rig moving;
        auto im = input(++moving.cycle, ++moving.frame, A0, kDoor, {{1.0f, kWallY, 0.9f}});
        im.masks->slices[0].trunc_frac = 0.5f;   // a mask clipped by the image border: the fit would refuse this frame
        moving.step(std::move(im));
        std::print("birth exclusion: claimed by a fridge -> {} birth(s); inadmissible frame -> {} birth(s), candidate views {}\n",
                   claimed.births.size(), moving.births.size(),
                   moving.core.candidates().empty() ? 0 : moving.core.candidates().front().belief.ledger.views().size());
        CHECK(claimed.births.empty());
        CHECK(moving.births.empty());
        CHECK(not moving.core.candidates().empty() and moving.core.candidates().front().belief.ledger.views().empty());
    }

    // ── a pending candidate is never removed (fix round 1, review I-3): no REMOVED row, no false-alarm trial, no C_delete ─────
    {
        Rig r;
        const DetectorTable d0 = r.core.detectors();
        for (int k = 0; k < 300; ++k)
            r.look(stand(4.0f, static_cast<float>(k % 7) * 20.0f - 60.0f), {1.0f, 0.5f}, {{1.0f, 0.5f, 0.9f}});
        const DetectorTable d1 = r.core.detectors();
        std::print("pending candidate: {} births, {} place rows, zed false_alarm {:.4f} -> {:.4f}, C_delete samples {}\n", r.births.size(),
                   r.core.places().rows().size(), d0.zed.false_alarm, d1.zed.false_alarm,
                   r.core.costs().n(CostEstimator::Kind::DeleteOrdinary));
        CHECK(r.births.empty());
        CHECK(r.core.places().rows().empty());
        CHECK(d1.zed.false_alarm == d0.zed.false_alarm and d1.ricoh.false_alarm == d0.ricoh.false_alarm);   // (fixed anyway)
        CHECK(r.core.costs().n(CostEstimator::Kind::DeleteOrdinary) == 0);
    }

    // ── the birth frame enters the newborn's belief ONCE (fix round 1, review m-9) ──────────────────────────────────────────
    {
        Rig r;
        r.look(A0, kDoor, {{1.0f, kWallY, 0.9f}});
        ++r.cycle; r.step(input(r.cycle, r.frame, A0, kDoor, {{1.0f, kWallY, 0.9f}}));   // the SAME frame id again
        r.settle(stand(6.0f, -75.0f));
        const auto* d = r.door();
        const double n = (d and not d->belief.ledger.views().empty()) ? d->belief.ledger.views().front().bern[0].n : -1.0;
        std::print("birth frame entries in the newborn's first view: {}\n", n);
        CHECK(n == 1.0);
    }

    // ── parked_stare_one_view: 20 Hz for 30 s from one standpoint moves L by exactly one view's llr ─────────────────────
    {
        Rig r;
        r.look(stand(6.0f, -75.0f), kDoor, {{1.0f, kWallY, 0.9f}});
        CHECK(r.door() != nullptr);
        r.settle(stand(6.0f, -75.0f));
        if (r.door())
        {
            r.door()->belief.carried += 10.0f;   // far from any line: this case is about counting, not deciding
            const float L0 = r.door()->belief.L();
            const std::size_t v0 = r.door()->belief.ledger.views().size();
            for (int k = 0; k < 600; ++k) r.look(stand(4.0f, 10.0f), kDoor, {}, 20.0);
            const auto& views = r.door()->belief.ledger.views();
            const float dL = r.door()->belief.L() - L0;
            std::print("parked_stare_one_view: 600 frames at 20 Hz -> {} new view(s), dL {:.6f}, that view's llr {:.6f}\n",
                       views.size() - v0, dL, views.back().llr);
            CHECK(views.size() == v0 + 1);
            CHECK(std::abs(dL - views.back().llr) < 1e-5f);
            CHECK(views.back().llr < 0.0f);
        }
    }

    // ── real_door_aspect_misses (spec test 4): one confirming hit view, then 20 miss views at 70° aspect at 1 m and 6 m ──
    {
        Bench b;
        const rc::DoorFitState th = shut_door();
        PlaceBelief pb;
        pb.prior = prior_of(th, kRoom, false, {0.2f, 0.1f}, 0, 0.05f, 0.03f, true, fit_size_prior(), 0.0f);
        b.frame(pb, th, A0, kDoor, true, false, {{{1.0f, kWallY}, -1.0f, 0}}, 1);
        const float hit_view = pb.ledger.views().back().llr;
        bool removed = false;
        float worst = 0.0f;
        for (int k = 0; k < 20; ++k)
        {
            const float range = k % 2 ? 6.0f : 1.0f, deg = (k / 2) % 2 ? 70.0f : -70.0f;
            const auto t = b.frame(pb, th, stand(range, deg), kDoor, true, false, {}, 2 + k);
            if (std::isfinite(t.zed_dl)) worst = std::min(worst, t.zed_dl);
            removed = removed or decide(pb.L(), Role::Ordinary, default_costs()).remove;
        }
        std::print("real_door_aspect_misses: hit view {:+.3f}, worst miss view {:+.4f}, L after 20 misses {:+.3f}, removed {}\n",
                   hit_view, worst, pb.L(), removed);
        CHECK(hit_view > 0.0f and -worst < hit_view);
        CHECK(not removed);
    }

    // ── ricoh_near_view_counts_once (replaces spec test 5): parked 1.4 m from a face-on door, the ricoh missing it for 60 s ─
    {
        Bench b;
        const rc::DoorFitState th = shut_door();
        PlaceBelief pb;
        pb.prior = prior_of(th, kRoom, false, {0.2f, 0.1f}, 0, 0.05f, 0.03f, true, fit_size_prior(), 0.0f);
        b.frame(pb, th, A0, kDoor, true, false, {{{1.0f, kWallY}, -1.0f, 0}}, 1);   // the door was confirmed once by the ZED
        b.pkt.ricoh_windows = {Vec2f(0.0f, 2.0f * std::numbers::pi_v<float>)};    // every strip processed
        const float L0 = pb.L();
        const std::size_t v0 = pb.ledger.views().size();
        for (int k = 0; k < 600; ++k)   // 60 s at 10 Hz; the ZED looks away
            b.frame(pb, th, stand(1.4f, 0.0f), {1.0f, -5.0f}, false, true, {}, 2);
        const float one = pb.ledger.views().back().llr;
        std::print("ricoh_near_view_counts_once: 600 ricoh frames at 1.4 m -> {} view(s), dL {:+.4f}, that view's miss llr {:+.4f}\n",
                   pb.ledger.views().size() - v0, pb.L() - L0, one);
        CHECK(pb.ledger.views().size() == v0 + 1);
        CHECK(std::abs((pb.L() - L0) - one) < 1e-5f and one < 0.0f);
        for (const float deg : {-50.0f, 50.0f})
            for (int k = 0; k < 600; ++k) b.frame(pb, th, stand(1.4f, deg), {1.0f, -5.0f}, false, true, {}, 2);
        const Verdict v = decide(pb.L(), Role::Route, default_costs());
        std::print("ricoh_near_view_counts_once: after 3 such standpoints L {:+.3f}, route delete line {:+.3f} -> remove {}\n",
                   pb.L(), v.delete_line, v.remove);
        CHECK(not v.remove);
    }

    // ── route_door_survives (spec test 6, core level): evidence that removes an Ordinary door leaves a route door alive ───
    {
        const auto run = [&](bool route, int max_views)
        {
            Rig r;
            r.look(stand(6.0f, -75.0f), kDoor, {{1.0f, kWallY, 0.9f}});
            r.settle(stand(6.0f, -75.0f));
            if (r.door()) r.door()->belief.carried += 0.5f - r.door()->belief.L();   // a door weakly believed in: L = +0.5
            int views = 0;
            for (int v = 0; v < max_views and r.removes.empty(); ++v, ++views)
                for (int k = 0; k < 5; ++k)
                {
                    auto in = input(++r.cycle, ++r.frame, stand(v % 2 ? 4.0f : 2.5f, -60.0f + 15.0f * static_cast<float>(v % 9)), kDoor, {});
                    if (route) in.route_faces = {"door_1"};
                    r.step(std::move(in));
                }
            return std::pair{views, r.removes.size()};
        };
        const auto [n_ord, rm_ord] = run(false, 200);
        const auto [n_route, rm_route] = run(true, n_ord);
        std::print("route_door_survives: ordinary removed after {} views ({}); route door after the same views: {} REMOVE\n", n_ord,
                   rm_ord, rm_route);
        CHECK(rm_ord == 1);
        CHECK(rm_route == 0);
    }

    // ── being_crossed_holds_evidence: during `executing` on its cross, evidence accumulates and no REMOVE happens ────────
    {
        Rig r;
        r.look(stand(6.0f, -75.0f), kDoor, {{1.0f, kWallY, 0.9f}});
        CHECK(r.door() != nullptr);
        r.settle(stand(6.0f, -75.0f));
        if (r.door()) r.door()->belief.carried += 0.3f - r.door()->belief.L();   // L = +0.3: the views below cross every line
        const std::size_t v0 = r.door() ? r.door()->belief.ledger.views().size() : 0;
        float L_min = 1e9f;
        for (int v = 0; v < 30 and r.door(); ++v)
            for (int k = 0; k < 5; ++k)
            {
                auto in = input(++r.cycle, ++r.frame, stand(v % 2 ? 4.0f : 2.5f, -60.0f + 15.0f * static_cast<float>(v % 9)), kDoor, {});
                in.affordances.push_back({.node_id = 555, .exists = true, .name = "aff_door_1_cross"});
                in.executing.push_back({.from = 8, .to = 555});
                r.step(std::move(in));
                if (r.door()) L_min = std::min(L_min, r.door()->belief.L());
            }
        std::print("being_crossed_holds_evidence: {} -> {} views, L min {:+.3f}, role {}, {} REMOVE\n", v0,
                   r.door() ? r.door()->belief.ledger.views().size() : 0, L_min,
                   r.door() ? static_cast<int>(r.door()->role) : -1, r.removes.size());
        CHECK(r.door() != nullptr and r.door()->belief.ledger.views().size() > v0);
        CHECK(L_min < 0.0f);   // below the creation line and the ordinary delete line: only the role keeps it
        CHECK(r.removes.empty());
        CHECK(r.door() and r.door()->role == Role::BeingCrossed);
    }

    // ── relocalising_robot_moves_nothing (Review Focus 1): σ_xy = 3 m, heading uniform — every view's llr < 0.05 ─────────
    {
        Bench b;
        const rc::DoorFitState th = shut_door();
        Mat6d C = Mat6d::Zero();
        C(0, 0) = C(1, 1) = 9.0;
        C(2, 2) = C(3, 3) = C(4, 4) = 1e-6;
        C(5, 5) = std::numbers::pi * std::numbers::pi / 3.0;   // a heading uniform on the circle: σ² = π²/3
        PlaceBelief pb;
        float worst = 0.0f, kappa = 1.0f;
        int v = 0;
        for (const float range : {4.0f, 6.0f})
            for (const float deg : {-60.0f, -30.0f, 0.0f, 30.0f, 60.0f})
            {
                const auto t = b.frame(pb, th, stand(range, deg), kDoor, true, false, {}, ++v, C);
                if (std::isfinite(t.zed_dl)) worst = std::max(worst, std::abs(t.zed_dl));
                if (std::isfinite(t.kappa)) kappa = std::min(kappa, t.kappa);
            }
        std::print("relocalising_robot_moves_nothing: {} miss views, worst |llr| {:.4f} (kappa down to {:.3f}); sum {:+.4f}\n", v, worst,
                   kappa, pb.ledger.sum());
        CHECK(worst < 0.05f);
        // a detection while lost is about this door only in proportion to kappa: one lost hit view cannot birth a door
        PlaceBelief hb;
        hb.prior = prior_of(th, kRoom, false, {0.2f, 0.1f}, 0, 0.05f, 0.03f, true, fit_size_prior(), 0.0f);
        b.frame(hb, th, A0, kDoor, true, false, {{{1.0f, kWallY}, -1.0f, 0}}, 1, C);
        std::print("relocalising_robot_moves_nothing: one lost HIT view llr {:+.3f} -> L {:+.3f}\n", hb.ledger.sum(), hb.L());
        CHECK(not decide(hb.L(), Role::Ordinary, default_costs()).exists);
    }

    // ── open_door_survives_face_on_views (Review Focus 2): φ posterior at 80°, the leaf mask on the open leaf ────────────
    {
        Bench b;
        rc::DoorFitState th = shut_door();
        const float d80 = 80.0f * std::numbers::pi_v<float> / 180.0f, d5 = 5.0f * std::numbers::pi_v<float> / 180.0f;
        th.leaf.phi = d80;
        th.leaf.swing = -1.0f;   // into the room (−y)
        th.leaf_pose = rc::door::leaf_pose(th.aperture, th.leaf);
        th.phi_est = d80;
        th.phi_curve = {{d80 - d5, 0.5f}, {d80, 1.0f}, {d80 + d5, 0.5f}};
        PlaceBelief pb;
        pb.prior = prior_of(th, kRoom, false, {0.2f, 0.1f}, 0, 0.05f, 0.03f, true, fit_size_prior(), 0.0f);
        pb.carried = 0.5f;   // a door barely believed in
        const Vec2f leaf_xy = th.leaf_pose.centre_xy;
        int removes = 0;
        for (int k = 0; k < 10; ++k)
        {
            b.frame(pb, th, stand(3.5f, -40.0f + 9.0f * static_cast<float>(k)), kDoor, true, false, {{leaf_xy, -1.0f, 0}}, 1 + k);
            removes += decide(pb.L(), Role::Ordinary, default_costs()).remove ? 1 : 0;
        }
        std::print("open_door_survives_face_on_views: 10 face-on views of an 80 deg leaf -> L {:+.3f}, {} REMOVE\n", pb.L(), removes);
        CHECK(removes == 0);
    }

    // ── determinism: two cores fed the same inputs write bit-identical existence rows ────────────────────────────────────
    {
        const auto run = [&]
        {
            Rig r;
            r.look(A0, kDoor, {{1.0f, kWallY, 0.9f}});
            for (int k = 0; k < 40; ++k) r.look(stand(3.0f + 0.05f * static_cast<float>(k), -30.0f + 2.0f * static_cast<float>(k)), kDoor, k % 3 ? std::vector<DoorDet>{} : std::vector<DoorDet>{{1.0f, kWallY, 0.9f}});
            return r.csv;
        };
        const auto a = run(), b = run();
        CHECK(not a.empty() and a == b);
        std::print("determinism: {} existence rows, identical {}\n", a.size(), a == b);
        // csv_classic_locale: the rows under es_ES carry decimal POINTS and as many fields as the header
        if (not a.empty())
        {
            const auto fields = [](const std::string& s) { return std::ranges::count(s, ',') + 1; };
            CHECK(a.back().find('.') != std::string::npos);
            CHECK(fields(a.back()) == fields(kExistenceCsvHeader));
            CHECK(a.back().back() == '\n');
            std::print("csv_classic_locale: last row: {}", a.back());
        }
    }

    // ── ImageSource (the tape path): patches + the recorded frame_ref reproduce the full frame inside the patch ──────────
    {
        cv::Mat full(120, 200, CV_8UC1);
        for (int y = 0; y < full.rows; ++y)
            for (int x = 0; x < full.cols; ++x)
                full.at<std::uint8_t>(y, x) = static_cast<std::uint8_t>((x * 37 + y * 91 + (x * y) % 13) & 0xFF);
        const auto ref = rc::edges::prepare_frame(full);
        ImageFrame g;
        g.format = PixelFormat::Gray8; g.width = full.cols; g.height = full.rows; g.stamp_ms = 5; g.frame_ref = ref.frame_ref;
        ImagePatch p;
        p.x0 = 40; p.y0 = 30; p.w = 60; p.h = 50;
        for (int y = p.y0; y < p.y0 + p.h; ++y)
            for (int x = p.x0; x < p.x0 + p.w; ++x) p.bytes.push_back(full.at<std::uint8_t>(y, x));
        g.patches.push_back(p);
        ImageSource src;
        const auto gen0 = src.rgb_generation();
        src.apply(g, std::nullopt);
        CHECK(src.rgb_generation() == gen0 + 1);
        CHECK(src.rgb().cols == full.cols and src.rgb().rows == full.rows);
        CHECK(src.frame_ref().has_value() and src.frame_ref().value() == ref.frame_ref);
        auto prep = rc::edges::prepare_frame(src.rgb());
        if (src.frame_ref().has_value()) prep.frame_ref = src.frame_ref().value();
        bool same = true;
        for (int y = p.y0 + 6; y < p.y0 + p.h - 6; ++y)
            for (int x = p.x0 + 6; x < p.x0 + p.w - 6; ++x)
                same = same and prep.gx.at<float>(y, x) == ref.gx.at<float>(y, x) and prep.gy.at<float>(y, x) == ref.gy.at<float>(y, x);
        CHECK(same);
    }

    // ── REVISIT SAMPLES (Task 10 fix round 1): one per place per visit, from a place that is BORN / found / removed in that visit,
    //    never from a single detection and never twice in a visit. The observable is the table's learned hazard (changes, exposure).
    constexpr std::int64_t kT0 = 1'700'000'000'000 - 60'000;   // the row was last sighted 60 s before the run starts
    const auto remembered_row = [&](Rig& r, bool removed)
    {
        PlaceRow row;
        row.room = "room_7"; row.name = "door_9"; row.xy = {1.0f, kWallY}; row.cov = Eigen::Matrix2f::Identity() * 0.04f;
        row.width = 0.9f; row.height = 2.05f; row.depth = 0.05f; row.L_evidence = removed ? -2.0f : 1.0f;
        row.last_seen_wall_ms = row.sighted_wall_ms = kT0;
        if (removed) row.removed_wall_ms = kT0;
        r.core.places().upsert(row);
    };
    {   // a candidate seeded from a remembered DOOR, born: ONE "found" sample (60 s exposure); the same visit's later frames add none
        Rig r;
        remembered_row(r, false);
        r.look(A0, kDoor, {{1.0f, kWallY, 0.9f}});
        CHECK(r.births.size() == 1);
        auto h = r.core.places().hazard_state();
        std::print("revisit_found: born -> changes {} exposure {:.2f} s, door '{}' descends from '{}'\n", h.changes, h.sum_elapsed_s,
                   r.door() ? r.door()->node_name : "?", r.door() ? r.door()->place_row_name : "?");
        CHECK(h.changes == 0.0 and h.sum_elapsed_s > 59.9 and h.sum_elapsed_s < 60.5);
        CHECK(r.door() != nullptr and r.door()->place_row_name == "door_9");
        const double e0 = h.sum_elapsed_s;
        for (int k = 0; k < 30; ++k) r.look(A0, kDoor, {{1.0f, kWallY, 0.9f}});   // re-detections in the SAME visit
        h = r.core.places().hazard_state();
        CHECK(h.sum_elapsed_s == e0 and h.changes == 0.0);
        const PlaceRow row9 = *r.core.places().find("room_7", "door_9");
        const std::int64_t later = row9.sighted_wall_ms + 1000;               // the birth moved the sighting to its frame
        CHECK(not r.core.places().sample_revisit(row9, 1, true, later));   // this visit's sample is taken
        CHECK(r.core.places().sample_revisit(row9, 2, true, later));       // a NEW visit gives one
    }
    {   // a candidate seeded from a remembered REMOVED place, born: that is a change (the world changed), once
        Rig r;
        remembered_row(r, true);
        r.look(A0, kDoor, {{1.0f, kWallY, 0.9f}});
        const auto h = r.core.places().hazard_state();
        std::print("revisit_removed_row_born: births {} -> changes {} exposure {:.2f} s\n", r.births.size(), h.changes, h.sum_elapsed_s);
        CHECK(r.births.size() == 1);                                   // N5: no vacuous guard - the birth must happen
        CHECK(h.changes == 1.0 and h.sum_elapsed_s > 59.9);
    }
    {   // a remembered door REMOVED in the visit (in view, judged absent): one "not found" sample; the removal's own row does not add
        Rig r;
        remembered_row(r, false);
        r.look(A0, kDoor, {{1.0f, kWallY, 0.9f}});
        CHECK(r.door() != nullptr);
        // the first visit's sample was the birth's (found); a restored door that was NOT sampled yet is the removal case:
        PlaceTable& pt = r.core.places();
        PlaceRow other = *pt.find("room_7", "door_9"); other.name = "door_8"; other.xy = {-3.0f, kWallY};
        pt.upsert(other);
        r.door()->place_row_name = "door_8";
        const auto before = pt.hazard_state();
        CostEstimator cheap;
        for (int k = 0; k < 3; ++k) cheap.add(CostEstimator::Kind::DeleteOrdinary, 0.1f, "test");   // losing a door is cheap: the line is high
        r.core.set_costs(cheap);
        for (int k = 0; k < 3 and r.removes.empty(); ++k) r.look(A0, kDoor);
        const auto after = pt.hazard_state();
        std::print("revisit_removed: {} REMOVE(s) -> changes {} -> {}, exposure {:.1f} -> {:.1f} s\n", r.removes.size(), before.changes,
                   after.changes, before.sum_elapsed_s, after.sum_elapsed_s);
        CHECK(r.removes.size() == 1 and after.changes == before.changes + 1.0 and after.sum_elapsed_s > before.sum_elapsed_s + 59.0);
        CHECK(not pt.sample_revisit(other, 1, false, kT0 + 1000));   // the visit's sample is spent: a re-detection adds none
    }

    // ── BIRTH AND SURVIVAL AGREE (fix round 3): a candidate born at L > 0 stays above its creation line on the first fitted update
    //    when nothing new was observed (the next frame looks away): the candidate was scored with the size the fit will give it ──
    for (const float w : {0.80f, 0.90f, 1.00f})
    {
        Rig r;
        r.look(A0, kDoor, {{1.0f, kWallY, w}});
        CHECK(r.births.size() == 1 and r.door() != nullptr);
        const float L_birth = r.door() ? r.door()->belief.L() : NAN;
        r.settle(A0);
        const float L_fit = r.door() ? r.door()->belief.L() : NAN;
        std::print("birth_survival_agree: w {:.2f} born L {:+.3f} -> after the first fitted update (nothing new observed) {:+.3f}; removes {}\n",
                   w, L_birth, L_fit, r.removes.size());
        CHECK(L_birth > 0.0f and L_fit > 0.0f and r.removes.empty());
    }

    // ── removal_rebirth_same_visit (re-review I2/I3, ruling): a door removed and re-born at the same place IN ONE VISIT is a flicker,
    //    not a revisit: 0 hazard samples, the learned hazard unchanged (the removal's row was written THIS visit). ──
    {
        Rig r;
        CostEstimator fixed;
        fixed.add(CostEstimator::Kind::Keep, 20.0f, "csv"); fixed.add(CostEstimator::Kind::DeleteOrdinary, 3.0f, "csv");
        r.core.set_costs(fixed);   // the delete line is +1.9: the newborn falls below it on its quiet frames
        const auto h0 = r.core.places().hazard_state();
        for (int k = 0; k < 24 and (r.births.size() < 2 or r.removes.empty()); ++k)
            r.look(stand(4.0f, static_cast<float>(k % 5) * 15.0f - 30.0f), kDoor,
                   k % 4 == 0 ? std::vector<DoorDet>{{1.0f, kWallY, 0.9f}} : std::vector<DoorDet>{});
        const auto h1 = r.core.places().hazard_state();
        std::print("removal_rebirth_same_visit: {} births, {} removes -> hazard changes {} -> {}, exposure {:.3f} -> {:.3f} s\n",
                   r.births.size(), r.removes.size(), h0.changes, h1.changes, h0.sum_elapsed_s, h1.sum_elapsed_s);
        CHECK(r.births.size() >= 2 and r.removes.size() >= 1);   // there WAS a removal then a re-birth
        CHECK(h1.changes == h0.changes and h1.sum_elapsed_s == h0.sum_elapsed_s);
    }

    // ── C_keep / C_delete are FIXED within a session (fix round 2, ruling): a run of flicker births and removals changes neither,
    //    and the events the offline tool needs are logged (removal, re-birth at the same place with its seconds, place key, stamp). ──
    {
        Rig r;
        CostEstimator fixed;
        fixed.add(CostEstimator::Kind::Keep, 20.0f, "csv"); fixed.add(CostEstimator::Kind::DeleteOrdinary, 3.0f, "csv");
        r.core.set_costs(fixed);
        const Costs c0 = r.core.costs().costs(rc::DoorConfig{}.exist_max_room_s);   // the delete line is +1.9: a door that falls below it after its birth is deleted
        for (int k = 0; k < 60; ++k)   // a hit view (born), then a few quiet frames (L is below the high line: removed), again and again
            r.look(stand(4.0f, static_cast<float>(k % 5) * 15.0f - 30.0f), kDoor,
                   k % 4 == 0 ? std::vector<DoorDet>{{1.0f, kWallY, 0.9f}} : std::vector<DoorDet>{});
        const Costs c1 = r.core.costs().costs(rc::DoorConfig{}.exist_max_room_s);
        int rm = 0, rb = 0;
        for (const auto& l : r.cost_rows) { rm += l.starts_with("remove,") ? 1 : 0; rb += l.starts_with("rebirth,") ? 1 : 0; }
        std::print("costs_fixed: {} births, {} removes -> events: {} remove, {} rebirth; C_keep {:.1f}->{:.1f}, C_delete {:.1f}->{:.1f}\n",
                   r.births.size(), r.removes.size(), rm, rb, c0.keep_s, c1.keep_s, c0.delete_ordinary_s, c1.delete_ordinary_s);
        CHECK(r.births.size() >= 2 and r.removes.size() >= 1);                      // there WAS flicker
        CHECK(c1.keep_s == c0.keep_s and c1.delete_ordinary_s == c0.delete_ordinary_s and c1.delete_route_s == c0.delete_route_s);
        CHECK(r.core.costs().n(CostEstimator::Kind::Keep) == 1 and r.core.costs().n(CostEstimator::Kind::DeleteOrdinary) == 1);
        CHECK(rm == static_cast<int>(r.removes.size()) and rb >= 1);               // every removal and re-birth is logged
        bool well_formed = true;
        for (const auto& l : r.cost_rows) well_formed = well_formed and std::ranges::count(l, ',') == 7 and l.ends_with("\n");
        CHECK(well_formed);
    }

    std::print("{}\n", fails == 0 ? "ALL PASS" : std::format("{} FAILURE(S)", fails));
    return fails == 0 ? 0 : 1;
}
