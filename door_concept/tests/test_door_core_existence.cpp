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
#include "../src/door_opening_edges.h"
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

struct DoorDet { float x = 1.0f, y = kWallY, w = 1.0f, conf = 0.9f; };

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
        s.label = "door"; s.class_id = 1000.f; s.confidence = d.conf;
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
    // These cases jump between standpoints from one cycle to the next: a teleport is not a turn, so turning is priced only where a case asks.
    static DoorCoreConfig config(bool turns) { DoorCoreConfig c; if (not turns) c.door.ai2_ang_lever_m = 0.0f; return c; }
    std::vector<std::string> csv, cost_rows;
    std::uint64_t cycle = 0;
    int frame = 0;
    explicit Rig(bool turns = false) : cfg(config(turns)), core(cfg)
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
    // ── the_blob_pays_its_measured_width (2026-11-09, clean19-21): the measurement is scored against the prior, not the measurement after
    //    that prior has pulled it (0.36 m pulled to 0.95 m paid -0.20 nats; five phantoms measuring 0.23-0.41 m were born on it)
    {
        Rig r;
        r.look(A0, kDoor, {{1.0f, kWallY, 0.36f}});
        std::print("narrow blob: 0.36 m candidate after one hit view -> {} birth(s), size term {:.2f}\n", r.births.size(),
                   r.core.candidates().empty() ? NAN : r.core.candidates().front().belief.prior.size);
        CHECK(r.births.empty());
        CHECK(not r.core.candidates().empty() and r.core.candidates().front().belief.prior.size < -3.0f);
    }
    // ── a_later_view_measures_the_width_again: one narrow glimpse must not hold a real door back; the size term is this frame's, as the band's
    {
        Rig r;
        r.look(A0, kDoor, {{1.0f, kWallY, 0.36f}});
        CHECK(r.births.empty());
        r.look(A0, kDoor, {{1.0f, kWallY, 1.00f}});
        std::print("narrow then full: {} birth(s)\n", r.births.size());
        CHECK(r.births.size() == 1);
    }
    // ── a_borderline_mask_is_not_a_door (2026-11-09, clean19-21: real doors were born from masks of 0.53-0.83, 11 of 14 phantoms below 0.53)
    {
        const ConfidenceModel m = default_detectors().confidence;
        CHECK(confidence_reliability(0.47f, m) < 0.2f);
        CHECK(confidence_reliability(0.75f, m) > 0.9f);
        Rig r;
        r.look(A0, kDoor, {{1.0f, kWallY, 1.0f, 0.47f}});
        std::print("borderline mask: conf 0.47 -> {} birth(s)\n", r.births.size());
        CHECK(r.births.empty());
    }
    // ── a_rotating_frame_says_little: the same confident mask, from a camera turning at about 2.5 rad/s, then at rest
    {
        const Vec2f aside = kDoor + Vec2f(1.0f, 0.0f);
        Rig still(true), turning(true);
        still.look(A0, kDoor);
        still.look(A0, kDoor, {{}});
        turning.look(A0, aside);
        turning.look(A0, kDoor, {{}});
        std::print("rotation: still -> {} birth(s), turning -> {} birth(s)\n", still.births.size(), turning.births.size());
        CHECK(still.births.size() == 1);
        CHECK(turning.births.empty());
        CHECK(rotation_reliability(0.0f, 2.0f, 0.6f) == 1.0f);
        CHECK(rotation_reliability(0.5f, 2.0f, 0.6f) < 0.3f);
    }
    {   // a 3 m candidate (size term −45 nats): vision saturates at −ln(lambda) (spec §3.1b), so no number of views creates it — only a physical test could (separate passes: views of different visits never merge)
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
        CHECK(not born and n == 200);                                        // 200 hit views: the saturation holds it under the size prior (it was born after enough with the old unbounded sum)
        CHECK(pb.L() < pb.prior.total() - float(std::log(lookalike_lambda(pb.lookalike, pb.prior.base_rate))) + 0.05f);
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
        CHECK(r.births.size() == 2);   // vision saturates near ln(0.01/0.002) = +1.6 for a further door: enough to be born (Pablo 2026-11-08, spec §3.1b)
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

    // ── the fit's own support (channel sup): x from the fit, one ℓ per view, no fit no term ──────────────────────────────────
    {   // a real door, well fitted (x ≈ 0.05), stays believed over the views of a tour; a fit explaining none of its support
        // (x ≈ 1) goes below the delete line within them (cmp_br3 door_6: birth (3.59,-1.91), fit 2.2 m off, L +5.51)
        const auto tour = [&](float x, bool with_support)
        {
            Rig r;
            CostEstimator costs;
            r.core.set_costs(CostEstimator::logged_default());
            r.look(A0, kDoor, {{1.0f, kWallY, 0.9f}});
            r.settle(A0);
            for (const float deg : {-40.0f, 40.0f, 0.0f})
                for (int k = 0; k < 8; ++k)
                {
                    if (with_support and r.door()) r.door()->pending_fit_support = FitSupport{x, 0};
                    r.look(stand(4.0f, deg), kDoor, {{1.0f, kWallY, 0.9f}});
                }
            const float L = r.door() ? r.door()->belief.L() : -99.0f;
            std::print("support_tour: x={:.2f} support={} -> door {} L={:+.2f} removes={}\n", x, with_support, r.door() ? "alive" : "removed", L, r.removes.size());
            return std::pair{r.door() != nullptr, L};
        };
        const auto [none_alive, none_L] = tour(0.05f, false);
        const auto [good_alive, good_L] = tour(0.05f, true);
        const auto [bad_alive, bad_L] = tour(0.99f, true);
        CHECK(none_alive and good_alive and good_L > 0.0f);   // x ≈ 0.05 leaves a real door believed
        CHECK(not bad_alive);                                 // x ≈ 1 removes it, by the costed line, within these three views
    }
    {   // a door with no fit gets no term: nothing pending, trace NaN, no ledger entry from the channel; pending is consumed once
        Rig r;
        r.look(A0, kDoor, {{1.0f, kWallY, 0.9f}});
        r.settle(A0);
        CHECK(r.door() != nullptr);
        if (r.door())
        {
            r.door()->pending_fit_support.reset();
            r.settle(A0);
            CHECK(not std::isfinite(r.door()->exist_trace.sup_x) and not std::isfinite(r.door()->exist_trace.sup_dl));
            const std::size_t n_csv = r.csv.size();
            r.door()->pending_fit_support = FitSupport{0.6f, 0};   // a moderate x: this door survives the term, and we read its trace
            r.look(stand(4.0f, 50.0f), kDoor);
            CHECK(r.door() != nullptr);
            if (r.door() == nullptr) { std::print("support_trace: door removed by x=0.6?\n"); }
            else
            {
            CHECK(std::isfinite(r.door()->exist_trace.sup_x) and std::abs(r.door()->exist_trace.sup_x - 0.6f) < 1e-6f);
            CHECK(r.door()->exist_trace.sup_dl < 0.0f);
            CHECK(not r.door()->pending_fit_support.has_value() or r.door()->pending_fit_support->x != 0.6f);   // consumed (the cycle's fit may have set a fresh one)
            CHECK(r.csv.size() > n_csv);
            // the columns are in the row: the last two fields of door_existence_log.csv are sup_x, sup_dl
            const std::string& row = r.csv.back();
            CHECK(row.substr(row.rfind(',') + 1) != "nan\n");
            std::print("support_trace: row={}", row);
            }
        }
    }
    {   // a candidate has no fit: observe_place without support adds no term (the brief: never a default)
        Bench b;
        const rc::DoorFitState th = shut_door();
        PlaceBelief pb;
        const auto t = b.frame(pb, th, A0, kDoor, true, false, {{{1.0f, kWallY}, -1.0f, 0}}, 0);
        CHECK(not std::isfinite(t.sup_x) and not std::isfinite(t.sup_dl));
        CHECK(pb.ledger.views().size() == 1 and pb.ledger.views()[0].sup.n == 0);
    }
    {   // a lost pose (κ → 0): the fit's x changes nothing (Review Focus 1)
        const rc::DoorConfig dc;
        Bench b;
        const rc::DoorFitState th = shut_door();
        Mat6d huge = Mat6d::Identity() * 1e6;
        PlaceBelief pb;
        std::vector<FrameDetection> nodets;
        const Mat4d body = body_pose(A0, kDoor);
        const std::optional<Mat4d> rtz = body * lift(1.2);
        const CameraIntrinsics cam = zed_cam();
        const ExistenceScene scene{b.cfg, rtz, cam, b.pkt, kRoom, rc::door::RoomBand{b.cfg.exist_wall_thickness_m, b.cfg.exist_boundary_sigma_m}, b.doors};
        ExistenceFrame fr{scene, b.cfg, b.det};
        fr.rt_cov = huge;
        const auto t = observe_place(pb, th, fr, 0, 0.0f, nullptr, -1, 0.0f, FitSupport{0.99f, 0});
        std::print("support_lost_pose: sup_x {} sup_dl {} views {}\n", t.sup_x, t.sup_dl, pb.ledger.views().size());
        CHECK(std::abs(pb.ledger.sum()) < 0.02f);   // κ for a σ = 1 km pose is ~1e-4: ℓ ≈ 0, no view of weight
    }

    {   // clean31 (real DOOR_2, fit 0.23 m from it): ONE new view whose only support row is a 0.16 m sliver on the jamb, x = 1.00.
        // The trial is worth the share of the opening the mask sampled: the sliver moves the ledger by about 16 % of a full mask's ℓ.
        const auto one_view = [&](float coverage)
        {
            Bench b;
            const rc::DoorFitState th = shut_door();
            PlaceBelief pb;
            const Mat4d body = body_pose(A0, kDoor);
            const std::optional<Mat4d> rtz = body * lift(1.2);
            const CameraIntrinsics cam = zed_cam();
            const ExistenceScene scene{b.cfg, rtz, cam, b.pkt, kRoom, rc::door::RoomBand{b.cfg.exist_wall_thickness_m, b.cfg.exist_boundary_sigma_m}, b.doors};
            ExistenceFrame fr{scene, b.cfg, b.det};
            const auto t = observe_place(pb, th, fr, 0, 0.0f, nullptr, -1, 0.0f, FitSupport{0.999999f, 0, coverage});
            std::print("support_coverage: c={:.2f} sup_dl {:+.2f} sup_c {:.2f}\n", coverage, t.sup_dl, t.sup_c);
            return t.sup_dl;
        };
        const float full = one_view(1.0f), sliver = one_view(0.16f), none = one_view(0.0f);
        CHECK(full < -5.0f);                                          // a full mask the fit explains none of: a strong "absent"
        CHECK(std::abs(sliver - 0.16f * full) < 0.02f * std::abs(full));   // a sliver is a sixth of that trial
        CHECK(1.10f + sliver > -1.08f);   // clean31: door_1 stood at L = +1.10; the sliver leaves it above the delete line (it was -6.64)
        CHECK(std::abs(none) < 1e-6f);                                // a mask entirely off the opening says nothing about it
    }
    {   // clean35 (real DOOR_2, born at its opening; a ONE-viewpoint hinge flip then slid the fit 0.6-0.8 m onto plain wall,
        // sigma_s 0.42 m): a support row scored on a window that may not be on the opening is a MIXTURE — with probability q
        // (aperture_overlap) it looked at the opening, otherwise at wall that a door and no door predict alike.
        const auto one_view = [&](float overlap)
        {
            Bench b;
            const rc::DoorFitState th = shut_door();
            PlaceBelief pb;
            const Mat4d body = body_pose(A0, kDoor);
            const std::optional<Mat4d> rtz = body * lift(1.2);
            const CameraIntrinsics cam = zed_cam();
            const ExistenceScene scene{b.cfg, rtz, cam, b.pkt, kRoom, rc::door::RoomBand{b.cfg.exist_wall_thickness_m, b.cfg.exist_boundary_sigma_m}, b.doors};
            ExistenceFrame fr{scene, b.cfg, b.det};
            const auto t = observe_place(pb, th, fr, 0, 0.0f, nullptr, -1, 0.0f, FitSupport{0.999999f, 0, 1.0f, overlap});
            std::print("support_overlap: q={:.2f} sup_dl {:+.2f} sup_q {:.2f}\n", overlap, t.sup_dl, t.sup_q);
            return t.sup_dl;
        };
        const float settled = one_view(1.0f), q = rc::door::aperture_overlap(0.95f, 0.42f), loose = one_view(q), lost = one_view(0.0f);
        std::print("aperture_overlap(0.95, 0.42) = {:.3f}\n", q);
        CHECK(settled < -5.0f);                                   // a settled fit: the full trial, as before
        CHECK(std::abs(loose - std::log1p(-q) - std::log(1.0 + q / (1.0 - q) * std::exp(settled))) < 1e-3f);   // the mixture
        CHECK(loose >= std::log1p(-q) - 1e-4f);                   // bounded by P(the window was on wall)
        CHECK(0.17f + loose > -1.08f);                            // clean35: door_2 stood at L = +0.17; it was removed at -1.15
        CHECK(std::abs(lost) < 1e-6f);                            // a window certainly off the opening says nothing about it
    }
    {   // aperture_overlap: E[(w - |d|)+]/w for d ~ N(0, sigma_s)
        CHECK(rc::door::aperture_overlap(1.0f, 0.0f) == 1.0f);
        CHECK(rc::door::aperture_overlap(0.0f, 0.1f) == 0.0f);
        CHECK(std::abs(rc::door::aperture_overlap(0.95f, 0.42f) - 0.65f) < 0.01f);   // the clean35 fit (hand: 0.650)
        CHECK(rc::door::aperture_overlap(1.0f, 0.01f) > 0.99f);
        CHECK(rc::door::aperture_overlap(1.0f, 100.0f) < 0.01f);
        CHECK(rc::door::aperture_overlap(1.0f, 0.2f) > rc::door::aperture_overlap(1.0f, 0.4f));
        // Monte-Carlo agreement at one point
        double acc = 0; std::uint32_t s = 12345;
        for (int i = 0; i < 200000; ++i)
        {
            s = s * 1664525u + 1013904223u; const double u1 = (s + 0.5) / 4294967296.0;
            s = s * 1664525u + 1013904223u; const double u2 = (s + 0.5) / 4294967296.0;
            const double d = 0.3 * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            acc += std::max(0.0, 0.8 - std::abs(d)) / 0.8;
        }
        CHECK(std::abs(rc::door::aperture_overlap(0.8f, 0.3f) - acc / 200000.0) < 0.005);
    }
    {   // the ledger's overlap persists (VL3); a VL2 record reads every frame as q = 1
        ViewLedger L;
        const std::size_t v = L.observe(Eigen::Vector2f(0.0f, -2.0f), Eigen::Vector2f(0.0f, 0.0f), Eigen::Vector2f(0.0f, 1.0f), 0.5f, 0.0f, 0);
        L.add_support(v, 0.9f, 1.0f, SupportModel{}, 0.5f);
        L.recompute(v, true);
        const auto s3 = L.serialise();
        const auto back = ViewLedger::parse(s3, true);
        CHECK(s3.starts_with("VL3"));
        CHECK(back.has_value() and std::abs(back->sum() - L.sum()) < 1e-5f);
        std::string s2 = s3; s2.replace(0, 3, "VL2"); s2 = s2.substr(0, s2.rfind(' '));   // drop sum_q: the old format
        const auto old = ViewLedger::parse(s2, true);
        ViewLedger full; const std::size_t vf = full.observe(Eigen::Vector2f(0.0f, -2.0f), Eigen::Vector2f(0.0f, 0.0f), Eigen::Vector2f(0.0f, 1.0f), 0.5f, 0.0f, 0);
        full.add_support(vf, 0.9f, 1.0f, SupportModel{}, 1.0f); full.recompute(vf, true);
        CHECK(old.has_value() and std::abs(old->sum() - full.sum()) < 1e-5f);
    }
    {   // aperture_coverage: the length of the opening sampled, not the count of returns; wall/jamb points beyond [0, w] add nothing
        rc::door::Aperture ap; ap.wall_O = {0.0f, 0.0f}; ap.wall_u = {1.0f, 0.0f}; ap.s = 2.0f; ap.w = 1.0f;
        std::vector<Eigen::Vector3f> full, sliver, wall, dense;
        for (int i = 0; i < 200; ++i) full.emplace_back(2.0f + 0.005f * i, 0.0f, 1.0f);
        for (int i = 0; i < 40; ++i) sliver.emplace_back(2.84f + 0.004f * i, 0.0f, 1.0f);       // 0.16 m at the far jamb
        for (int i = 0; i < 40; ++i) wall.emplace_back(3.05f + 0.01f * i, 0.0f, 1.0f);          // beside the opening
        for (int i = 0; i < 4000; ++i) dense.emplace_back(2.84f + 0.00004f * i, 0.0f, 1.0f);    // the same 0.16 m, 100x the returns
        const float cf = rc::door::aperture_coverage(ap, full), cs = rc::door::aperture_coverage(ap, sliver);
        const float cw = rc::door::aperture_coverage(ap, wall), cd = rc::door::aperture_coverage(ap, dense);
        std::print("aperture_coverage: full {:.2f} sliver {:.2f} wall {:.2f} dense {:.2f}\n", cf, cs, cw, cd);
        CHECK(cf > 0.95f);
        CHECK(std::abs(cs - 0.16f) < 0.035f);   // one cell (kRegSigmaRM) either way
        CHECK(cw == 0.0f);
        CHECK(std::abs(cd - cs) < 0.035f);
        CHECK(rc::door::aperture_coverage(ap, {}) == 0.0f);
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
            const float L0 = float(r.door()->belief.Sc());   // the view counts into Sc (the belief's L is saturated, spec §3.1b)
            const std::size_t v0 = r.door()->belief.ledger.views().size();
            for (int k = 0; k < 600; ++k) r.look(stand(4.0f, 10.0f), kDoor, {}, 20.0);
            const auto& views = r.door()->belief.ledger.views();
            const float dL = float(r.door()->belief.Sc()) - L0;
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
        const float L0 = float(pb.Sc());   // (Sc: the views' own sum; the belief's L is saturated, spec §3.1b)
        const std::size_t v0 = pb.ledger.views().size();
        for (int k = 0; k < 600; ++k)   // 60 s at 10 Hz; the ZED looks away
            b.frame(pb, th, stand(1.4f, 0.0f), {1.0f, -5.0f}, false, true, {}, 2);
        std::print("ricoh_near_view_counts_once: 600 ricoh frames at 1.4 m -> {} new view(s), dSc {:+.4f}\n", pb.ledger.views().size() - v0, float(pb.Sc()) - L0);
        CHECK(pb.ledger.views().size() == v0);              // ★the ricoh's missed frames are no trial at all (Pablo 2026-11-08): no view, no evidence
        CHECK(float(pb.Sc()) == L0);
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
            if (r.door()) { auto& b = r.door()->belief; b.carried_sc += 0.5f - float(b.Sc()); b.carried += 0.5f - b.L(); }   // a door weakly believed in: Sc = +0.5 and L = +0.5 (the misses below must reach the belief, spec §3.1b)
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
        if (r.door()) { auto& b = r.door()->belief; b.carried_sc += 0.3f - float(b.Sc()); b.carried += 0.3f - b.L(); }   // Sc = +0.3 and L = +0.3: the views below cross every line
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

    // ── crossed_door_has_no_delete_line (Pablo 2026-11-08): a crossing fact is proof, latched: even after the fact goes away and with L very low, no REMOVE ──
    {
        Rig r;
        r.look(stand(6.0f, -75.0f), kDoor, {{1.0f, kWallY, 0.9f}});
        CHECK(r.door() != nullptr);
        r.settle(stand(6.0f, -75.0f));
        CHECK(r.door() and not r.door()->belief.crossed);
        if (r.door())
        {
            const auto id = r.door()->node_id;
            auto in = input(++r.cycle, ++r.frame, stand(6.0f, -75.0f), kDoor, {});
            in.backed_faces = {id};                                // the exit fact, one cycle only
            r.step(std::move(in));
            CHECK(r.door() and r.door()->belief.crossing_protected() and not r.door()->belief.crossed);   // clean16: pending, not yet proof
        }
        if (r.door()) r.door()->belief.carried = -100.0f;          // the band of a swap (clean1 door_9), and more: no line can catch it
        for (int v = 0; v < 12 and r.door(); ++v)
            for (int k = 0; k < 5; ++k)
                r.step(input(++r.cycle, ++r.frame, stand(v % 2 ? 4.0f : 2.5f, -60.0f + 15.0f * static_cast<float>(v % 9)), kDoor, {}));
        CHECK(r.door() != nullptr);
        CHECK(r.removes.empty());
        CHECK(r.door() and r.door()->role == Role::BeingCrossed and std::isinf(r.door()->exist_trace.delete_line));
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
    // ── a_place_refuted_by_its_height_is_not_reborn (2026-11-13, clean60 door_7, the GlassBay pane): a born door fitted 1.00 x 2.62 m
    //    (σ_h 0.024) was removed by its size term, its row keeping vision's +L (the size term is not in L_evidence); the next candidate,
    //    which measures no height, was born again — 18 times. The row's remembered height is the candidate's height measurement.
    for (const bool tall : {true, false})
    {
        Rig r;
        PlaceRow row;
        row.room = "room_7"; row.name = "door_7"; row.xy = {1.0f, kWallY}; row.cov = Eigen::Matrix2f::Identity() * 0.04f;
        row.width = 1.00f; row.height = tall ? 2.62f : 2.10f; row.height_sd = 0.024f; row.depth = 0.05f;
        row.L_evidence = 1.5f;                                   // clean60: zed +0.88, sup +0.66 — vision said door-shaped
        row.last_seen_wall_ms = row.sighted_wall_ms = kT0; row.removed_wall_ms = kT0;
        r.core.places().upsert(row);
        r.look(A0, kDoor, {{1.0f, kWallY, 1.00f}});
        const float size = r.core.candidates().empty() ? NAN : r.core.candidates().front().belief.prior.size;
        std::print("height_refuted_place ({}): births {}, candidate size term {:+.2f}\n", tall ? "2.62 m" : "2.10 m", r.births.size(),
                   r.births.empty() ? size : 0.0f);
        if (tall)
        {
            CHECK(r.births.empty());
            CHECK(not r.core.candidates().empty() and size < -9.0f);   // ≈ −19.4 × age factor (60 s at the 1/600 s prior hazard)
            CHECK(not r.core.candidates().empty() and not r.core.candidate_is_open(r.core.candidates().front()));   // not a question to chase
        }
        else
            CHECK(r.births.size() == 1);                         // a door-tall remembered place is born as before
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
            r.look(stand(3.0f + 0.25f * static_cast<float>(k / 5), static_cast<float>(k % 5) * 15.0f - 30.0f), kDoor,
                   k % 4 == 0 ? std::vector<DoorDet>{{1.0f, kWallY, 0.9f, 0.667f}} : std::vector<DoorDet>{});   // conf 0.667: reliability 0.9, the weak hit this flicker needs
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
            r.look(stand(3.0f + 0.25f * static_cast<float>(k / 5), static_cast<float>(k % 5) * 15.0f - 30.0f), kDoor,
                   k % 4 == 0 ? std::vector<DoorDet>{{1.0f, kWallY, 0.9f, 0.667f}} : std::vector<DoorDet>{});   // conf 0.667: reliability 0.9, the weak hit this flicker needs
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

    // ── clean25: one clipped mask does not answer the question. A blob's width is a lower bound, so the widest it ever looked decides ──
    {
        Rig r;
        CostEstimator costs;
        costs.add(CostEstimator::Kind::Keep, 20.0f, "csv"); costs.add(CostEstimator::Kind::DeleteOrdinary, 30.0f, "csv");
        r.core.set_costs(costs);
        for (int k = 0; k < 8; ++k) r.look(stand(3.0f, 0.0f), kDoor, {{1.0f, kWallY, 1.0f, 0.46f}});
        r.look(stand(3.0f, 0.0f), kDoor, {{1.0f, kWallY, 0.08f, 0.46f}});   // the frame's size term sinks L far below the line
        CHECK(r.core.candidates().size() == 1);
        if (r.core.candidates().size() == 1)
        {
            const auto& c = r.core.candidates().front();
            std::print("clipped frame: L={:+.2f}, size {:+.2f}, best {:+.2f}\n", c.belief.L(), c.belief.prior.size, c.best_size);
            CHECK(c.belief.L() < -2.0f);
            CHECK(r.core.candidate_is_open(c));
            CHECK(r.core.candidate_question_p(c) > 0.3f);     // what a look is worth is judged the same way as the question
        }
        Rig narrow;
        narrow.core.set_costs(costs);
        for (int k = 0; k < 8; ++k) narrow.look(stand(3.0f, 0.0f), kDoor, {{1.0f, kWallY, 0.12f, 0.46f}});   // never wider: it is not a door-sized thing
        CHECK(narrow.core.candidates().size() == 1);
        if (narrow.core.candidates().size() == 1) CHECK(not narrow.core.candidate_is_open(narrow.core.candidates().front()));
    }

    // ── a promoted room with no door: places along its walls are open questions, retired once a door is known (clean27, 2026-11-09) ──
    {
        Rig r;
        r.look(stand(3.0f, 0.0f), kDoor);
        const std::string room = r.core.room_name();
        CHECK(r.core.hypothesise_place(room, Vec2f(-2.0f, 5.0f), true));
        CHECK(r.core.hypothesise_place(room, Vec2f(-2.05f, 5.0f), true));     // the same place: one question
        CHECK(r.core.hypothesise_place(room, Vec2f(4.0f, 5.0f)));             // not a wall seed
        CHECK(not r.core.hypothesise_place("another_room", Vec2f(0.0f, 5.0f)));
        CHECK(r.core.candidates().size() == 2);
        if (r.core.candidates().size() == 2) CHECK(r.core.candidate_is_open(r.core.candidates().front()));
        r.core.retire_wall_seeds(room);
        CHECK(r.core.candidates().size() == 1 and not r.core.candidates().front().wall_seed);
    }

    // ── an undecided candidate is an open question, and only a new standpoint can answer it (clean23, 2026-11-09) ──
    {
        Rig r;
        CostEstimator costs;
        costs.add(CostEstimator::Kind::Keep, 20.0f, "csv"); costs.add(CostEstimator::Kind::DeleteOrdinary, 30.0f, "csv");   // delete line -ln(1.5) = -0.41: between the open L and the refused L of this case
        r.core.set_costs(costs);
        for (int k = 0; k < 8; ++k) r.look(stand(3.0f, 0.0f), kDoor, {{1.0f, kWallY, 1.0f, 0.46f}});   // one view of a borderline mask: not born, not refused
        CHECK(r.births.empty());
        CHECK(r.core.candidates().size() == 1);
        if (r.core.candidates().size() == 1)
        {
            const auto& c = r.core.candidates().front();
            std::print("open question: L={:+.2f}, views {}\n", c.belief.L(), c.belief.ledger.views().size());
            CHECK(r.core.candidate_is_open(c));
            CHECK(not r.core.candidate_view_is_new(c, stand(3.0f, 0.0f)));    // the view it already has
            CHECK(r.core.candidate_view_is_new(c, stand(3.0f, 60.0f)));       // another bearing is another trial
        }
        // an empty view of the place, and more of them: the costed line refuses it, the question is closed
        for (int v = 0; v < 6; ++v)
            for (int k = 0; k < 6; ++k) r.look(stand(2.5f + 0.3f * static_cast<float>(v), -60.0f + 24.0f * static_cast<float>(v)), kDoor);
        CHECK(r.births.empty());
        if (r.core.candidates().size() == 1)
        {
            std::print("refused: L={:+.2f}\n", r.core.candidates().front().belief.L());
            CHECK(not r.core.candidate_is_open(r.core.candidates().front()));
        }
    }

    // ── a_door_at_the_end_of_a_wall (2026-11-11, clean50 door_3): the existence prior carries the extent term (place_prior.h), the
    //    wall from the fit's own definition (door_carrying_wall). The top wall ends at x = −5: a 1.0 m door centred at −4.70 has its
    //    west jamb 0.20 m past the end and pays; the same door at −4.45 (5 cm of wall beside its post) barely pays; mid-wall pays
    //    nothing; a proto outline (a frontier, no wall end known) pays nothing; and the term is REPLACED with the prior (total()).
    {
        const auto prior_at = [&](float x, bool proto)
        {
            const rc::DoorFitState th = typical_door_at({x, kWallY}, 1.0f, rc::DoorConfig{}.door_prior_h_m, kRoom);
            return prior_of(th, kRoom, proto, {0.2f, 0.1f}, 0, 0.05f, 0.03f, true, fit_size_prior(), 0.0f, 0.05f, 0.05f);
        };
        const auto over = prior_at(-4.70f, false), truth = prior_at(-4.45f, false), mid = prior_at(1.0f, false), front = prior_at(-4.70f, true);
        std::print("door at the end of a wall: extent over {:+.3f} truth {:+.3f} mid-wall {:+.4f} proto {:+.3f}\n",
                   over.extent, truth.extent, mid.extent, front.extent);
        CHECK(over.extent < -1.5f);
        CHECK(truth.extent > -0.5f and truth.extent < 0.0f);
        CHECK(std::abs(mid.extent) < 1e-4f);
        CHECK(front.extent == 0.0f);
        CHECK(std::abs(over.total() - (over.band + over.base_rate + over.size + over.extent)) < 1e-5f);
    }

    std::print("{}\n", fails == 0 ? "ALL PASS" : std::format("{} FAILURE(S)", fails));
    return fails == 0 ? 0 : 1;
}
