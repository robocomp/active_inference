// test_door_core_existence.cpp — the DoorCore's EXISTENCE slice on synthetic inputs (Task 5 of the door-core replay
// plan). update_existence_beliefs moved out of SpecificWorker into the core (src/core/door_existence.{h,cpp}): the
// silhouette, the image-geometry evidence, P(detect), the removal decision. This drives it with the fit test's scene —
// a door in the top wall of a 10 x 10 m room, the robot still at (1, 0.5) looking +y — and a synthetic mask that lights
// the door's projected pixels (present) or lights nothing on it (absent):
//   1. PRESENT then ABSENT: L rises from the birth seed to the clamp while the mask lights the door, then falls under
//      the silhouette's absence and the door is REMOVED once the shared debounce has accrued RemoveFrames ideal looks:
//      one REMOVE decision, the world's before_remove() BEFORE it and remove_door() AFTER it, the door gone from the
//      world, and the removal cycle's Existence row at or below logit(RemovalProb).
//   2. "AN OBSERVATION, NOT A CYCLE": one Existence row per door per NEW mask frame; a repeated frame id integrates
//      nothing and writes no row.
//   3. TURNED AWAY = HOLD: the same absence with the camera looking -y (the door behind it) moves L by nothing.
//   4. DETERMINISM: two cores fed the same inputs write bit-identical Existence rows.
//   5. The existence CSV row (etc/door_existence_log.csv) is formatted in the classic locale under es_ES.
//   6. ImageSource (the tape path): a gray frame rebuilt from ROI patches + the recorded frame_ref scores exactly as
//      the full frame does inside the patch (prepare_frame's blur + Sobel see the same pixels).
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_core_existence.cpp \
//       src/core/door_core.cpp src/core/door_fit.cpp src/core/door_existence.cpp src/core/door_cycle_io.cpp \
//       src/door_belief.cpp src/door_model.cpp src/door_frame_evidence.cpp ../common/contour_edge/contour_edge_check.cpp \
//       ../common/contour_edge/contour_edge_project.cpp ../common/contour_edge/contour_depth_check.cpp \
//       -lzstd -lopencv_core -lopencv_imgproc -o /tmp/test_door_core_existence && /tmp/test_door_core_existence
#include <clocale>
#include <cmath>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "../src/core/door_core.h"
#include "../src/core/door_existence.h"
#include "../../common/contour_edge/contour_edge_check.h"

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
constexpr int kW = 1280, kH = 720;
constexpr float kF = 700.f;
constexpr float kRobotY = 0.5f;    // 4.5 m from the door: its apparent size sits inside the detector envelope

static Mat4d pose(double x, double y, double z, bool turned_away)
{
    Mat4d T = Mat4d::Identity();
    if (turned_away) { T(0, 0) = -1.0; T(1, 1) = -1.0; }   // yaw 180°: zed +y points to -y
    T(0, 3) = x; T(1, 3) = y; T(2, 3) = z;
    return T;
}

// The door's projected pixel box from (1, kRobotY, 1.2) looking +y (the pinhole the core uses).
static void door_pixels(std::vector<Vec2f>& px)
{
    const float Y = kWallY - kRobotY;
    const float c0 = kW / 2 + kF * (kDoorX - 0.5f - 1.0f) / Y, c1 = kW / 2 + kF * (kDoorX + 0.5f - 1.0f) / Y;
    const float r0 = kH / 2 - kF * (2.1f - 1.2f) / Y, r1 = kH / 2 - kF * (0.0f - 1.2f) / Y;
    for (float r = r0; r < r1; r += 3.0f)
        for (float c = c0; c < c1; c += 3.0f)
            px.emplace_back(c, r);
}

enum class Phase { Present, Absent };

static DoorCycleInput cycle_input(std::uint64_t cycle, int frame_id, Phase phase, bool turned_away = false)
{
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = static_cast<std::int64_t>(cycle) * 100'000'000;   // 10 Hz
    in.t_wall_ms = 1'700'000'000'000 + static_cast<std::int64_t>(cycle) * 100;
    in.robot.present = true;
    in.robot.room_T_body = pose(1.0, kRobotY, 0.0, turned_away);
    in.robot.body_T_zed = pose(0.0, 0.0, 1.2, false);
    in.camera.present = true;
    in.camera.width = kW; in.camera.height = kH; in.camera.focal_x = kF; in.camera.focal_y = kF;

    MaskPacket p;
    p.valid = true;
    p.frame_id = frame_id;
    p.timestamp_ms = 1000 + static_cast<std::uint64_t>(frame_id);
    if (phase == Phase::Present)
    {
        for (int i = 0; i <= 20; ++i)
            for (int j = 0; j <= 21; ++j)
            {
                const float jitter = 0.004f * static_cast<float>(((i * 7 + j * 13) % 5) - 2);
                p.support_points.emplace_back(kDoorX - 0.5f + 0.05f * static_cast<float>(i), kWallY + jitter,
                                              0.1f * static_cast<float>(j));
            }
        MaskSlice s;
        s.label = "door"; s.class_id = 1000.f; s.confidence = 0.9f;
        s.centroid = {kDoorX, kWallY, 1.05f}; s.bbox_min = {kDoorX - 0.5f, kWallY - 0.01f, 0.0f};
        s.bbox_max = {kDoorX + 0.5f, kWallY + 0.01f, 2.1f};
        s.support_begin = 0; s.support_end = p.support_points.size();
        s.trunc_frac = 0.0f; s.range = kWallY - kRobotY; s.centroid_radius = 0.1f; s.source = 0; s.has_depth = true;
        door_pixels(p.mask_pixels);
        s.pixel_begin = 0; s.pixel_end = p.mask_pixels.size();
        p.slices.push_back(s);
    }
    else
    {
        // A small unrelated mask in the image corner: the frame carries mask pixels (so the silhouette channel can
        // speak) and none of them is on the door.
        MaskSlice s;
        s.label = "plant"; s.class_id = 5.f; s.confidence = 0.8f; s.source = 0; s.has_depth = true;
        for (int r = 0; r < 30; r += 3) for (int c = 0; c < 30; c += 3) p.mask_pixels.emplace_back(c, r);
        s.pixel_begin = 0; s.pixel_end = p.mask_pixels.size();
        p.slices.push_back(s);
    }
    in.masks = std::move(p);
    return in;
}

struct Run
{
    std::vector<rc::door::DoorDecision> exist, removes;
    std::vector<std::string> csv;
    std::vector<std::string> order;     // "REMOVE <id>" interleaved with the world's events
    int birth_cycle = -1;
};

// `present` cycles with the door lit, then `absent` cycles without; `repeat_at` > 0 re-sends that cycle's frame id.
static Run run(int present, int absent, bool turned_away = false, int repeat_at = -1, ExistWorld* out_world = nullptr)
{
    DoorCoreConfig cfg;
    ExistWorld local;
    ExistWorld& w = out_world ? *out_world : local;
    w.cfg = cfg.door;
    DoorCore core(cfg);
    w.core = &core;
    Run r;
    core.connect(&w, {.decision = [&](const rc::door::DoorDecision& d)
                      {
                          if (d.kind == static_cast<std::uint16_t>(K::Remove))
                              w.events.push_back("REMOVE " + std::to_string(d.door_id));
                      },
                      .diag = [&](const DiagRow& d) { if (d.stream == "door_existence_log.csv") r.csv.push_back(d.line); }});
    int frame = 0;
    for (int c = 1; c <= present + absent; ++c)
    {
        const bool repeat = c == repeat_at;
        if (not repeat) ++frame;
        const Phase ph = c <= present ? Phase::Present : Phase::Absent;
        const auto out = core.step(cycle_input(static_cast<std::uint64_t>(c), frame, ph, turned_away and c > present));
        for (const auto& d : out.decisions)
        {
            if (d.kind == static_cast<std::uint16_t>(K::Existence)) r.exist.push_back(d);
            if (d.kind == static_cast<std::uint16_t>(K::Remove)) r.removes.push_back(d);
            if (d.kind == static_cast<std::uint16_t>(K::Birth)) r.birth_cycle = c;
        }
    }
    r.order = w.events;
    w.core = nullptr;
    return r;
}

static std::vector<double> L_of(const Run& r) { std::vector<double> v; for (const auto& d : r.exist) v.push_back(d.a); return v; }

int main()
{
    if (std::setlocale(LC_ALL, "es_ES.UTF-8") == nullptr) std::setlocale(LC_ALL, "");
    const rc::DoorConfig dcfg;
    const double L_remove = std::log(dcfg.exist_removal_prob / (1.0 - dcfg.exist_removal_prob));

    // ── 1. present → absent: L to the clamp, then down, then ONE removal through the world ──────────────────────────
    {
        ExistWorld w;
        const Run r = run(40, 80, false, -1, &w);
        const auto L = L_of(r);
        std::print("birth at cycle {}, {} Existence rows, {} REMOVE\n", r.birth_cycle, r.exist.size(), r.removes.size());
        CHECK(r.birth_cycle > 0);
        CHECK(not L.empty());
        double L_max = -1e9;
        std::size_t i_max = 0;
        for (std::size_t i = 0; i < L.size(); ++i) if (L[i] > L_max) { L_max = L[i]; i_max = i; }
        if (not L.empty())
            std::print("L first {:.3f}, max {:.3f} (row {}), last {:.3f}\n", L.front(), L_max, i_max, L.back());
        CHECK(L_max >= dcfg.exist_max_logodds - 1e-3);        // confirmed to the clamp while lit
        CHECK(r.removes.size() == 1);
        CHECK(w.st.empty());                                   // the world removed it
        if (r.removes.size() == 1)
        {
            const auto& rm = r.removes.front();
            CHECK(rm.cycle > 40);                              // only after the door stopped being lit
            CHECK(rm.note.starts_with("logodds "));
            // the order: before_remove → the REMOVE decision → remove_door
            const std::vector<std::string> want{"before " + std::to_string(rm.door_id), "REMOVE " + std::to_string(rm.door_id),
                                                "remove " + std::to_string(rm.door_id)};
            CHECK(r.order == want);
            // the removal cycle's Existence row carries the belief that removed it
            const auto& last = r.exist.back();
            CHECK(last.cycle == rm.cycle and last.door_id == rm.door_id);
            CHECK(last.a <= L_remove + 1e-6);
            CHECK(last.b == 1.0 / (1.0 + std::exp(-last.a)) or std::abs(last.b - 1.0 / (1.0 + std::exp(-last.a))) < 1e-6);
        }
        // CSV rows exist (the throttle forces one on the removal cycle) and are classic-locale under es_ES
        CHECK(not r.csv.empty());
        if (not r.csv.empty())
        {
            std::print("existence csv last row: {}", r.csv.back());
            CHECK(r.csv.back().find('.') != std::string::npos);
            CHECK(r.csv.back().back() == '\n');
        }
    }

    // ── 2. one Existence row per NEW frame: a repeated frame id integrates nothing ─────────────────────────────────
    {
        const Run a = run(40, 0);
        const Run b = run(40, 0, false, /*repeat_at=*/30);
        CHECK(not a.exist.empty());
        CHECK(b.exist.size() + 1 == a.exist.size());
        bool row_at_30 = false;
        for (const auto& d : b.exist) row_at_30 = row_at_30 or d.cycle == 30;
        CHECK(not row_at_30);
    }

    // ── 3. turned away: the door is not probed, L HOLDS, nothing is removed ────────────────────────────────────────
    {
        const Run t = run(40, 60, /*turned_away=*/true);
        CHECK(t.removes.empty());
        double L40 = 0, Lend = 0;
        for (const auto& d : t.exist) { if (d.cycle == 40) L40 = d.a; Lend = d.a; }
        std::print("turned away: L at 40 {:.4f}, at the end {:.4f}\n", L40, Lend);
        CHECK(Lend == L40);
    }

    // ── 4. determinism ─────────────────────────────────────────────────────────────────────────────────────────────
    {
        const Run a = run(40, 80), b = run(40, 80);
        CHECK(not a.exist.empty());
        CHECK(a.exist == b.exist);
        CHECK(a.removes == b.removes);
        CHECK(a.csv == b.csv);
    }

    // ── 6. ImageSource: patches + the recorded frame_ref reproduce the full frame inside the patch ──────────────────
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
        src.apply(g, std::nullopt);
        CHECK(src.rgb().cols == full.cols and src.rgb().rows == full.rows);
        CHECK(src.frame_ref().has_value() and src.frame_ref().value() == ref.frame_ref);
        auto prep = rc::edges::prepare_frame(src.rgb());
        if (src.frame_ref().has_value()) prep.frame_ref = src.frame_ref().value();
        bool same = true;
        for (int y = p.y0 + 6; y < p.y0 + p.h - 6; ++y)
            for (int x = p.x0 + 6; x < p.x0 + p.w - 6; ++x)
                same = same and prep.gx.at<float>(y, x) == ref.gx.at<float>(y, x) and prep.gy.at<float>(y, x) == ref.gy.at<float>(y, x);
        CHECK(same);
        CHECK(prep.frame_ref == ref.frame_ref);
    }

    if (fails == 0) std::print("test_door_core_existence: ALL PASS\n");
    else            std::print("test_door_core_existence: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
