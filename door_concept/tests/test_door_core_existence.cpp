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
#include <charconv>
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

// The door's projected pixel box from (robot_x, kRobotY, 1.2) looking +y (the pinhole the core uses), the leaf at leaf_y.
static void door_pixels(std::vector<Vec2f>& px, float leaf_y = kWallY, float robot_x = 1.0f)
{
    const float Y = leaf_y - kRobotY;
    const float c0 = kW / 2 + kF * (kDoorX - 0.5f - robot_x) / Y, c1 = kW / 2 + kF * (kDoorX + 0.5f - robot_x) / Y;
    const float r0 = kH / 2 - kF * (2.1f - 1.2f) / Y, r1 = kH / 2 - kF * (0.0f - 1.2f) / Y;
    for (float r = r0; r < r1; r += 3.0f)
        for (float c = c0; c < c1; c += 3.0f)
            px.emplace_back(c, r);
}

enum class Phase { Present, Absent };
// Task D-C: the robot↔room RT covariance the inputs carry (σ of x, y in m and of yaw in rad); < 0 ⇒ none recorded.
static double g_pose_sd_xy = -1.0, g_pose_sd_yaw = -1.0;

// leaf_y: where the leaf really is (Task 5b: the wall line is the room polygon's face, the leaf may stand off it); robot_x:
// the robot's x (1 = square-on; off it the view is oblique and a plane at the wrong depth projects to the wrong pixels).
static DoorCycleInput cycle_input(std::uint64_t cycle, int frame_id, Phase phase, bool turned_away = false,
                                  float leaf_y = kWallY, float robot_x = 1.0f, bool sweep = false)
{
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = static_cast<std::int64_t>(cycle) * 100'000'000;   // 10 Hz
    in.t_wall_ms = 1'700'000'000'000 + static_cast<std::int64_t>(cycle) * 100;
    in.robot.present = true;
    in.robot.room_T_body = pose(robot_x, kRobotY, 0.0, turned_away);
    in.robot.body_T_zed = pose(0.0, 0.0, 1.2, false);
    if (g_pose_sd_xy >= 0.0)
    {
        Mat6d C = Mat6d::Zero();
        C(0, 0) = C(1, 1) = g_pose_sd_xy * g_pose_sd_xy;
        C(2, 2) = C(3, 3) = C(4, 4) = 1e-6;
        C(5, 5) = g_pose_sd_yaw * g_pose_sd_yaw;
        in.robot.rt_cov = C;
    }
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
                p.support_points.emplace_back(kDoorX - 0.5f + 0.05f * static_cast<float>(i), leaf_y + jitter,
                                              0.1f * static_cast<float>(j));
            }
        MaskSlice s;
        s.label = "door"; s.class_id = 1000.f; s.confidence = 0.9f;
        s.centroid = {kDoorX, leaf_y, 1.05f}; s.bbox_min = {kDoorX - 0.5f, leaf_y - 0.01f, 0.0f};
        s.bbox_max = {kDoorX + 0.5f, leaf_y + 0.01f, 2.1f};
        s.support_begin = 0; s.support_end = p.support_points.size();
        s.trunc_frac = 0.0f; s.range = leaf_y - kRobotY; s.centroid_radius = 0.1f; s.source = 0; s.has_depth = true;
        door_pixels(p.mask_pixels, leaf_y, robot_x);
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
    if (sweep and phase == Phase::Present)
    {
        // the low LiDAR (as test_door_core_fit): rays stop on the wall beside the doorway and on the SHUT leaf inside it —
        // the WALL at the leaf's plane too: the polygon line is off the real wall (tourR2's DOOR_0 wall registered 5-7 cm
        // into the room, its leaf 0.10-0.15 m), which is what the LiDAR registration measures and the mask fit's d carries
        BpearlSweep b;
        b.room_fresh = true;
        b.origin_room = {robot_x, kRobotY, 0.35f};
        for (int k = 0; k <= 80; ++k)
        {
            const float x = kDoorX - 2.0f + 0.05f * static_cast<float>(k);
            const bool in_aperture = x > kDoorX - 0.48f and x < kDoorX + 0.48f;
            for (const float z : {0.4f, 0.8f})
                b.pts_room.emplace_back(x, leaf_y, z);
        }
        in.bpearl = std::move(b);
    }
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
// sweep: the robot walks a triangle wave x ∈ 1 ± 1.5 m (period 12 frames) — every few frames a NEW view of the door, so
// evidence is from new looks, not one look repeated (Task 5b item 4: the existence channels pool the looks of one view —
// door_view_pool.h — so a parked robot's repeats are one observation). false = parked.
static Run run(int present, int absent, bool turned_away = false, int repeat_at = -1, ExistWorld* out_world = nullptr,
               bool sweep = true)
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
        float rx = 1.0f;
        if (sweep)
        {
            const int k = c % 12;
            rx = 1.0f + 1.5f * (k < 6 ? (static_cast<float>(k) / 3.0f - 1.0f) : (1.0f - static_cast<float>(k - 6) / 3.0f));
        }
        const auto out = core.step(cycle_input(static_cast<std::uint64_t>(c), frame, ph, turned_away and c > present, kWallY, rx));
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
        const Run r = run(40, 200, false, -1, &w);   // 200: the sweep spends most frames off-centre (P(detect) ≈ 0.03-0.3)
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

    // ── 7. Task 5b: a shut door 0.12 m OFF the polygon's wall line on the room side (the tapes' DOOR_0), seen obliquely ──
    //    The fit must put the leaf's plane there (DoorBeliefState::d ≈ −0.12, out of the room is +) without stretching w,
    //    and the silhouette — rendered on that plane — must read the leaf OCCUPIED: occ ≫ free on its rows, L confirmed.
    for (const float robot_x : {1.0f, -1.0f})
    {
        DoorCoreConfig cfg;
        ExistWorld w;
        w.cfg = cfg.door;
        DoorCore core(cfg);
        w.core = &core;
        std::vector<std::string> csv;
        std::vector<double> L;
        core.connect(&w, {.decision = [&](const rc::door::DoorDecision& d)
                          { if (d.kind == static_cast<std::uint16_t>(K::Existence)) L.push_back(d.a); },
                          .diag = [&](const DiagRow& d) { if (d.stream == "door_existence_log.csv") csv.push_back(d.line); }});
        for (int c = 1; c <= 60; ++c)
            (void)core.step(cycle_input(static_cast<std::uint64_t>(c), c, Phase::Present, false, kWallY - 0.12f, robot_x, true));
        CHECK(w.st.size() == 1);
        if (w.st.size() != 1) continue;
        const auto& b = w.st.begin()->second.ai2_belief;
        const float sd_d = b.recess_sd(), sd_w = std::sqrt(b.covariance()(1, 1));
        std::print("door off the line, robot x {:+.0f}: d {:+.3f}±{:.3f} (truth -0.12), w {:.3f}±{:.3f} (truth 1.00), phi {:.3f}, L {:.2f}\n",
                   robot_x, b.recess(), sd_d, b.state().w, sd_w, b.params().leaf.phi, L.empty() ? -99.0 : L.back());
        // 3 σ, not 2: square-on, the low LiDAR's leaf angle shares the off-line returns with d for a while (phi 0.11 rad
        // there, 0.03 oblique) — the field sees the plane only through d, not as a quantity of its own (report, concerns).
        CHECK(std::abs(b.recess() + 0.12f) <= 3.0f * sd_d);
        CHECK(std::abs(b.state().w - 1.0f) <= 2.0f * sd_w + 0.01f);
        // a parked camera's looks pool to ONE observation (door_view_pool.h): confirmed above the birth seed, not to the clamp
        CHECK(not L.empty() and L.back() > dcfg.exist_birth_logodds);
        // the silhouette's own counts on every forced CSV row: sil_occ (col 11) ≫ sil_free (col 12)
        const auto col = [](const std::string& line, int k)
        {
            std::size_t a = 0;
            for (int i = 0; i < k; ++i) a = line.find(',', a) + 1;
            double v = 0.0;
            std::from_chars(line.data() + a, line.data() + line.find(',', a), v);
            return v;
        };
        int n_rows = 0, n_occ = 0;
        for (const auto& line : csv)
        {
            const double occ = col(line, 10), fr = col(line, 11);
            if (occ + fr <= 0.0) continue;
            ++n_rows;
            n_occ += occ >= 5.0 * fr ? 1 : 0;
            std::print("    sil_occ {:.1f} sil_free {:.1f}\n", occ, fr);
        }
        CHECK(n_rows > 0 and n_occ == n_rows);
    }

    // ── 8. Task 5b item 4: a PARKED robot's repeated looks are one look repeated (Existence.FrameCorrelation, measured
    //    0.78 on tourR1+R2 parked pairs; the looks of one view are pooled, door_view_pool.h — a moving phantom is
    //    removed: tests/test_door_existence_moving_phantom.cpp): the parked
    //    absence that removes the door from NEW views (section 1, the sweep) must NOT remove it from one unchanging view,
    //    and L stays above the removal boundary.
    {
        ExistWorld w;
        const Run r = run(40, 80, false, -1, &w, /*sweep=*/false);
        const auto L = L_of(r);
        std::print("parked absence: {} REMOVE, L last {:.3f} (removal boundary {:.3f})\n", r.removes.size(),
                   L.empty() ? -99.0 : L.back(), L_remove);
        CHECK(r.removes.empty());
        CHECK(not L.empty() and L.back() > L_remove);
    }

    // ── 9. Task D-C (rtD1i door_5): a door RESTORED below the removal bound is removed only by data that push it further
    //    down. It entered at L0 = −3.5 (a remembered −4 decayed by the hazard, door_identity_table.h) and the parked looks
    //    that follow are net POSITIVE (the door is lit, pooled by view to one observation): the old rule counted those looks
    //    as sustained condemnation (the prior alone sat below logit(RemovalProb)) and removed it; now it must survive. The
    //    same restored door then truly ABSENT from new views (the sweep) is still removed.
    for (const bool absent_after : {false, true})
    {
        DoorCoreConfig cfg;
        ExistWorld w;
        w.cfg = cfg.door;
        DoorCore core(cfg);
        w.core = &core;
        std::vector<rc::door::DoorDecision> removes;
        std::vector<double> Ls;
        core.connect(&w, {.decision = [&](const rc::door::DoorDecision& d)
                          {
                              if (d.kind == static_cast<std::uint16_t>(K::Remove)) removes.push_back(d);
                              if (d.kind == static_cast<std::uint16_t>(K::Existence)) Ls.push_back(d.a);
                          }, .diag = {}});
        int c = 0;
        for (; c < 40; ++c)   // born and fitted, parked square-on
            (void)core.step(cycle_input(static_cast<std::uint64_t>(c + 1), c + 1, Phase::Present));
        CHECK(w.st.size() == 1);
        if (w.st.size() != 1) continue;
        auto& inst = w.st.begin()->second;
        // the restore (rc::door::seed_restored_instance's existence half): the remembered prior, the debounce afresh
        inst.existence.set(-3.5f);
        inst.existence_entry_L = inst.existence.logodds();
        inst.existence_debounce = {};
        inst.silhouette_view_pool = {};
        inst.image_view_pool = {};
        const double L_entry = inst.existence.logodds();
        Ls.clear();
        for (int k = 0; k < 240; ++k, ++c)
        {
            float rx = 1.0f;
            if (absent_after) { const int q = k % 12; rx = 1.0f + 1.5f * (q < 6 ? (static_cast<float>(q) / 3.0f - 1.0f) : (1.0f - static_cast<float>(q - 6) / 3.0f)); }
            (void)core.step(cycle_input(static_cast<std::uint64_t>(c + 1), c + 1, absent_after ? Phase::Absent : Phase::Present,
                                        false, kWallY, rx));
            if (not removes.empty()) break;
        }
        double L_min = 1e9, L_max = -1e9;
        for (const double v : Ls) { L_min = std::min(L_min, v); L_max = std::max(L_max, v); }
        std::print("restored at L {:.2f} then {}: {} REMOVE, L range [{:.2f}, {:.2f}] (bound {:.2f})\n", L_entry,
                   absent_after ? "absent from new views" : "lit, parked", removes.size(), L_min, L_max, L_remove);
        if (absent_after)
            CHECK(removes.size() == 1);                       // data against it since entry ⇒ removed
        else
        {
            CHECK(L_min >= L_entry - 1e-6);                   // the data since entry are net positive …
            CHECK(removes.empty());                           // … so the remembered prior alone does not remove it
        }
    }

    // ── 10. Task D-C (rtD1i door_5): THE IMAGE CHANNELS ARE WEIGHTED BY THE POSE'S UNCERTAINTY. The door was removed on
    //     looks taken from a pose 4 m wrong. pose_overlap_factor κ — the expected fraction of the predicted image that lands
    //     on the door when the bearing is uncertain — is 1 for a precise pose and falls toward 0 as the pose spreads, and it
    //     multiplies P(detect) and the image evidence: the same absence (section 1's sweep) that removes the door from a
    //     well-localised robot (σ 3 cm, 0.6°: the tapes' typical RT covariance) must not move it from a robot whose pose is
    //     spread over metres.
    {
        const Mat4d cam = pose(1.0, kRobotY, 1.2, false);
        const Eigen::Vector2f door(kDoorX, kWallY);
        const auto k_of = [&](double sxy, double syaw)
        {
            Mat6d C = Mat6d::Zero(); C(0, 0) = C(1, 1) = sxy * sxy; C(5, 5) = syaw * syaw;
            return pose_overlap_factor(C, cam, door, 0.5f * std::hypot(1.0f, 2.1f)).kappa;
        };
        const float k0 = k_of(0.0, 0.0), k3 = k_of(0.03, 0.0105), k30 = k_of(0.30, 0.05), k4 = k_of(4.0, 1.0);
        std::print("pose overlap κ: exact {:.4f}, σ 3 cm/0.6° {:.4f}, 30 cm/3° {:.4f}, 4 m/57° {:.4f}; no covariance {:.1f}\n",
                   k0, k3, k30, k4, pose_overlap_factor(std::nullopt, cam, door, 1.0f).kappa);
        CHECK(k0 == 1.0f);
        CHECK(k3 > 0.97f and k3 < 1.0f);
        CHECK(k30 < k3 and k4 < k30);
        CHECK(k4 < 0.15f);
        CHECK(pose_overlap_factor(std::nullopt, cam, door, 1.0f).kappa == 1.0f);
        CHECK(pose_overlap_factor(std::nullopt, cam, door, 1.0f).spread_m == 0.0f);
        for (const double sd : {0.03, 4.0})
        {
            g_pose_sd_xy = sd; g_pose_sd_yaw = sd < 1.0 ? 0.0105 : 1.0;
            ExistWorld w;
            const Run r = run(40, 200, false, -1, &w);
            const auto L = L_of(r);
            double L40 = 0.0;
            for (const auto& d : r.exist) if (d.cycle <= 40) L40 = d.a;
            std::print("absence from new views, pose σ {:.2f} m: {} REMOVE, L at the end of presence {:.2f}, last {:.2f}\n", sd,
                       r.removes.size(), L40, L.empty() ? -99.0 : L.back());
            if (sd < 1.0) CHECK(r.removes.size() == 1);
            else { CHECK(r.removes.empty()); CHECK(not L.empty() and L.back() > L_remove); }
        }
        g_pose_sd_xy = g_pose_sd_yaw = -1.0;
    }

    // ── 11. Task D-C (rtD1i door_5, visit 1): A DOOR HIDDEN BEHIND A PARTITION IS NOT SEEN, SO IT IS NOT ABSENT. DOOR_0's
    //     hinge jamb stands 7 cm from the corner where a partition meets its wall; rc::occlusion::walls_block skipped every
    //     edge within 0.30 m of a sample as "the target's own wall", so both faces of the partition were skipped for the
    //     hinge half of the door, and from behind the partition the frame and leaf samples there read VISIBLE and empty —
    //     the image channels walked the shut door +4 → −4 while it was hidden. Here: the door confirmed square-on, then the
    //     robot steps 2.2 m west, behind a partition whose face is 2 cm from the jamb (room polygon with that spur): the
    //     doorway frame and the leaf silhouette must be wholly HIDDEN from there (no sample visible ⇒ no absence charged),
    //     and visible again from the open side.
    {
        ExistWorld w;
        w.polygon = {{-5.f, -5.f}, {5.f, -5.f}, {5.f, 5.f}, {0.48f, 5.f}, {0.48f, 1.5f}, {0.38f, 1.5f}, {0.38f, 5.f}, {-5.f, 5.f}};
        DoorCoreConfig cfg;
        w.cfg = cfg.door;
        DoorCore core(cfg);
        w.core = &core;
        std::vector<rc::door::DoorDecision> removes;
        std::vector<double> Lh;
        core.connect(&w, {.decision = [&](const rc::door::DoorDecision& d)
                          {
                              if (d.kind == static_cast<std::uint16_t>(K::Remove)) removes.push_back(d);
                              if (d.kind == static_cast<std::uint16_t>(K::Existence)) Lh.push_back(d.a);
                          }, .diag = {}});
        int c = 1;
        for (; c <= 40; ++c)   // confirmed, the robot at (1.6, 0.5): the partition is west of its line of sight
            (void)core.step(cycle_input(static_cast<std::uint64_t>(c), c, Phase::Present, false, kWallY, 1.6f));
        CHECK(w.st.size() == 1);
        if (w.st.size() == 1)
        {
            // the doorway FRAME probe (jambs + lintel), seen from behind the partition: every sample hidden
            const auto& inst = w.st.begin()->second;
            const DoorCycleInput in = cycle_input(static_cast<std::uint64_t>(c), c, Phase::Absent, false, kWallY, -1.2f);
            const auto T = CameraModel::room_T_zed(in.robot);
            const std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> doors{{w.st.begin()->first, &w.st.begin()->second}};
            const ExistenceScene behind{cfg.door, T, in.camera, *in.masks, w.polygon, core.fit().room_band(), doors};
            const auto fp = compute_frame_probe(behind, inst);
            const auto sil = compute_silhouette_existence(behind, inst, nullptr);
            // and from the open side (x 1.6), the same probes see it
            const DoorCycleInput in2 = cycle_input(static_cast<std::uint64_t>(c), c, Phase::Absent, false, kWallY, 1.6f);
            const ExistenceScene open_side{cfg.door, CameraModel::room_T_zed(in2.robot), in2.camera, *in2.masks, w.polygon,
                                           core.fit().room_band(), doors};
            const auto fp2 = compute_frame_probe(open_side, inst);
            std::print("hidden behind a partition 2 cm from its jamb: frame visible {:.2f} ({} samples), silhouette detectable {}/{}; "
                       "from the open side frame visible {:.2f}\n", fp.vis_frac(), fp.samples.size(), sil.n_detectable, sil.n_total,
                       fp2.vis_frac());
            CHECK(fp.vis_frac() == 0.0f and fp.samples.empty());
            CHECK(sil.n_detectable == 0);
            CHECK(fp2.vis_frac() > 0.5f);
        }
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
