// test_door_absence_detectability.cpp — Task 15: ABSENCE IS WEIGHTED BY P(detect) IN EVERY CHANNEL, the image geometry included.
//
// Live recog_proto5 (2026-10-29): DOOR_2, restored as door_4 and OPEN (phi ≈ 85-90°), was removed "for absence" while the
// robot approached it for `cross door_4` (1.6-3 m from it), and the same hole was re-born at once as door_39 with a fresh,
// shut-looking belief. Its existence trace (door_existence_log.csv): L lost 34.7 nats over its life and only 0.28 of them on
// rows whose silhouette P(detect) exceeded 0.02 — the silhouette (the channel WITH a detection model) never charged it. The
// IMAGE-GEOMETRY channel (doorway frame + leaf contour/depth, door_view_pool.h) did: −1.90 (frame exc −9.2) at cycle 8549,
// −1.54 / −1.69 (frame dep −1.00) at 9147/9148 with p_detect 0.000 and the leaf fully occluded (occl 420), each new view a
// fresh refutation at the full llr = log(pd/pc); at L = −4 the wall-band prior's per-cycle look (1 − π = 0.78) completed the
// removal debounce in 26 cycles with no sensor look at all.
//
// The scene is test_door_core_existence.cpp's (a door in the top wall of a 10 x 10 m room, camera 1.2 m high looking +y),
// with a synthetic ZED DEPTH frame added so the image-geometry channel speaks:
//   CONFIRM  the hole reads DEEPER than the wall (a doorway: the next room behind it);
//   REFUTE   the hole reads NEARER than the wall (live 9148's dep −1.00: something stands in front of the opening).
//   A. the door_4 trace, synthesised: confirmed from the detectable far view (masks lit), then the robot APPROACHES to 1.1-2 m —
//      the door overflows the detector envelope (roi_fill > MaxFill ⇒ P(detect) ≈ 0), no mask lights it — while the depth
//      REFUTES it from a new view every frame. L must not fall below the level it had when the approach began (nothing the
//      detector could have resolved has said "absent"). Before the fix it walked to the −4 floor (RED).
//   B. A PHANTOM seen from DETECTABLE viewpoints (the far sweep, as test_door_core_existence §1 — the recog_try23/26 phantoms
//      door_11/door_12 on room_1's south wall were removed from such views) with the depth refuting: still REMOVED.
//   C. door_view_pool.h's look weight: w = 1 on every look is bit-identical to the unweighted pool; w = 0 changes nothing
//      (neither speaks nor dilutes); a fresh view's refutation at w = p says p of it.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_absence_detectability.cpp \
//       src/core/door_core.cpp src/core/door_fit.cpp src/core/door_existence.cpp src/core/door_cycle_io.cpp \
//       src/door_belief.cpp src/door_model.cpp src/door_frame_evidence.cpp ../common/contour_edge/contour_edge_check.cpp \
//       ../common/contour_edge/contour_edge_project.cpp ../common/contour_edge/contour_depth_check.cpp \
//       -lzstd -lopencv_core -lopencv_imgproc -o /tmp/test_door_absence_detectability && /tmp/test_door_absence_detectability
#include <clocale>
#include <cmath>
#include <cstring>
#include <limits>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>
#include "../src/core/door_core.h"
#include "../src/core/door_existence.h"
#include "../src/door_view_pool.h"

using namespace rc::door::core;
using K = rc::door::DoorDecisionKind;
#ifdef TASK15_RED_BASELINE
#define IMG_ABS_W(s) -9.0f   // the pre-Task-15 state has no such field
#else
#define IMG_ABS_W(s) (s).dbg_img_absence_w
#endif

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

struct ExistWorld final : TrackerWorld
{
    DoorCore* core = nullptr;
    rc::DoorConfig cfg;
    std::vector<Vec2f> polygon{{-5.f, -5.f}, {5.f, -5.f}, {5.f, 5.f}, {-5.f, 5.f}};
    mutable std::unordered_map<std::uint64_t, rc::DoorFitState> st;
    std::uint64_t next_id = 1000;
    std::uint64_t room_id() const override { return 7; }
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
    void before_remove(std::uint64_t) override {}
    void remove_door(std::uint64_t id, float) override { st.erase(id); }
};

constexpr float kDoorX = 1.0f, kWallY = 4.99f, kWallLine = 5.0f, kCamZ = 1.2f;
constexpr int kW = 1280, kH = 720;
constexpr float kF = 700.f;

static Mat4d pose(double x, double y, double z)
{
    Mat4d T = Mat4d::Identity();
    T(0, 3) = x; T(1, 3) = y; T(2, 3) = z;
    return T;
}

// The door's projected pixel box (the mask the detector would return) from (rx, ry) looking +y.
static void door_pixels(std::vector<Vec2f>& px, float rx, float ry)
{
    const float Y = kWallY - ry;
    const float c0 = kW / 2 + kF * (kDoorX - 0.5f - rx) / Y, c1 = kW / 2 + kF * (kDoorX + 0.5f - rx) / Y;
    const float r0 = kH / 2 - kF * (2.1f - kCamZ) / Y, r1 = kH / 2 - kF * (0.0f - kCamZ) / Y;
    for (float r = std::max(0.f, r0); r < std::min<float>(kH, r1); r += 3.0f)
        for (float c = std::max(0.f, c0); c < std::min<float>(kW, c1); c += 3.0f)
            px.emplace_back(c, r);
}

enum class Hole { Deeper, Nearer };

// The ZED depth frame (forward distance) from (rx, ry, kCamZ) looking +y: the wall plane y = kWallLine everywhere, and inside
// the aperture x ∈ [kDoorX ± 0.5], z ∈ [0, 2.1] either the next room 3 m behind it (Deeper) or a surface at 0.55 of the wall's
// distance in front of it (Nearer).
static rc::door::core::ImageFrame depth_frame(float rx, float ry, Hole hole, std::uint64_t stamp)
{
    rc::door::core::ImageFrame f;
    f.format = rc::door::core::PixelFormat::DepthF32;
    f.width = kW; f.height = kH; f.stamp_ms = stamp;
    rc::door::core::ImagePatch p;
    p.x0 = 0; p.y0 = 0; p.w = kW; p.h = kH;
    p.bytes.resize(static_cast<std::size_t>(kW) * kH * sizeof(float));
    std::vector<float> d(static_cast<std::size_t>(kW) * kH);
    const float t_wall = kWallLine - ry;
    for (int r = 0; r < kH; ++r)
        for (int c = 0; c < kW; ++c)
        {
            const float x = (static_cast<float>(c) - kW / 2) / kF, z = -(static_cast<float>(r) - kH / 2) / kF;
            const float X = rx + x * t_wall, Z = kCamZ + z * t_wall;
            float t = t_wall;
            if (X > kDoorX - 0.5f and X < kDoorX + 0.5f and Z > 0.0f and Z < 2.1f)
                t = hole == Hole::Deeper ? t_wall + 3.0f : 0.55f * t_wall;
            d[static_cast<std::size_t>(r) * kW + c] = t;
        }
    std::memcpy(p.bytes.data(), d.data(), p.bytes.size());
    f.patches.push_back(std::move(p));
    return f;
}

static DoorCycleInput cycle_input(std::uint64_t cycle, int frame_id, bool lit, float rx, float ry, Hole hole)
{
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = static_cast<std::int64_t>(cycle) * 100'000'000;
    in.t_wall_ms = 1'700'000'000'000 + static_cast<std::int64_t>(cycle) * 100;
    in.robot.present = true;
    in.robot.room_T_body = pose(rx, ry, 0.0);
    in.robot.body_T_zed = pose(0.0, 0.0, kCamZ);
    in.camera.present = true;
    in.camera.width = kW; in.camera.height = kH; in.camera.focal_x = kF; in.camera.focal_y = kF;
    MaskPacket p;
    p.valid = true;
    p.frame_id = frame_id;
    p.timestamp_ms = 1000 + static_cast<std::uint64_t>(frame_id);
    if (lit)
    {
        for (int i = 0; i <= 20; ++i)
            for (int j = 0; j <= 21; ++j)
            {
                const float jitter = 0.004f * static_cast<float>(((i * 7 + j * 13) % 5) - 2);
                p.support_points.emplace_back(kDoorX - 0.5f + 0.05f * static_cast<float>(i), kWallY + jitter, 0.1f * static_cast<float>(j));
            }
        MaskSlice s;
        s.label = "door"; s.class_id = 1000.f; s.confidence = 0.9f;
        s.centroid = {kDoorX, kWallY, 1.05f}; s.bbox_min = {kDoorX - 0.5f, kWallY - 0.01f, 0.0f};
        s.bbox_max = {kDoorX + 0.5f, kWallY + 0.01f, 2.1f};
        s.support_begin = 0; s.support_end = p.support_points.size();
        s.trunc_frac = 0.0f; s.range = kWallY - ry; s.centroid_radius = 0.1f; s.source = 0; s.has_depth = true;
        door_pixels(p.mask_pixels, rx, ry);
        s.pixel_begin = 0; s.pixel_end = p.mask_pixels.size();
        p.slices.push_back(s);
    }
    else
    {
        // a small unrelated mask in the image corner: the frame carries mask pixels (the silhouette can speak), none on the door
        MaskSlice s;
        s.label = "plant"; s.class_id = 5.f; s.confidence = 0.8f; s.source = 0; s.has_depth = true;
        for (int r = 0; r < 30; r += 3) for (int c = 0; c < 30; c += 3) p.mask_pixels.emplace_back(c, r);
        s.pixel_begin = 0; s.pixel_end = p.mask_pixels.size();
        p.slices.push_back(s);
    }
    in.masks = std::move(p);
    in.depth = depth_frame(rx, ry, hole, 1000 + static_cast<std::uint64_t>(frame_id));
    return in;
}

struct Look { float rx, ry; bool lit; Hole hole; };
struct Trace
{
    std::vector<double> L;   // one per Existence decision
    std::vector<int> cycle;
    int removes = 0;
    double L_at(int c) const { double v = std::numeric_limits<double>::quiet_NaN(); for (std::size_t i = 0; i < L.size(); ++i) if (cycle[i] <= c) v = L[i]; return v; }
};

// Drive the core over `looks`; print a few channel numbers every `every` cycles from `print_from` on.
// top_y: the room polygon's top face (the wall line the door is in).
static Trace drive(const std::vector<Look>& looks, int print_from, int every, float top_y = kWallLine)
{
    DoorCoreConfig cfg;
    ExistWorld w;
    w.cfg = cfg.door;
    w.polygon = {{-5.f, -5.f}, {5.f, -5.f}, {5.f, top_y}, {-5.f, top_y}};
    DoorCore core(cfg);
    w.core = &core;
    Trace t;
    core.connect(&w, {.decision = [&](const rc::door::DoorDecision& d)
                      {
                          if (d.kind == static_cast<std::uint16_t>(K::Existence)) { t.L.push_back(d.a); t.cycle.push_back(static_cast<int>(d.cycle)); }
                          if (d.kind == static_cast<std::uint16_t>(K::Remove)) ++t.removes;
                      },
                      .diag = [](const DiagRow&) {}});
    for (std::size_t k = 0; k < looks.size(); ++k)
    {
        const int c = static_cast<int>(k) + 1;
        const auto& lk = looks[k];
        (void)core.step(cycle_input(static_cast<std::uint64_t>(c), c, lk.lit, lk.rx, lk.ry, lk.hole));
        if (c >= print_from and (c - print_from) % every == 0)
            for (const auto& [id, s] : w.st)
                std::print("    cycle {:3d} robot ({:+.2f},{:.2f}) L {:+.2f} | p_detect {:.3f} p_env {:.2f} ndet {} | img dL {:+.2f} w_abs {:+.2f} "
                           "frame vis {:.2f} dep {:+.2f} | band p {:.2f} streak {:.2f}\n", c, lk.rx, lk.ry, s.existence.logodds(), s.dbg_sil_pdetect,
                           s.dbg_p_env, s.dbg_sil_ndet, s.dbg_edge_delta, IMG_ABS_W(s), s.dbg_frame_vis, s.dbg_frame_depth, s.dbg_band_p,
                           s.existence_debounce.streak);
    }
    w.core = nullptr;
    return t;
}

// the far, detectable sweep of test_door_core_existence (x ∈ 1 ± 1.5 m, period 12, 4.5 m from the door)
static float sweep_x(int c)
{
    const int k = c % 12;
    return 1.0f + 1.5f * (k < 6 ? (static_cast<float>(k) / 3.0f - 1.0f) : (1.0f - static_cast<float>(k - 6) / 3.0f));
}

int main()
{
    if (std::setlocale(LC_ALL, "es_ES.UTF-8") == nullptr) std::setlocale(LC_ALL, "");
    const rc::DoorConfig dcfg;
    const double L_remove = std::log(dcfg.exist_removal_prob / (1.0 - dcfg.exist_removal_prob));

    // ── C. the view pool's look weight ─────────────────────────────────────────────────────────────────────────────────
    // (-DTASK15_RED_BASELINE: build against the pre-Task-15 sources, whose pool has no weight, to show A failing.)
#ifndef TASK15_RED_BASELINE
    {
        // the unweighted pool, as it was before Task 15 (door_view_pool.h at 9d45e5ef), for the bit-identity check
        struct Old
        {
            bool have = false; float bearing = 0, range = 0, n = 0, mean = 0, counted = 0;
            float add(const Eigen::Vector2f& obs, const Eigen::Vector2f& tgt, float R, float v, float llr)
            {
                const Eigen::Vector2f dv = obs - tgt; const float d = dv.norm();
                const float b = std::atan2(dv.y(), dv.x()); const float Rr = std::max(R, 1e-3f);
                float s = 0.0f;
                if (have)
                {
                    const float half = std::max(std::atan2(Rr, d), 1e-3f);
                    const float db = std::remainder(b - bearing, 2.0f * std::numbers::pi_v<float>);
                    const float dr = d - range;
                    s = std::exp(-std::min((db / half) * (db / half) + (dr / Rr) * (dr / Rr), 60.0f));
                }
                const float n_old = s * n; n = n_old + 1.0f; mean = (n_old * mean + v) / n;
                if (have) { bearing += std::remainder(b - bearing, 2.0f * std::numbers::pi_v<float>) / n; range += (d - range) / n; }
                else { bearing = b; range = d; have = true; }
                const float c = llr * std::tanh(mean);
                const float pending = c - s * counted; counted = c; return pending;
            }
        } old;
        rc::door::ViewPool pool;
        bool identical = true;
        const Eigen::Vector2f door(1.0f, 5.0f);
        for (int k = 0; k < 200; ++k)
        {
            const Eigen::Vector2f cam(sweep_x(k) + 0.01f * static_cast<float>(k % 7), 0.5f + 0.02f * static_cast<float>(k % 11));
            const float v = std::sin(0.37f * static_cast<float>(k)) * 2.0f;
            const float a = old.add(cam, door, 1.17f, v, 2.83f);
            const float b = pool.add(cam, door, 1.17f, v, 2.83f);   // default w = 1
            pool.commit(b);
            identical = identical and a == b;
        }
        std::print("C. w = 1 on every look: {} to the unweighted pool over 200 looks\n", identical ? "bit-identical" : "DIFFERENT");
        CHECK(identical);

        rc::door::ViewPool p0;
        const float c0 = p0.add({1.0f, 0.5f}, door, 1.17f, +2.0f, 2.83f);
        p0.commit(c0);
        const auto before = p0;
        const float z = p0.add({1.0f, 0.5f}, door, 1.17f, -5.0f, 2.83f, 0.0f);   // the same view, refuted unresolvably
        std::print("C. w = 0: ΔL {:+.3f}, pool n {:.3f}→{:.3f} mean {:+.3f}→{:+.3f}\n", z, before.n, p0.n, before.mean, p0.mean);
        CHECK(z == 0.0f and p0.n == before.n and p0.mean == before.mean and p0.counted == before.counted);

        rc::door::ViewPool pf;
        const float full = rc::door::ViewPool{}.add({1.0f, 0.5f}, door, 1.17f, -1.0f, 2.83f);
        const float part = pf.add({1.0f, 0.5f}, door, 1.17f, -1.0f, 2.83f, 0.25f);
        std::print("C. a fresh view's refutation: w 1 → {:+.3f}, w 0.25 → {:+.3f}\n", full, part);
        CHECK(std::abs(part - 0.25f * full) < 1e-5f);
    }
#endif

    // ── A. the door_4 trace: confirmed far, then APPROACHED where it is undetectable, the depth refuting every new view ──
    //    (The live REMOVE was then completed by the wall-band prior's look, door_4 sitting 0.2 m inside room_3's polygon; this
    //    scene's fit seats the aperture on the polygon face, π = 1, so the RED here is L at the −4 floor without the REMOVE.)
    {
        const float top_y = kWallLine;
        std::vector<Look> looks;
        for (int c = 1; c <= 40; ++c) looks.push_back({sweep_x(c), 0.5f, true, Hole::Deeper});
        constexpr int kApproach = 60;
        for (int k = 0; k < kApproach; ++k)
        {
            // y 3.0 → 3.9 (2.0 → 1.1 m from the wall), x swinging ±0.8 m: a new view of the door every frame or two
            const float ry = 3.0f + 0.9f * static_cast<float>(k) / static_cast<float>(kApproach - 1);
            const int m = k % 8;
            const float rx = kDoorX + 0.8f * (m < 4 ? (static_cast<float>(m) / 2.0f - 1.0f) : (1.0f - static_cast<float>(m - 4) / 2.0f));
            looks.push_back({rx, ry, false, Hole::Nearer});
        }
        std::print("A. polygon face y={:.2f}: confirmed from the far sweep, then approached to 1.1-2 m (undetectable), the hole reading NEARER:\n", top_y);
        const Trace t = drive(looks, 40, 6, top_y);
        const double L_before = t.L_at(40);
        double L_min = 1e9;
        for (std::size_t i = 0; i < t.L.size(); ++i) if (t.cycle[i] > 40) L_min = std::min(L_min, t.L[i]);
        std::print("A. L when the approach began {:+.3f}; lowest during it {:+.3f}; removal bound {:+.3f}; REMOVE {}\n",
                   L_before, L_min, L_remove, t.removes);
        CHECK(std::isfinite(L_before) and L_before > L_remove);
        CHECK(t.removes == 0);
        // Nothing the detector could resolve said "absent": the belief holds where the approach found it. (The first approach
        // frame is still marginally detectable — p_detect 0.005 — and charges its due, ~0.01 nats; before the fix L hit −4.)
        CHECK(L_min >= L_before - 0.05);
    }

    // ── B. a PHANTOM seen from DETECTABLE viewpoints, the depth refuting: still removed ──────────────────────────────────
    {
        std::vector<Look> looks;
        for (int c = 1; c <= 40; ++c) looks.push_back({sweep_x(c), 0.5f, true, Hole::Deeper});
        for (int c = 41; c <= 240; ++c) looks.push_back({sweep_x(c), 0.5f, false, Hole::Nearer});
        std::print("B. a phantom: lit 40 frames, then absent from the detectable far sweep with the hole reading NEARER:\n");
        const Trace t = drive(looks, 41, 40);
        std::print("B. REMOVE {} (last L {:+.3f})\n", t.removes, t.L.empty() ? -99.0 : t.L.back());
        CHECK(t.removes == 1);
    }

    std::print("{}\n", fails == 0 ? "ALL PASS" : std::format("{} FAILURE(S)", fails));
    return fails == 0 ? 0 : 1;
}
