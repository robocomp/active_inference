// test_door_absence_detectability.cpp — Task 15: ABSENCE IS WEIGHTED BY P(detect) IN EVERY CHANNEL, the image geometry included,
// and a static prior is never a look.
//
// Live recog_proto5 (2026-10-29): DOOR_2, restored as door_4 and OPEN (phi ≈ 85-90°), was removed "for absence" while the
// robot approached it for `cross door_4` (1.6-3 m from it), and the same hole was re-born at once as door_39 with a fresh,
// shut-looking belief. Its existence trace (door_existence_log.csv): L lost 34.7 nats over its life and only 0.28 of them on
// rows whose silhouette P(detect) exceeded 0.02. The IMAGE-GEOMETRY channel (doorway frame + leaf contour/depth,
// door_view_pool.h) had no detection model: −1.90 (frame exc −9.2, vis 0.43) at cycle 8549, −1.54 / −1.69 (frame dep −1.00) at
// 9147/9148 with p_detect 0.000 and the leaf occluded (occl 420); and once L sat at −4 the WALL-BAND PRIOR (door_4 0.2 m inside
// room_3's polygon, π 0.15-0.23) advanced the removal debounce by 1 − π every cycle and completed it in 26 cycles with no
// sensor look at all.
//
// The scene is test_door_core_existence.cpp's (a door in the top wall y = 5 of a 10 x 10 m room, camera 1.2 m high looking +y)
// with a synthetic ZED DEPTH frame so the image-geometry channel speaks:
//   DEEPER   the hole reads deeper than the wall (a doorway: the next room behind it)               — confirms
//   NEARER   the hole reads nearer than the wall (live 9148's dep −1.00: something stands in it)     — refutes
//   FLAT     a blank wall, the would-be opening ±4 cm off it, the sign alternating frame to frame   — zero-mean
//   A1. envelope overflow: confirmed from the far sweep, then approached and PASSED THROUGH (3.0 → 6.0 m, the opening at
//       y = 5) from views where the doorway overflows the detector envelope, the depth refuting: L must not fall.
//   A2. the live mechanism: after the confirmation room_3's polygon is re-estimated 0.2 m beyond the door (band π ≈ 0.15, the
//       term enters L once), then the robot approaches from 3 m to 1.6 m with a PERSON covering the lower part of the doorway
//       (the silhouette occluded, the frame partly visible) and the hole reading nearer, and passes through: NOT removed (the
//       band prior is not a look), L not below the level the prior left it at.
//   A3. a BLANK-WALL phantom approached at close range with zero-mean frame noise: L must not RISE (a weight on one sign only
//       is a ratchet).
//   B.  a phantom refuted from DETECTABLE viewpoints (the far sweep): still REMOVED — and the image channel's refutations
//       there count (Σ of its negative ΔL < −1, look weight > 0.2), not only the silhouette's.
//   D.  recog_try23 door_11: a phantom 0.6 m INSIDE the polygon (π ≈ 0) is removed by its prior L plus the real detectable
//       looks of its absence; the same phantom never looked at (camera turned away) is NOT removed by the prior alone.
//   C.  door_view_pool.h's look weight: w = 1 is bit-identical to the unweighted pool; w = 0 changes nothing; w = p says p of a
//       fresh view.
//
// Build + run (from door_concept/; any output path — door_concept/build is the agent's own build dir):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_absence_detectability.cpp \
//       src/core/door_core.cpp src/core/door_fit.cpp src/core/door_existence.cpp src/core/door_cycle_io.cpp \
//       src/door_belief.cpp src/door_model.cpp src/door_frame_evidence.cpp ../common/contour_edge/contour_edge_check.cpp \
//       ../common/contour_edge/contour_edge_project.cpp ../common/contour_edge/contour_depth_check.cpp \
//       -lzstd -lopencv_core -lopencv_imgproc -o build/test_door_absence_detectability && build/test_door_absence_detectability
// -DTASK15_RED_BASELINE builds against sources without the Task 15 fields (prints −9 for them and skips C), for the RED runs.
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
#define IMG_W(s) -9.0f
#define AP_ENV(s) -9.0f
#else
#define IMG_W(s) (s).dbg_img_look_w
#define AP_ENV(s) (s).dbg_p_env_aperture
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

static Mat4d pose(double x, double y, double z, bool turned_away = false)
{
    Mat4d T = Mat4d::Identity();
    if (turned_away) { T(0, 0) = -1.0; T(1, 1) = -1.0; }   // yaw 180°: zed +y points to −y
    T(0, 3) = x; T(1, 3) = y; T(2, 3) = z;
    return T;
}

// The door's projected pixel box from (rx, ry) looking +y, clipped to the image; rows with z in [z0, z1] only.
static void door_pixels(std::vector<Vec2f>& px, float rx, float ry, float z0 = 0.0f, float z1 = 2.1f)
{
    const float Y = kWallY - ry;
    if (Y < 0.3f) return;
    const float c0 = kW / 2 + kF * (kDoorX - 0.5f - rx) / Y, c1 = kW / 2 + kF * (kDoorX + 0.5f - rx) / Y;
    const float r0 = kH / 2 - kF * (z1 - kCamZ) / Y, r1 = kH / 2 - kF * (z0 - kCamZ) / Y;
    for (float r = std::max(0.f, r0); r < std::min<float>(kH, r1); r += 3.0f)
        for (float c = std::max(0.f, c0); c < std::min<float>(kW, c1); c += 3.0f)
            px.emplace_back(c, r);
}

enum class Hole { Deeper, Nearer, Flat };

// The ZED depth frame (forward distance) from (rx, ry, kCamZ) looking +y. Past the wall (the robot through the doorway) the
// camera sees the next room: 3 m everywhere.
static ImageFrame depth_frame(float rx, float ry, Hole hole, std::uint64_t stamp)
{
    ImageFrame f;
    f.format = PixelFormat::DepthF32;
    f.width = kW; f.height = kH; f.stamp_ms = stamp;
    ImagePatch p;
    p.x0 = 0; p.y0 = 0; p.w = kW; p.h = kH;
    p.bytes.resize(static_cast<std::size_t>(kW) * kH * sizeof(float));
    std::vector<float> d(static_cast<std::size_t>(kW) * kH);
    const float t_wall = kWallLine - ry;
    // FLAT: a blank wall whose depth over the would-be opening carries a per-frame offset of ±4 cm, its sign alternating with
    // the frame (exactly zero-mean over any two frames) — the noise the frame channel sees on a blank wall, both signs.
    const float flat_bias = (stamp % 2 == 0 ? 1.0f : -1.0f) * 0.04f;
    for (int r = 0; r < kH; ++r)
        for (int c = 0; c < kW; ++c)
        {
            float t = 3.0f;
            if (t_wall > 0.05f)
            {
                const float x = (static_cast<float>(c) - kW / 2) / kF, z = -(static_cast<float>(r) - kH / 2) / kF;
                const float X = rx + x * t_wall, Z = kCamZ + z * t_wall;
                t = t_wall;
                const bool in_hole = X > kDoorX - 0.5f and X < kDoorX + 0.5f and Z > 0.0f and Z < 2.1f;
                if (hole == Hole::Flat)
                { if (in_hole) t += flat_bias; }
                else if (in_hole)
                    t = hole == Hole::Deeper ? t_wall + 3.0f : 0.55f * t_wall;
            }
            d[static_cast<std::size_t>(r) * kW + c] = t;
        }
    std::memcpy(p.bytes.data(), d.data(), p.bytes.size());
    f.patches.push_back(std::move(p));
    return f;
}

struct Look
{
    float rx = 1.0f, ry = 0.5f;
    bool lit = false;           // the detector returns a door mask
    Hole hole = Hole::Deeper;
    bool person = false;        // a person mask covers the doorway's lower 1.5 m (an occluder, not a door)
    bool turned_away = false;
    float top_y = kWallLine;    // the room polygon's top face this cycle (its re-estimation moves it)
};

static DoorCycleInput cycle_input(std::uint64_t cycle, int frame_id, const Look& lk)
{
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = static_cast<std::int64_t>(cycle) * 100'000'000;
    in.t_wall_ms = 1'700'000'000'000 + static_cast<std::int64_t>(cycle) * 100;
    in.robot.present = true;
    in.robot.room_T_body = pose(lk.rx, lk.ry, 0.0, lk.turned_away);
    in.robot.body_T_zed = pose(0.0, 0.0, kCamZ);
    in.camera.present = true;
    in.camera.width = kW; in.camera.height = kH; in.camera.focal_x = kF; in.camera.focal_y = kF;
    MaskPacket p;
    p.valid = true;
    p.frame_id = frame_id;
    p.timestamp_ms = 1000 + static_cast<std::uint64_t>(frame_id);
    if (lk.lit)
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
        s.trunc_frac = 0.0f; s.range = kWallY - lk.ry; s.centroid_radius = 0.1f; s.source = 0; s.has_depth = true;
        door_pixels(p.mask_pixels, lk.rx, lk.ry);
        s.pixel_begin = 0; s.pixel_end = p.mask_pixels.size();
        p.slices.push_back(s);
    }
    {
        // a small unrelated mask in the image corner: the packet carries mask pixels (the silhouette can speak)
        MaskSlice s;
        s.label = "plant"; s.class_id = 5.f; s.confidence = 0.8f; s.source = 0; s.has_depth = true;
        s.pixel_begin = p.mask_pixels.size();
        for (int r = 0; r < 30; r += 3) for (int c = 0; c < 30; c += 3) p.mask_pixels.emplace_back(c, r);
        s.pixel_end = p.mask_pixels.size();
        p.slices.push_back(s);
    }
    if (lk.person and not lk.turned_away)
    {
        MaskSlice s;
        s.label = "person"; s.class_id = 0.f; s.confidence = 0.9f; s.source = 0; s.has_depth = true;
        s.pixel_begin = p.mask_pixels.size();
        door_pixels(p.mask_pixels, lk.rx, lk.ry, 0.0f, 1.5f);
        s.pixel_end = p.mask_pixels.size();
        p.slices.push_back(s);
    }
    in.masks = std::move(p);
    in.depth = depth_frame(lk.rx, lk.ry, lk.hole, 1000 + static_cast<std::uint64_t>(frame_id));
    return in;
}

struct Row { int c = 0; double L = 0, img_dL = 0, img_w = 0, band_p = 1, streak = 0, p_det = 0, ap_env = 0; };
struct Trace
{
    std::vector<Row> rows;   // the door's state after each cycle it was alive
    int removes = 0;
    double L_at(int c) const { double v = std::numeric_limits<double>::quiet_NaN(); for (const auto& r : rows) if (r.c <= c) v = r.L; return v; }
    double L_min_after(int c) const { double v = 1e9; for (const auto& r : rows) if (r.c > c) v = std::min(v, r.L); return v; }
    double L_max_after(int c) const { double v = -1e9; for (const auto& r : rows) if (r.c > c) v = std::max(v, r.L); return v; }
};

static Trace drive(const std::vector<Look>& looks, int print_from, int every)
{
    DoorCoreConfig cfg;
    ExistWorld w;
    w.cfg = cfg.door;
    DoorCore core(cfg);
    w.core = &core;
    Trace t;
    core.connect(&w, {.decision = [&](const rc::door::DoorDecision& d)
                      { if (d.kind == static_cast<std::uint16_t>(K::Remove)) ++t.removes; },
                      .diag = [](const DiagRow&) {}});
    for (std::size_t k = 0; k < looks.size(); ++k)
    {
        const int c = static_cast<int>(k) + 1;
        const auto& lk = looks[k];
        w.polygon = {{-5.f, -5.f}, {5.f, -5.f}, {5.f, lk.top_y}, {-5.f, lk.top_y}};
        (void)core.step(cycle_input(static_cast<std::uint64_t>(c), c, lk));
        for (const auto& [id, s] : w.st)
        {
            t.rows.push_back({c, s.existence.logodds(), s.dbg_edge_delta, IMG_W(s), s.dbg_band_p, s.existence_debounce.streak,
                              s.dbg_sil_pdetect, AP_ENV(s)});
            if (c >= print_from and (c - print_from) % every == 0)
                std::print("    cycle {:3d} robot ({:+.2f},{:.2f}) L {:+.2f} | p_detect {:.3f} ndet {} occl {} | img dL {:+.2f} w {:+.2f} "
                           "ap_env {:+.2f} frame vis {:.2f} dep {:+.2f} | band p {:.2f} streak {:.2f}\n", c, lk.rx, lk.ry,
                           s.existence.logodds(), s.dbg_sil_pdetect, s.dbg_sil_ndet, s.dbg_sil_noccl, s.dbg_edge_delta, IMG_W(s),
                           AP_ENV(s), s.dbg_frame_vis, s.dbg_frame_depth, s.dbg_band_p, s.existence_debounce.streak);
        }
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
// a lateral swing of ±a m, period 8: a new view of the door every frame or two
static float swing(int k, float a)
{
    const int m = k % 8;
    return kDoorX + a * (m < 4 ? (static_cast<float>(m) / 2.0f - 1.0f) : (1.0f - static_cast<float>(m - 4) / 2.0f));
}
static std::vector<Look> confirmed_far(int n, Hole hole = Hole::Deeper)
{
    std::vector<Look> v;
    for (int c = 1; c <= n; ++c) v.push_back({.rx = sweep_x(c), .ry = 0.5f, .lit = true, .hole = hole});
    return v;
}

int main()
{
    if (std::setlocale(LC_ALL, "es_ES.UTF-8") == nullptr) std::setlocale(LC_ALL, "");
    const rc::DoorConfig dcfg;
    const double L_remove = std::log(dcfg.exist_removal_prob / (1.0 - dcfg.exist_removal_prob));

    // ── C. the view pool's look weight ─────────────────────────────────────────────────────────────────────────────────
#ifndef TASK15_RED_BASELINE
    {
        struct Old   // the unweighted pool as it was before Task 15, for the bit-identity check
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
        p0.commit(p0.add({1.0f, 0.5f}, door, 1.17f, +2.0f, 2.83f));
        const auto before = p0;
        const float z = p0.add({1.0f, 0.5f}, door, 1.17f, -5.0f, 2.83f, 0.0f);
        CHECK(z == 0.0f and p0.n == before.n and p0.mean == before.mean and p0.counted == before.counted);
        rc::door::ViewPool pf;
        const float full = rc::door::ViewPool{}.add({1.0f, 0.5f}, door, 1.17f, -1.0f, 2.83f);
        const float part = pf.add({1.0f, 0.5f}, door, 1.17f, -1.0f, 2.83f, 0.25f);
        std::print("C. w = 0 changes nothing; a fresh view's refutation: w 1 → {:+.3f}, w 0.25 → {:+.3f}\n", full, part);
        CHECK(std::abs(part - 0.25f * full) < 1e-5f);
    }
#endif

    // ── A1. envelope overflow: approached and passed THROUGH, the depth refuting ────────────────────────────────────────
    {
        auto looks = confirmed_far(40);
        for (int k = 0; k < 90; ++k)   // y 3.0 → 6.0: through the opening at y = 5 (x within ±0.3 of its centre)
            looks.push_back({.rx = swing(k, 0.3f), .ry = 3.0f + 3.0f * static_cast<float>(k) / 89.0f, .hole = Hole::Nearer});
        std::print("A1. confirmed far, then approached 2 m → through the doorway → 1 m beyond (undetectable), hole NEARER:\n");
        const Trace t = drive(looks, 40, 10);
        const double L0 = t.L_at(40), Lmin = t.L_min_after(40);
        std::print("A1. L at the approach {:+.3f}, lowest after {:+.3f}, REMOVE {}\n", L0, Lmin, t.removes);
        CHECK(std::isfinite(L0) and L0 > L_remove);
        CHECK(t.removes == 0);
        // Its prior-driven level: what a door is born with (the birth seed), plus only what a resolving look said. The
        // refutations from where the doorway is still (barely) inside the envelope count at that weight — small.
        CHECK(Lmin > dcfg.exist_birth_logodds);
    }

    // ── A2. the live mechanism: polygon 0.2 m beyond the door, a person in the doorway, 3 → 1.6 m, then through ─────────
    {
        auto looks = confirmed_far(40);
        for (int c = 31; c <= 40; ++c) looks[static_cast<std::size_t>(c - 1)].top_y = kWallLine + 0.2f;   // re-estimated
        for (int k = 0; k < 60; ++k)
            looks.push_back({.rx = swing(k, 0.6f), .ry = 2.0f + 1.4f * static_cast<float>(k) / 59.0f, .hole = Hole::Nearer,
                             .person = true, .top_y = kWallLine + 0.2f});
        for (int k = 0; k < 40; ++k)
            looks.push_back({.rx = swing(k, 0.2f), .ry = 3.4f + 2.4f * static_cast<float>(k) / 39.0f, .hole = Hole::Nearer,
                             .person = true, .top_y = kWallLine + 0.2f});
        std::print("A2. polygon re-estimated 0.2 m beyond the door; a person in the doorway; 3 → 1.6 m, then through:\n");
        const Trace t = drive(looks, 40, 10);
        const double L0 = t.L_at(40), Lmin = t.L_min_after(40);
        double band = 1.0;
        for (const auto& r : t.rows) if (r.c == 40) band = r.band_p;
        std::print("A2. band π {:.2f}; L after the prior entered {:+.3f}; lowest after {:+.3f}; REMOVE {}\n", band, L0, Lmin, t.removes);
        CHECK(band < 0.5);              // the scene does put the door off the band, as live
        CHECK(t.removes == 0);          // the prior is in L once; it is not a look
        CHECK(Lmin > dcfg.exist_birth_logodds);   // as A1: never below the birth seed on looks that could not resolve it
    }

    // ── A3. a blank-wall phantom approached at close range with zero-mean frame noise: L must not rise ───────────────────
    {
        auto looks = confirmed_far(40, Hole::Flat);
        for (int k = 0; k < 80; ++k)
            looks.push_back({.rx = swing(k, 0.8f), .ry = 3.0f + 0.9f * static_cast<float>(k % 40) / 39.0f, .hole = Hole::Flat});
        std::print("A3. a phantom on a blank wall, then close undetectable views with zero-mean depth noise:\n");
        const Trace t = drive(looks, 40, 20);
        const double L0 = t.L_at(40), Lmax = t.L_max_after(40);
        std::print("A3. L at the approach {:+.3f}, highest after {:+.3f}\n", L0, Lmax);
        CHECK(std::isfinite(L0));
        CHECK(Lmax <= L0 + 0.05);
    }

    // ── B. a phantom refuted from DETECTABLE viewpoints: removed, and the image channel's refutations count ─────────────
    {
        auto looks = confirmed_far(40);
        for (int c = 41; c <= 240; ++c) looks.push_back({.rx = sweep_x(c), .ry = 0.5f, .hole = Hole::Nearer});
        std::print("B. a phantom: lit 40 frames, then absent from the detectable far sweep, the hole reading NEARER:\n");
        const Trace t = drive(looks, 41, 40);
        double neg = 0.0, w_max = 0.0;
        for (const auto& r : t.rows)
            if (r.c > 40 and r.img_dL < 0.0) { neg += r.img_dL; w_max = std::max(w_max, r.img_w); }
        std::print("B. REMOVE {}; the image channel's refutations Σ {:+.2f} nats at look weight up to {:.2f}\n", t.removes, neg, w_max);
        CHECK(t.removes == 1);
        CHECK(neg < -1.0);
        CHECK(w_max > 0.2);
    }

    // ── D. recog_try23 door_11: off the band (π ≈ 0) — removed by its prior L + real detectable looks; never by the prior alone ─
    for (const bool looked_at : {true, false})
    {
        auto looks = confirmed_far(40);   // (lit; the polygon moves when the detector stops firing, so the term is what it says)
        for (int c = 41; c <= 240; ++c)
            looks.push_back({.rx = sweep_x(c), .ry = 0.5f, .hole = Hole::Flat, .turned_away = not looked_at, .top_y = kWallLine + 0.6f});
        std::print("D. a phantom 0.6 m inside the polygon, then {}:\n", looked_at ? "absent from the detectable far sweep" : "never looked at (camera turned away)");
        const Trace t = drive(looks, 41, 100);
        double band = 1.0;
        for (const auto& r : t.rows) if (r.c == 41) band = r.band_p;
        std::print("D. band π {:.3f}; REMOVE {}; L last {:+.2f}\n", band, t.removes, t.rows.empty() ? -99.0 : t.rows.back().L);
        CHECK(band < 0.05);
        CHECK(t.removes == (looked_at ? 1 : 0));
    }

    std::print("{}\n", fails == 0 ? "ALL PASS" : std::format("{} FAILURE(S)", fails));
    return fails == 0 ? 0 : 1;
}
