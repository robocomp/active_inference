// test_door_core_fit.cpp — the DoorCore's FIT slice on synthetic inputs (Task 4b of the door-core replay plan).
//
// The fit (observe → run_inference → estimate_phi, with the pre-belief geometry and the read-back) moved out of the
// DSR-bound DoorFitter into the core (src/core/door_fit.h), so step() now produces every door's belief from the input
// alone. This drives it with a door seen through the top wall of a 10 x 10 m room:
//   1. BIRTH → FIT: the tracker births the door, the core initialises its belief against the wall it was seen through
//      and the aperture converges onto the true doorway (centre, wall line); the belief is the one the tracker then
//      associates with (TrackedDoor::belief non-null).
//   2. PHI from the low LiDAR: a sweep whose rays stop on a SHUT leaf drives phi to ~0 and narrows its width; the same
//      door with the rays flying THROUGH the aperture reads a clearly larger angle. One Phi decision per door per cycle,
//      carrying the fit state's own numbers (d = -1: the camera does not score phi).
//   3. DETERMINISM — the reason the core exists: two cores fed the same input produce bit-identical Phi decisions even
//      when the second one runs with real sleeps between cycles (the fit reads time ONLY from the input; the old fitter
//      read steady_clock mid-cycle, so a replay could never reproduce its dt).
//   4. FOREIGN CLAIMS: a point another object already explains is dropped by the fit (the core's one claim set per
//      cycle — the worker's second graph walk is gone).
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_core_fit.cpp src/core/door_core.cpp src/core/door_fit.cpp \
//       src/core/door_existence.cpp src/core/door_cycle_io.cpp src/door_belief.cpp src/door_model.cpp src/door_frame_evidence.cpp \
//       ../common/contour_edge/contour_edge_check.cpp ../common/contour_edge/contour_edge_project.cpp \
//       ../common/contour_edge/contour_depth_check.cpp -lzstd -lopencv_core -lopencv_imgproc -o /tmp/test_door_core_fit && /tmp/test_door_core_fit
#include <chrono>
#include <clocale>
#include <cmath>
#include <print>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "../src/core/door_core.h"

using namespace rc::door::core;
using K = rc::door::DoorDecisionKind;

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
    std::optional<GhostMatch> match_ghost(const Vec2f&) const override { return std::nullopt; }
    std::optional<IdentityMatch> match_identity(const Vec2f&) const override { return std::nullopt; }
    std::vector<std::string> reserved_names() const override { return {}; }
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
    void reacquire(std::uint64_t, const GhostMatch&) override {}
    void forget_ghost(const Vec2f&) override {}
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

// The door: an aperture in the top wall (y = 5), centred at x = 1.0, 1.0 m wide, 2.1 m tall; the robot stands still at
// (1, 2) with the zed 1.2 m up, looking +y (zed +y = forward).
constexpr float kDoorX = 1.0f, kWallY = 4.99f;

static Mat4d translation(double x, double y, double z)
{
    Mat4d T = Mat4d::Identity();
    T(0, 3) = x; T(1, 3) = y; T(2, 3) = z;
    return T;
}

enum class Leaf { None, Shut, Open };

static DoorCycleInput cycle_input(std::uint64_t cycle, int frame_id, Leaf leaf, bool claim = false)
{
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = static_cast<std::int64_t>(cycle) * 100'000'000;   // 10 Hz
    in.t_wall_ms = 1'700'000'000'000 + static_cast<std::int64_t>(cycle) * 100;
    in.robot.present = true;
    in.robot.room_T_body = translation(1.0, 2.0, 0.0);
    in.robot.body_T_zed = translation(0.0, 0.0, 1.2);
    in.camera.present = true;
    in.camera.width = 1280; in.camera.height = 720; in.camera.focal_x = 700.f; in.camera.focal_y = 700.f;

    MaskPacket p;
    p.valid = true;
    p.frame_id = frame_id;
    p.timestamp_ms = 1000 + static_cast<std::uint64_t>(frame_id);
    for (int i = 0; i <= 20; ++i)
        for (int j = 0; j <= 21; ++j)
        {
            const float jitter = 0.004f * static_cast<float>(((i * 7 + j * 13) % 5) - 2);   // deterministic ±8 mm
            p.support_points.emplace_back(kDoorX - 0.5f + 0.05f * static_cast<float>(i), kWallY + jitter,
                                          0.1f * static_cast<float>(j));
        }
    MaskSlice s;
    s.label = "door"; s.class_id = 1000.f; s.confidence = 0.9f;
    s.centroid = {kDoorX, kWallY, 1.05f}; s.bbox_min = {kDoorX - 0.5f, kWallY - 0.01f, 0.0f};
    s.bbox_max = {kDoorX + 0.5f, kWallY + 0.01f, 2.1f};
    s.support_begin = 0; s.support_end = p.support_points.size();
    s.trunc_frac = 0.0f; s.range = 3.0f; s.centroid_radius = 0.1f; s.source = 0; s.has_depth = true;
    p.slices.push_back(s);
    in.masks = std::move(p);

    if (claim)   // a foreign object (another agent's cabinet) standing in front of the right jamb
    {
        ObjectInfo o;
        o.id = 77; o.name = "cabinet_1"; o.type = "box";
        o.width_m = 0.4f; o.depth_m = 0.4f; o.height_m = 2.5f;
        o.room_T_node = translation(kDoorX + 0.4, kWallY, 0.0);
        in.objects.push_back(o);
    }

    if (leaf != Leaf::None)
    {
        BpearlSweep b;
        b.room_fresh = true;
        b.origin_room = {1.0f, 2.0f, 0.35f};
        for (int k = 0; k <= 80; ++k)
        {
            const float x = -1.0f + 0.05f * static_cast<float>(k);
            const bool in_aperture = x > kDoorX - 0.48f and x < kDoorX + 0.48f;
            for (const float z : {0.4f, 0.8f})
            {
                if (leaf == Leaf::Open and in_aperture)
                    b.pts_room.emplace_back(1.0f + (x - 1.0f) * 2.0f, 8.0f, z);   // flew through, stopped in the next room
                else
                    b.pts_room.emplace_back(x, kWallY + 0.02f, z);               // stopped on the wall / the shut leaf
            }
        }
        in.bpearl = std::move(b);
    }
    return in;
}

struct Run
{
    std::vector<rc::door::DoorDecision> phi;
    std::vector<std::string> stdout_lines;
    int birth_cycle = -1;
};

static Run run(Leaf leaf, int cycles, bool sleep_between = false, bool claim = false, FitWorld* out_world = nullptr)
{
    DoorCoreConfig cfg;
    FitWorld local;
    FitWorld& w = out_world ? *out_world : local;
    w.cfg = cfg.door;
    DoorCore core(cfg);
    w.core = &core;
    Run r;
    core.connect(&w, {.decision = {}, .diag = [&](const DiagRow& d) { if (d.stream == "stdout") r.stdout_lines.push_back(d.line); }});
    for (int c = 1; c <= cycles; ++c)
    {
        const auto out = core.step(cycle_input(static_cast<std::uint64_t>(c), c, leaf, claim));
        for (const auto& d : out.decisions)
        {
            if (d.kind == static_cast<std::uint16_t>(K::Phi)) r.phi.push_back(d);
            if (d.kind == static_cast<std::uint16_t>(K::Birth)) r.birth_cycle = c;
        }
        if (sleep_between) std::this_thread::sleep_for(std::chrono::milliseconds(7));
    }
    w.core = nullptr;
    return r;
}

int main()
{
    if (std::setlocale(LC_ALL, "es_ES.UTF-8") == nullptr) std::setlocale(LC_ALL, "");

    // ── 1. birth → fit: the aperture converges onto the doorway ─────────────────────────────────────────────────────
    {
        FitWorld w;
        const Run r = run(Leaf::Shut, 40, false, false, &w);
        std::print("birth at cycle {}, {} Phi rows, {} fits published\n", r.birth_cycle, r.phi.size(), w.after_fit_calls);
        CHECK(r.birth_cycle > 0);
        CHECK(w.st.size() == 1);
        if (w.st.size() == 1)
        {
            const auto& f = w.st.begin()->second;
            const auto& s = f.model.state();
            std::print("aperture centre ({:.4f}, {:.4f}) w {:.3f} h {:.3f} matched {} | phi {:.4f} sigma {:.4f} measured {}\n",
                       s.ap_cx, s.ap_cy, s.w, s.h, f.matched_frames, f.phi_est, f.phi_sigma, f.phi_measured);
            CHECK(f.ai2_initialized);
            CHECK(f.matched_frames >= 30);
            CHECK(std::abs(s.ap_cx - kDoorX) < 0.05f);
            CHECK(std::abs(s.ap_cy - 5.0f) < 0.12f);        // on the top wall's line (the belief seats it in the wall band)
            CHECK(std::abs(s.w - 1.0f) < 0.10f);
            // 2. a shut leaf: phi ~ 0, measured, its width well below "nothing known" (2.0944)
            CHECK(f.phi_measured);
            CHECK(f.phi_est < 0.35f);
            CHECK(f.phi_sigma < 1.0f);
            // the tracker now associates with the belief the fit built
            const auto d = w.doors();
            CHECK(d.size() == 1 and d[0].belief != nullptr and d[0].ai2_initialized);
        }
        // one Phi row per door per cycle from its birth on, carrying the state's own numbers
        CHECK(static_cast<int>(r.phi.size()) == 40 - r.birth_cycle + 1);
        if (not r.phi.empty() and w.st.size() == 1)
        {
            const auto& last = r.phi.back();
            const auto& f = w.st.begin()->second;
            CHECK(last.a == static_cast<double>(f.phi_est) and last.b == static_cast<double>(f.phi_sigma));
            CHECK(last.c == 1.0 and last.d == -1.0);
            CHECK(last.cycle == 40 and last.name == f.node_name);
        }
    }

    // ── 2b. the same doorway with the rays flying through: a clearly larger angle ──────────────────────────────────
    {
        FitWorld ws, wo;
        (void) run(Leaf::Shut, 40, false, false, &ws);
        (void) run(Leaf::Open, 40, false, false, &wo);
        if (ws.st.size() == 1 and wo.st.size() == 1)
        {
            const float ps = ws.st.begin()->second.phi_est, po = wo.st.begin()->second.phi_est;
            std::print("phi shut {:.4f} rad, open {:.4f} rad\n", ps, po);
            CHECK(po > ps + 0.3f);
        }
        else CHECK(false);
    }

    // ── 3. determinism: the fit reads time only from the input ──────────────────────────────────────────────────────
    {
        const Run a = run(Leaf::Shut, 30);
        const Run b = run(Leaf::Shut, 30, /*sleep_between=*/true);
        CHECK(not a.phi.empty());
        CHECK(a.phi == b.phi);   // bit for bit: phi_est, phi_sigma, measured, support
    }

    // ── 4. a point another object already explains is not evidence for the door ──────────────────────────────────
    {
        const Run c = run(Leaf::None, 20, false, /*claim=*/true);
        bool dropped = false;
        for (const auto& l : c.stdout_lines) dropped = dropped or l.find("[exclusion] point dropped") != std::string::npos;
        CHECK(dropped);
        const Run n = run(Leaf::None, 20, false, false);
        bool none = true;
        for (const auto& l : n.stdout_lines) none = none and l.find("[exclusion] point dropped") == std::string::npos;
        CHECK(none);
    }

    if (fails == 0) std::print("test_door_core_fit: ALL PASS\n");
    else            std::print("test_door_core_fit: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
