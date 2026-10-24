// test_door_open_leaf.cpp — Task D2 (gold12): an OPEN leaf is a door. Four things, all on the real DoorCore / helpers:
//   1. CROSSING CREDIT: a door re-made across a crossing keeps its L and gains ln(CrossPassablePrior/leak) (crossing_logodds), and is
//      NOT re-seeded at BirthLogodds by the existence slice once it is marked seeded.
//   2. RICOH ABSENCE, OPEN-AWARE: a ricoh-born door whose leaf stands open (phi 90 deg) under hundreds of ricoh 'looked, not
//      detected' keeps its L (~0 dL); the same door with a shut leaf is removed (the old channel, unchanged).
//   3. RICOH BIRTH WEIGHT: a slice a leaf-width off the wall gets a non-zero weight under the open-leaf prior (and 0 under a
//      closed prior, as before); a far-off-wall slice does not; an on-wall slice is unchanged.
//   4. the leaf-closedness covariate: cos^2 over the posterior (shut 1, 90 deg 0, smooth between).
// Build + run: see test_door_ricoh_lifecycle.cpp's g++ line with this file in its place.
#include <charconv>
#include <clocale>
#include <cmath>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>
#include "../src/core/door_core.h"
#include "../src/core/door_existence.h"
#include "../src/door_leaf_visibility.h"

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

// One ricoh frame: the robot at `cam`; the door slice (source 1, LiDAR-filled depth) when `door`; a depth-less ricoh slice of
// something else always (so the packet says "the 360 ran"). No ZED mask pixels.
static DoorCycleInput ricoh_cycle(int k, Vec2f cam, bool door, float conf = 0.9f, float dvar = 0.02f)
{
    DoorCycleInput in;
    in.cycle = static_cast<std::uint64_t>(k);
    in.t_steady_ns = static_cast<std::int64_t>(k) * 100'000'000;
    in.t_wall_ms = 1'700'000'000'000 + static_cast<std::int64_t>(k) * 100;
    in.robot.present = true;
    in.robot.room_T_body = Mat4d::Identity();
    in.robot.room_T_body->coeffRef(0, 3) = cam.x(); in.robot.room_T_body->coeffRef(1, 3) = cam.y();
    in.robot.body_T_zed = Mat4d::Identity();
    in.camera.present = true; in.camera.width = 1280; in.camera.height = 720; in.camera.focal_x = 700.f; in.camera.focal_y = 700.f;
    MaskPacket p;
    p.valid = true; p.frame_id = k; p.timestamp_ms = 1000 + static_cast<std::uint64_t>(k);
    if (door)
    {
        for (int i = 0; i <= 20; ++i)
            for (int j = 0; j <= 21; ++j)
                p.support_points.emplace_back(kDoorX - 0.5f + 0.05f * static_cast<float>(i), kWallY, 0.1f * static_cast<float>(j));
        MaskSlice s;
        s.label = "door"; s.class_id = 1000.f; s.confidence = conf;
        s.centroid = {kDoorX, kWallY, 1.05f}; s.bbox_min = {kDoorX - 0.5f, kWallY - 0.01f, 0.0f}; s.bbox_max = {kDoorX + 0.5f, kWallY + 0.01f, 2.1f};
        s.support_begin = 0; s.support_end = p.support_points.size();
        s.range = (Vec2f(kDoorX, kWallY) - cam).norm(); s.source = 1; s.has_depth = true; s.depth_var = dvar;
        p.slices.push_back(s);
    }
    MaskSlice o;
    o.label = "plant"; o.class_id = 5.f; o.confidence = 0.8f; o.source = 1; o.has_depth = false;
    p.slices.push_back(o);
    in.masks = std::move(p);
    return in;
}
static Vec2f viewpoint(int k, float r = 1.5f) { return Vec2f(r * std::cos(0.45f * k), -1.0f + r * std::sin(0.45f * k)); }

struct Tally { int births = 0, removes = 0, first_birth = -1, first_remove = -1; };

int main()
{
    if (std::setlocale(LC_ALL, "es_ES.UTF-8") == nullptr) std::setlocale(LC_ALL, "");
    constexpr float kPi = 3.14159265f;
    {   // 4. closedness
        CHECK(std::abs(rc::door::leaf_closedness({}, 0.0f) - 1.0f) < 1e-5f);
        CHECK(std::abs(rc::door::leaf_closedness({}, kPi / 2) - 0.0f) < 1e-5f);
        CHECK(std::abs(rc::door::leaf_closedness({{0.0f, 1.0f}, {kPi / 2, 1.0f}}, 0.0f) - 0.5f) < 1e-5f);
        CHECK(rc::door::leaf_closedness({}, 0.8f) > rc::door::leaf_closedness({}, 1.2f));   // smooth, monotone
        CHECK(std::abs(rc::door::ricoh_detect_scale(0.0f, 0.05f) - 0.05f) < 1e-6f and rc::door::ricoh_detect_scale(1.0f, 0.05f) == 1.0f);
    }
    {   // 1. crossing credit
        const float dl = rc::door::crossing_logodds(0.5f, 0.05f);
        std::print("1: crossing ln(0.5/0.05) = {:.3f}\n", dl);
        CHECK(dl > 2.2f and dl < 2.4f);
        CHECK(rc::door::crossing_logodds(0.01f, 0.05f) == 0.0f);                       // a happened crossing never counts against
        CHECK(rc::door::crossing_logodds(0.2f, 0.05f) < rc::door::crossing_logodds(0.5f, 0.05f));   // from the likelihood ratio, not a constant
        const float L_src = 1.83f;
        float Lout[3] = {0, 0, 0};
        for (int carry = 0; carry < 3; ++carry)   // the re-made door with / without the carried belief (one world each: two doors at one place would merge)
        {
            DoorCoreConfig cfg; ExistWorld w; w.cfg = cfg.door; DoorCore core(cfg); w.core = &core; core.connect(&w);
            rc::DoorFitState inst;
            rc::DoorState s; s.w = cfg.door.door_prior_w_m; s.h = cfg.door.door_prior_h_m; s.thickness = cfg.door.door_thickness_m;
            core.fit().init_instance(inst, 1, "door_1", s);
            if (carry)   // 1: carried L; 2: unseeded source - BirthLogodds + the crossing (specificworker I2)
            {
                inst.existence.set((carry == 1 ? L_src : cfg.door.exist_birth_logodds) + dl);
                inst.existence_seeded = true; inst.existence_entry_L = inst.existence.logodds(); inst.existence_since_entry = 0.0f;
            }
            w.st.emplace(1, std::move(inst));
            core.step(ricoh_cycle(1, viewpoint(1), false));
            CHECK(w.st.contains(1));
            Lout[carry] = w.st.contains(1) ? w.st.at(1).existence.logodds() : -99.0f;
            w.core = nullptr;
        }
        std::print("1: re-made door L = {:.2f} (carried), {:.2f} (fresh, re-seeded)\n", Lout[1], Lout[0]);
        CHECK(std::abs(Lout[1] - std::min(L_src + dl, 4.0f)) < 1e-4f and Lout[1] > Lout[0] + 1.0f);   // kept, not re-seeded at BirthLogodds
        CHECK(std::abs(Lout[2] - (1.0f + dl)) < 1e-4f and Lout[2] > Lout[0]);   // unseeded source: seed + crossing
        CHECK(std::abs(Lout[0] - 1.0f) < 1e-4f);                                                      // the old path, for contrast
    }
    {   // 3. ricoh birth weight (slice vs the wall at y = 5)
        const std::vector<Vec2f> poly{{-5.f, -5.f}, {5.f, -5.f}, {5.f, 5.f}, {-5.f, 5.f}};
        const Vec2f cam(0.f, 0.f);
        const auto rr = [&](float y, float p_open) { return ricoh_range(poly, cam, Vec2f(0.f, y), 0.02f, 0.03f, 0.01f, 0.04f, 0.01f, true, p_open, 0.9f); };
        const auto on = rr(4.99f, 0.0f), on_open = rr(4.99f, 0.2f);
        const auto off_closed = rr(4.1f, 0.0f), off_open = rr(4.1f, 0.2f), off_mid = rr(4.3f, 0.2f), far_open = rr(2.5f, 0.2f);
        std::print("3: on-wall w={:.4f}/{:.4f}  0.9m-off: closed-prior w={:.2e} open-prior w={:.4f} (agree_open {:.3f})  0.7m-off w={:.4f}  2.5m-off w={:.2e}\n",
                   on.weight, on_open.weight, off_closed.weight, off_open.weight, off_open.agree_open, off_mid.weight, far_open.weight);
        CHECK(std::abs(on.weight - on_open.weight) < 0.02f * on.weight);   // an on-wall slice is unchanged by the prior
        CHECK(off_closed.weight < 1e-6f);                                  // the old behaviour: an off-wall slice births nothing
        CHECK(off_open.weight > 1e-3f and off_mid.weight > 0.1f * on.weight);   // the open leaf off the wall now counts
        CHECK(off_open.weight < on.weight and off_mid.weight < on.weight);      // ...but never dominates an on-wall slice
        CHECK(far_open.weight < 1e-6f);                                    // a random far slice is still nothing
        CHECK(off_mid.rho_open > 0.9f and std::abs(off_mid.xy.y() - 5.0f) < 0.3f);   // placed at the wall (the door), not at the leaf
    }
    {   // 2. ricoh absence of an open leaf
        const float phis[2] = {0.0f, kPi / 2};
        float Lend[2], L100[2], L0[2]; int removes[2];
        for (int open = 0; open < 2; ++open)
        {
            DoorCoreConfig cfg; ExistWorld w; w.cfg = cfg.door; DoorCore core(cfg); w.core = &core; core.connect(&w);
            Tally t;
            for (int k = 1; k <= 300; ++k) for (const auto& d : core.step(ricoh_cycle(k, viewpoint(k, 3.0f), true)).decisions)
                if (d.kind == static_cast<std::uint16_t>(K::Birth)) ++t.births;
            CHECK(t.births == 1 and w.st.size() == 1);
            Tally t2;
            L0[open] = w.st.empty() ? 0.0f : w.st.begin()->second.existence.logodds();
            L100[open] = L0[open];
            for (int k = 301; k <= 1500 and not w.st.empty(); ++k)
            {
                if (k == 401) L100[open] = w.st.begin()->second.existence.logodds();
                // the leaf-angle posterior is IDENTIFIED (by the LiDAR in the live agent): shut, or open at 90 deg
                for (auto& [id, st] : w.st) { st.phi_est = phis[open]; st.phi_curve = {{phis[open], 1.0f}}; }
                for (const auto& d : core.step(ricoh_cycle(k, viewpoint(k, 3.0f), false)).decisions)
                    if (d.kind == static_cast<std::uint16_t>(K::Remove)) ++t2.removes;
            }
            removes[open] = t2.removes;
            Lend[open] = w.st.empty() ? -99.0f : w.st.begin()->second.existence.logodds();
            std::print("2: leaf {}: L {:.2f} -> {:.2f} after 100 absent frames -> {:.2f} at end; removes {}\n", open ? "OPEN (90 deg)" : "SHUT",
                       L0[open], L100[open], Lend[open], removes[open]);
            w.core = nullptr;
        }
        CHECK(removes[0] == 1);                                  // shut leaf: the old behaviour, removed
        CHECK(removes[1] == 0);                                  // open leaf: not removed by 1200 'looked, not detected' frames
        CHECK((L0[1] - L100[1]) < 0.1f * (L0[0] - L100[0]));     // ...and in the same 100 frames lost <10% of what the shut one lost
    }
    if (fails == 0) std::print("test_door_open_leaf: ALL PASS\n");
    else            std::print("test_door_open_leaf: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
