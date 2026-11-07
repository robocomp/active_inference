// test_door_open_leaf.cpp — Task D2 (gold12): an OPEN leaf is a door. Four things, all on the real DoorCore / helpers:
//   1. CROSSING CREDIT: a door re-made across a crossing keeps its evidence and gains ln(CrossPassablePrior/leak) (crossing_logodds).
//   2. RICOH ABSENCE, OPEN-AWARE: a ricoh-born door whose leaf stands open (phi 90 deg) under hundreds of ricoh 'looked, not
//      detected' frames keeps most of its L; the same door with a shut leaf pays more (per view, not per frame).
//   3. RICOH BIRTH WEIGHT: a slice a leaf-width off the wall gets a non-zero weight under the open-leaf prior (and 0 under a
//      closed prior, as before); a far-off-wall slice does not; an on-wall slice is unchanged.
//   4. the leaf-closedness covariate: cos^2 over the posterior (shut 1, 90 deg 0, smooth between).
// Build + run: see test_door_core_existence.cpp's g++ line with this file in its place.
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
        // The re-made door's belief (Task 9: carried_belief): the source face's evidence plus the crossing, an empty ledger; a face never
        // known carries nothing, so the new room's prior plus the crossing stands alone. (The full core path is test_carried_door.)
        const float L_src = 1.83f;
        rc::door::core::PlaceRow src; src.L_evidence = L_src;
        const auto carried = rc::door::core::carried_belief(&src, dl);
        CHECK(std::abs(carried.L_evidence() - (L_src + dl)) < 1e-5f and carried.ledger.views().empty());
        rc::door::core::PlacePrior room_prior; room_prior.band = -0.5f;
        const auto unknown = rc::door::core::carried_belief(nullptr, dl, room_prior);
        CHECK(std::abs(unknown.L() - (room_prior.total() + dl)) < 1e-5f and unknown.carried == 0.0f);
        std::print("1: carried L = {:.2f}; never-known face L = {:.2f}\n", carried.L(), unknown.L());
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
            L0[open] = w.st.empty() ? 0.0f : w.st.begin()->second.belief.L();
            L100[open] = L0[open];
            for (int k = 301; k <= 1500 and not w.st.empty(); ++k)
            {
                if (k == 401) L100[open] = w.st.begin()->second.belief.L();
                // the leaf-angle posterior is IDENTIFIED (by the LiDAR in the live agent): shut, or open at 90 deg
                for (auto& [id, st] : w.st) { st.phi_est = phis[open]; st.phi_curve = {{phis[open], 1.0f}}; }
                for (const auto& d : core.step(ricoh_cycle(k, viewpoint(k, 3.0f), false)).decisions)
                    if (d.kind == static_cast<std::uint16_t>(K::Remove)) ++t2.removes;
            }
            removes[open] = t2.removes;
            Lend[open] = w.st.empty() ? -99.0f : w.st.begin()->second.belief.L();
            std::print("2: leaf {}: L {:.2f} -> {:.2f} after 100 absent frames -> {:.2f} at end; removes {}\n", open ? "OPEN (90 deg)" : "SHUT",
                       L0[open], L100[open], Lend[open], removes[open]);
            w.core = nullptr;
        }
        // The new model (Task 10): misses are Bernoulli trials per VIEW (view_ledger.h), so 1200 'looked, not detected' frames from a
        // circle of standpoints are a handful of views, not 1200 observations - a door seen on 300 frames is not deleted by them,
        // shut or open. What the leaf angle decides is how much each missed view COSTS: the shut leaf is fully visible from the
        // ricoh, the open one (edge-on, off the wall) is marginalised away (closedness), so it keeps nearly all of its L.
        CHECK(Lend[0] < L0[0] - 0.1f);                           // the shut leaf pays for its absences
        CHECK(removes[0] == 0 and removes[1] == 0);              // ...but a few missed views do not delete a door seen 300 times
        CHECK((L0[1] - Lend[1]) < 0.5f * (L0[0] - Lend[0]));     // the open leaf loses less than half of what the shut one lost
        CHECK(Lend[1] > Lend[0]);
    }
    if (fails == 0) std::print("test_door_open_leaf: ALL PASS\n");
    else            std::print("test_door_open_leaf: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
