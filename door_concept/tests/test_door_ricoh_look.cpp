// test_door_ricoh_look.cpp — Q2 follow-up: a door only the 360 camera has reported elicits a frontal ZED look BECAUSE its existence is
// open, with no 'ricoh-born' rule: the existence mutual information is part of the planner's gain (EpistemicPlanner + ExistenceLook).
//   NOTE the numbers: the geometric gain of an unfitted 360-born aperture is already ~1.8 nats; existence adds ~0.18 at p=0.73 (its maximum
//   mutual information is ~0.2), so the look wins by that margin over the same look at a settled door and over the approach offer (<= 1).
//   1. p_exists after a ricoh birth (BirthLogodds 1 => 0.73): the look's gain is above the same belief with existence settled, and above the
//      typical approach offer (DoorAffordance.ValueApproach x P(complete)); its viewpoint is frontal (within 15 deg of the wall normal).
//   2. after a simulated ZED confirmation (p -> 0.99) the existence term, and so the gain, falls back to the geometric one.
//   3. a phantom: the ZED looks and the door is not there - the SAME existence belief that prices the look drives p below the removal
//      floor, and the look's value is gone with it (existence settled the other way).
// Build + run (from door_concept/): g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_ricoh_look.cpp src/door_belief.cpp \
//   src/epistemic_planner.cpp -o /tmp/test_door_ricoh_look && /tmp/test_door_ricoh_look
#include <cmath>
#include <print>
#include <random>
#include <vector>
#include "../src/door_belief.h"
#include "../src/door_config.h"
#include "../src/door_existence_info.h"
#include "../src/epistemic_planner.h"
#include "../../common/existence_belief/existence_belief.h"

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

static rc::DoorBeliefParams params()
{
    rc::DoorBeliefParams p;
    p.wall_O = {4.031f, 2.538f};
    p.wall_u = Eigen::Vector2f(-1.0f, -0.0005f).normalized();
    p.sigma_base_m = 0.03f; p.clutter_frac = 0.10f; p.clutter_scale_m = 0.12f; p.thickness = 0.05f;
    p.prior_s_std = 0.60f; p.prior_w_std = 0.06f; p.prior_h_std = 0.08f; p.tpl_w = 1.0f; p.tpl_h = 2.1f;
    p.process_std_s = 0.005f; p.process_std_w = 0.0005f; p.process_std_h = 0.0005f;
    p.common_mode_s_std = 0.03f; p.common_mode_wh_std = 0.02f; p.gn_iters = 4;
    p.recess_prior_m = 0.0f; p.recess_prior_std = std::sqrt(0.2f * 0.2f / 12.0f + 0.1f * 0.1f);
    return p;
}


// ── a FITTED door (12 full-mask frames from a ZED viewpoint): the geometry is settled, so its D-optimal gain is small ───────────────
constexpr float kFx = 700.f, kCx = 960.f;
static std::vector<Eigen::Vector3f> leaf_points(const rc::DoorBeliefParams& p, float s, float w, float h, int n, std::mt19937& rng)
{
    rc::door::Aperture a;
    a.wall_O = p.wall_O; a.wall_u = p.wall_u; a.s = s; a.w = w; a.h = h; a.thickness = p.thickness; a.depth = 0.f;
    const rc::door::LeafPose L = rc::door::leaf_pose(a, p.leaf);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::normal_distribution<float> N(0.f, 0.006f);
    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i < n; ++i)
        pts.push_back(rc::door::leaf_point(L, (U(rng) - 0.5f) * w, (i % 2 == 0 ? 1.f : -1.f) * L.half_t + N(rng), U(rng) * h));
    return pts;
}
static rc::MaskBearings camera(const rc::DoorBeliefParams& p, float s_c, float range)
{
    const Eigen::Vector2f u = p.wall_u, n(-u.y(), u.x());
    const Eigen::Vector2f c = p.wall_O + s_c * u + range * n, f = -n, r(f.y(), -f.x());
    rc::MaskBearings mb;
    Eigen::Matrix3f R; R << r.x(), r.y(), 0, f.x(), f.y(), 0, 0, 0, 1;
    const Eigen::Vector3f C(c.x(), c.y(), 1.0f);
    mb.zed_T_room.leftCols<3>() = R; mb.zed_T_room.col(3) = -R * C;
    mb.fx = kFx; mb.cx = kCx; mb.jamb_depth = 0.0f;
    return mb;
}
static rc::DoorBelief fitted(std::mt19937& rng)
{
    const auto p = params();
    const float s_true = 3.06f, w_true = 1.0f, range = 2.5f;
    rc::DoorBelief truth({s_true, w_true, 2.1f, 0.f}, p);
    rc::DoorBelief b({3.10f, 1.0f, 2.1f, 0.f}, p);
    b.set_hinge_prob(1.0f); b.set_leaf_phi_sigma(0.146f);
    for (int k = 0; k < 12; ++k)
    {
        rc::MaskBearings mb = camera(p, s_true + 0.5f * w_true, range);
        const auto [lo, hi] = truth.predicted_column_extent(truth.state(), mb);
        mb.lo = lo; mb.hi = hi;
        const float R = 0.00235f, ppm = kFx / range;
        mb.var = ppm * ppm * R + 1.0f / 12.0f;
        mb.var_common = ppm * ppm * 0.0005f + b.hull_mismatch_var(mb);
        rc::DoorFrame f;
        f.points = leaf_points(p, s_true, w_true, 2.07f, 3000, rng);
        f.R.assign(f.points.size(), R); f.chain_cov_s = 0.0122f; f.chain_cov_n = 0.0122f;
        f.bearings = mb;
        b.update(f);
    }
    return b;
}

int main()
{
    const auto p = params();
    rc::DoorBelief b({3.10f, 1.0f, 2.1f, 0.f}, p);          // a freshly 360-born aperture: only the prior, no ZED mask yet
    b.set_hinge_prob(1.0f); b.set_leaf_phi_sigma(0.146f);
    const std::vector<Eigen::Vector2f> room{{-1.f, -6.f}, {8.f, -6.f}, {8.f, 2.538f}, {-1.f, 2.538f}};
    rc::EpistemicPlanner planner(1.8f);
    rc::nbv::Sensor sensor; sensor.hfov_rad = 1.2f; sensor.vfov_rad = 0.9f; sensor.height_m = 1.2f;
    const auto look = [&](float p_exists) { const rc::ExistenceLook e{p_exists, 0.85f, 0.05f}; return planner.compute(b, 0.02f, 0.03f, sensor, {}, room, nullptr, &e); };
    const auto off_normal_deg = [&](const rc::EpistemicProposal& pr)
    {
        const auto ap = b.aperture();
        const Eigen::Vector2f d = Eigen::Vector2f(pr.epistemic_target_x_m, pr.epistemic_target_y_m) - ap.centre_xy();
        return std::acos(std::clamp(std::abs(d.dot(ap.across_u())) / std::max(1e-6f, d.norm()), 0.f, 1.f)) * 57.2958f;
    };

    // 1. ricoh-only birth: existence prior is the birth log-odds (Existence.BirthLogodds = 1)
    const float p_birth = 1.0f / (1.0f + std::exp(-1.0f));
    const auto geom = planner.compute(b, 0.02f, 0.03f, sensor, {}, room);                // geometry only (existence not in question)
    const auto born = look(p_birth);
    const auto settled = look(0.999f);
    rc::DoorPragmaticCfg pg;                                                              // the approach offer on the same scale
    const float approach = pg.value_approach;                                             // x P(complete) <= 1
    std::print("1: look gain: geometry only {:.3f} | ricoh-born (p={:.2f}) {:.3f} | existence settled {:.3f} nats; approach offer <= {:.3f}; viewpoint {:.1f} deg off the normal\n",
               geom.epistemic_gain, p_birth, born.epistemic_gain, settled.epistemic_gain, approach, off_normal_deg(born));
    CHECK(geom.valid and born.valid and settled.valid);
    CHECK(born.epistemic_gain > settled.epistemic_gain + 0.1f);
    CHECK(born.epistemic_gain > geom.epistemic_gain);
    CHECK(born.epistemic_gain > approach);
    CHECK(off_normal_deg(born) < 15.0f);
    // 2. a ZED confirmation drives p_exists up through the shared existence belief; the look's value falls with it
    rc::exist::ExistenceBelief eb; eb.set(1.0f);
    float prev = look(eb.p_exists()).epistemic_gain, first = prev;
    for (int k = 0; k < 6; ++k)
    {
        eb.integrate(1.0f, std::log(0.85f / 0.05f));                                       // a ZED hit: delta = p_vis * llr
        const float g = look(eb.p_exists()).epistemic_gain;
        CHECK(g <= prev + 1e-4f);
        prev = g;
    }
    std::print("2: after 6 ZED confirmations p={:.3f}: look gain {:.3f} -> {:.3f}\n", eb.p_exists(), first, prev);
    CHECK(first - prev > 0.05f);                                                            // the existence part of the value is gone
    CHECK(std::abs(prev - settled.epistemic_gain) < 0.05f * first);                        // ...leaving only the geometric gain
    // 3. phantom: the ZED looks and the door is not there - misses drive the same belief down; the look is worth nothing in both tails
    rc::exist::ExistenceBelief ph; ph.set(1.0f);
    for (int k = 0; k < 8; ++k) ph.integrate(1.0f, std::log(0.15f / 0.95f));
    const float g_ph = look(ph.p_exists()).epistemic_gain;
    std::print("3: phantom after 8 ZED misses: p={:.4f} (L={:.2f}), look gain {:.3f}\n", ph.p_exists(), ph.logodds(), g_ph);
    CHECK(ph.p_exists() < 0.05f and ph.logodds() <= -3.99f);
    CHECK(born.epistemic_gain - g_ph > 0.05f);                                              // existence settled the other way: its part is gone too
    // (the removal itself - Existence floor -4 reached by ZED absence, once, no rebirth - is test_door_core_existence / test_door_ricoh_lifecycle B)
    // 4. SETTLED GEOMETRY (the review's case): after a ZED fit the D-optimal gain is small, so existence is what is left to buy - it must change the offer
    {
        std::mt19937 rng(11);
        rc::DoorBelief fb = fitted(rng);
        const auto go = [&](const rc::ExistenceLook* e) { return planner.compute(fb, 0.02f, 0.03f, sensor, {}, room, nullptr, e); };
        const rc::ExistenceLook open_{p_birth, 0.85f, 0.05f}, shut_{0.999f, 0.85f, 0.05f};
        const auto g0 = go(nullptr), g_open = go(&open_), g_shut = go(&shut_);
        std::print("4: fitted door: geometry-only look {:.3f} nats | existence open (p={:.2f}) {:.3f} | existence settled {:.3f}; approach offer <= {:.3f}\n",
                   g0.epistemic_gain, p_birth, g_open.epistemic_gain, g_shut.epistemic_gain, approach);
        CHECK(g0.valid and g_open.valid and g_shut.valid);
        CHECK(g0.epistemic_gain < 0.5f * approach);                                         // the geometry alone would not justify a look
        CHECK(g_open.epistemic_gain - g_shut.epistemic_gain > 0.05f);                       // open existence raises the offer...
        CHECK(g_open.epistemic_gain > 1.5f * g0.epistemic_gain);                            // ...to well above what geometry alone is worth
        // the ORDER of offers: any competing offer worth between the two (e.g. 0.20 nats, the scale of a remembered-door look) is beaten by the
        // look at the door whose existence is open and beats the look at the same door once existence is settled - existence changes the order.
        const float rival = 0.5f * (g_shut.epistemic_gain + g_open.epistemic_gain);
        std::print("   rival offer {:.3f}: open-existence look {} it, settled look {} it\n", rival,
                   g_open.epistemic_gain > rival ? "beats" : "loses to", g_shut.epistemic_gain > rival ? "beats" : "loses to");
        CHECK(g_open.epistemic_gain > rival and g_shut.epistemic_gain < rival);
        // the viewpoint: existence is the same question from every face, so it does not move a frontal choice (reported, not asserted to differ)
        std::print("   viewpoint open ({:.2f},{:.2f}) vs settled ({:.2f},{:.2f})\n", g_open.epistemic_target_x_m, g_open.epistemic_target_y_m,
                   g_shut.epistemic_target_x_m, g_shut.epistemic_target_y_m);
    }
    if (fails == 0) std::print("test_door_ricoh_look: ALL PASS\n"); else std::print("test_door_ricoh_look: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
