// Offline harness for src/core/door_mask_causes.h (the four causes of a door-class mask, one-model spec §3.2) and the
// DoorBelief four-branch leaf marginal (leaf_mixture). The scene: a wall along +x at y = 5, a door 0.85 m wide.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I../common tests/test_door_mask_causes.cpp src/door_belief.cpp -o /tmp/test_door_mask_causes && /tmp/test_door_mask_causes
#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/core/door_mask_causes.h"
#include "../src/door_leaf_channel.h"

using namespace rc::door::core;
using V3 = Eigen::Vector3f;
static int g_fail = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static constexpr float kSigma = 0.03f, kR = kSigma * kSigma;

static rc::DoorBelief make_door(float s, float w = 0.85f, rc::door::HingeSide hinge = rc::door::HingeSide::Near, float swing = +1.0f)
{
    rc::DoorBeliefParams p;
    p.thickness = 0.05f; p.floor_z = 0.0f;
    p.wall_u = Eigen::Vector2f(1.0f, 0.0f);
    p.wall_O = Eigen::Vector2f(0.0f, 5.0f);
    p.leaf = {0.0f, hinge, swing};
    return rc::DoorBelief({s, w, 2.05f}, p);
}

/// A leaf channel whose four filters hold the leaf at `phi` (rad, on the 5-degree grid) and whose static beliefs are
/// P(hinge Near) = p_near, P(swing +1) = p_plus.
static rc::door::LeafChannel make_channel(float phi, double p_near, double p_plus)
{
    rc::door::LeafChannel ch;
    const auto lp = rc::door::gaussian_log_prior(0.0f, 2.0944f, 25, phi, 0.02f);
    for (auto& f : ch.phi) f.reset(lp);
    ch.hinge.prior = std::log(p_near / (1.0 - p_near));
    ch.swing.prior = std::log(p_plus / (1.0 - p_plus));
    return ch;
}

/// Points sampled on a leaf of `door` hung from `hinge` and swung to `swing` at `phi`.
static std::vector<V3> leaf_points(const rc::DoorBelief& door, rc::door::HingeSide hinge, float swing, float phi)
{
    rc::door::LeafState ls{phi, hinge, swing};
    const auto pose = rc::door::leaf_pose(door.aperture(), ls);
    std::vector<V3> v;
    for (int i = 0; i <= 12; ++i)
        for (int j = 0; j <= 12; ++j)
        {
            const float lx = pose.half_w * (-1.0f + 2.0f * float(i) / 12.0f), lz = pose.centre_z + pose.half_h * (-0.9f + 1.8f * float(j) / 12.0f);
            v.push_back(rc::door::leaf_point(pose, lx, 0.0f, lz));
        }
    return v;
}

static const rc::DoorBeliefParams kBP{};
static const double kLogKappa = std::log(0.194), kLogPd = std::log(0.938);

static MaskCauses explain(const std::vector<V3>& pts, const std::vector<DoorCauseInput>& doors, double log_birth)
{
    MaskCauseModel m;
    m.log_birth = log_birth; m.log_kappa = kLogKappa; m.log_pd_new = kLogPd; m.new_sd_m = 0.15; m.clutter_len_m = 5.5;
    return causes_of_mask(0, pts, kR, doors, m, kBP);
}

int main()
{
    const double log_birth_second = std::log(0.012);   // a second door on a wall that already has a certain door (wall_base_rate)

    // leaf_either_swing_explains: the leaf stands side-on at 90 degrees on the swing side the belief gives 0.1; today's
    // single-swing marginal calls every point clutter.
    {
        const auto door = make_door(1.0f);
        const auto ch = make_channel(1.5708f, 0.99999, 0.9);   // swing believed +1 at 0.9; the leaf really swings -1
        const auto pts = leaf_points(door, rc::door::HingeSide::Near, -1.0f, 1.5708f);
        // the single-swing marginal (today's): the believed side only
        rc::DoorBelief b = door;
        b.set_hinge_prob(0.99999f);
        std::vector<std::pair<float, float>> curve;
        for (int k = 0; k < 25; ++k) curve.push_back({2.0944f * float(k) / 24.0f, k == 18 ? 1.0f : 1e-6f});
        const float x_old = b.clutter_fraction_leaf_marginal(pts, kR, curve);
        const float x_new = b.clutter_fraction_leaf_marginal(pts, kR, ch);
        std::printf("leaf side-on, wrong swing: x_old %.3f x_new %.3f\n", x_old, x_new);
        CHECK(x_old > 0.5f);
        CHECK(x_new < 0.1f);
        DoorCauseInput d; d.id = 7; d.belief = &door; d.leaf = &ch; d.log_r = std::log(0.9); d.log_pd = kLogPd;
        const auto mc = explain(pts, {d}, log_birth_second);
        const double rho_leaf = mc.rho_kind(CauseKind::Leaf), rho_clut = mc.rho_kind(CauseKind::Clutter);
        std::printf("  rho leaf %.3f aperture %.3f new %.3f clutter %.3f\n", rho_leaf, mc.rho_kind(CauseKind::Aperture), mc.rho_kind(CauseKind::NewDoor), rho_clut);
        CHECK(rho_leaf > rho_clut);
        CHECK(rho_leaf > mc.rho_kind(CauseKind::NewDoor));
    }

    // known_aperture_takes_mask: a mask on a shut door's aperture.
    {
        const auto door = make_door(1.0f);
        const auto ch = make_channel(0.0f, 0.99999, 0.9);
        const auto pts = leaf_points(door, rc::door::HingeSide::Near, +1.0f, 0.0f);
        DoorCauseInput d; d.id = 7; d.belief = &door; d.leaf = &ch; d.log_r = std::log(0.95); d.log_pd = kLogPd;
        const auto mc = explain(pts, {d}, log_birth_second);
        std::printf("known aperture: rho aperture %.4f new %.4f clutter %.4f\n", mc.rho_kind(CauseKind::Aperture), mc.rho_kind(CauseKind::NewDoor), mc.rho_kind(CauseKind::Clutter));
        CHECK(mc.rho_kind(CauseKind::Aperture) > 0.9);
        CHECK(mc.rho_kind(CauseKind::NewDoor) < 0.05);
        double s = 0; for (const double r : mc.rho) s += r;
        CHECK(std::abs(s - 1.0) < 1e-9);
    }

    // shared_mask_splits (Review Focus 2): two doors equally close share one mask.
    {
        const auto a = make_door(0.9f), b = make_door(1.1f);   // apertures [0.9,1.75] and [1.1,1.95]: centred 1.325 / 1.525 around the mask's 1.425
        const auto ch = make_channel(0.0f, 0.99999, 0.9);
        std::vector<V3> pts;
        for (int i = 0; i <= 12; ++i)
            for (int j = 0; j <= 12; ++j)
                pts.emplace_back(1.325f + 0.2f * (-1.0f + 2.0f * float(i) / 12.0f), 5.0f, 0.2f + 1.6f * float(j) / 12.0f);
        DoorCauseInput da, db;
        da.id = 1; da.belief = &a; da.leaf = &ch; da.log_r = std::log(0.9); da.log_pd = kLogPd; da.place_var = 0.01f;
        db = da; db.id = 2; db.belief = &b;
        // both doors lie symmetrically about the mask's centre; give the second the mirrored belief explicitly
        const auto c = make_door(1.0f), d = make_door(1.0f);
        da.belief = &c; db.belief = &d;
        const auto mc = explain(pts, {da, db}, log_birth_second);
        const double ra = mc.rho_door(1), rb = mc.rho_door(2);
        std::printf("shared mask: rho door1 %.3f door2 %.3f new %.3f\n", ra, rb, mc.rho_kind(CauseKind::NewDoor));
        CHECK(std::abs(ra - 0.5 * (ra + rb)) < 0.05 and std::abs(rb - 0.5 * (ra + rb)) < 0.05);
        CHECK(ra + rb > 0.9);
    }

    // clutter takes a mask nobody explains
    {
        const auto door = make_door(1.0f);
        const auto ch = make_channel(0.0f, 0.99999, 0.9);
        std::vector<V3> pts;
        for (int i = 0; i < 40; ++i) pts.emplace_back(-3.0f + 0.01f * float(i), 2.0f, 1.0f);   // 3 m off the wall, nowhere a door is
        DoorCauseInput d; d.id = 7; d.belief = &door; d.leaf = &ch; d.log_r = std::log(0.9); d.log_pd = kLogPd;
        const auto mc = explain(pts, {d}, std::log(1e-4));
        std::printf("nowhere: rho clutter %.3f\n", mc.rho_kind(CauseKind::Clutter));
        CHECK(mc.rho_kind(CauseKind::Clutter) > 0.9);
    }
    std::printf(g_fail ? "FAILED\n" : "ALL PASS\n");
    return g_fail != 0;
}
