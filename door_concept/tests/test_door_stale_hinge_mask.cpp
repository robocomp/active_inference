// test_door_stale_hinge_mask — the mask frame must weight its s-evidence with the hinge posterior AFTER this frame's own
// leaf evidence (clean60 2026-11-13, door_3: s 0.23 -> 0.42 m in ONE mask frame, +0.2 m from the truth, after 10 s unobserved;
// the same frame's leaf evidence then flipped the hinge). door_fit.cpp now steps the leaf channel BEFORE the mask update, so
// DoorBelief::set_hinge_prob / set_leaf_phi_sigma receive the fresh posterior. This test pins the belief-side consequence the
// reorder relies on (the belief marginalises the mask over BOTH jambs, weight p_hinge_):
//   (1) truth: leaf hung from the FAR jamb, phi 0.17. A confident WRONG hinge (stale P 0.95 for Near) lets one mask frame move s;
//       the same frame under the ambiguous posterior (P 0.5) never moves it more. ★ At belief level the effect is a few cm
//       (the 0.19 m of clean60 also needed the live frame's chain/offset terms); the live ORDER is not exercised here.
//   (2) a well-open leaf (phi 1.0) with a confident, correct hinge still localises s as before.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tests/test_door_stale_hinge_mask.cpp src/door_belief.cpp -o /tmp/test_door_stale_hinge_mask && /tmp/test_door_stale_hinge_mask
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <print>
#include <random>
#include <vector>

#include "../src/door_belief.h"

namespace
{
int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what)
{
    ++g_checks;
    if (not ok) { ++g_fail; std::println("  FAIL: {}", what); }
}
const bool kTrace = std::getenv("STALE_HINGE_TRACE") != nullptr;
constexpr float kT = 0.20f, kSigmaB = 0.10f;   // Existence.WallThicknessM / BoundarySigmaM (the live config)

// The tapes' wall: room_1's y = 2.538 face, run along −x; across_u = (−u.y, u.x) points INTO the room.
rc::DoorBeliefParams tape_params()
{
    rc::DoorBeliefParams p;
    p.wall_O = {4.031f, 2.538f};
    p.wall_u = Eigen::Vector2f(-1.0f, -0.0005f).normalized();
    p.sigma_base_m = 0.03f; p.clutter_frac = 0.10f; p.clutter_scale_m = 0.12f;
    p.thickness = 0.05f;
    p.prior_s_std = 0.60f; p.prior_w_std = 0.06f; p.prior_h_std = 0.08f;
    p.tpl_w = 1.0f; p.tpl_h = 2.1f;
    p.process_std_s = 0.005f; p.process_std_w = 0.0005f; p.process_std_h = 0.0005f;
    p.common_mode_s_std = 0.03f;
    p.common_mode_wh_std = 0.02f;   // the live config (AI2CommonModeSizeStd)
    p.gn_iters = 4;
    p.recess_prior_m = 0.0f;        // DoorFit::make_belief_params
    p.recess_prior_std = std::sqrt(kT * kT / 12.0f + kSigmaB * kSigmaB);
    return p;
}

// Points on BOTH faces of the true leaf (aperture [s, s+w] at depth d beyond the line, angle phi), 0..h tall.
// mid_plane: every point on the leaf's mid-plane (what a mask's depth gives at a few mm noise — the tapes' D1/D2 frames).
std::vector<Eigen::Vector3f> leaf_points(const rc::DoorBeliefParams& p, float s, float w, float h, float d, float phi,
                                         int n, std::mt19937& rng, bool mid_plane = false)
{
    rc::door::Aperture a;
    a.wall_O = p.wall_O; a.wall_u = p.wall_u; a.s = s; a.w = w; a.h = h; a.thickness = p.thickness; a.depth = -d;
    rc::door::LeafState ls = p.leaf;
    ls.phi = phi;
    const rc::door::LeafPose L = rc::door::leaf_pose(a, ls);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::normal_distribution<float> N(0.f, 0.006f);
    std::vector<Eigen::Vector3f> pts;
    pts.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const float lx = (U(rng) - 0.5f) * w;
        const float ly = (mid_plane ? 0.f : (i % 2 == 0 ? 1.f : -1.f) * L.half_t) + N(rng);
        pts.push_back(rc::door::leaf_point(L, lx, ly, U(rng) * h));
    }
    return pts;
}

rc::DoorFrame frame_of(const std::vector<Eigen::Vector3f>& pts, float R, float chain)
{
    rc::DoorFrame f;
    f.points = pts;
    f.R.assign(pts.size(), R);
    f.chain_cov_s = chain;
    f.chain_cov_n = chain;   // an isotropic chain: the same variance across the wall (Task D-C round 2)
    return f;
}

float sd(const rc::DoorBelief& b, int i) { return std::sqrt(std::max(0.f, b.covariance_full()(i, i))); }

// One mask frame on a belief that believes `believed`, with hinge probability P; returns the s step.
float one_frame(rc::door::HingeSide believed, float P, float phi_true, float phi_sigma, float s_off, int seed)
{
    auto p = tape_params();
    p.prior_s_std = 0.153f;                       // clean60: sigma_s 0.153 after 10 s unobserved
    p.leaf.hinge = believed; p.leaf.swing = +1.f; p.leaf.phi = phi_true;
    auto pt = p; pt.leaf.hinge = rc::door::HingeSide::Far;
    std::mt19937 rng(static_cast<unsigned>(seed));
    const auto pts = leaf_points(pt, 3.06f, 1.0f, 2.07f, 0.0f, phi_true, 3000, rng);
    rc::DoorBelief b({3.06f + s_off, 1.0f, 2.07f, p.recess_prior_m}, p);
    b.set_hinge_prob(P);
    b.set_leaf_phi_sigma(phi_sigma);
    const float s0 = b.state().s;
    b.update(frame_of(pts, 0.00235f, 0.0122f));
    return b.state().s - s0;
}
}  // namespace

int main()
{
    // (1) ambiguous hinge, nearly shut leaf: error from the truth after one frame (start offset `off`), stale vs fresh posterior
    int n_better = 0, n_worse = 0;
    float worst_stale = 0.f, worst_fresh = 0.f;
    for (int seed = 1; seed <= 6; ++seed)
        for (const float off : {-0.1f, 0.f, 0.1f})
        {
            const float stale = std::abs(one_frame(rc::door::HingeSide::Near, 0.95f, 0.17f, 0.001f, off, seed));
            const float fresh = std::abs(one_frame(rc::door::HingeSide::Near, 0.50f, 0.17f, 0.05f, off, seed));
            worst_stale = std::max(worst_stale, stale); worst_fresh = std::max(worst_fresh, fresh);
            n_better += fresh < stale - 1e-3f; n_worse += fresh > stale + 1e-3f;
            if (kTrace) std::println("  seed {} off {:+.2f}: |ds| stale {:.3f} fresh {:.3f}", seed, off, stale, fresh);
        }
    std::println("(1) wrong confident hinge: worst one-frame |ds| stale P {:.3f} m; ambiguous posterior {:.3f} m ({} better, {} worse of 18)",
                 worst_stale, worst_fresh, n_better, n_worse);
    check(n_better >= 6, "(1) ... and moves it less in the cases the wrong hinge pulls");
    check(worst_fresh <= worst_stale, "(1) worst case no worse");

    // (2) well-open leaf, confident correct hinge: one frame still pulls s toward the truth, as before the change
    float least_pull = 1e9f;
    for (int seed = 1; seed <= 6; ++seed)
        for (const float off : {-0.1f, 0.1f})
        {
            const float ds = one_frame(rc::door::HingeSide::Far, 1.0f, 1.0f, 0.02f, off, seed);
            least_pull = std::min(least_pull, -ds * (off > 0 ? 1.f : -1.f));   // movement toward the truth
        }
    std::println("(2) open leaf, confident hinge: least movement toward the truth in one frame {:.3f} m", least_pull);
    check(least_pull > 0.03f, "(2) an open leaf with a confident hinge still localises s");

    std::println("{}", g_fail == 0 ? "test_door_stale_hinge_mask: ALL PASS" : "test_door_stale_hinge_mask: FAILED");
    return g_fail == 0 ? 0 : 1;
}
