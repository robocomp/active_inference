// test_door_hull_midpoint — Task W (+ fix round 1, I-2): the sides-midpoint factor carries the variance a PARTIAL mask implies
// (DoorBelief::hull_mismatch_var = (hull width − mask width)²/12), and that term must not cost the belief anything it had.
//   (a) ab7_A1's first frame: a 1.00 m door, a mask 58% of the predicted hull, its centre 0.62 hull-widths off, 2.8 m away.
//       WITHOUT the term w goes 1.00 -> 1.39 (the synthetic collapse, here upward); WITH it w stays within 1.5 prior σ (0.09 m).
//   (b)(c) a door whose opening is GENUINELY narrower (1.00 -> 0.75 m, the far jamb comes in), full masks, near (2.0 m) and far
//       (4.5 m). ⚠ MEASURED: this belief does NOT follow a narrower width — with the term or without it (w 0.97-0.99 after 12
//       frames, σ_w 0.017): w is a strong template (Door.PriorWidthStd 0.06, folded in EVERY frame by accumulate_extra "what
//       makes the strong size prior stick"), the leaf's SDF has no free-space term (a panel longer than the points is not
//       penalised) and the midpoint of a one-sided narrowing is the only width-sensitive channel. So the assertion is the one
//       that CAN hold: the term does not make the response worse than the control (|w_with − w_truth| ≤ |w_without − w_truth|
//       + 0.02). A narrower CLEAR SPAN is carried by the leaf angle (LiDAR rays → phi curve → clear_width) and the anchor's
//       wall-gap width, neither of which this term touches.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tests/test_door_hull_midpoint.cpp src/door_belief.cpp -o /tmp/test_door_hull_midpoint && /tmp/test_door_hull_midpoint
#include <cmath>
#include <cstdlib>
#include <print>
#include <random>
#include <vector>

#include "../src/door_belief.h"

namespace
{
int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what) { ++g_checks; if (not ok) { ++g_fail; std::println("  FAIL: {}", what); } }
constexpr float kFx = 700.f, kCx = 960.f;

rc::DoorBeliefParams params()
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
// Shut leaf (phi 0) points on both faces of the aperture [s, s+w], 0..h tall.
std::vector<Eigen::Vector3f> leaf_points(const rc::DoorBeliefParams& p, float s, float w, float h, int n, std::mt19937& rng)
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
// The camera `range` m in front of the aperture centre, looking at it (zed: X right, Y forward, Z up).
rc::MaskBearings camera(const rc::DoorBeliefParams& p, float s_c, float range)
{
    const Eigen::Vector2f u = p.wall_u, n(-u.y(), u.x());   // across_u: into the room
    const Eigen::Vector2f c = p.wall_O + s_c * u + range * n, f = -n, r(f.y(), -f.x());
    rc::MaskBearings mb;
    Eigen::Matrix3f R; R << r.x(), r.y(), 0, f.x(), f.y(), 0, 0, 0, 1;
    const Eigen::Vector3f C(c.x(), c.y(), 1.0f);
    mb.zed_T_room.leftCols<3>() = R; mb.zed_T_room.col(3) = -R * C;
    mb.fx = kFx; mb.cx = kCx; mb.jamb_depth = 0.0f;
    return mb;
}

struct Run { float w = 0, sd_w = 0; };
// 12 frames of a door whose TRUE aperture is [s_true, s_true + w_true]; the mask is `mask_frac` of the true hull, centred
// (1 = a full mask). `with_term` = add hull_mismatch_var to the sides' common mode (the control turns it off).
Run run(float w_true, float range, float mask_frac, bool with_term, std::mt19937& rng, float off_frac = 0.0f)
{
    auto p = params();
    const float s_true = 3.06f;   // the near jamb stays where the 1.00 m door had it: the far jamb comes in (a narrower opening, not a shifted one)
    rc::DoorBelief truth({s_true, w_true, 2.1f, 0.f}, p);
    rc::DoorBelief b({3.10f, 1.0f, 2.1f, 0.f}, p);
    b.set_hinge_prob(1.0f); b.set_leaf_phi_sigma(0.146f);
    for (int k = 0; k < 12; ++k)
    {
        rc::MaskBearings mb = camera(p, s_true + 0.5f * w_true, range);
        const auto [lo, hi] = truth.predicted_column_extent(truth.state(), mb);
        const float mid = 0.5f * (lo + hi) + off_frac * (hi - lo), half = 0.5f * mask_frac * (hi - lo);
        mb.lo = mid - half; mb.hi = mid + half;
        const float R = 0.00235f, ppm = kFx / range;
        mb.var = ppm * ppm * R + 1.0f / 12.0f;
        mb.var_common = ppm * ppm * 0.0005f + (with_term ? b.hull_mismatch_var(mb) : 0.0f);
        rc::DoorFrame f;
        f.points = leaf_points(p, s_true, w_true, 2.07f, 3000, rng);
        f.R.assign(f.points.size(), R); f.chain_cov_s = 0.0122f; f.chain_cov_n = 0.0122f;
        f.bearings = mb;
        b.update(f);
    }
    return {b.state().w, std::sqrt(std::max(0.f, b.covariance_full()(1, 1)))};
}
}  // namespace

int main()
{
    std::mt19937 rng(11);
    std::println("(a) far, small partial mask of a 1.00 m door (58% of the predicted hull, centred 0.62 hull-widths to one side, 2.8 m — ab7_A1's first frame): w stays within 1.5 prior sigma");
    {
        const auto with = run(1.0f, 2.8f, 0.58f, true, rng, 0.62f);
        const auto wout = run(1.0f, 2.8f, 0.58f, false, rng, 0.62f);
        std::println("    with the term w {:.3f} (sd {:.3f}); without {:.3f}", with.w, with.sd_w, wout.w);
        check(std::abs(with.w - 1.0f) < 1.5f * 0.06f, "a partial far mask moved w by more than 1.5 prior sigma");
        check(std::abs(wout.w - 1.0f) > 1.5f * 0.06f, "the control does not collapse: this synthetic no longer discriminates");
    }
    std::println("(b) the opening GENUINELY narrows 1.00 -> 0.75 m, near (2.0 m), full masks: the term costs nothing");
    {
        const auto with = run(0.75f, 2.0f, 1.0f, true, rng);
        const auto wout = run(0.75f, 2.0f, 1.0f, false, rng);
        std::println("    with the term w {:.3f} (sd {:.3f}); without {:.3f}  [truth 0.75]", with.w, with.sd_w, wout.w);
        check(std::abs(with.w - 0.75f) <= std::abs(wout.w - 0.75f) + 0.02f, "the term made following a narrower opening worse (near)");
    }
    std::println("(c) the same narrower door, FAR (4.5 m), full mask");
    {
        const auto with = run(0.75f, 4.5f, 1.0f, true, rng);
        const auto wout = run(0.75f, 4.5f, 1.0f, false, rng);
        std::println("    with the term w {:.3f} (sd {:.3f}); without {:.3f}  [truth 0.75]", with.w, with.sd_w, wout.w);
        check(std::abs(with.w - 0.75f) <= std::abs(wout.w - 0.75f) + 0.02f, "the term made following worse at range");
    }
    std::println("(d) a FULL mask of the true 1.00 m door is not diluted: the term is ~0 when mask = hull");
    {
        const auto with = run(1.0f, 2.0f, 1.0f, true, rng);
        std::println("    w {:.3f} (sd {:.3f})", with.w, with.sd_w);
        check(std::abs(with.w - 1.0f) < 0.03f, "a full mask of the true hull moved w");
    }
    std::println("{} / {} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
