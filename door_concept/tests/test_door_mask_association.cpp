// test_door_mask_association — a mask frame the door model explains as CLUTTER must not move the aperture (explore_tour8,
// 2026-11-06: 313 mask points at 5.68 m, clutter=100% — another doorway seen through the crossed one — still moved door_3's
// aperture 0.8 m off its wall through the mask-sides midpoint factor; the wall-band prior then fell to 0.01 and existence
// REMOVED the door). The midpoint factor is weighted by the share of the frame's points the door explains, as the points are.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_mask_association.cpp src/door_belief.cpp \
//       -o /tmp/test_door_mask_association && /tmp/test_door_mask_association
#include <cmath>
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
    p.common_mode_s_std = 0.03f; p.common_mode_wh_std = 0.02f; p.gn_iters = 4;
    return p;
}
rc::MaskBearings camera(const rc::DoorBeliefParams& p, float s_c, float range)
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
// Shut-leaf points on the aperture [s, s+w] (as test_door_look_vs_approach.cpp).
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
// A mask whose sides say the door is `shift` m along the wall from where the belief has it.
rc::DoorFrame frame_with_shifted_mask(const rc::DoorBeliefParams& p, float s_belief, float shift, std::vector<Eigen::Vector3f> pts)
{
    rc::DoorBelief truth({s_belief + shift, 1.0f, 2.1f, 0.f}, p);
    truth.set_leaf_phi(0.0f);
    rc::MaskBearings mb = camera(p, s_belief + 0.5f, 3.0f);
    const auto [lo, hi] = truth.predicted_column_extent(truth.state(), mb);
    mb.lo = lo; mb.hi = hi; mb.var = 1.0f; mb.var_common = 4.0f;
    rc::DoorFrame f;
    f.points = std::move(pts);
    f.R.assign(f.points.size(), 0.0009f);
    f.chain_cov_s = 0.0004f; f.chain_cov_n = 0.0004f;
    f.bearings = mb;
    return f;
}
}  // namespace

int main()
{
    std::println("test_door_mask_association");
    std::mt19937 rng(7);
    const auto p = params();
    const float s0 = 3.10f, shift = 0.6f;
    {   // all_clutter_mask_does_not_move_the_aperture: the mask's points are 5 m behind the wall (another doorway)
        rc::DoorBelief b({s0, 1.0f, 2.1f, 0.f}, p);
        b.set_hinge_prob(1.0f); b.set_leaf_phi(0.0f);
        std::vector<Eigen::Vector3f> far;
        const Eigen::Vector2f n(-p.wall_u.y(), p.wall_u.x());
        std::uniform_real_distribution<float> U(-0.5f, 0.5f);
        for (int i = 0; i < 313; ++i)
        {
            const Eigen::Vector2f q = p.wall_O + (s0 + 0.5f + U(rng)) * p.wall_u - 5.0f * n;
            far.emplace_back(q.x(), q.y(), 1.0f + U(rng));
        }
        b.update(frame_with_shifted_mask(p, s0, shift, far));
        std::println("  all-clutter frame: s {:.3f} -> {:.3f} (mask implied {:+.2f} m)", s0, b.state().s, shift);
        check(std::abs(b.state().s - s0) < 0.05f, "all_clutter_mask_does_not_move_the_aperture");
    }
    {   // control: the same mask sides with points ON the leaf (the mask is this door's) still pull s toward them
        rc::DoorBelief b({s0, 1.0f, 2.1f, 0.f}, p);
        b.set_hinge_prob(1.0f); b.set_leaf_phi(0.0f);
        b.update(frame_with_shifted_mask(p, s0, shift, leaf_points(p, s0, 1.0f, 2.07f, 313, rng)));
        std::println("  on-door frame: s {:.3f} -> {:.3f}", s0, b.state().s);
        check(b.state().s - s0 > 0.02f, "a mask on the door still moves s toward its sides");
    }
    std::println("{}/{} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
