// test_door_width_error_diag — door-passable spec step 0 (2026-11-05): the width model error's INPUTS are exposed
// (DoorBelief::last_width_error) so the live log can say what inflated sigma_w. The scene is warm59's: the leaf 120 deg open
// on the FAR side of the wall, the camera 1.3 m in front on the near side, the mask = the doorway between the jambs.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_width_error_diag.cpp src/door_belief.cpp \
//       -o /tmp/test_door_width_error_diag && /tmp/test_door_width_error_diag
#include <cmath>
#include <print>

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
}  // namespace

int main()
{
    std::println("test_door_width_error_diag");
    const auto p = params();
    const rc::DoorBeliefState st{3.10f, 1.0f, 2.1f, 0.f};
    // the doorway the camera sees: the hull of a SHUT leaf (its corners are the jambs)
    rc::DoorBelief shut(st, p);
    shut.set_leaf_phi(0.0f);
    rc::MaskBearings mb = camera(p, st.s + 0.5f * st.w, 1.3f);
    const auto [jlo, jhi] = shut.predicted_column_extent(shut.state(), mb);
    mb.lo = jlo + 2.f; mb.hi = jhi - 2.f; mb.var = 1.0f; mb.var_common = 0.f;

    rc::DoorBelief b(st, p);
    b.set_hinge_prob(1.0f);
    b.set_leaf_phi(2.09f);
    b.set_leaf_swing(-1.0f);   // opens AGAINST the wall normal: the far side, away from the camera
    rc::DoorFrame f; f.R.assign(0, 0.f); f.bearings = mb;
    b.update(f);
    const auto& d = b.last_width_error();
    std::println("  r_w {:.1f} px, r_w_jambs {:.1f} px, J_w {:.1f} px/m, R_w {:.2f} px2, model sd_w {:.3f} m, post sd_w {:.3f} m",
                 d.r_w, d.r_w_jambs, d.J_w, d.R_w, std::sqrt(d.var_model_w), std::sqrt(d.var_post_w));
    check(d.valid, "the diagnostic was not filled by a mask frame");
    check(std::isfinite(d.r_w) and std::isfinite(d.r_w_jambs), "residuals not finite");
    check(std::abs(d.r_w_jambs) < 5.f, "the jambs-only hull does not match the doorway mask");
    check(d.r_w > 50.f, "the far leaf edge does not widen the hull (warm59 signature absent)");
    check(d.var_post_w > 0.f and std::isfinite(d.var_post_w), "var_post_w (the prior-to-frame posterior variance) missing");
    std::println("{}/{} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
