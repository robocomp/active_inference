// test_door_look_vs_approach — Task E: a poor door estimate makes the EPISTEMIC action (look) beat the goal-directed one.
//   1. HONEST UNCERTAINTY. A belief whose masks disagree with it (a partial far mask; the Task-W midpoint term ablated, which
//      reproduces the ab7 collapse) REPORTS a wider sigma_w (DoorBelief::covariance_reported) than one its masks agree with.
//   2. ONE SCALE. The look's value (EpistemicPlanner gain: D-optimal expected information on [s, w, h], nats) and the approach's
//      (DoorAffordance.ValueApproach, nats, x P(complete | the reported width sigma)) compared as the
//      controller compares them: degraded belief => look > approach, and the approach is LOWERED, not withdrawn; good belief =>
//      approach > look. The look's viewpoint is frontal at a good standoff without any perpendicular rule.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_look_vs_approach.cpp src/door_belief.cpp src/epistemic_planner.cpp \
//       -o /tmp/test_door_look_vs_approach && /tmp/test_door_look_vs_approach
#include <cmath>
#include <cstdlib>
#include <print>
#include <random>
#include <vector>

#include "../src/door_belief.h"
#include "../src/door_config.h"
#include "../src/door_approach_pose.h"
#include "../src/door_dof.h"
#include "../src/epistemic_planner.h"

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

// 12 frames of a door whose TRUE aperture is [3.06, 3.06 + w_true]; the mask is `mask_frac` of the true hull, centred
// (1 = a full mask) and `off_frac` of the hull off-centre. `with_term` = the Task-W midpoint term (the control turns it OFF
// to reproduce the collapse). The belief is returned: the test reads what it REPORTS.
rc::DoorBelief run(float w_true, float range, float mask_frac, bool with_term, std::mt19937& rng, float off_frac = 0.0f)
{
    auto p = params();
    const float s_true = 3.06f;
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
    return b;
}
float sd_w(const rc::DoorBelief& b) { return std::sqrt(std::max(0.f, b.covariance_reported()(1, 1))); }

// The two prices on ONE scale (nats), exactly as the agent computes them for a robot `robot_range` m in front of the door.
struct Prices { float look_gain = 0, approach_value = 0, look_standoff = 0, look_off_normal_deg = 0; rc::EpistemicProposal prop; };
Prices prices(const rc::DoorBelief& b, float robot_range)
{
    Prices out;
    const std::vector<Eigen::Vector2f> room{{-1.f, -6.f}, {8.f, -6.f}, {8.f, 2.538f}, {-1.f, 2.538f}};
    rc::EpistemicPlanner planner(1.8f);
    rc::nbv::Sensor sensor;
    sensor.hfov_rad = 1.2f; sensor.vfov_rad = 0.9f; sensor.height_m = 1.2f;
    std::vector<rc::nbv::Obstacle> walls;
    out.prop = planner.compute(b, 0.02f, 0.03f, sensor, walls, room);
    out.look_gain = out.prop.valid ? out.prop.epistemic_gain : 0.f;
    // the approach, as SpecificWorker::step_pragmatics prices it
    rc::DoorPragmaticCfg pg;
    const rc::door::Aperture ap = b.aperture();
    rc::door::ApproachProblem apb;
    apb.side = +1.f;
    apb.d_min = ap.w + 0.5f * pg.robot_passage_width_m + pg.swing_clearance_m;   // rc::door::safe_standoff (door_pragmatics.h pulls DSR)
    apb.d_max = std::max(pg.actuation_reach_m, apb.d_min + 0.30f);
    apb.sigma = pg.pose_sigma_floor_m;
    apb.sigma_arrival = rc::door::arrival_sigma(Eigen::Matrix3f::Zero(), 0.5f * (apb.d_min + apb.d_max), pg.pose_sigma_floor_m);
    apb.needed_w = pg.robot_passage_width_m + pg.passage_margin_m;
    apb.sigma_w = sd_w(b);
    apb.memory_s = pg.approach_refusal_memory_s;
    apb.footprint_r = 0.5f * pg.robot_passage_width_m + pg.passage_margin_m;
    const auto choice = rc::door::choose_approach_pose(ap, apb, rc::door::ApproachMemory{}, 0);
    const Eigen::Vector2f robot = ap.centre_xy() + robot_range * ap.across_u();
    const float p_reach = rc::door::reach_prob_at(ap, apb, robot);
    (void)p_reach;
    out.approach_value = pg.value_approach * choice.p_complete_live;
    if (out.prop.valid)
    {
        const Eigen::Vector2f v(out.prop.epistemic_target_x_m, out.prop.epistemic_target_y_m);
        const Eigen::Vector2f d = v - ap.centre_xy();
        out.look_standoff = d.norm();
        out.look_off_normal_deg = std::acos(std::clamp(std::abs(d.dot(ap.across_u())) / std::max(1e-6f, d.norm()), 0.f, 1.f)) * 57.2958f;
    }
    return out;
}
}  // namespace


int main()
{
    std::mt19937 rng(11);
    std::println("1. HONEST UNCERTAINTY: a belief the mask disagrees with REPORTS it");
    const auto good     = run(1.0f, 2.0f, 1.0f, true, rng);                 // a full mask of the true door
    const auto partial  = run(1.0f, 2.8f, 0.58f, true, rng, 0.62f);        // ab7_A1's first frame, persisting
    const auto collapse = run(1.0f, 2.8f, 0.58f, false, rng, 0.62f);       // the Task-W term ablated: the collapse
    std::println("    reported sd_w: good {:.3f} (w {:.3f}) | partial mask {:.3f} (w {:.3f}) | ablated {:.3f} (w {:.3f}); posterior sd_w good {:.3f} ablated {:.3f}",
                 sd_w(good), good.state().w, sd_w(partial), partial.state().w, sd_w(collapse), collapse.state().w,
                 std::sqrt(good.covariance()(1, 1)), std::sqrt(collapse.covariance()(1, 1)));
    check(sd_w(good) < 0.06f, "a belief its mask agrees with reports more than its prior sigma");
    check(sd_w(partial) > 2.f * sd_w(good), "a partial-mask belief does not look more than 2x less certain than a full-mask one");
    check(sd_w(collapse) > 0.15f, "the ablated (collapsed) belief is still reported as certain");
    check(sd_w(collapse) > 2.f * std::sqrt(collapse.covariance()(1, 1)), "the reported sd is not larger than the posterior's own");

    // the REAL collapse direction (ab7: w 1.00 -> 0.75): the belief is sitting at 0.75 when a partial mask arrives
    rc::DoorBelief narrow({3.10f, 0.75f, 2.1f, 0.f}, params());
    narrow.set_hinge_prob(1.0f); narrow.set_leaf_phi_sigma(0.146f);
    {
        const auto p = params();
        rc::DoorBelief truth({3.06f, 1.0f, 2.1f, 0.f}, p);
        rc::MaskBearings mb = camera(p, 3.56f, 2.8f);
        const auto [lo, hi] = truth.predicted_column_extent(truth.state(), mb);
        const float mid = 0.5f * (lo + hi) + 0.62f * (hi - lo), half = 0.29f * (hi - lo);
        mb.lo = mid - half; mb.hi = mid + half;
        const float R = 0.00235f, ppm = kFx / 2.8f;
        mb.var = ppm * ppm * R + 1.0f / 12.0f; mb.var_common = ppm * ppm * 0.0005f;
        rc::DoorFrame f;
        f.points = leaf_points(p, 3.06f, 1.0f, 2.07f, 3000, rng);
        f.R.assign(f.points.size(), R); f.chain_cov_s = 0.0122f; f.chain_cov_n = 0.0122f; f.bearings = mb;
        narrow.update(f);
    }
    std::println("    off-template (w {:.3f}) after one partial mask: reported sd_w {:.3f}", narrow.state().w, sd_w(narrow));
    check(sd_w(narrow) > 0.0f, "reported sd_w must exist");

    std::println("1b. EVIDENCE-DRIVEN: the error decays as views agree; a door the model cannot follow keeps it, but a look from where it persisted buys nothing");
    {
        const auto agree = run(0.95f, 2.0f, 1.0f, true, rng);   // an off-template (0.95 vs 1.00) door the belief CAN follow
        std::println("    truth 0.95, 12 full masks: w {:.3f}, reported sd_w {:.3f}, model error var {:.2e}", agree.state().w, sd_w(agree), agree.model_error_var().second);
        check(sd_w(agree) < 0.06f and agree.model_error_var().second < 4e-4f, "agreeing views did not bring the reported sigma down");
        const auto stuck = run(0.80f, 2.0f, 1.0f, true, rng);   // 0.80: the strong template cannot follow it, every mask disagrees
        const Eigen::Vector2f c = stuck.aperture().centre_xy(), n = stuck.aperture().across_u();
        std::println("    truth 0.80 (template-locked at w {:.3f}): reported sd_w {:.3f}; resolvable var_w from where it was seen {:.5f} vs reported {:.5f}",
                     stuck.state().w, sd_w(stuck), stuck.covariance_resolvable(c + 2.0f * n)(1, 1), stuck.covariance_reported()(1, 1));
        check(stuck.covariance_resolvable(c + 2.0f * n)(1, 1) < 0.5f * stuck.covariance_reported()(1, 1),
              "a persistent disagreement still counts as information to be had from the viewpoint it persisted at");
    }
    std::println("1c. NEAR-ZERO LEVER: an edge-on view injects no error (information form), it does not blow up");
    {
        auto p = params();
        rc::DoorBelief b({3.10f, 1.0f, 2.1f, 0.f}, p);
        rc::MaskBearings mb = camera(p, 3.56f, 2.8f);
        // camera far along the wall: grazing, the hull's width barely depends on w
        mb = camera(p, 3.56f + 40.0f, 0.05f);
        const auto [lo, hi] = b.predicted_column_extent(b.state(), mb);
        mb.lo = lo + 5.f; mb.hi = hi - 40.f; mb.var = 1.f; mb.var_common = 0.f;
        rc::DoorFrame f; f.R.assign(0, 0.f); f.bearings = mb;
        b.update(f);
        const float v = b.model_error_var().second;
        std::println("    grazing view, 45 px disagreement: model error var_w {:.2e}", v);
        check(std::isfinite(v) and v < 0.25f, "a near-zero lever blew the model error up");
    }
    std::println("1d. UNRESOLVABLE: error that persisted from a viewpoint is not information a look from there would buy");
    {
        const auto& b = partial;   // persistent partial mask from 2.8 m in front of the aperture centre
        const Eigen::Vector2f c = b.aperture().centre_xy(), n = b.aperture().across_u();
        const float at_same = b.covariance_resolvable(c + 2.8f * n)(1, 1), far_other = b.covariance_resolvable(c + 2.8f * Eigen::Vector2f(n.y(), -n.x()) * 0.9f + 0.5f * n)(1, 1);
        std::println("    resolvable var_w at the viewpoint it was measured from {:.4f}; from another aspect {:.4f}; reported {:.4f}", at_same, far_other, b.covariance_reported()(1, 1));
        check(at_same < b.covariance_reported()(1, 1) - 1e-4f, "error measured from this very viewpoint still counts as resolvable");
        check(far_other > at_same, "a different viewpoint does not keep more of the error to resolve");
    }

    std::println("2. ONE SCALE: look gain vs approach value (nats), robot 4 m in front of the door");
    const auto pg_ = prices(good, 4.0f), pp_ = prices(partial, 4.0f), pc_ = prices(narrow, 4.0f);
    const auto show = [](const char* n, const Prices& x)
    { std::println("    {:9}: look {:.3f} nats at {:.2f} m, {:.1f} deg off the normal | approach {:.3f}", n, x.look_gain, x.look_standoff, x.look_off_normal_deg, x.approach_value); };
    show("good", pg_); show("partial", pp_); show("off-tmpl", pc_);
    check(pg_.prop.valid and pp_.prop.valid and pc_.prop.valid, "the planner refused a proposal");
    check(pg_.approach_value > pg_.look_gain, "good belief: approach does not beat the look");
    check(pp_.look_gain > pp_.approach_value, "partial belief: the look does not beat the approach");
    check(pc_.look_gain > pc_.approach_value, "off-template belief: the look does not beat the approach");
    check(pc_.approach_value > 0.f, "a poor estimate WITHDREW the approach's value to exactly 0 (must only lower it)");
    check(pc_.look_off_normal_deg < 15.f, "the look's viewpoint is not within 15 deg of the door normal");
    std::println("    (the standoff is the detector envelope's argmax for THIS stub sensor: {:.2f} m)", pc_.look_standoff);
    std::println("3. THE OFFER IS NOT WITHDRAWN BY A NARROW-LOOKING WIDTH THAT IS UNCERTAIN (w 0.75, needed {:.2f})", 0.70f + 0.15f);
    {
        rc::DoorPragmaticCfg pg;
        rc::door::Aperture ap = good.aperture();
        ap.w = 0.75f;
        const auto offer = [&](float sigma_w)
        {
            rc::door::ApproachProblem apb;
            apb.side = +1.f;
            apb.d_min = ap.w + 0.5f * pg.robot_passage_width_m + pg.swing_clearance_m;
            apb.d_max = std::max(pg.actuation_reach_m, apb.d_min + 0.30f);
            apb.sigma = pg.pose_sigma_floor_m; apb.sigma_arrival = pg.pose_sigma_floor_m;
            apb.needed_w = pg.robot_passage_width_m + pg.passage_margin_m; apb.sigma_w = sigma_w;
            apb.footprint_r = 0.5f * pg.robot_passage_width_m + pg.passage_margin_m;
            const auto c = rc::door::choose_approach_pose(ap, apb, rc::door::ApproachMemory{}, 0);
            const auto f = rc::door::choose_approach_pose(ap, apb, rc::door::ApproachMemory{}, 0);
            return std::pair{rc::door::approach_offer_prob(c, f), c.p_complete_live};
        };
        const auto [p0, v0] = offer(0.0f);
        const auto [p1, v1] = offer(0.25f);
        std::println("    sigma_w 0: offer p {:.2f}, P(complete) {:.3f} | sigma_w 0.25: offer p {:.2f}, P(complete) {:.3f}", p0, v0, p1, v1);
        check(p0 < pg.withdraw_prob, "control: a confidently narrow width should withdraw the approach");
        check(p1 >= pg.offer_prob and v1 > 0.f, "an uncertain narrow-looking width withdrew the approach (it must only lower its value)");
        // P(passable) marginalised over the width: a wide-open leaf (phi 90 deg) of a 0.75 +- 0.25 aperture
        const std::vector<std::pair<float, double>> open_leaf{{1.5708f, 0.0}};
        const auto l0 = rc::door::passable_logit(open_leaf, 0.75f, 0.85f, 0.0f);
        const auto l1 = rc::door::passable_logit(open_leaf, 0.75f, 0.85f, 0.25f);
        check(l0.has_value() and l1.has_value() and std::isfinite(*l1) and rc::door::sigmoid(*l1) > 0.2 and rc::door::sigmoid(*l1) < 0.5,
              "P(passable) is not the width-marginalised probability (expected 0.2..0.5 for 0.75 +- 0.25 against 0.85)");
    }
    std::println("{} / {} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
