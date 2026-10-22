// test_door_look_loop — Task E fix round 2: the look <-> approach LOOP, with the REAL producer in the loop.
// DoorBelief (the fit, fed synthetic masks from wherever the robot stands) -> EpistemicPlanner (covariance_resolvable) -> the
// approach offer's price -> the controller's DoorPassage::choose_look_or_step. The door is PERSISTENTLY OFF-TEMPLATE: its true
// width (0.80 m) is one the strong template (1.00 m) cannot follow, so every full mask disagrees with the belief for ever.
// Asserts: the look's resolvable gain falls once the look was taken; the passage reaches the approach (-> Cross) in a bounded
// number of cycles; no look -> look-from-another-face alternation.
// Build (from controller/tests):
//   g++ -std=c++23 -O1 -fPIC -I/usr/include/eigen3 -I../../common -DSHADOW_JSON="\"$PWD/../../robot_concept/shadow.json\"" \
//     $(pkg-config --cflags Qt6Core) test_door_look_loop.cpp ../../door_concept/src/door_belief.cpp ../../door_concept/src/epistemic_planner.cpp \
//     -o /tmp/tdll -ldsr_api -ldsr_core -lfastcdr -lfastdds $(pkg-config --libs Qt6Core) && /tmp/tdll
#include "../src/controller_door_passage.h"
#include "../../door_concept/src/door_belief.h"
#include "../../door_concept/src/door_config.h"
#include "../../door_concept/src/door_approach_pose.h"
#include "../../door_concept/src/epistemic_planner.h"

#include <print>
#include <random>

namespace
{
int fails = 0, checks = 0;
void check(bool ok, const char* what) { ++checks; if (not ok) { ++fails; std::println("  FAIL: {}", what); } }
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
// a camera at `pos` (room frame, 1 m high) looking at `target`
rc::MaskBearings camera_at(const Eigen::Vector2f& pos, const Eigen::Vector2f& target)
{
    const Eigen::Vector2f f = (target - pos).normalized(), r(f.y(), -f.x());
    rc::MaskBearings mb;
    Eigen::Matrix3f R; R << r.x(), r.y(), 0, f.x(), f.y(), 0, 0, 0, 1;
    const Eigen::Vector3f C(pos.x(), pos.y(), 1.0f);
    mb.zed_T_room.leftCols<3>() = R; mb.zed_T_room.col(3) = -R * C;
    mb.fx = kFx; mb.cx = kCx; mb.jamb_depth = 0.0f;
    return mb;
}
}  // namespace

static void scenario(const char* name, bool always_partial)
{
    std::println("{}", name);
    std::mt19937 rng(5);
    const auto p = params();
    const float s_true = 3.06f, w_true = 0.80f;
    rc::DoorBelief truth({s_true, w_true, 2.1f, 0.f}, p);
    rc::DoorBelief b({3.10f, 1.0f, 2.1f, 0.f}, p);
    b.set_hinge_prob(1.0f); b.set_leaf_phi_sigma(0.146f);

    const std::vector<Eigen::Vector2f> room{{-1.f, -6.f}, {8.f, -6.f}, {8.f, 2.538f}, {-1.f, 2.538f}};
    rc::EpistemicPlanner planner(1.8f);
    rc::nbv::Sensor sensor; sensor.hfov_rad = 1.92f; sensor.vfov_rad = 1.22f; sensor.height_m = 1.0f;
    planner.set_detector_envelope(rc::detect::DetectorEnvelope{0.10f, 0.60f, 0.06f});
    rc::DoorPragmaticCfg pg;
    const float lambda = 0.2f, margin = 0.5f;

    // frames from where the robot stands (5 per cycle): the full mask of the TRUE door, which the template-locked belief disagrees with
    const auto observe = [&](const Eigen::Vector2f& pos, float frac = 1.0f, float off = 0.0f)
    {
        for (int k = 0; k < 5; ++k)
        {
            rc::MaskBearings mb = camera_at(pos, truth.aperture().centre_xy());
            const auto [lo, hi] = truth.predicted_column_extent(truth.state(), mb);
            if (not (std::isfinite(lo) and std::isfinite(hi))) return;
            const float mid = 0.5f * (lo + hi) + off * (hi - lo), half = 0.5f * frac * (hi - lo);
            mb.lo = mid - half; mb.hi = mid + half;
            const float R = 0.00235f, ppm = kFx / std::max(0.5f, (truth.aperture().centre_xy() - pos).norm());
            mb.var = ppm * ppm * R + 1.0f / 12.0f; mb.var_common = ppm * ppm * 0.0005f + b.hull_mismatch_var(mb);
            rc::DoorFrame f;
            f.points = leaf_points(p, s_true, w_true, 2.07f, 2000, rng);
            f.R.assign(f.points.size(), R); f.chain_cov_s = 0.0122f; f.chain_cov_n = 0.0122f; f.bearings = mb;
            b.update(f);
        }
    };

    Eigen::Vector2f robot = b.aperture().centre_xy() + 4.0f * b.aperture().across_u();
    observe(robot, 0.58f, 0.62f);   // ab7_A1's first sight: a small partial mask, far, off to one side
    bool look_incumbent = false;
    int looks = 0, cycles = 0, consecutive_looks = 0, max_consecutive = 0;
    bool reached = false;
    float gain_before_look = 0.f, gain_after_look = 0.f;
    std::vector<Eigen::Vector2f> look_views;
    for (; cycles < 30 and not reached; ++cycles)
    {
        const auto prop = planner.compute(b, 0.02f, 0.03f, sensor, {}, room);
        const float look_gain = prop.valid ? prop.epistemic_gain : 0.f;
        // the approach offer, as SpecificWorker prices it
        const rc::door::Aperture ap = b.aperture();
        rc::door::ApproachProblem apb;
        apb.side = +1.f;
        apb.d_min = ap.w + 0.5f * pg.robot_passage_width_m + pg.swing_clearance_m;
        apb.d_max = std::max(pg.actuation_reach_m, apb.d_min + 0.30f);
        apb.sigma = pg.pose_sigma_floor_m; apb.sigma_arrival = pg.pose_sigma_floor_m;
        apb.needed_w = pg.robot_passage_width_m + pg.passage_margin_m;
        apb.sigma_w = std::sqrt(std::max(0.0f, b.covariance_reported()(1, 1)));
        apb.footprint_r = 0.5f * pg.robot_passage_width_m + pg.passage_margin_m;
        const auto ch = rc::door::choose_approach_pose(ap, apb, rc::door::ApproachMemory{}, 0);
        const float appr_value = pg.value_approach * ch.p_complete_live;
        const Eigen::Vector2f appr_xy(ch.pose.x, ch.pose.y), look_xy(prop.epistemic_target_x_m, prop.epistemic_target_y_m);
        const auto c = rc::DoorPassage::choose_look_or_step(appr_value, (robot - appr_xy).norm(),
                                                            prop.valid and look_gain > 0.f ? std::optional<float>(look_gain) : std::nullopt,
                                                            (robot - look_xy).norm(), lambda, margin, look_incumbent);
        std::println("  cycle {:2}: look {:.3f} nats at ({:.2f},{:.2f}) | approach {:.3f} | sd_w rep {:.3f} err {:.1e} | -> {}", cycles, look_gain,
                     look_xy.x(), look_xy.y(), appr_value, apb.sigma_w, b.model_error_var().second, c.look ? "LOOK" : "APPROACH");
        look_incumbent = c.look;
        if (c.look)
        {
            if (looks == 0) gain_before_look = look_gain;
            ++looks; ++consecutive_looks; max_consecutive = std::max(max_consecutive, consecutive_looks);
            look_views.push_back(look_xy);
            robot = look_xy;                 // drives to the viewpoint...
            if (always_partial) observe(robot, 0.58f, 0.62f); else observe(robot);   // ...and sees the (persistently off-template) door from there
            const auto after = planner.compute(b, 0.02f, 0.03f, sensor, {}, room);
            if (looks == 1) gain_after_look = after.valid ? after.epistemic_gain : 0.f;
            look_incumbent = false;          // the look completed; the producer re-offers whatever gain is left
        }
        else
        {
            consecutive_looks = 0;
            robot = appr_xy; if (always_partial) observe(robot, 0.58f, 0.62f); else observe(robot);
            reached = true;                  // the approach is taken: the passage goes on to open/cross
        }
    }
    std::println("  {} look(s), approach taken at cycle {}, look gain {:.3f} -> {:.3f} after the first look", looks, cycles, gain_before_look, gain_after_look);
    check(reached and cycles <= 8, "the passage did not reach its approach (-> Cross) in a bounded number of cycles");
    check(looks == 0 or gain_after_look < gain_before_look, "the resolvable gain did not fall after the look");
    check(max_consecutive <= 1 and looks <= 2, "look -> look from another face alternation");
}

int main()
{
    scenario("A. persistent off-template width (0.80 vs 1.00): full masks from every viewpoint disagree", false);
    scenario("B. WORST CASE: the mask is a persistent partial hull from EVERY viewpoint (no view ever resolves it)", true);
    std::println("test_door_look_loop: {} checks, {} failures", checks, fails);
    return fails == 0 ? 0 : 1;
}
