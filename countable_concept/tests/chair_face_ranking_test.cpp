// chair_face_ranking_test — the chair's NBV face ranking on the countable binary (shared-engine K2b; spec G5: "the chair and
// table self_test face-ranking checks must keep passing").
//
// Which check this is. The plan (Task 11) names "ChairBelief::self_test, including its NBV face-ranking check"; the legacy
// ChairBelief::self_test (chair_concept/src/chair_belief.cpp) has NO face-ranking check — it tests the SDF, pose recovery,
// association, the 4-way orientation, the rig prior and the reported covariance. The legacy chair's face ranking is its
// EpistemicPlanner, pinned against a verbatim pre-move copy by chair_concept/tests/planner_ab_test.cpp. The countable binary's
// equivalent is the toy-box ranking of common/ai_belief/tests/offer_return_view_test.cpp (c): seen only from one face, a box
// ranks another face first. This test is that check for the CHAIR on the countable binary — the seat/backrest/legs parts on
// the seat_top proportion (EngineQ), the class's own priors (chair_concept/etc/countable.toml) — through the agent's own
// EpistemicPlanner::gain_at and compute, i.e. the engine's predicted_gain.
//
// Prediction, stated before running (spec G5, the sign claim of offer_return_view_test (a)): after parked camera updates from
// the front, the shared camera error of that view is known (ρ = 1 along a zero path), so a look from the SEEN face promises
// less than a look from each of the other three faces, and the planner's own offer is not on the seen face.
// Run: LANG=es_ES.UTF-8 ./chair_face_ranking_test MANIFEST_DIR   (the class values: chair_concept/etc/countable.toml)
#include <array>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include "countable_belief.h"
#include "epistemic_planner.h"
#include <filesystem>
#include "countable_config.h"
#include <genericworker.h>   // the test stub: the real ConfigLoader (manifest_stub/)
#include "beam_fixture.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

// The truth: a dining chair at (0.4, −0.3), yaw 0 (front −y, backrest +y), w 0.50, h 0.47, H 0.95, seat top 0.48·H.
static constexpr float tx = 0.40f, ty = -0.30f, tw = 0.50f, th = 0.47f, tH = 0.95f, tq = 0.48f;
static const Eigen::Vector2f kRobot(tx, ty - 1.6f);   // parked in front

// The chair's class values come from chair_concept/etc/countable.toml through the agent's OWN loader (rc::load_countable_config,
// the real ConfigLoader, from the class folder as cwd, under es_ES), mapped onto the belief exactly as CountableFitter::run_inference
// maps them (fix round 1: no hand-copied values that could drift from the config). The beam config is the test fixture's.
static CountableBeliefParams params(const std::string& dir)
{
    std::filesystem::current_path(std::filesystem::path(REPO_DIR) / "chair_concept");
    ConfigLoader cl;
    cl.load("etc/countable.toml");
    const rc::CountableConfig c = rc::load_countable_config(cl);
    CountableBeliefParams p;
    p.sigma_base_m = c.ai2_sigma_base_m;
    p.prior_footprint_m = c.ai2_prior_footprint_m; p.prior_footprint_std = c.ai2_prior_footprint_std;
    p.prior_depth_m = c.ai2_prior_depth_m; p.prior_depth_std = c.ai2_prior_depth_std;
    p.prior_height_m = c.ai2_prior_height_m; p.prior_height_std = c.ai2_prior_height_std;
    p.plaus_height_min = c.plaus_height_min; p.plaus_height_soft = c.plaus_height_soft;
    p.wall_flush_prior = c.ai2_wall_flush_prior;
    p.room_axis_std = c.ai2_room_axis_std_deg * 0.017453293f;
    p.volatility_infer = c.ai2_volatility_infer;
    p.move_rate_per_day = c.move_rate_per_day; p.move_scale_m = c.move_scale_m; p.move_scale_rad = c.move_scale_rad;
    p.wall_standoff_m = c.wall_standoff_m; p.wall_standoff_std_m = c.wall_standoff_std_m;
    p.vertical_extent_m = c.vertical_extent_m;
    p.shape_parts = c.shape_parts; p.solid_parts = c.solid_parts; p.shape_proportions = c.shape_proportions;
    p.layer_spread_m = c.layer_spread_m;
    p.persistent = c.lidar_measures_on_gated;
    p.beam = rc::test::test_beam_config();
    (void)dir;
    return p;
}

// What a camera in front sees of the truth: the seat's front edge, the seat top, the backrest's front face, the front legs.
static CountableFrame front_frame(const CountableBeliefParams& p)
{
    CountableFrame f;
    const float st = tq * tH, su = (tq - 0.066f) * tH, yb = 0.5f * th - 0.097f * th, yf = -0.5f * th;
    const float leg = 0.076f * tw;
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 3; ++j)        // seat front edge
        f.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 12.0f, ty + yf, su + (st - su) * (j + 0.5f) / 3.0f);
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 6; ++j)        // seat top, front edge to the backrest
        f.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 12.0f, ty + yf + (yb - yf) * (j + 0.5f) / 6.0f, st);
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 8; ++j)        // backrest front face
        f.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 12.0f, ty + yb, st + (tH - st) * (j + 0.5f) / 8.0f);
    for (const float sx : {-1.0f, 1.0f}) for (int j = 0; j < 8; ++j)  // the two front legs' front faces
        for (int i = 0; i < 2; ++i)
            f.points.emplace_back(tx + sx * (0.5f * tw - 0.5f * leg) + leg * (i - 0.5f) * 0.5f, ty + yf, su * (j + 0.5f) / 8.0f);
    f.R.assign(f.points.size(), p.sigma_base_m * p.sigma_base_m);
    f.z_top_obs = tH;
    const float pp = p.common_mode_pos_std * p.common_mode_pos_std, ss = p.common_mode_size_std * p.common_mode_size_std,
                yy = p.common_mode_yaw_std * p.common_mode_yaw_std;
    f.sig2_cam = {pp, pp, ss, ss, ss, yy};
    f.sig2_L   = {pp, pp, ss, ss, ss, yy};
    f.sigma_pose = Eigen::Vector3d(1e-4, 1e-4, 0.0).asDiagonal();
    f.rho_cam.fill(1.0f); f.rho_L.fill(1.0f); f.rho_chain = 1.0f;
    f.robot_xy = kRobot;
    return f;
}

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    if (argc < 2) { std::printf("usage: %s MANIFEST_DIR\n", argv[0]); return 2; }
    const CountableBeliefParams p = params(argv[1]);
    CHECK(p.shape_parts.size() == 6 and p.solid_parts and p.shape_proportions.size() == 1 and p.persistent);

    CountableBelief b(CountableBeliefState{tx + 0.03f, ty - 0.02f, 0.92f, 0.48f, 0.46f, 0.0f}, p);
    CHECK(b.uses_q());   // the parts on the seat_top proportion: the EngineQ instantiation
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    const CountableFrame f = front_frame(p);
    for (int k = 0; k < 20; ++k) b.update(f);
    const auto& s = b.state();
    std::printf("fit: cx %.3f cy %.3f H %.3f w %.3f h %.3f yaw %.3f q %.3f (truth %.2f %.2f %.2f %.2f %.2f 0 %.2f)\n",
                s.cx, s.cy, s.H, s.w, s.h, s.yaw, b.proportion_value(0), tx, ty, tH, tw, th, tq);

    rc::nbv::Sensor cam;                     // the ZED as the agent models it (gain_query_test's values)
    cam.hfov_rad = 1.92f; cam.vfov_rad = 1.18f; cam.height_m = 0.98f;
    EpistemicPlanner planner;
    const Eigen::Vector2f c(tx, ty);
    // the four faces' standpoints, 1.6 m out along each outward normal, facing the centre: front (−y, seen), right, back, left
    const std::array<Eigen::Vector2f, 4> normal{Eigen::Vector2f(0, -1), Eigen::Vector2f(1, 0), Eigen::Vector2f(0, 1), Eigen::Vector2f(-1, 0)};
    const std::array<const char*, 4> name{"front (seen)", "right", "back", "left"};
    std::array<float, 4> g{};
    for (int k = 0; k < 4; ++k)
    {
        const Eigen::Vector2f sp = c + 1.6f * normal[k];
        g[k] = planner.gain_at(b, sp, std::atan2(c.y() - sp.y(), c.x() - sp.x()), cam);
        std::printf("  %-13s promised %.6f nats\n", name[k], g[k]);
        CHECK(std::isfinite(g[k]) and g[k] >= 0.0f);
    }
    for (int k = 1; k < 4; ++k) CHECK(g[k] > g[0]);   // every unseen face ranks above the seen one

    // the planner's own offer: its standpoint's face (the outward normal closest to the centre→standpoint direction) is not the front
    const EpistemicProposal prop = planner.compute(b, 0.0f, p.sigma_base_m, cam);
    CHECK(prop.valid);
    if (prop.valid)
    {
        const Eigen::Vector2f d = Eigen::Vector2f(prop.epistemic_target_x_m, prop.epistemic_target_y_m) - c;
        int face = 0;
        for (int k = 1; k < 4; ++k) if (normal[k].dot(d) > normal[face].dot(d)) face = k;
        std::printf("  offer at (%.3f, %.3f) promised %.6f nats: the %s face\n", prop.epistemic_target_x_m, prop.epistemic_target_y_m,
                    prop.promised_nats, name[face]);
        CHECK(face != 0);
    }
    std::printf(bad ? "chair_face_ranking_test FAILED (%d)\n" : "chair_face_ranking_test OK\n", bad);
    return bad ? 1 : 0;
}
