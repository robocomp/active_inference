// table_face_ranking_test — the table's NBV face ranking on the countable binary (shared-engine K2c; spec G5: "the chair and
// table self_test face-ranking checks must keep passing").
//
// Which check this is. The legacy TableBelief::self_test (table_concept/src/table_belief.cpp, case (f)) ranks faces by the D-optimal
// gain ½·ln det(I + Σ·ΔI) of the legacy belief's own information model; that belief and its predicted_information do not exist on
// the countable binary, so the legacy check cannot be run against it. The countable equivalent is the one K2b wrote for the chair
// (chair_face_ranking_test, after offer_return_view_test (c)): the agent's own EpistemicPlanner::gain_at and compute, i.e. the
// engine's predicted_gain, on the TABLE — top + four legs on the top_under proportion (EngineQ), symmetry c2v, the class's own
// priors loaded from table_concept/etc/countable.toml through the agent's loader.
//
// Prediction, stated before running (spec G5, the sign claim of offer_return_view_test (a)): after parked camera updates from the
// −y face, the shared camera error of that view is known (ρ = 1 along a zero path), so a look from the SEEN face promises less than
// a look from each of the other three faces, and the planner's own offer is not on the seen face.
// Run: LANG=es_ES.UTF-8 ./table_face_ranking_test MANIFEST_DIR   (the class values: table_concept/etc/countable.toml)
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

// The truth: a dining table at (0.4, −0.3), yaw 0, top 1.60 x 0.90, top surface at 0.74, a 3 cm top (top_under q = 1 − 0.03/0.74),
// legs laid by the manifest's fractions (0.048 of each extent, centres at ±0.441).
static constexpr float tx = 0.40f, ty = -0.30f, tw = 1.60f, th = 0.90f, tH = 0.74f, tq = 1.0f - 0.03f / 0.74f;
static constexpr float kOut = 2.0f;                    // standpoints 2 m out from the centre along each face normal
static const Eigen::Vector2f kRobot(tx, ty - kOut);   // parked in front of the −y face

// The table's class values come from table_concept/etc/countable.toml through the agent's OWN loader (rc::load_countable_config,
// the real ConfigLoader, from the class folder as cwd, under es_ES), mapped onto the belief exactly as CountableFitter::run_inference
// maps them (fix round 1: no hand-copied values that could drift from the config). The beam config is the test fixture's.
static CountableBeliefParams params(const std::string& dir)
{
    std::filesystem::current_path(std::filesystem::path(REPO_DIR) / "table_concept");
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
    p.symmetric = c.shape_symmetric;
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

// What a camera in front (z 0.98 m, above the top) sees of the truth: the top surface, the top's front edge, the four legs' −y faces
// (the back legs between the front ones, under the top).
static CountableFrame front_frame(const CountableBeliefParams& p)
{
    CountableFrame f;
    const float tu = tq * tH, lx = 0.048f * tw, ly = 0.048f * th, ax = 0.441f * tw, ay = 0.441f * th;
    for (int i = 0; i < 16; ++i) for (int j = 0; j < 9; ++j)        // top surface
        f.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 16.0f, ty - 0.5f * th + th * (j + 0.5f) / 9.0f, tH);
    for (int i = 0; i < 16; ++i) for (int j = 0; j < 2; ++j)        // the top's front edge
        f.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 16.0f, ty - 0.5f * th, tu + (tH - tu) * (j + 0.5f) / 2.0f);
    for (const float sx : {-1.0f, 1.0f}) for (const float sy : {-1.0f, 1.0f})   // the legs' −y faces
        for (int j = 0; j < 8; ++j) for (int i = 0; i < 2; ++i)
            f.points.emplace_back(tx + sx * ax + lx * (i - 0.5f) * 0.5f, ty + sy * ay - 0.5f * ly, tu * (j + 0.5f) / 8.0f);
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
    CHECK(p.shape_parts.size() == 5 and p.solid_parts and p.shape_proportions.size() == 1 and p.persistent and p.symmetric);

    CountableBelief b(CountableBeliefState{tx + 0.03f, ty - 0.02f, 0.72f, 1.55f, 0.88f, 0.0f}, p);
    CHECK(b.uses_q());   // the parts on the top_under proportion: the EngineQ instantiation
    b.set_room_interior(Eigen::Vector2f(0.0f, -4.0f));
    const CountableFrame f = front_frame(p);
    for (int k = 0; k < 20; ++k) b.update(f);
    const auto& s = b.state();
    std::printf("fit: cx %.3f cy %.3f H %.3f w %.3f h %.3f yaw %.3f q %.3f (truth %.2f %.2f %.2f %.2f %.2f 0 %.2f)\n",
                s.cx, s.cy, s.H, s.w, s.h, s.yaw, b.proportion_value(0), tx, ty, tH, tw, th, tq);

    rc::nbv::Sensor cam;                     // the ZED as the agent models it (gain_query_test's values)
    cam.hfov_rad = 1.92f; cam.vfov_rad = 1.18f; cam.height_m = 0.98f;
    EpistemicPlanner planner;
    const Eigen::Vector2f c(tx, ty);
    // the four faces' standpoints, 2 m out along each outward normal, facing the centre: front (−y, seen), right, back, left
    const std::array<Eigen::Vector2f, 4> normal{Eigen::Vector2f(0, -1), Eigen::Vector2f(1, 0), Eigen::Vector2f(0, 1), Eigen::Vector2f(-1, 0)};
    const std::array<const char*, 4> name{"front (seen)", "right", "back", "left"};
    std::array<float, 4> g{};
    for (int k = 0; k < 4; ++k)
    {
        const Eigen::Vector2f sp = c + kOut * normal[k];
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
    std::printf(bad ? "table_face_ranking_test FAILED (%d)\n" : "table_face_ranking_test OK\n", bad);
    return bad ? 1 : 0;
}
