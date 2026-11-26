// table_symmetry_test — K2c: why the countable engine needs no quotient chart for the table, checked on the class's own data.
//
// The decision (task-K2c-report.md §2): a table has no front, so (w, h, yaw), (w, h, yaw + π) and (h, w, yaw ± π/2) are one
// table. The group acts freely on the (w, h, yaw) chart (every non-identity element moves yaw), so a Gaussian on one sheet IS a
// Gaussian on the quotient, PROVIDED every term of the model is invariant under the group — then nothing tells the sheets apart
// and the four "front modes" of the countable belief are one body, not four hypotheses. Three checks:
//   (a) the manifest: table.concept.toml declares [shape] symmetry = "c2v" and its parts pass the loader's invariance check; a
//       class with a front (the chair's parts) declared c2v is refused; the other classes are "none";
//   (b) the solid: the parts SDF (common/ai_belief/parts_geometry.h, the camera SDF the engine uses) is the same function of the
//       point at θ and at each group image of θ — within float roundoff (bound derived below, before running); the chair's is not;
//   (c) the belief: with symmetric = true the front-mode posterior is a point mass, the REPORTED yaw variance is the within-sheet
//       Σ(5,5), and a front cue rotates nothing; with symmetric = false (the same belief otherwise) the undecided modes add their
//       variance (π²/4 for a rectangle: modes {0, π} at ½ each) and the cue rotates the box;
//   (d) the whole update is equivariant (fix round 1): a belief built from the TABLE's own loaded params
//       (table_concept/etc/countable.toml through rc::load_countable_config) and its group image (w↔h, yaw + π/2), given the SAME
//       frames — a table seen from the front with a room wall staged 5 cm behind one side — end at states that map onto each other.
//       Bound, stated before running: the persistent engine stops when the undamped step is below gn_tol_sigma·σ_k on every θ
//       component and then applies it, so two runs of the same problem can differ by at most that step: |Δθ_k| ≤ gn_tol_sigma·σ_k
//       (σ_k the posterior's, at the end), per component after the group map (yaw compared modulo π/2's image).
// Run: LANG=es_ES.UTF-8 ./table_symmetry_test
#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <numbers>
#include <random>
#include <unistd.h>
#include "countable_belief.h"
#include "countable_vertical_span.h"
#include "ai_belief/parts_geometry.h"
#include "shape_parts.h"
#include "countable_config.h"
#include <genericworker.h>   // the test stub: the real ConfigLoader (manifest_stub/)
#include "beam_fixture.h"

#ifndef MANIFEST_DIR
#error "MANIFEST_DIR must name common/concept_manifest"
#endif
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static std::string mf(const char* cls) { return std::string(MANIFEST_DIR) + "/" + cls + ".concept.toml"; }

static bool symmetry_throws(const std::string& body)
{
    const std::string p = (std::filesystem::temp_directory_path() / std::format("table_symmetry_test_{}.toml", ::getpid())).string();
    { std::ofstream o(p); o << body; }
    bool threw = false;
    try { rc::manifest::load_shape_symmetry(p); } catch (const std::runtime_error&) { threw = true; }
    std::filesystem::remove(p);
    return threw;
}

// max |sdf(p; θ) − sdf(p; g·θ)| over random points, for the three non-identity group elements
static float max_sdf_gap(const std::vector<rc::manifest::ShapePart>& parts, float H, float w, float h, float yaw)
{
    using rc::ai::parts::sdf;
    const rc::ai::parts::Table T{.parts = parts, .solid_is_parts = true};
    const rc::countable::VerticalSpan span{};
    constexpr float cx = 0.7f, cy = -0.4f, pi = std::numbers::pi_v<float>;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> Ux(cx - 1.5f, cx + 1.5f), Uy(cy - 1.5f, cy + 1.5f), Uz(-0.1f, 1.0f);
    float gap = 0.0f;
    for (int i = 0; i < 20000; ++i)
    {
        const Eigen::Vector3f p(Ux(rng), Uy(rng), Uz(rng));
        const float d0 = sdf(p, cx, cy, H, w, h, yaw, T, span);
        gap = std::max({gap, std::abs(d0 - sdf(p, cx, cy, H, w, h, yaw + pi, T, span)),
                             std::abs(d0 - sdf(p, cx, cy, H, h, w, yaw + 0.5f * pi, T, span)),
                             std::abs(d0 - sdf(p, cx, cy, H, h, w, yaw - 0.5f * pi, T, span))});
    }
    return gap;
}

// The table's params as the agent builds them (CountableFitter::run_inference's mapping, as in table_face_ranking_test).
static rc::CountableBeliefParams table_params()
{
    const auto cwd = std::filesystem::current_path();
    std::filesystem::current_path(std::filesystem::path(REPO_DIR) / "table_concept");
    ConfigLoader cl;
    cl.load("etc/countable.toml");
    const rc::CountableConfig c = rc::load_countable_config(cl);
    std::filesystem::current_path(cwd);
    rc::CountableBeliefParams p;
    p.sigma_base_m = c.ai2_sigma_base_m;
    p.prior_footprint_m = c.ai2_prior_footprint_m; p.prior_footprint_std = c.ai2_prior_footprint_std;
    p.prior_depth_m = c.ai2_prior_depth_m; p.prior_depth_std = c.ai2_prior_depth_std;
    p.prior_height_m = c.ai2_prior_height_m; p.prior_height_std = c.ai2_prior_height_std;
    p.plaus_height_min = c.plaus_height_min; p.plaus_height_soft = c.plaus_height_soft;
    p.wall_flush_prior = c.ai2_wall_flush_prior;
    p.wall_precision = c.ai2_wall_precision; p.wall_parallel_precision = c.ai2_wall_parallel_precision;
    p.wall_reach_m = c.ai2_wall_reach_m; p.wall_no_cross_precision = c.ai2_wall_no_cross_precision;
    p.wall_explain_frac = c.ai2_wall_explain_frac; p.wall_explain_sigma_m = c.ai2_wall_explain_sigma_m;
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
    return p;
}

// A dining table at (0.4, −0.3), yaw 0, top 1.60 x 0.90 at 0.74, a 3 cm top, seen from the −y side (as table_face_ranking_test),
// with a room wall staged 5 cm behind its +y face (normal −y, into the room).
static rc::CountableFrame wall_frame(const rc::CountableBeliefParams& p)
{
    constexpr float tx = 0.40f, ty = -0.30f, tw = 1.60f, th = 0.90f, tH = 0.74f, tq = 1.0f - 0.03f / 0.74f;
    rc::CountableFrame f;
    const float tu = tq * tH, lx = 0.048f * tw, ly = 0.048f * th, ax = 0.441f * tw, ay = 0.441f * th;
    for (int i = 0; i < 16; ++i) for (int j = 0; j < 9; ++j)
        f.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 16.0f, ty - 0.5f * th + th * (j + 0.5f) / 9.0f, tH);
    for (int i = 0; i < 16; ++i) for (int j = 0; j < 2; ++j)
        f.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 16.0f, ty - 0.5f * th, tu + (tH - tu) * (j + 0.5f) / 2.0f);
    for (const float sx : {-1.0f, 1.0f}) for (const float sy : {-1.0f, 1.0f})
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
    f.robot_xy = Eigen::Vector2f(tx, ty - 2.0f);
    f.wall.ok = true; f.wall.p = Eigen::Vector2f(tx, ty + 0.5f * th + 0.05f); f.wall.n = Eigen::Vector2f(0.0f, -1.0f);
    f.wall.sigma_m = 0.02f; f.wall.length_m = 3.0f; f.wall.room = 1; f.wall.edge = 0;
    return f;
}

int main()
{
    std::setlocale(LC_ALL, "");
    using namespace rc::manifest;

    // (a) the manifest
    const auto table = load_shape_parts(mf("table"));
    CHECK(table.size() == 5 and load_solid_is_parts(mf("table")));
    CHECK(load_shape_symmetry(mf("table")) == Symmetry::c2v);
    CHECK(parts_invariant_c2v(table));
    for (const char* cls : {"chair", "bed", "nightstand", "toilet", "refrigerator", "hood"})
        CHECK(load_shape_symmetry(mf(cls)) == Symmetry::none);
    CHECK(not parts_invariant_c2v(load_shape_parts(mf("chair"))));
    const std::string leg = "[[shape.part]]\nname = \"l\"\nx_f = 0.4\ny_f = 0.4\nw_f = 0.1\nh_f = 0.1\nz0_f = 0\nz1_f = 1\n";
    CHECK(symmetry_throws("[shape]\nsymmetry = \"c2v\"\n" + leg));                  // one corner leg: not invariant
    CHECK(symmetry_throws("[shape]\nsymmetry = \"d4\"\n"));                         // an unknown group
    CHECK(symmetry_throws("[shape]\nsymmetry = c2v\n"));                            // unquoted
    CHECK(not symmetry_throws("[shape]\nsymmetry = \"none\"\n" + leg));

    // (b) the solid. Bound, stated before running: points within 2.2 m of the centre; the rotation of yaw + π/2 in float is off by
    // at most an ulp of the angle (≤ 2.4e-7 rad for |angle| < 4), which moves a local coordinate by ≤ 2.2 · 2 · 2.4e-7 ≈ 1.1e-6 m;
    // the ~10 float operations of to_local / part_sdf on values ≤ 2.2 m add ≤ 10 · 2.4e-7 ≈ 2.4e-6 m. Total < 4e-6 m; use 1e-5 m.
    constexpr float kTol = 1e-5f;
    const float g_table = max_sdf_gap(table, 0.74f, 1.60f, 0.90f, 0.3f);
    const float g_square = max_sdf_gap(table, 0.45f, 0.80f, 0.80f, -1.1f);
    const float g_chair = max_sdf_gap(load_shape_parts(mf("chair")), 0.95f, 0.50f, 0.47f, 0.3f);
    std::printf("max |sdf(p; θ) − sdf(p; g·θ)|: table 1.60 x 0.90 %.2e m, table 0.80 x 0.80 %.2e m, chair %.3f m (bound %.0e m)\n",
                g_table, g_square, g_chair, kTol);
    CHECK(g_table < kTol and g_square < kTol);
    CHECK(g_chair > kTol);   // the check can tell: a class with a front is not invariant

    // (c) the belief's reported yaw and the front resolver
    for (const bool sym : {true, false})
    {
        rc::CountableBeliefParams p;
        p.symmetric = sym;
        rc::CountableBelief b(rc::CountableBeliefState{0.7f, -0.4f, 0.74f, 1.60f, 0.90f, 0.3f}, p);
        const auto post = b.front_posterior();
        const float within = b.covariance()(5, 5), reported = b.yaw_marginal_var();
        std::printf("symmetric=%d: front posterior (%.2f %.2f %.2f %.2f), yaw var within %.5f reported %.5f\n", int(sym),
                    post[0], post[1], post[2], post[3], within, reported);
        rc::FrontCue cue;                         // a decisive cue: "the door faces −(the current front)", i.e. mode 2
        cue.bearing_rad = 0.3f + 0.5f * std::numbers::pi_v<float>;
        cue.confidence = 1.0f;
        bool rotated = false;
        for (int k = 0; k < 10; ++k) { cue.view_bearing_rad = -1.5f + 0.3f * float(k); rotated = b.resolve_front(cue) or rotated; }
        if (sym)
        {
            CHECK(post[0] == 1.0f and post[1] == 0.0f and post[2] == 0.0f and post[3] == 0.0f);
            CHECK(reported == within);
            CHECK(not rotated and b.state().yaw == 0.3f and b.state().w == 1.60f);
            CHECK(not b.resolve_front_geometric());
        }
        else
        {   // rectangle (|w−h|/(w+h) = 0.28): modes {0, π} undecided at ½ each ⇒ + (π/2)² = π²/4 (float rounding of ~2.47 ≪ 1e-5)
            CHECK(std::abs((reported - within) - std::numbers::pi_v<float> * std::numbers::pi_v<float> / 4.0f) < 1e-5f);
            CHECK(rotated);
        }
    }
    // (d) equivariance of the whole update on the table's own params
    {
        const rc::CountableBeliefParams p = table_params();
        CHECK(p.symmetric and p.persistent and p.shape_parts.size() == 5);
        constexpr float hp = 0.5f * std::numbers::pi_v<float>;
        rc::CountableBelief A(rc::CountableBeliefState{0.43f, -0.32f, 0.72f, 1.55f, 0.88f, 0.0f}, p);
        rc::CountableBelief B(rc::CountableBeliefState{0.43f, -0.32f, 0.72f, 0.88f, 1.55f, hp}, p);   // the image under swap∘r_π/2
        for (auto* b : {&A, &B}) b->set_room_interior(Eigen::Vector2f(0.0f, -4.0f));
        const rc::CountableFrame f = wall_frame(p);
        for (int k = 0; k < 10; ++k) { A.update(f); B.update(f); }
        const auto& a = A.state(); const auto& bs = B.state();
        const auto S = A.covariance();
        const double tol = p.gn_tol_sigma;
        const float dyaw = std::remainder(bs.yaw - a.yaw - hp, std::numbers::pi_v<float>);   // modulo the r_π image
        const std::array<float, 6> d{bs.cx - a.cx, bs.cy - a.cy, bs.H - a.H, bs.w - a.h, bs.h - a.w, dyaw};
        const std::array<int, 6> k_of{0, 1, 2, 4, 3, 5};   // B's component ↔ A's
        std::printf("equivariance: A (%.4f %.4f %.4f %.4f %.4f %.4f q %.4f)  B (%.4f %.4f %.4f %.4f %.4f %.4f q %.4f)\n", a.cx, a.cy, a.H,
                    a.w, a.h, a.yaw, A.proportion_value(0), bs.cx, bs.cy, bs.H, bs.w, bs.h, bs.yaw, B.proportion_value(0));
        for (int i = 0; i < 6; ++i)
        {
            const double bound = tol * std::sqrt(double(S(k_of[i], k_of[i])));
            std::printf("  component %d: |Δ| %.3e  bound %.3e\n", i, std::abs(double(d[i])), bound);
            CHECK(std::abs(double(d[i])) <= bound);
        }
    }
    std::printf(bad ? "table_symmetry_test FAILED (%d)\n" : "table_symmetry_test OK\n", bad);
    return bad ? 1 : 0;
}
