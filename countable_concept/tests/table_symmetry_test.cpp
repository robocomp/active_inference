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
//       variance (π²/4 for a rectangle: modes {0, π} at ½ each) and the cue rotates the box.
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
    std::printf(bad ? "table_symmetry_test FAILED (%d)\n" : "table_symmetry_test OK\n", bad);
    return bad ? 1 : 0;
}
