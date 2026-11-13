// test_colour_name.cpp — the appearance belief's brightness channel and the fixed-palette colour name.
// Build+run: common/appearance_belief/tests/run_tests.sh
#include <cmath>
#include <cstdio>
#include <numeric>
#include <optional>
#include <string>

#include "../appearance_belief.h"
#include "../colour_name.h"

static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (not (c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

using rc::appearance::AppearanceBelief;
using rc::appearance::name_colour;

namespace
{
/// An observed linear sRGB triple → what retina publishes: chromaticity and brightness.
Eigen::Vector3f chroma_of(float r, float g, float b) { const float s = r + g + b; return {r / s, g / s, b / s}; }
float lum_of(float r, float g, float b) { return (r + g + b) / 3.0f; }

/// A belief fed `frames` identical frames of one colour, the per-cell scatter `cell_var` (chroma) and `lum_var`.
AppearanceBelief seen(float r, float g, float b, int frames = 20, float cell_var = 1e-4f, float lum_var = 1e-3f)
{
    AppearanceBelief a;
    for (int i = 0; i < frames; ++i)
    {
        a.update(chroma_of(r, g, b), Eigen::Vector3f::Constant(cell_var), 30.0f);
        a.update_lum(lum_of(r, g, b), lum_var, 30.0f);
    }
    return a;
}

std::string name_of(const AppearanceBelief& a) { return name_colour(a).name; }
}   // namespace

int main()
{
    // ── brightness channel ──────────────────────────────────────────────────────────────────────────────
    {
        AppearanceBelief a;
        CHECK(not a.lum_initialized());
        a.update(chroma_of(0.5f, 0.5f, 0.5f), Eigen::Vector3f::Constant(1e-4f), 30.0f);
        CHECK(a.initialized() and not a.lum_initialized());   // an old retina (no brightness) leaves it unknown
        a.update_lum(0.4f, 1e-3f, 30.0f);
        CHECK(a.lum_initialized());
        CHECK(std::fabs(a.lum() - 0.4f) < 1e-3f);
        const float v1 = a.lum_variance();
        a.update_lum(0.4f, 1e-3f, 30.0f);
        CHECK(a.lum_variance() < v1);                          // evidence sharpens it
        const float v2 = a.lum_variance();
        a.inflate_for_age(10.0f);
        CHECK(a.lum_variance() > v2);                          // time widens it, like the chromaticity
        a.update_lum(0.6f, 1e-3f, 0.0f);                        // no cells: no information
        CHECK(std::fabs(a.lum() - 0.4f) < 1e-3f);
    }

    // ── naming: the Webots chairs, as lit (shading factor 0.6) ───────────────────────────────────────────
    CHECK(name_of(seen(0.60f, 0.18f, 0.60f)) == "magenta");    // world: color 1 0.3 1
    CHECK(name_of(seen(0.70f, 0.70f, 0.07f)) == "yellow");     // world: color 1 1 0.1
    {
        const auto c = name_colour(seen(0.60f, 0.18f, 0.60f));
        CHECK(c.prob > 0.5f);
        const float total = std::accumulate(c.probs.begin(), c.probs.end(), 0.0f);
        CHECK(std::fabs(total - 1.0f) < 1e-4f);
    }

    // ── brightness separates what chromaticity cannot ────────────────────────────────────────────────────
    CHECK(name_of(seen(0.90f, 0.90f, 0.90f)) == "white");
    CHECK(name_of(seen(0.50f, 0.50f, 0.50f)) == "grey");
    // A dark surface: retina's quantisation floor makes its chromaticity scatter large.
    CHECK(name_of(seen(0.07f, 0.06f, 0.06f, 20, 2e-2f)) == "black");
    CHECK(name_of(seen(0.45f, 0.28f, 0.15f)) == "brown");
    CHECK(name_of(seen(0.95f, 0.55f, 0.10f)) == "orange");
    CHECK(name_of(seen(0.15f, 0.55f, 0.20f)) == "green");
    CHECK(name_of(seen(0.15f, 0.30f, 0.80f)) == "blue");
    CHECK(name_of(seen(0.75f, 0.10f, 0.10f)) == "red");

    // ── no brightness measured: the neutral terms share the probability (no guess) ─────────────────────
    {
        AppearanceBelief a;
        for (int i = 0; i < 20; ++i) a.update(chroma_of(0.5f, 0.5f, 0.5f), Eigen::Vector3f::Constant(1e-4f), 30.0f);
        const auto c = name_colour(a);
        CHECK(c.name == "white" or c.name == "grey" or c.name == "black");
        CHECK(c.prob < 0.5f);
    }

    // ── pure black: retina gives brightness but NO chromaticity (every pixel R+G+B = 0) ────────────────────
    {
        AppearanceBelief a;
        for (int i = 0; i < 20; ++i) a.update_lum(0.0f, 1e-5f, 30.0f);
        CHECK(not a.initialized() and a.lum_initialized());
        CHECK(name_colour(a).name == "black");
        const float v = a.lum_variance();
        a.inflate_for_age(10.0f);
        CHECK(a.lum_variance() > v);                           // brightness drifts even without chromaticity
    }

    // ── nothing measured: no name ────────────────────────────────────────────────────────────────────────
    CHECK(name_colour(AppearanceBelief{}).name.empty());

    std::printf("test_colour_name: %d checks, %d failures\n", checks, fails);
    return fails == 0 ? 0 : 1;
}
