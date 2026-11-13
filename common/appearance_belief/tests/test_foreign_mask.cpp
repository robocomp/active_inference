// test_foreign_mask.cpp — one frame whose mask is not the object (the wall, the table) must not rename it.
// Build+run: common/appearance_belief/tests/run_tests.sh
//
// Measured case (2026-11-11, chat_e2e8/e2e9 colour traces): chair_35 (yellow, chromaticity 0.434/0.429/0.137,
// brightness 0.74) received single frames of chromaticity 0.346/0.334/0.320, brightness 0.84, 221 cells with a
// tiny spread: a neutral surface credited to the chair. Each one renamed it "grey"/"white" at p ~0.4, because a
// frame's noise was its between-cell spread only, ~1e-5, and the belief's own variance after 0.1 s of drift was
// larger, so the newest frame won outright.
#include <cmath>
#include <cstdio>
#include <string>

#include "../appearance_belief.h"
#include "../colour_name.h"

static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (not (c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

using rc::appearance::AppearanceBelief;
using rc::appearance::name_colour;

namespace
{
struct Frame { Eigen::Vector3f chroma; float var, n, lum, lum_var, lum_n; };
const Frame kYellow{{0.434f, 0.429f, 0.137f}, 2e-5f, 20.0f, 0.737f, 1e-4f, 20.0f};
const Frame kMagenta{{0.409f, 0.185f, 0.406f}, 7e-5f, 30.0f, 0.776f, 5e-3f, 30.0f};
const Frame kWall{{0.346f, 0.334f, 0.320f}, 3e-4f, 221.0f, 0.840f, 2e-2f, 221.0f};

/// One cycle as chair_concept runs it: 0.1 s of drift, then the frame.
void step(AppearanceBelief& a, const Frame& f)
{
    a.inflate_for_age(0.1f);
    a.observe(f.chroma, Eigen::Vector3f::Constant(f.var), f.n, f.lum, f.lum_var, f.lum_n);
}
}   // namespace

int main()
{
    // ── a foreign frame among the object's frames does not rename it ─────────────────────────────────────
    {
        AppearanceBelief a;
        for (int i = 0; i < 30; ++i) step(a, kYellow);
        CHECK(name_colour(a).name == "yellow");
        CHECK(a.last_weight() > 0.9f);                     // its own frames count fully
        step(a, kWall);
        CHECK(a.last_weight() < 0.1f);                     // the wall frame is explained as not-this-object
        CHECK(name_colour(a).name == "yellow");
        CHECK(name_colour(a).prob > 0.9f);
        step(a, kYellow);
        CHECK(name_colour(a).name == "yellow");
    }

    // ── a lasting change is still followed (repainted / the first view was wrong) ──────────────────────
    {
        AppearanceBelief a;
        for (int i = 0; i < 30; ++i) step(a, kYellow);
        for (int i = 0; i < 100; ++i) step(a, kMagenta);   // 10 s of a different colour
        CHECK(name_colour(a).name == "magenta");
    }
    {
        AppearanceBelief a;
        step(a, kWall);                                    // the very first frame is the wrong mask
        for (int i = 0; i < 100; ++i) step(a, kYellow);
        CHECK(name_colour(a).name == "yellow");
    }

    // ── no brightness in the frame (an old retina): chromaticity alone decides the weight ───────────────
    {
        AppearanceBelief a;
        for (int i = 0; i < 30; ++i)
        { a.inflate_for_age(0.1f); a.observe(kYellow.chroma, Eigen::Vector3f::Constant(kYellow.var), kYellow.n, 0.0f, 0.0f, 0.0f); }
        CHECK(a.initialized() and not a.lum_initialized());
        a.inflate_for_age(0.1f);
        a.observe(kWall.chroma, Eigen::Vector3f::Constant(kWall.var), kWall.n, 0.0f, 0.0f, 0.0f);
        CHECK(a.last_weight() < 0.1f);
        CHECK(std::fabs(a.map()[2] - kYellow.chroma[2]) < 0.01f);
    }

    std::printf("test_foreign_mask: %d checks, %d failures\n", checks, fails);
    return fails == 0 ? 0 : 1;
}
