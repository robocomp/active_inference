// test_mask_color_summary.cpp — retina's per-mask colour summary: chromaticity (existing) + brightness (2026-11-11).
// Build+run: retina/tests/run_mask_color_test.sh
#include <cmath>
#include <cstdio>

#include "../src/mask_color_summary.h"

static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (not (c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

using rc::retina::mask_color_summary;

namespace
{
/// A 200x200 image of one flat BGR colour with a full-frame mask.
struct Scene { cv::Mat bgr, mask; };
Scene flat(int b, int g, int r)
{
    return {cv::Mat(200, 200, CV_8UC3, cv::Scalar(b, g, r)), cv::Mat(200, 200, CV_8UC1, cv::Scalar(255))};
}
}   // namespace

int main()
{
    // Existing behaviour kept: a magenta surface (R=153 G=46 B=153) → chromaticity, n_eff > 0.
    {
        const auto s = flat(153, 46, 153);
        const auto c = mask_color_summary(s.bgr, s.mask, 0, 0, 199, 199, 10);
        CHECK(c.n_eff > 0.0f);
        CHECK(std::fabs(c.chroma[0] - 153.0f / 352.0f) < 1e-3f);
        CHECK(std::fabs(c.chroma[1] - 46.0f / 352.0f) < 1e-3f);
        // NEW: brightness = (R+G+B)/(3*255)
        CHECK(std::fabs(c.lum - 352.0f / 765.0f) < 1e-3f);
        CHECK(c.lum_var > 0.0f);                 // never zero: the quantisation floor
        CHECK(c.lum_var < 1e-4f);                // a flat surface is tight
        CHECK(c.lum_n == c.n_eff);
    }
    // Pure black: no chromaticity at all (n_eff 0), but brightness IS measured — a black chair must be nameable.
    {
        const auto s = flat(0, 0, 0);
        const auto c = mask_color_summary(s.bgr, s.mask, 0, 0, 199, 199, 10);
        CHECK(c.n_eff == 0.0f);
        CHECK(c.lum_n > 0.0f);
        CHECK(c.lum < 1e-3f);
    }
    // A brightness GRADIENT across the mask (shading): the between-cell spread shows in lum_var. (Not a two-level
    // split: the robust median/MAD rightly treats a minority level as outliers.)
    {
        auto s = flat(0, 0, 0);
        for (int x = 0; x < 200; ++x) s.bgr.col(x).setTo(cv::Scalar(30 + x, 30 + x, 30 + x));
        const auto c = mask_color_summary(s.bgr, s.mask, 0, 0, 199, 199, 10);
        CHECK(c.lum_var > 1e-2f);
    }
    // Empty mask: nothing at all.
    {
        auto s = flat(100, 100, 100);
        s.mask.setTo(0);
        const auto c = mask_color_summary(s.bgr, s.mask, 0, 0, 199, 199, 10);
        CHECK(c.n_eff == 0.0f and c.lum_n == 0.0f);
    }
    std::printf("test_mask_color_summary: %d checks, %d failures\n", checks, fails);
    return fails == 0 ? 0 : 1;
}
