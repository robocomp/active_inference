// mask_membership_test — m = pi_mask * Phi(s/sigma_px) (spec 2026-11-15 lidar-measures G2). Run: ./mask_membership_test
#include <cmath>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>
#include "countable_mask_membership.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

// A filled rectangle [c0, c0+w) x [r0, r0+h) sampled every `step` pixels in both axes.
static std::vector<Eigen::Vector2f> rect(int c0, int r0, int w, int h, int step = 1)
{
    std::vector<Eigen::Vector2f> v;
    for (int r = r0; r < r0 + h; r += step)
        for (int c = c0; c < c0 + w; c += step)
            v.emplace_back(static_cast<float>(c), static_cast<float>(r));
    return v;
}

static void inside_one_boundary_half_outside_zero()
{
    const auto px = rect(100, 50, 40, 30);
    const auto mm = MaskMembership::build(px);
    CHECK(not mm.empty());
    CHECK(mm.stride_px() == 1.f);
    const float q = mm.stride_px();
    CHECK(membership_weight(mm.signed_distance_px(120, 65), 2.f) > 0.999f);
    CHECK(membership_weight(mm.signed_distance_px(100 - 10, 65), 2.f) < 1e-3f);
    CHECK(std::abs(membership_weight(mm.signed_distance_px(139.5f, 65), 2.f) - 0.5f) < 0.02f);
    CHECK(std::abs(mm.signed_distance_px(139, 65) - (0.5f * q)) < 0.05f);   // an edge pixel reads +q/2
}

static void symmetric_in_s()
{
    for (float s : {0.3f, 1.f, 4.f})
        CHECK(std::abs(membership_weight(s, 2.f) + membership_weight(-s, 2.f) - 1.f) < 1e-6f);
}

static void sigma_grows_as_z_shrinks()
{
    const float far = projection_sigma_px(900.f, 4.f, 0.05f, 0.02f, 0.f, 1.f);
    const float near = projection_sigma_px(900.f, 1.f, 0.05f, 0.02f, 0.f, 1.f);
    CHECK(near > far);
    CHECK(std::abs(projection_sigma_px(900.f, 2.f, 0.f, 0.f, 0.f, 3.f) - 3.f) < 1e-5f);   // only the stride left
}

static void stride_of_subsampled_mask()
{
    const auto mm = MaskMembership::build(rect(99, 51, 42, 30, 3));   // columns/rows every 3 px
    CHECK(mm.stride_px() == 3.f);
    const float q = mm.stride_px();
    CHECK(std::abs(mm.signed_distance_px(99 + 3 * 13, 51 + 3 * 5) - 0.5f * q) < 0.05f);   // right edge sample: +q/2
    // interior sample (cell 7,5 of the 14x10 grid): 5 cells to the nearest outside cell ⇒ 5q − q/2
    CHECK(std::abs(mm.signed_distance_px(99 + 3 * 6, 51 + 3 * 4) - (5.f * q - 0.5f * q)) < 0.05f);
    CHECK(std::abs(mm.signed_distance_px(99, 60) - 0.5f * q) < 0.05f);                      // left edge sample
    CHECK(mm.signed_distance_px(120, 66) > 0.f);
    CHECK(mm.signed_distance_px(80, 66) < 0.f);
}

static void thin_mask()
{
    const auto mm = MaskMembership::build(rect(10, 10, 60, 1));   // one row
    CHECK(std::abs(mm.signed_distance_px(40, 10) - 0.5f) < 0.05f);
    CHECK(mm.signed_distance_px(40, 14) < -3.f);
    CHECK(std::isfinite(mm.signed_distance_px(40, 10.5f)));
}

static void outside_raster_continuous()
{
    const auto mm = MaskMembership::build(rect(100, 50, 40, 30));
    // Across the one-cell raster border (the raster spans cols 99..140, rows 49..80).
    float prev = mm.signed_distance_px(95.f, 65.f);
    for (float c = 95.f; c <= 99.f; c += 0.01f)
    {
        const float s = mm.signed_distance_px(c, 65.f);
        CHECK(std::abs(s - prev) < 0.1f);
        prev = s;
    }
    CHECK(membership_weight(mm.signed_distance_px(300.f, 65.f), 2.f) < 1e-6f);
}

static Projector fixed(float c, float r, float z) { return [=](const Eigen::Vector3f&) { return std::optional<Eigen::Vector3f>(Eigen::Vector3f(c, r, z)); }; }

static void make_membership_nullopt_is_zero()
{
    const auto mm = MaskMembership::build(rect(100, 50, 40, 30));
    const auto f = make_membership([](const Eigen::Vector3f&) { return std::optional<Eigen::Vector3f>{}; }, mm, 900.f,
                                   0.05f, 0.02f, 0.f, 1.f);
    CHECK(f(Eigen::Vector3f::Zero()) == 0.f);
}

static void make_membership_behind_is_zero()
{
    // The projector refuses depth < 0.20 m, as CountableProjection::project_to_zed does; the weight is then exactly 0.
    const auto mm = MaskMembership::build(rect(100, 50, 40, 30));
    const Projector behind = [](const Eigen::Vector3f& p) -> std::optional<Eigen::Vector3f>
    { if (p.y() < 0.20f) return std::nullopt; return Eigen::Vector3f(120, 65, p.y()); };
    const auto f = make_membership(behind, mm, 900.f, 0.001f, 0.f, 0.f, 1.f);
    CHECK(f(Eigen::Vector3f(0, -1.f, 0)) == 0.f);
    CHECK(f(Eigen::Vector3f(0, 2.f, 0)) > 0.999f);
}

static void pi_mask_scales()
{
    const auto mm = MaskMembership::build(rect(100, 50, 40, 30));
    const auto f = make_membership(fixed(120, 65, 2.f), mm, 900.f, 0.001f, 0.f, 0.f, 0.6f);
    CHECK(std::abs(f(Eigen::Vector3f::Zero()) - 0.6f) < 1e-3f);
}

int main()
{
    inside_one_boundary_half_outside_zero();
    symmetric_in_s();
    sigma_grows_as_z_shrinks();
    stride_of_subsampled_mask();
    thin_mask();
    outside_raster_continuous();
    make_membership_nullopt_is_zero();
    make_membership_behind_is_zero();
    pi_mask_scales();
    std::printf(bad == 0 ? "mask_membership_test: PASS\n" : "mask_membership_test: %d FAILED\n", bad);
    return bad == 0 ? 0 : 1;
}
