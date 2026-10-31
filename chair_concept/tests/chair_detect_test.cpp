// chair_detect_test: the chair's P(detect) is the shared countable formula with wall line of sight.
// OLD_ copies the RETIRED private formula (centre ramp x range ramp, no walls) so the difference is shown here and nowhere else.
#include "chair_detect.h"
#include <cmath>
#include <cstdio>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
using namespace rc;

// Retired zed_detectability, config as etc/config.toml set it: EdgeOffset 1, RangeFull 4, RangeRef 7.
static float old_pd(float off_x, float off_y, float range)
{
    const float pc = std::clamp(1.0f - std::hypot(off_x, off_y) / 1.0f, 0.0f, 1.0f);
    const float pr = std::clamp((7.0f - range) / (7.0f - 4.0f), 0.0f, 1.0f);
    return pc * pr;
}

// A pinhole ZED (1280x720, f = 448 px (robot_concept/shadow.json cam_rgb_focalx = cam_rgb_focaly = 448)) looking along +x from the origin at a 0.60 x 1.25 m chair (manifest [prior.template], width x top) at `range`, `off` px right of centre.
static ChairDetectInputs view(float range, float lateral_m)
{
    const float f = 448.0f, W = 1280.0f, H = 720.0f;
    ChairDetectInputs in;
    in.roi_valid = true;
    in.roi_fill = std::max(0.60f * f / range / W, 1.25f * f / range / H);
    in.in_fov = 1.0f;
    in.roi_offset_x = (lateral_m / range * f) / (0.5f * W);
    in.cam_xy = {0.0f, 0.0f};
    in.chair_xy = {range, lateral_m};
    return in;
}

int main()
{
    const rc::detect::DetectorEnvelope env{};
    const std::vector<Eigen::Vector2f> room{{-1, -5}, {6, -5}, {6, 5}, {-1, 5}};   // camera and chair inside
    // two rooms: a partition at x = 3 with no gap; chair at x = 4.5 in the next room, camera at the origin
    const std::vector<Eigen::Vector2f> two_rooms{{-1, -5}, {3, -5}, {3, 5}, {-1, 5}};   // the camera's room only
    // Table of before/after for the report (columns: range, lateral, old, new)
    for (const float r : {1.0f, 2.0f, 3.0f, 4.0f, 6.0f})
        for (const float lat : {0.0f, 0.5f})
        {
            const auto in = view(r, lat);
            std::printf("range %.1f lateral %.1f  old %.3f  new %.3f\n", r, lat, old_pd(in.roi_offset_x, 0.0f, r), chair_p_detect(in, env, room));
        }
    { // clear chair at 2 m: within 0.15 of the old value (regression guard, not a model parameter)
        const auto in = view(2.0f, 0.0f);
        const float o = old_pd(in.roi_offset_x, 0.0f, 2.0f), n = chair_p_detect(in, env, room);
        std::printf("clear 2 m: old %.3f new %.3f (fill %.3f)\n", o, n, in.roi_fill);
        CHECK(std::fabs(o - n) < 0.15f);
    }
    { // near-side distance where P falls below 0.5 (for the report)
        float r = 6.0f;
        while (r > 0.3f and chair_p_detect(view(r, 0.0f), env, room) >= 0.5f) r -= 0.01f;
        std::printf("near-side P<0.5 below %.2f m\n", r);
    }
    { // behind a wall: the target is in the next room, no gap. Old said > 0.5 (it had no walls); new says ~0.
        const auto in = view(2.0f, 0.0f);
        ChairDetectInputs behind = in;
        behind.chair_xy = {4.5f, 0.0f};
        behind.roi_fill = view(4.5f, 0.0f).roi_fill;
        const float o = old_pd(behind.roi_offset_x, 0.0f, 4.5f);
        const float n = chair_p_detect(behind, env, two_rooms);
        std::printf("behind wall: old %.3f new %.3f\n", o, n);
        CHECK(o > 0.5f);
        CHECK(n < 1e-6f);
        // no polygon known: wall factor 1
        CHECK(chair_p_detect(behind, env, {}) > 0.1f);
    }
    { // in_fov and central each multiply P exactly
        const auto base = view(3.0f, 0.0f);
        const float p0 = chair_p_detect(base, env, room);
        auto half = base; half.in_fov = 0.5f;
        CHECK(std::fabs(chair_p_detect(half, env, room) - 0.5f * p0) < 1e-6f);
        auto cen = base; cen.roi_offset_x = 0.5f;   // central = 1 - 0.5
        CHECK(std::fabs(chair_p_detect(cen, env, room) - 0.5f * p0) < 1e-6f);
    }
    { // a chair standing flush against the far wall is not hidden by its own wall
        auto in = view(5.9f, 0.0f);
        CHECK(chair_p_detect(in, env, room) > 0.0f);
    }
    { // no ROI -> 0 ; off the edge of the image -> 0 ; overflowing the frame -> ~0 (envelope upper shoulder)
        auto in = view(2.0f, 0.0f); in.roi_valid = false;
        CHECK(chair_p_detect(in, env, room) == 0.0f);
        auto near_in = view(0.6f, 0.0f);
        CHECK(chair_p_detect(near_in, env, room) < 0.05f);
    }
    std::printf(bad ? "chair_detect_test: %d FAILED\n" : "chair_detect_test: OK\n", bad);
    return bad != 0;
}
