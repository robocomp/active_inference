// test_bottle_contour — the bottle's classifier-free contour channel, geometry + scoring, no DSR, no network.
//
//   1. the outline is the cylinder's silhouette: width ~ 2 f r / z, and a slid null exists on both sides;
//   2. a rim point behind the camera = NOT MEASURED (empty face), never a refutation;
//   3. drawn cylinder on a plain background: RGB excess is positive at the true pose and ~0 on blank;
//   4. the resolvability weight p_resolve = r/(r+hypot(sigma, 2px·2r/w_px)) falls CONTINUOUSLY with range.
//
// Build: see tests/CMakeLists.txt (target test_bottle_contour). Pass = exit code 0.

#include <cmath>
#include <cstdio>
#include <optional>

#include <opencv2/imgproc.hpp>

#include "../src/bottle_cylinder_contour.h"
#include "../../common/contour_edge/contour_edge_check.h"

namespace
{
int g_fail = 0;
#define CHECK(cond, ...) do { if (not (cond)) { ++g_fail; std::printf("FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// Pinhole camera at the origin looking along +y (the ZED convention: x right, y forward, z up).
constexpr double F = 700.0, CX = 640.0, CY = 360.0;
rc::edges::PointProjector pinhole(double cam_z)
{
    return [cam_z](const Eigen::Vector3d& P) -> std::optional<rc::edges::ProjectedVertex>
    {
        const double y = P.y();
        if (y <= 0.2) return std::nullopt;
        rc::edges::ProjectedVertex v;
        v.px = cv::Point2f(static_cast<float>(CX + F * P.x() / y), static_cast<float>(CY - F * (P.z() - cam_z) / y));
        v.depth_m = static_cast<float>(y);
        return v;
    };
}

float p_resolve(float r, float sigma_pos, float w_px)
{
    const float tol_m = 2.0f * (2.0f * r) / w_px;
    return r / (r + std::hypot(sigma_pos, tol_m));
}
}

int main()
{
    const float r = 0.035f, z0 = 0.80f, z1 = 1.00f;
    const Eigen::Vector3d cam(0, 0, 1.2);

    // 1. silhouette width and the null
    {
        const auto cs = rc::cylinder_contour_set(0.0f, 1.5f, z0, z1, r, cam, pinhole(cam.z()));
        CHECK(cs.face.valid(), "face must be valid");
        const cv::Rect bb = cv::boundingRect(cs.face.px);
        const double expect_w = 2.0 * F * r / 1.5;
        CHECK(std::abs(bb.width - expect_w) < 2.5, "width %d vs %.1f", bb.width, expect_w);
        CHECK(cs.controls.size() == 4, "controls %zu", cs.controls.size());
        // the null is the same shape displaced sideways: same height, different x
        const cv::Rect c0 = cv::boundingRect(cs.controls.front().px);
        CHECK(std::abs(c0.height - bb.height) <= 2, "control height %d vs %d", c0.height, bb.height);
        CHECK(std::abs(c0.x - bb.x) > bb.width / 2, "control not displaced");
    }
    // 2. behind the camera = not measured
    {
        const auto cs = rc::cylinder_contour_set(0.0f, 0.1f, z0, z1, r, cam, pinhole(cam.z()));
        CHECK(not cs.face.valid(), "a rim behind the camera must leave the face empty");
    }
    // 3. image evidence: a bright cylinder on a dark wall is a real boundary; blank is not
    {
        const auto cs = rc::cylinder_contour_set(0.0f, 1.2f, z0, z1, r, cam, pinhole(cam.z()));
        CHECK(cs.face.valid(), "face valid");
        cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));
        cv::fillConvexPoly(img, cs.face.px, cv::Scalar(220, 220, 220));
        // add texture elsewhere so the frame reference is not degenerate
        cv::rectangle(img, cv::Rect(100, 100, 120, 200), cv::Scalar(200, 200, 200), -1);
        std::vector<std::vector<cv::Point>> ctl;
        for (const auto& c : cs.controls) ctl.push_back(c.px);
        const auto on = rc::edges::contour_edge_support(img, cs.face.px, ctl);
        CHECK(on.n_samples > 0 and on.n_controls > 0, "scored n=%d ctl=%d", on.n_samples, on.n_controls);
        CHECK(on.excess > 0.5f, "true-pose excess %.3f must be clearly positive", on.excess);

        cv::Mat blank(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));
        cv::rectangle(blank, cv::Rect(100, 100, 120, 200), cv::Scalar(200, 200, 200), -1);
        const auto off = rc::edges::contour_edge_support(blank, cs.face.px, ctl);
        CHECK(std::abs(off.excess) < 0.2f, "blank excess %.3f must be ~0", off.excess);
        CHECK(on.excess > off.excess + 0.5f, "on %.3f vs blank %.3f", on.excess, off.excess);
    }
    // 4. resolvability falls continuously with range (no step)
    {
        float prev = 2.0f, max_step = 0.0f;
        for (float z = 0.5f; z <= 10.0f; z += 0.25f)
        {
            const float w_px = static_cast<float>(2.0 * F * r / z);
            const float p = p_resolve(r, 0.02f, w_px);
            CHECK(p <= prev + 1e-6f and p >= 0.0f and p <= 1.0f, "non-monotone at %.2f m: %.4f after %.4f", z, p, prev);
            if (prev <= 1.0f) max_step = std::max(max_step, prev - p);
            prev = p;
        }
        const float near = p_resolve(r, 0.02f, static_cast<float>(2.0 * F * r / 0.8));
        const float far  = p_resolve(r, 0.02f, static_cast<float>(2.0 * F * r / 8.0));
        std::printf("p_resolve near(0.8 m)=%.3f far(8 m)=%.3f max step per 0.25 m=%.3f\n", near, far, max_step);
        // Near, the belief's own position sigma (2 cm vs r = 3.5 cm) is the limit; far, the 2 px edge tolerance
        // takes over. A bottle only a pixel or two wide cannot be localised on at all.
        const float speck = p_resolve(r, 0.02f, 1.0f);
        std::printf("p_resolve speck(1 px)=%.3f\n", speck);
        CHECK(std::abs(near - 0.635f) < 0.01f, "near %.3f", near);
        CHECK(far < near, "far %.3f must be below near %.3f", far, near);
        CHECK(speck < 0.25f, "a 1 px bottle weight %.3f must be ~0", speck);
        CHECK(max_step < 0.1f, "step %.3f is not continuous", max_step);
    }

    std::printf(g_fail == 0 ? "ALL PASS\n" : "%d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
