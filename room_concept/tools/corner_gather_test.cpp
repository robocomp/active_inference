/*
 *  corner_gather_test.cpp — does the layout-wide explain-away do what it was built to do?
 *
 *  The live tour measured the DISEASE precisely (tmp/corner_probe.csv, 55803 candidates): 15.6% of
 *  corner evaluations fit a line PERPENDICULAR to their own model edge, because a gather band wide
 *  enough to reach the real wall through the layout's error is also wide enough to swallow a face at
 *  right angles to it. It also measured the first CURE failing in a new way: a plain softmax over
 *  edges let a COLLINEAR neighbour — the same physical surface, split in two by a trace vertex —
 *  take half of every point, and corner 1 starved on 5.6% of its returns.
 *
 *  The second cure makes a rival steal only in proportion to 1 − |t_claim·t_rival|. That is a claim
 *  about geometry, and geometry can be tested without the robot. Three scenarios, each an isolated
 *  version of one live failure, each scored on the thing that actually matters — how far the detected
 *  corner lands from the corner that generated the points:
 *
 *    CLEAN        a plain rectangle. Neither path may damage it.
 *    STEP         the notch: a 0.8 m face at right angles, inside the neighbouring wall's band.
 *                 This is corners 13 / 27 / 11.
 *    SUBDIVIDED   a wall cut in two by a collinear vertex 0.5 m from the corner, so the corner's own
 *                 edge is short and its continuation competes for the same returns. This is corner 1,
 *                 and the case the disagreement factor exists for.
 *
 *  Run both arms:  ./bin/corner_gather_test           (committed gather)
 *                  RC_CORNER_EXPLAIN_AWAY=1 ./bin/corner_gather_test
 */
#include "corner_detector.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using rc::CornerDetector;
using Eigen::Vector2f;

namespace
{
    /// Sample returns along every segment of a closed surface, with Gaussian perpendicular noise.
    /// The SURFACE is what the lidar sees; the POLYGON handed to the detector may differ from it
    /// (extra collinear vertices), which is exactly what the subdivided case is about.
    std::vector<Eigen::Vector3f> scan(const std::vector<Vector2f>& surface, float sigma, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> noise(0.f, sigma);
        std::vector<Eigen::Vector3f> out;
        const std::size_t n = surface.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            const Vector2f a = surface[i], b = surface[(i + 1) % n];
            const Vector2f ab = b - a;
            const float len = ab.norm();
            if (len < 1e-6f) continue;
            const Vector2f t = ab / len;
            const Vector2f nrm(-t.y(), t.x());
            const int steps = std::max(2, static_cast<int>(len / 0.01f));   // ~1 cm spacing
            for (int k = 0; k <= steps; ++k)
            {
                const Vector2f p = a + t * (len * static_cast<float>(k) / steps) + nrm * noise(rng);
                out.emplace_back(p.x(), p.y(), 0.f);
            }
        }
        return out;
    }

    struct Case
    {
        std::string name;
        std::vector<Vector2f> polygon;   // what the detector is told
        std::vector<Vector2f> surface;   // what the lidar sees
        Vector2f robot;
    };

    void run(const Case& c)
    {
        CornerDetector::Params p;                     // defaults: band 0.35, base 0.04, map 0.06
        CornerDetector det(p);
        det.set_model_corners(c.polygon);

        // ⚠ detect() takes the cloud ALREADY IN THE ROBOT FRAME — it only drops z, it does not
        // transform. Handing it world coordinates silently offsets every gather by the robot's
        // position, which starves the bands and lands the fits on the wrong walls; with the robot at
        // the origin the bug is invisible, which is how the first version of this test passed two
        // cases and produced pure noise on the third.
        auto to_robot = [&](const std::vector<Eigen::Vector3f>& w)
        {
            std::vector<Eigen::Vector3f> r;
            r.reserve(w.size());
            for (const auto& q : w) r.emplace_back(q.x() - c.robot.x(), q.y() - c.robot.y(), q.z());
            return r;
        };

        // 20 draws, because a 2 mm difference on one noise seed is not a result.
        double err_tot = 0.0; int err_cnt = 0, coll_tot = 0, fb_tot = 0, det_tot = 0;
        double kept_tot = 0.0; int kept_n = 0;
        for (unsigned seed = 1; seed <= 20; ++seed)
        {
            const auto r = det.detect(to_robot(scan(c.surface, 0.02f, seed)),
                                      c.robot.x(), c.robot.y(), 0.f, Eigen::Matrix3f::Zero());
            det_tot += r.corners_detected;
            for (const auto& pr : r.probes)
            {
                err_tot += std::hypot(pr.nu_x, pr.nu_y); err_cnt++;
                if (std::min(pr.ori[0], pr.ori[1]) < 0.01f) coll_tot++;
                if (pr.propagated == 0) fb_tot++;
                for (int k = 0; k < 2; ++k)
                    if (pr.nraw[k] > 0) { kept_tot += static_cast<double>(pr.npts[k]) / pr.nraw[k]; kept_n++; }
            }
        }

        const auto pts = to_robot(scan(c.surface, 0.02f, 1));
        const auto res = det.detect(pts, c.robot.x(), c.robot.y(), 0.f, Eigen::Matrix3f::Zero());

        std::printf("\n  %-11s  %zu returns, %d in FOV, %d detected, %d accepted"
                    " (rej: fewpts %d, dist %d, convex %d, occl %d)\n",
                    c.name.c_str(), pts.size(), res.corners_in_fov, res.corners_detected,
                    res.corners_accepted, res.rej_fewpoints, res.rej_dist, res.rej_convex,
                    res.rej_occluded);
        std::printf("    %-6s %8s %7s %7s %13s %13s %6s %8s %11s\n",
                    "corner", "err(m)", "ori_in", "ori_out", "in n(eff/raw)", "out n(eff/raw)",
                    "prop", "sig_det", "rival in/out");
        double err_sum = 0.0; int err_n = 0, collapsed = 0, fellback = 0;
        for (const auto& pr : res.probes)
        {
            const float err = std::hypot(pr.nu_x, pr.nu_y);
            const float sdet = std::sqrt(std::max(0.f, (pr.sdet_xx + pr.sdet_yy) * 0.5f));
            const auto kept = [](int a, int b) { return b > 0 ? static_cast<float>(a) / b : 0.f; };
            (void)kept;
            std::printf("    %-6d %8.4f %7.3f %7.3f %6d/%-6d %6d/%-6d %6d %8.3f %5d/%-5d\n",
                        pr.model_index, err, pr.ori[0], pr.ori[1],
                        pr.npts[0], pr.nraw[0], pr.npts[1], pr.nraw[1],
                        pr.propagated, sdet, pr.rival[0], pr.rival[1]);
            err_sum += err; err_n++;
            if (std::min(pr.ori[0], pr.ori[1]) < 0.01f) collapsed++;
            if (pr.propagated == 0) fellback++;
        }
        (void)err_sum; (void)err_n; (void)collapsed; (void)fellback;
        std::printf("    == over 20 seeds: mean corner error %.4f m | collapsed %d/%d (%.0f%%)"
                    " | fallback %d/%d | responsibility kept %.2f\n",
                    err_cnt ? err_tot / err_cnt : 0.0, coll_tot, err_cnt,
                    err_cnt ? 100.0 * coll_tot / err_cnt : 0.0, fb_tot, err_cnt,
                    kept_n ? kept_tot / kept_n : 0.0);
    }
} // namespace

int main()
{
    const bool on = std::getenv("RC_CORNER_EXPLAIN_AWAY") != nullptr;
    std::printf("=== corner gather: explain-away %s ===\n", on ? "ON" : "OFF (committed)");

    // A plain room. Nothing here may get worse.
    const std::vector<Vector2f> rect{{-3,-2}, {3,-2}, {3,2}, {-3,2}};

    // The notch. (0.5,2)->(0.5,1.2) is a 0.8 m face at right angles to the wall y = 1.2, and it sits
    // inside that wall's 0.35 m gather band — the live corner-13 geometry.
    const std::vector<Vector2f> step{{-3,-2}, {3,-2}, {3,2}, {0.5,2}, {0.5,1.2}, {-3,1.2}};

    // A wall cut in two 0.5 m before the corner. The extra vertex is collinear, so it never becomes a
    // landmark, but it stays in the polygon and its edge competes for the same returns.
    const std::vector<Vector2f> split{{-3,-2}, {2.5,-2}, {3,-2}, {3,2}, {-3,2}};
    const std::vector<Vector2f> split_surface{{-3,-2}, {3,-2}, {3,2}, {-3,2}};

    run({"CLEAN",      rect,  rect,          {0.f, 0.f}});
    run({"STEP",       step,  step,          {0.f, 0.f}});
    run({"SUBDIVIDED", split, split_surface, {1.5f, 0.f}});
    return 0;
}
