#include "contour_edge_check.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace rc::edges
{

namespace
{

// Mean ACROSS-boundary gradient along a polygon's edges.
//
// ★The projection is only across the edge normal. A boundary that happens to lie along a strong
// gradient running parallel to it (a wall/floor join under a door, a skirting board) contributes
// nothing, because such a gradient is not evidence for THIS boundary — only a step across the predicted
// line is. Taking |grad| instead would score any busy region highly and turn clutter into confirmation.
float mean_across_boundary_gradient(const cv::Mat& gx, const cv::Mat& gy,
                                    const std::vector<cv::Point>& poly, int& n_out)
{
    n_out = 0;
    if (poly.size() < 3)
        return 0.0f;
    double acc = 0.0;
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const cv::Point a = poly[i], b = poly[(i + 1) % poly.size()];
        const double len = std::hypot(static_cast<double>(b.x - a.x), static_cast<double>(b.y - a.y));
        if (len < 4.0)
            continue;
        // Unit normal to this edge.
        const double nx = -(b.y - a.y) / len, ny = (b.x - a.x) / len;
        const int steps = std::min(160, std::max(8, static_cast<int>(len / 3.0)));
        for (int s = 0; s < steps; ++s)
        {
            const double t = (s + 0.5) / steps;
            const int px = static_cast<int>(std::lround(a.x + t * (b.x - a.x)));
            const int py = static_cast<int>(std::lround(a.y + t * (b.y - a.y)));
            if (px < 1 or py < 1 or px >= gx.cols - 1 or py >= gx.rows - 1)
                continue;
            // ★Take the STRONGEST across-boundary response within ±2 px of the predicted line, not the
            // value exactly on it. The projection carries real error — pose covariance, the door's own
            // fitted width, RT lag — and demanding a sub-pixel hit would score a correct prediction that
            // is 2 px off as no better than a wrong one, which is a measurement of our calibration
            // rather than of the world.
            float best = 0.0f;
            for (int d = -2; d <= 2; ++d)
            {
                const int qx = std::clamp(px + static_cast<int>(std::lround(d * nx)), 1, gx.cols - 2);
                const int qy = std::clamp(py + static_cast<int>(std::lround(d * ny)), 1, gx.rows - 2);
                const float proj = std::abs(static_cast<float>(gx.at<float>(qy, qx) * nx
                                                             + gy.at<float>(qy, qx) * ny));
                best = std::max(best, proj);
            }
            acc += best;
            ++n_out;
        }
    }
    return n_out > 0 ? static_cast<float>(acc / n_out) : 0.0f;
}

}   // namespace

std::vector<std::vector<cv::Point>> make_side_controls(const std::vector<cv::Point>& poly,
                                                       int frame_w, int frame_h)
{
    std::vector<std::vector<cv::Point>> out;
    if (poly.size() < 3)
        return out;
    const cv::Rect bb = cv::boundingRect(poly);
    if (bb.width < 8)
        return out;
    // ±1 and ±1.6 widths: near enough to be the same wall and lighting, far enough not to overlap the
    // object itself. Two distances so a control that lands on a second door (or a window) cannot alone
    // decide the comparison.
    for (const double k : {-1.6, -1.0, 1.0, 1.6})
    {
        const int dx = static_cast<int>(std::lround(k * bb.width));
        std::vector<cv::Point> c;
        c.reserve(poly.size());
        bool inside = true;
        for (const auto& p : poly)
        {
            const cv::Point q(p.x + dx, p.y);
            if (q.x < 0 or q.x >= frame_w or q.y < 0 or q.y >= frame_h) { inside = false; break; }
            c.push_back(q);
        }
        if (inside)
            out.push_back(std::move(c));
    }
    return out;
}

ContourEdgeScore contour_edge_support(const cv::Mat& img,
                                      const std::vector<cv::Point>& poly,
                                      const std::vector<std::vector<cv::Point>>& controls)
{
    ContourEdgeScore out;
    if (img.empty() or poly.size() < 3)
        return out;

    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;
    // Light blur before Sobel: without it the score is dominated by sensor/compression noise, which is
    // uniform across the frame and would therefore wash out the very contrast this is built on.
    cv::Mat blurred, gx, gy;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Sobel(blurred, gx, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, gy, CV_32F, 0, 1, 3);

    int n_true = 0;
    out.s_true = mean_across_boundary_gradient(gx, gy, poly, n_true);
    out.n_samples = n_true;
    if (n_true == 0)
        return out;   // nothing measured — the caller must not read this as "no support"

    double ctl_acc = 0.0;
    int    ctl_n   = 0;
    for (const auto& c : controls)
    {
        int n_c = 0;
        const float s = mean_across_boundary_gradient(gx, gy, c, n_c);
        if (n_c > 0) { ctl_acc += s; ++ctl_n; }
    }
    out.s_control  = ctl_n > 0 ? static_cast<float>(ctl_acc / ctl_n) : 0.0f;
    out.n_controls = ctl_n;

    // The frame's own gradient level, as the unit `excess` is measured in. Mean |grad| over the whole
    // image: cheap, and it tracks scene contrast, exposure and blur together — which is precisely what
    // has to be divided out for "this boundary is stronger than usual" to mean the same thing in a dim
    // corridor and a bright room.
    cv::Mat mag;
    cv::magnitude(gx, gy, mag);
    out.frame_ref = static_cast<float>(cv::mean(mag)[0]);
    out.excess = (ctl_n > 0 and out.frame_ref > 1e-3f)
                 ? (out.s_true - out.s_control) / out.frame_ref
                 : 0.0f;   // no controls, or a frame with no gradient at all ⇒ says nothing

    // Neutral 0.5 when there are no usable controls: with nothing to compare against, this frame says
    // nothing, and saying nothing must look different from saying "no".
    const float denom = out.s_true + out.s_control;
    out.support = (ctl_n > 0 and denom > 1e-6f) ? (out.s_true / denom) : 0.5f;
    return out;
}

}   // namespace rc::edges
