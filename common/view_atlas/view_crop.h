/*
 * view_crop.h — the view-atlas crop and colour histogram (spec: VIEW_ATLAS.md §2). Must match
 * common/view_atlas/tools/build_dataset.py::make_crop exactly, or the fitted constants describe a
 * different descriptor than the one the agent computes.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace rc::atlas
{
struct Crop
{
    cv::Mat rgb224;    // 224x224 RGB, background = ImageNet mean
    cv::Mat bgr_box;   // the mask's bounding box, original pixels (for the colour histogram)
    cv::Mat mask_box;  // CV_8U 0/255, same size as bgr_box
    int     mask_px = 0;
    float   trunc_frac = 0.f;   // fraction of mask pixels on the image border row/column
};

inline std::optional<Crop> make_crop(const cv::Mat& bgr, const std::vector<Eigen::Vector2f>& mask_px)
{
    if (bgr.empty() or mask_px.empty()) return std::nullopt;
    cv::Mat mask = cv::Mat::zeros(bgr.size(), CV_8U);
    int x0 = bgr.cols, y0 = bgr.rows, x1 = -1, y1 = -1, n = 0, border = 0;
    for (const auto& p : mask_px)
    {
        const int x = int(std::lround(p.x())), y = int(std::lround(p.y()));
        if (x < 0 or y < 0 or x >= bgr.cols or y >= bgr.rows) continue;
        if (mask.at<std::uint8_t>(y, x) == 0) { ++n; if (x == 0 or y == 0 or x == bgr.cols - 1 or y == bgr.rows - 1) ++border; }
        mask.at<std::uint8_t>(y, x) = 255;
        x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y);
    }
    if (n == 0) return std::nullopt;
    Crop c;
    c.mask_px = n;
    c.trunc_frac = float(border) / float(n);
    const cv::Rect box(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    c.bgr_box = bgr(box).clone();
    c.mask_box = mask(box).clone();

    const float cx = 0.5f * float(x0 + x1 + 1), cy = 0.5f * float(y0 + y1 + 1);
    const float s = 1.15f * float(std::max(box.width, box.height));
    const int X1 = int(std::max(cx - s / 2, 0.f)), Y1 = int(std::max(cy - s / 2, 0.f));
    const int X2 = int(std::min(cx + s / 2, float(bgr.cols))), Y2 = int(std::min(cy + s / 2, float(bgr.rows)));
    const cv::Rect sq(X1, Y1, std::max(X2 - X1, 1), std::max(Y2 - Y1, 1));
    cv::Mat crop = bgr(sq).clone();
    crop.setTo(cv::Scalar(104, 116, 124), mask(sq) == 0);
    cv::Mat r;
    cv::resize(crop, r, {224, 224}, 0, 0, cv::INTER_AREA);
    cv::cvtColor(r, c.rgb224, cv::COLOR_BGR2RGB);
    return c;
}

/// 16 hue x 4 saturation over mask pixels, sqrt, L2-normalised; zeros when the mask is empty.
inline Eigen::VectorXf colour_hist(const cv::Mat& bgr_box, const cv::Mat& mask_box)
{
    Eigen::VectorXf h = Eigen::VectorXf::Zero(64);
    if (bgr_box.empty()) return h;
    cv::Mat hsv;
    cv::cvtColor(bgr_box, hsv, cv::COLOR_BGR2HSV);
    for (int y = 0; y < hsv.rows; ++y)
        for (int x = 0; x < hsv.cols; ++x)
            if (mask_box.at<std::uint8_t>(y, x))
            {
                const auto v = hsv.at<cv::Vec3b>(y, x);
                h[(v[0] * 16 / 180) * 4 + v[1] * 4 / 256] += 1.f;
            }
    h = h.cwiseSqrt();
    const float nrm = h.norm();
    if (nrm > 0.f) h /= nrm;
    return h;
}
} // namespace rc::atlas
