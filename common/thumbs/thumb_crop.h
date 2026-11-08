/*
 * common/thumbs/thumb_crop.h — cut one object's box out of a BGR frame as a small JPEG. No model. Producer side (retina).
 */
#pragma once

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rc::thumbs
{
/// Inclusive pixel box, grown by a small margin and clipped to the image. Empty when nothing of the box is in the image.
inline std::vector<std::uint8_t> crop_thumbnail(const cv::Mat& bgr, int x0, int y0, int x1, int y1, int max_side, int jpeg_quality)
{
    if (bgr.empty() or x1 < x0 or y1 < y0) return {};
    static constexpr double kMargin = 0.1;   // a little of the surroundings, for the eye
    const int mx = static_cast<int>((x1 - x0 + 1) * kMargin), my = static_cast<int>((y1 - y0 + 1) * kMargin);
    const cv::Rect box = cv::Rect(cv::Point(x0 - mx, y0 - my), cv::Point(x1 + mx + 1, y1 + my + 1)) & cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (box.empty()) return {};
    const double scale = static_cast<double>(max_side) / std::max(box.width, box.height);
    cv::Mat small;
    cv::resize(bgr(box), small, cv::Size(std::max(1, static_cast<int>(box.width * scale + 0.5)), std::max(1, static_cast<int>(box.height * scale + 0.5))), 0, 0, cv::INTER_AREA);
    std::vector<std::uint8_t> jpeg;
    cv::imencode(".jpg", small, jpeg, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality});
    return jpeg;
}
}   // namespace rc::thumbs
