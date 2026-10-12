#pragma once
/*
 * image_prep.h — a camera frame made small enough to show the vision model (look() tool, chat_session.h).
 *
 * The frame is downscaled so its LONGEST side is at most max_side (never upscaled; aspect kept; INTER_AREA = proper
 * averaging when shrinking) and JPEG-encoded, to keep the base64 request small.
 * ★It does NOT buy tokens with qwen3-vl under ollama (measured 2026-10-12): ollama resizes every image to about a
 * megapixel itself, so 384x216, 768x432 and 1280x720 all cost ~1050 prompt tokens, and 1920x960 costs ~1800.
 * Sending LESS than ~1280 px wide only throws detail away before ollama upsamples it again.
 * Pure OpenCV, no Qt, no DSR: tests/test_look.cpp exercises it offline.
 */

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace rc::chat
{

struct PreparedImage
{
    std::vector<unsigned char> jpeg;   ///< the encoded bytes
    int width = 0;                     ///< after scaling
    int height = 0;
};

/// Scale size (w, h) so the longest side is at most max_side (max_side <= 0 ⇒ unchanged). Never upscales.
[[nodiscard]] inline cv::Size fit_longest_side(int w, int h, int max_side)
{
    const int longest = std::max(w, h);
    if (max_side <= 0 or longest <= max_side) return {w, h};
    const double s = static_cast<double>(max_side) / static_cast<double>(longest);
    return {std::max(1, static_cast<int>(std::lround(w * s))), std::max(1, static_cast<int>(std::lround(h * s)))};
}

/// bgr: 8-bit BGR (CV_8UC3) or grey (CV_8UC1). Returns nullopt for an empty/unsupported image or a failed encode.
[[nodiscard]] inline std::optional<PreparedImage> prepare_image(const cv::Mat& bgr, int max_side, int jpeg_quality)
{
    if (bgr.empty() or bgr.depth() != CV_8U or not (bgr.channels() == 3 or bgr.channels() == 1)) return std::nullopt;
    const cv::Size target = fit_longest_side(bgr.cols, bgr.rows, max_side);
    cv::Mat small;
    if (target == bgr.size()) small = bgr;
    else cv::resize(bgr, small, target, 0, 0, cv::INTER_AREA);
    PreparedImage out;
    // libjpeg's own valid range; outside it imencode would clamp silently anyway.
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, std::clamp(jpeg_quality, 1, 100)};
    if (not cv::imencode(".jpg", small, out.jpeg, params) or out.jpeg.empty()) return std::nullopt;
    out.width = small.cols;
    out.height = small.rows;
    return out;
}

}   // namespace rc::chat
