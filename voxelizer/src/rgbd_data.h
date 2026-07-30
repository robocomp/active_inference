#pragma once

#include <opencv2/core.hpp>
#include <vector>

/// Lightweight RGBD bundle read from DSR. Depth is stored per pixel in camera
/// metric units; XYZ is computed only where needed (e.g. YOLO mask pixels).
struct RGBDData
{
    /// CV_8UC3, height × width, **BGR** — OpenCV's native order, which is what every consumer here
    /// wants (drawing, imwrite, the ONNX pre-processing that flips to RGB itself).
    ///
    /// This member was called `rgb` while holding BGR. Nothing was wrong with the DATA — every
    /// consumer already converted correctly — but the name inverted the one fact a reader needs, and
    /// a channel swap is silent: it does not crash, it does not assert, it just makes red things blue.
    /// It is spelled `bgr` now so the call sites read as what they are, e.g. mask_color_summary(bgr,…)
    /// whose parameter was already named `bgr`. Match `ZedRgbFrame::bgr`, which always had it right.
    cv::Mat              bgr;
    std::vector<float>   depth;   ///< dense, row*width+col index, camera depth
    float                focal_x = 0.f;
    float                focal_y = 0.f;
    int                 width  = 0;
    int                 height = 0;
};
