#pragma once

#include <opencv2/core.hpp>
#include <vector>

/// Lightweight RGBD bundle read from DSR. Depth is stored per pixel in camera
/// metric units; XYZ is computed only where needed (e.g. YOLO mask pixels).
struct RGBDData
{
    cv::Mat              rgb;     ///< CV_8UC3, height × width
    std::vector<float>   depth;   ///< dense, row*width+col index, camera depth
    float                focal_x = 0.f;
    float                focal_y = 0.f;
    int                 width  = 0;
    int                 height = 0;
};
