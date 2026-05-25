#pragma once

#include <opencv2/core.hpp>
#include <vector>

/// Simple 3D point (camera-frame, metres) stored dense at one entry per pixel.
struct PointXYZ
{
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
};

/// Lightweight RGBD bundle read from DSR (replaces RoboCompCameraRGBDSimple::TRGBD).
struct RGBDData
{
    cv::Mat             rgb;     ///< CV_8UC3, height × width
    std::vector<PointXYZ> points; ///< dense, row*width+col index, camera frame
    int                 width  = 0;
    int                 height = 0;
};
