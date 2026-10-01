/*
 * zed_frame_dump.h — save ZED RGB frames with the pose they were captured from.
 *
 * Feeds the WAF pilot (world-model head trained offline on (image, pose) pairs). Off by default: it
 * writes a JPEG per kept frame, which is a disk and main-thread cost nobody should pay unless they
 * are recording a dataset on purpose.
 *
 * A frame is kept when the CAMERA has moved or turned enough since the last kept frame. That gate
 * decides what the dataset CONTAINS (parked frames are copies of each other and would dominate it),
 * not any model quantity, so it is a dataset-design knob in the same sense as depth_dataset's
 * pose_close().
 *
 * One CSV row per saved image, written only after the image is on disk, so every row has its file.
 * Poses are logged three ways because the pilot needs to choose between them offline:
 *   cam_*   room_T_zed at the capture stamp (what the frames should be keyed on)
 *   rob_*   room_T_robot = room_T_zed · robot_T_zed⁻¹ (the localisation estimate)
 *   gt_*    Webots ground truth EXACTLY as published (world frame, inverted angle sign — see
 *           place_stage.cpp); a latest-value attribute, so gt_ts is logged for the latency check.
 *           gt_valid = 0 on the real robot.
 *
 * Diagnostic only. Nothing here is published or read by any agent.
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include <Eigen/Dense>
#include <dsr/api/dsr_eigen_defs.h>

namespace rc::diag
{

struct ZedDumpGroundTruth
{
    float x = 0.f, y = 0.f, angle = 0.f;
    std::uint64_t stamp = 0;
    bool valid = false;
};

class ZedFrameDump
{
public:
    // Disabled until configure() is called with enabled=true; a disabled instance touches no file
    // and costs one bool test per frame.
    void configure(bool enabled, std::string dir, int jpeg_quality, float min_move_m, float min_turn_rad);
    bool enabled() const { return enabled_; }

    // Saves the frame if the camera moved enough since the last saved one. `bgr` is only read.
    // Returns true when a frame was written.
    bool maybe_save(std::uint64_t stamp, const cv::Mat& bgr, const Mat::RTMat& room_T_zed,
                    const std::optional<Mat::RTMat>& robot_T_zed, const ZedDumpGroundTruth& gt);

    std::size_t saved() const { return saved_; }

private:
    bool          enabled_      = false;
    std::string   dir_;
    int           jpeg_quality_ = 92;
    float         min_move_m_   = 0.05f;
    float         min_turn_rad_ = 0.0524f;
    std::ofstream csv_;
    bool          have_last_    = false;
    float         last_x_ = 0.f, last_y_ = 0.f, last_yaw_ = 0.f;
    std::size_t   saved_        = 0;
};

} // namespace rc::diag
