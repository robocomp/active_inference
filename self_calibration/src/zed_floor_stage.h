/*
 * zed_floor_stage — the ZED camera's roll, pitch and height from the FLOOR in its depth image (plan item 4.1).
 *
 * The depth image is back-projected through the pinhole model into the ZED node frame (X right, Y forward = the
 * depth value, Z up — NOT optical z-forward), mapped into the robot (Shadow) frame through the CURRENT graph
 * mount, and fed to the SAME floor factor the bpearl uses (rc::mountf::FloorPlaneMount, common/sensor_calib/
 * mount_factors.h): the floor is z = 0 in the robot frame, a mixture over "floor / not floor" does the outlier
 * rejection (walls, furniture) — there is no height band, gate or range cut here. `stride` and `rate_hz` are
 * COMPUTE budgets (pixels back-projected per frame / frames fed per second), they say nothing about the data.
 *
 * Info3 order (droll, dpitch, dz), robot axes, about the ZED origin: the CORRECTION that restores the floor to
 * z = 0, so a planted mount error reads back as its NEGATIVE (as for the bpearl).
 *
 * dz IS NOT THE MOUNT HEIGHT ALONE (until 4.2): the depth value is scaled by the camera's depth-scale error s,
 * which moves the floor to s·h below the camera, so dz = (height error) + (s - 1)·h. Roll and pitch are
 * unaffected by a uniform depth scale (it dilates the floor plane about the camera, tilt-preserving).
 *
 * ESTIMATE + LOG ONLY. Pure: no Qt, no DSR — the agent hands it plain data (sensor_inputs.h structs).
 */
#pragma once
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "../../common/sensor_calib/mount_factors.h"
#include "sensor_inputs.h"

namespace sc
{
// Depth image (FORMAT_DEPTH_F32 metres, or FORMAT_Z16 millimetres) -> points in the ZED node frame
// (X right, Y forward = the depth value, Z up), every `stride`-th pixel; non-finite and <= 0 depths are skipped.
// X = (u - cx) d / fx, Z = (cy - v) d / fy with cx, cy = size/2; K belongs to the RGB model and is scaled
// uniformly when the depth image has another size. Returns empty for an unknown format or missing intrinsics.
[[nodiscard]] std::vector<Eigen::Vector3f> depth_to_points(const ImageIn& depth, const PinholeIntrinsics& K, int stride);

class ZedFloorStage
{
public:
    struct Config
    {
        double rate_hz = 1.0;                    // frames fed per second: a COMPUTE budget
        int    stride = 4;                       // every stride-th pixel in u and v: a COMPUTE budget
        std::string log_dir = "tmp/zed_floor";   // "" = no files
    };
    explicit ZedFloorStage(Config cfg);

    // Feed one depth frame (robot_from_sensor must be set; room_T_robot only keys the floor's place cell).
    // False when skipped (unresolved mount, rate budget, no usable points).
    bool observe(const ImageIn& depth, const PinholeIntrinsics& K, const std::optional<Eigen::Affine3d>& room_T_robot);

    [[nodiscard]] rc::mountf::Info3 info() const { return floor_.info(); }
    [[nodiscard]] long frames() const { return n_; }
    [[nodiscard]] std::string summary() const;   // "[zed-floor] roll ± pitch ± deg  z ± mm (n frames)"
    void log_state(std::int64_t stamp_ms);       // append to <log_dir>/zed_floor_<ts>.csv (no-op without a log_dir)

private:
    Config cfg_;
    rc::mountf::FloorPlaneMount floor_;
    std::int64_t last_ms_ = 0;
    long n_ = 0;
    std::ofstream csv_;
};
}  // namespace sc
