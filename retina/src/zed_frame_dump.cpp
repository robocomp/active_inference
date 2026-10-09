#include "zed_frame_dump.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <locale>
#include <numbers>
#include <print>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace rc::diag
{

namespace
{
float yaw_of(const Mat::RTMat& T)
{
    const auto& R = T.linear();
    return static_cast<float>(std::atan2(R(1, 0), R(0, 0)));
}

float wrap_pi(float a)
{
    return std::remainder(a, 2.0f * std::numbers::pi_v<float>);
}
} // namespace

void ZedFrameDump::configure(bool enabled, std::string dir, int jpeg_quality, float min_move_m,
                             float min_turn_rad)
{
    enabled_      = enabled;
    dir_          = std::move(dir);
    jpeg_quality_ = std::clamp(jpeg_quality, 50, 100);
    min_move_m_   = min_move_m;
    min_turn_rad_ = min_turn_rad;
    if (not enabled_)
        return;

    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    const auto csv_path = std::format("{}/poses.csv", dir_);
    const bool fresh = not std::filesystem::exists(csv_path);
    csv_.open(csv_path, std::ios::app);
    if (not csv_)
    {
        std::println(stderr, "[zed-dump] could not open {} — dump disabled", csv_path);
        enabled_ = false;
        return;
    }
    csv_.imbue(std::locale::classic());   // decimal points regardless of LANG (CLAUDE.md)
    if (fresh)
        csv_ << "stamp_ms,file,cam_x,cam_y,cam_z,cam_yaw,rob_x,rob_y,rob_yaw,rob_valid,"
                "gt_x,gt_y,gt_angle,gt_ts,gt_valid\n";
    std::println("[zed-dump] recording ZED frames to {}/ (move ≥ {:.2f} m or turn ≥ {:.1f}°)",
                 dir_, min_move_m_, min_turn_rad_ * 180.0f / std::numbers::pi_v<float>);
}

bool ZedFrameDump::maybe_save(std::uint64_t stamp, const cv::Mat& bgr, const Mat::RTMat& room_T_zed,
                              const std::optional<Mat::RTMat>& robot_T_zed, const ZedDumpGroundTruth& gt)
{
    if (not enabled_ or bgr.empty())
        return false;
    // ZedSource leaves room_T_sensor at its Identity default when room<-zed did not resolve. A real
    // camera pose is never exactly Identity (the lens sits ~0.95 m above the floor), so this is the
    // "no pose" case, and a frame without a pose is useless to the dataset.
    if (room_T_zed.matrix() == Mat::RTMat::Identity().matrix())
        return false;

    const auto  t   = room_T_zed.translation();
    const float cx  = static_cast<float>(t.x());
    const float cy  = static_cast<float>(t.y());
    const float yaw = yaw_of(room_T_zed);
    if (have_last_)
    {
        const bool moved  = std::hypot(cx - last_x_, cy - last_y_) >= min_move_m_;
        const bool turned = std::abs(wrap_pi(yaw - last_yaw_)) >= min_turn_rad_;
        if (not moved and not turned)
            return false;
    }

    const auto file = std::format("{}.jpg", stamp);
    const std::vector<int> jpeg{cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
    if (not cv::imwrite(std::format("{}/{}", dir_, file), bgr, jpeg))
    {
        std::println(stderr, "[zed-dump] could not write {}/{}", dir_, file);
        return false;
    }

    float rx = 0.f, ry = 0.f, ryaw = 0.f;
    if (robot_T_zed)
    {
        const Mat::RTMat room_T_robot = room_T_zed * robot_T_zed->inverse();
        rx = static_cast<float>(room_T_robot.translation().x());
        ry = static_cast<float>(room_T_robot.translation().y());
        ryaw = yaw_of(room_T_robot);
    }
    csv_ << std::format("{},{},{:.4f},{:.4f},{:.4f},{:.5f},{:.4f},{:.4f},{:.5f},{},{:.4f},{:.4f},{:.5f},{},{}\n",
                        stamp, file, cx, cy, static_cast<float>(t.z()), yaw, rx, ry, ryaw,
                        robot_T_zed ? 1 : 0, gt.x, gt.y, gt.angle, gt.stamp, gt.valid ? 1 : 0);
    csv_.flush();   // survives a crash mid-drive

    have_last_ = true;
    last_x_ = cx; last_y_ = cy; last_yaw_ = yaw;
    if (++saved_ % 100 == 0)
        std::println("[zed-dump] {} frames saved", saved_);
    return true;
}

} // namespace rc::diag
