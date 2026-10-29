#include "zed_floor_stage.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <locale>


namespace sc
{
namespace
{
constexpr double kRad2Deg = 180.0 / M_PI;
// The producer's depth format codes (common/media_transport/generated/idl/image_frame.hpp), repeated here so the
// stage stays free of the DDS-generated header.
}

std::vector<Eigen::Vector3f> depth_to_points(const ImageIn& depth, const PinholeIntrinsics& K, int stride)
{
    std::vector<Eigen::Vector3f> out;
    if (depth.width == 0 or depth.height == 0 or not (K.fx > 0.0) or not (K.fy > 0.0) or K.width <= 0.0 or K.height <= 0.0)
        return out;
    const bool f32 = depth.format == kFormatDepthF32, z16 = depth.format == kFormatZ16;
    if (not f32 and not z16)
        return out;
    const std::size_t bpp = f32 ? 4 : 2;
    const std::size_t step = depth.step ? depth.step : depth.width * bpp;
    if (depth.data.size() < step * (depth.height - 1) + depth.width * bpp)
        return out;
    const double sx = depth.width / K.width, sy = depth.height / K.height;   // uniform when the aspect is the same
    // Different aspect ratios (sx != sy) mean the depth image is most likely a CROP of the RGB frame, not a rescale of
    // it, and a crop moves the principal point instead of scaling the focal lengths. The per-axis scaling is kept
    // (nothing better is known), but say so once, with both sizes.
    static bool warned_aspect = false;
    if (not warned_aspect and std::abs(sx / sy - 1.0) > 1e-3)
    {
        warned_aspect = true;
        std::fprintf(stderr, "[zed-floor] WARNING depth image %zux%zu and RGB intrinsics %gx%g have different aspect ratios "
                     "(sx %.4f, sy %.4f): probably a crop, not a scale; per-axis scaling kept\n",
                     std::size_t(depth.width), std::size_t(depth.height), K.width, K.height, sx, sy);
    }
    const double fx = K.fx * sx, fy = K.fy * sy, cx = 0.5 * depth.width, cy = 0.5 * depth.height;
    const std::size_t k = static_cast<std::size_t>(std::max(1, stride));
    out.reserve((depth.width / k + 1) * (depth.height / k + 1));
    for (std::size_t v = 0; v < depth.height; v += k)
    {
        const std::uint8_t* row = depth.data.data() + v * step;
        for (std::size_t u = 0; u < depth.width; u += k)
        {
            double d;
            if (f32) { float f; std::memcpy(&f, row + u * 4, 4); d = f; }
            else     { std::uint16_t z; std::memcpy(&z, row + u * 2, 2); d = z * 1e-3; }
            if (not std::isfinite(d) or not (d > 0.0))
                continue;
            out.emplace_back(static_cast<float>((u - cx) * d / fx), static_cast<float>(d),
                             static_cast<float>((cy - v) * d / fy));
        }
    }
    return out;
}

ZedFloorStage::ZedFloorStage(Config cfg) : cfg_(std::move(cfg)) {}

bool ZedFloorStage::observe(const ImageIn& depth, const PinholeIntrinsics& K, const std::optional<Eigen::Affine3d>& room_T_robot)
{
    if (not depth.robot_from_sensor.has_value())
        return false;
    const double period_ms = cfg_.rate_hz > 0.0 ? 1000.0 / cfg_.rate_hz : 0.0;
    if (last_ms_ != 0 and static_cast<double>(depth.stamp_ms - last_ms_) < period_ms)
        return false;
    auto pts = depth_to_points(depth, K, cfg_.stride);
    if (pts.empty())
        return false;
    last_ms_ = depth.stamp_ms;
    const Eigen::Affine3f& T = *depth.robot_from_sensor;
    for (auto& p : pts)
        p = T * p;
    std::int64_t place = -1;
    if (room_T_robot.has_value())
    {
        const auto& R = room_T_robot->linear();
        place = rc::mountf::place_key(static_cast<float>(room_T_robot->translation().x()),
                                      static_cast<float>(room_T_robot->translation().y()),
                                      static_cast<float>(std::atan2(R(1, 0), R(0, 0))));
    }
    floor_.observe_sweep(pts, T.translation(), place);
    ++n_;
    return true;
}

std::string ZedFloorStage::summary() const
{
    const auto i = floor_.info();
    const Eigen::Vector3d m = i.mean(), s = i.sigma();
    return std::format("[zed-floor] roll {:+.3f}±{:.3f} pitch {:+.3f}±{:.3f} deg  z {:+.1f}±{:.1f} mm (z includes depth scale; roll/pitch assume a UNIFORM depth scale, a range-dependent depth error bends the floor and leaks into pitch, see 4.2) ({} frames)",
                       m(0) * kRad2Deg, s(0) * kRad2Deg, m(1) * kRad2Deg, s(1) * kRad2Deg, m(2) * 1e3, s(2) * 1e3, n_);
}

void ZedFloorStage::log_state(std::int64_t stamp_ms)
{
    if (cfg_.log_dir.empty())
        return;
    if (not csv_.is_open())
    {
        std::error_code ec;
        std::filesystem::create_directories(cfg_.log_dir, ec);
        char stamp[32];
        const std::time_t now_t = std::time(nullptr);
        std::strftime(stamp, sizeof stamp, "%Y-%m-%d_%H-%M-%S", std::localtime(&now_t));
        csv_.imbue(std::locale::classic());   // CLAUDE.md: data files never acquire a comma separator (before open)
        csv_.open(cfg_.log_dir + "/zed_floor_" + stamp + ".csv");
        csv_ << "stamp_ms,droll,dpitch,dz_incl_depth_scale,sd_droll,sd_dpitch,sd_dz,n_frames,floor_share,floor_sigma\n";
    }
    if (not csv_.is_open())
        return;
    const auto i = floor_.info();
    const Eigen::Vector3d m = i.mean(), s = i.sigma();
    csv_ << stamp_ms << ',' << m(0) << ',' << m(1) << ',' << m(2) << ',' << s(0) << ',' << s(1) << ',' << s(2) << ',' << n_
         << ',' << floor_.floor_share() << ',' << floor_.sigma_floor() << '\n';
    csv_.flush();
}
}  // namespace sc
