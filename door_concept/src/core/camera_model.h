// camera_model.h — DSR-free camera projection for the door core: DSR::CameraAPI::project (cortex/api/dsr_camera_api.cpp),
// line for line, for the three models CameraAPI knows (pinhole = the zed; equirectangular / cylindrical = the 360 ricoh).
// The core cannot include DSR, so this reproduces the arithmetic; tests/test_camera_model.cpp pins it against the
// formulas quoted from CameraAPI. Camera frame: X right, Y forward, Z up.
#pragma once
#include "door_cycle_io.h"
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <Eigen/Dense>

namespace rc::door::core
{
enum class Projection : std::uint8_t { Pinhole, Equirectangular, Cylindrical };   // == DSR::CameraAPI::ProjectionModel

struct CameraSpec
{
    Projection model = Projection::Pinhole;
    std::uint32_t width = 0, height = 0;
    float focal_x = 0.f, focal_y = 0.f;
    float az_sign = 1.f, az_offset = 0.f;           // 360 models: the panorama's column convention (mirror + seam zero)
};

/// CameraIntrinsics::projection_model holds the CameraAPI enum value (0 pinhole, 1 equirect, 2 cylindrical).
[[nodiscard]] inline CameraSpec camera_spec(const CameraIntrinsics& c)
{
    CameraSpec s;
    s.model = c.projection_model == 1 ? Projection::Equirectangular
            : c.projection_model == 2 ? Projection::Cylindrical : Projection::Pinhole;
    s.width = c.width;  s.height = c.height;
    s.focal_x = c.focal_x;  s.focal_y = c.focal_y;
    s.az_sign = c.equirect_az_sign;  s.az_offset = c.equirect_az_offset;
    return s;
}

/// Pixel of a point in the camera frame. nullopt: behind a pinhole (y <= 0), or at the origin of a 360 camera.
/// (CameraAPI returns the image centre at the 360 origin and a sign-flipped, meaningless pixel behind a pinhole; a
/// caller that reprojects a door needs to know the point has no pixel.)
[[nodiscard]] inline std::optional<Eigen::Vector2d> project(const CameraSpec& c, const Eigen::Vector3d& p)
{
    constexpr double two_pi = 2.0 * std::numbers::pi;
    const double W = static_cast<double>(c.width), H = static_cast<double>(c.height);
    auto column = [&]
    {
        double u = ((c.az_sign * std::atan2(p.x(), p.y()) + c.az_offset) / two_pi + 0.5) * W;
        u = std::fmod(u, W);
        if (u < 0.0) u += W;
        return u;
    };
    switch (c.model)
    {
    case Projection::Equirectangular:
    {
        const double r = p.norm();
        if (r < 1e-9) return std::nullopt;
        return Eigen::Vector2d(column(), (std::asin(-p.z() / r) / std::numbers::pi + 0.5) * H);
    }
    case Projection::Cylindrical:
    {
        const double rho = std::hypot(p.x(), p.y());
        if (p.norm() < 1e-9) return std::nullopt;
        const double fv = W / two_pi;
        const double v = (rho > 1e-9) ? (0.5 * H - fv * (p.z() / rho)) : (p.z() >= 0.0 ? 0.0 : H);
        return Eigen::Vector2d(column(), v);
    }
    case Projection::Pinhole:
        break;
    }
    if (p.y() <= 0.0) return std::nullopt;
    // CameraAPI: centre = width / 2 (unsigned, stored float, passed as int); u = fx X/Y + cx, v = -fy Z/Y + cy.
    const int cx = static_cast<int>(static_cast<float>(c.width / 2u)), cy = static_cast<int>(static_cast<float>(c.height / 2u));
    return Eigen::Vector2d(c.focal_x * p.x() / p.y() + cx, -c.focal_y * p.z() / p.y() + cy);
}
}  // namespace rc::door::core
