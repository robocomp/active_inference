#include "countable_viewpoint.h"

#include <cmath>

namespace rc {

void accumulate_path(ViewpointPath& p, const Eigen::Vector2f& sensor_xy_room,
                     const Eigen::Vector2f& step_xy_room, const Eigen::Vector2f& object_xy)
{
    const Eigen::Vector2f r = sensor_xy_room - object_xy;
    const float z = r.norm();
    // Degenerate geometry (the sensor sits at the object centre: r̂ undefined), not a tuning threshold.
    if (z < 1e-3f)
        return;
    const Eigen::Vector2f rhat = r / z;
    const Eigen::Vector2f that(-rhat.y(), rhat.x());
    p.dphi_rad += std::abs(step_xy_room.dot(that)) / z;
    p.dz_m     += std::abs(step_xy_room.dot(rhat));
}

Eigen::Vector2f odometry_step_room(float v_adv, float v_side, float yaw_room, float dt_s)
{
    const float c = std::cos(yaw_room), s = std::sin(yaw_room);
    const Eigen::Vector2f robot(v_side * dt_s, v_adv * dt_s);   // x right, y forward
    return {c * robot.x() - s * robot.y(), s * robot.x() + c * robot.y()};
}

float rho_camera(const ViewpointPath& p, float z_m, float w_perp_m)
{
    const float half_angle = std::atan(w_perp_m / (2.f * z_m));
    return std::exp(-std::abs(p.dphi_rad) / half_angle - std::abs(p.dz_m) / z_m);
}

float rho_lidar_lateral(const ViewpointPath& p, float g)
{
    if (g <= 0.f) return 1.f;
    return std::exp(-std::abs(p.dphi_rad) / g);
}

float rho_lidar_height(const ViewpointPath& p, float z_m, float eps_rad, float g)
{
    if (g <= 0.f) return 1.f;
    return std::exp(-std::abs(p.dz_m) * std::abs(std::tan(eps_rad)) / (g * z_m));
}

float rho_chain(const Eigen::Vector2f& c, float var_xx, float var_yy)
{
    float e = 0.f;
    const float cs[2] = {c.x(), c.y()}, vs[2] = {var_xx, var_yy};
    for (int i = 0; i < 2; ++i)
    {
        if (vs[i] == 0.f)
        {
            if (cs[i] != 0.f) return 0.f;
            continue;
        }
        e += cs[i] * cs[i] / vs[i];
    }
    return std::exp(-0.5f * e);
}

}  // namespace rc
