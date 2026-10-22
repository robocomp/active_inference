/*
 * object_approach_geometry.h — where to stand to approach an object, and from which sides. SHARED, PURE (Eigen only).
 * Tested in tests/test_object_approach_geometry.cpp. Plan: docs/superpowers/plans/2026-10-18-object-approach-affordance.md.
 *
 * ONE STANDOFF RULE FOR EVERY OBJECT. For an approach direction d (unit, from the object's centre outwards):
 *
 *     D(d) = h(d) + clearance + k·σ(d)
 *
 *   · h(d): the footprint box's SUPPORT FUNCTION, the exact centre-to-edge distance along d. The half diagonal (what
 *     chat used) is right only towards a corner and stands too far off every face.
 *   · clearance: half the robot body diagonal (0.46 × 0.65 m → 0.398 m). An affordance ends with a facing rotation
 *     in place, run with no obstacle check (controller config §GoalFacingYawEnabled), so the standpoint must leave
 *     the body's swept circle free. NOTHING ELSE IS ADDED: the controller's ComfortStandoff is an MPPI comfort cost,
 *     and stacking it is the 0.95 m mistake controller_types.h records. A person's clearance is social instead.
 *   · σ(d): the object's position spread along d plus the robot's own; k = 2 (~95 %), the figure residual_concept
 *     (PublishExtentSigmaK) and the controller's σ-padding use.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace rc::pragmatic::approach
{

using Vec2 = Eigen::Vector2f;

/// The footprint box (half sizes along the object's own x, y; rotated by yaw) measured from its centre along `dir`.
[[nodiscard]] inline float support(const Vec2& half_extent, float yaw, const Vec2& dir)
{
    const float phi = std::atan2(dir.y(), dir.x()) - yaw;
    return half_extent.x() * std::abs(std::cos(phi)) + half_extent.y() * std::abs(std::sin(phi));
}

/// σ along `dir`: the object's position covariance projected on it, plus the robot's own position σ.
[[nodiscard]] inline float sigma_along(const Eigen::Matrix2f& obj_cov, const Vec2& dir, float robot_sigma_m)
{
    const float v = dir.dot(obj_cov * dir) + robot_sigma_m * robot_sigma_m;
    return std::sqrt(std::max(0.0f, v));
}

struct StandCfg
{
    float clearance_m = 0.398f;   ///< half the robot body diagonal (see the header); a person's social distance
    float sigma_k     = 2.0f;
};

struct Pose
{
    float x = 0.0f, y = 0.0f, yaw = 0.0f;
};

/// Where to stand to approach the object from `dir`, facing its centre.
[[nodiscard]] inline Pose standpoint(const Vec2& centre, float yaw, const Vec2& half_extent, const Vec2& dir,
                                     const Eigen::Matrix2f& obj_cov, float robot_sigma_m, const StandCfg& cfg)
{
    const Vec2  d = dir.normalized();
    const float D = support(half_extent, yaw, d) + cfg.clearance_m + cfg.sigma_k * sigma_along(obj_cov, d, robot_sigma_m);
    const Vec2  s = centre + D * d;
    return {s.x(), s.y(), std::atan2(-d.y(), -d.x())};
}

/// One side an object can be approached from: its name and its outward direction (room frame).
struct Dir
{
    std::string kind;
    Vec2        dir;
};

/// A fronted object's four sides. Left and right are the OBJECT's own: a chair's left as one sits in it.
[[nodiscard]] inline std::vector<Dir> fronted_sides(float front_yaw)
{
    constexpr float q = std::numbers::pi_v<float> / 2;
    const auto at = [](float a) { return Vec2{std::cos(a), std::sin(a)}; };
    return {{"front", at(front_yaw)}, {"left", at(front_yaw + q)}, {"back", at(front_yaw + 2 * q)},
            {"right", at(front_yaw - q)}};
}

/// Is a fronted object's front known? Its sides are 90° apart, so "front" names the right side while the yaw error
/// stays within ±45°; at 2σ (~95 %, k as above) that is 2·σ_yaw < π/4. Otherwise the front may be a side, and the
/// object is approached by its box faces instead.
[[nodiscard]] inline bool front_known(float yaw_sigma)
{
    return std::isfinite(yaw_sigma) and 2.0f * yaw_sigma < std::numbers::pi_v<float> / 4;
}

/// A frontless object's four box faces, each named by the room axis its outward normal is nearest (xp, yp, xn, yn).
/// A table's fitted yaw has a 180° symmetry, so names from its own axes would swap on a refit; room-axis names do
/// not. Near 45° a face would be renamed on noise, so a name is kept until the yaw is past 45° by 2σ of itself.
class FaceFrame
{
public:
    std::vector<Dir> update(float yaw, float yaw_sigma)
    {
        constexpr float q = std::numbers::pi_v<float> / 2;
        const int best = static_cast<int>(std::lround(yaw / q));
        if (not branch_set_)
        {
            branch_     = best;
            branch_set_ = true;
        }
        else if (std::abs(wrap(yaw - static_cast<float>(branch_) * q)) > q / 2 + 2.0f * yaw_sigma)
            branch_ = best;
        static constexpr std::array<const char*, 4> names{"xp", "yp", "xn", "yn"};
        std::vector<Dir> out;
        for (int a = 0; a < 4; ++a)
        {
            // axis a takes the face normal yaw + (a − branch)·π/2: the box's true normal, not the room axis
            const float n = yaw + static_cast<float>(a - branch_) * q;
            out.push_back({names[static_cast<std::size_t>(a)], Vec2{std::cos(n), std::sin(n)}});
        }
        return out;
    }

private:
    static float wrap(float a) { return std::remainder(a, 2.0f * std::numbers::pi_v<float>); }
    int  branch_     = 0;
    bool branch_set_ = false;
};

/// The side whose direction points most towards the robot: a frontless object's default side.
[[nodiscard]] inline const Dir& facing(const std::vector<Dir>& sides, const Vec2& centre, const Vec2& robot)
{
    const Vec2 to_robot = robot - centre;
    std::size_t best = 0;
    for (std::size_t i = 1; i < sides.size(); ++i)
        if (sides[i].dir.dot(to_robot) > sides[best].dir.dot(to_robot)) best = i;
    return sides[best];
}

/// A side's precondition: its standpoint inside the room polygon and at least `margin_m` from every wall, so the
/// body can turn there.
[[nodiscard]] inline bool fits_in_room(const Vec2& p, const std::vector<Vec2>& polygon, float margin_m)
{
    if (polygon.size() < 3) return false;
    bool inside = false;
    float min_d = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++)
    {
        const Vec2& a = polygon[j];
        const Vec2& b = polygon[i];
        if ((b.y() > p.y()) != (a.y() > p.y())
            and p.x() < (a.x() - b.x()) * (p.y() - b.y()) / (a.y() - b.y()) + b.x())
            inside = not inside;
        const Vec2  ab = b - a;
        const float t  = std::clamp((p - a).dot(ab) / std::max(ab.squaredNorm(), 1e-12f), 0.0f, 1.0f);
        min_d = std::min(min_d, (p - (a + t * ab)).norm());
    }
    return inside and min_d >= margin_m;
}

/// P(d ≤ R) for a distance known to ±σ (normal CDF), as door_pragmatics.h's prob_within.
[[nodiscard]] inline float prob_within(float d, float R, float sigma)
{
    const float s = std::max(sigma, 1e-3f);
    return 0.5f * std::erfc(-(R - d) / (s * std::numbers::sqrt2_v<float>));
}

/// The offer's value falls as the robot arrives: arriving is what satisfies an approach, so an approach the robot
/// already stands at is worth nothing (the door's ping-pong lesson). `arrival_m` is the controller's goal threshold.
[[nodiscard]] inline float arrival_value(float value, float dist_m, float arrival_m, float sigma_m)
{
    return value * (1.0f - prob_within(dist_m, arrival_m, sigma_m));
}

}   // namespace rc::pragmatic::approach
