// seen_map.h — Task R: what the FORWARD CAMERA has already looked at. Eigen + std only.
// The camera sees ahead of the base (a half-angle wedge to a range). A cell that was inside that wedge at some tick has been SEEN;
// travelling over seen space is not blind (the camera, not only the LiDAR band, checked it: leaf, jamb, low obstacle). Backing out
// of a dead end along the way in is travel over seen space; backing into a room the camera has never faced is not.
// Cells are the mapping resolution (0.1 m), not a decision threshold. STATIC-WORLD assumption: nothing expires (a door that closed
// behind the robot is still "seen" — flagged; the LiDAR contact model still sees any solid return).
#pragma once
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <Eigen/Dense>
#include "policy_types.h"

namespace rc::policy
{
class SeenMap
{
public:
    explicit SeenMap(float cell_m = 0.1f) : cell_(cell_m) {}
    // Mark every cell inside the camera wedge (|bearing| <= half_fov, range <= range_m) at `pose` (yaw = forward heading).
    void mark(const Pose2& pose, float half_fov_rad, float range_m)
    {
        if (not (half_fov_rad >= 0.f) or not (range_m > 0.f)) return;
        const int n = int(std::ceil(range_m / cell_));
        const int cx = int(std::floor(pose.p.x() / cell_)), cy = int(std::floor(pose.p.y() / cell_));
        const float ch = std::cos(pose.yaw), sh = std::sin(pose.yaw);
        for (int ix = cx - n; ix <= cx + n; ++ix)
            for (int iy = cy - n; iy <= cy + n; ++iy)
            {
                const float dx = (float(ix) + 0.5f) * cell_ - pose.p.x(), dy = (float(iy) + 0.5f) * cell_ - pose.p.y();
                const float r = std::hypot(dx, dy);
                if (r > range_m) continue;
                if (r > 0.5f * cell_ and std::abs(std::atan2(-sh * dx + ch * dy, ch * dx + sh * dy)) > half_fov_rad) continue;
                cells_.insert(key(ix, iy));
            }
    }
    bool seen(const Eigen::Vector2f& q) const
    { return cells_.contains(key(int(std::floor(q.x() / cell_)), int(std::floor(q.y() / cell_)))); }
    void clear() { cells_.clear(); }
    std::size_t size() const { return cells_.size(); }
private:
    static std::int64_t key(int ix, int iy) { return (std::int64_t(ix) << 32) ^ std::int64_t(std::uint32_t(iy)); }
    float cell_;
    std::unordered_set<std::int64_t> cells_;
};
}  // namespace rc::policy
