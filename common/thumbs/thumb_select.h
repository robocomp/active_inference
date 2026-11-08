/*
 * common/thumbs/thumb_select.h — which views of an object are worth keeping: the best one per viewing-angle bin.
 * Geometry and bookkeeping only (no image processing, no model), so it runs in the concept agents.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <vector>

namespace rc::thumbs
{
inline constexpr int kViewBins = 8;

/// The bin of the camera's direction as seen from the object, in the object's own frame (its yaw).
inline int view_bin(float cam_x, float cam_y, float obj_x, float obj_y, float obj_yaw, int n_bins)
{
    const float two_pi = 2.f * std::numbers::pi_v<float>;
    float a = std::fmod(std::atan2(cam_y - obj_y, cam_x - obj_x) - obj_yaw, two_pi);
    if (a < 0.f) a += two_pi;
    return std::min(static_cast<int>(a / two_pi * static_cast<float>(n_bins)), n_bins - 1);
}

class ThumbSet
{
public:
    struct View { float score = 0.f; std::vector<std::uint8_t> jpeg; };

    /// Keep this view if it is the best seen in its bin. True when the set changed.
    bool offer(int bin, float score, std::vector<std::uint8_t> jpeg)
    {
        if (jpeg.empty()) return false;
        const auto it = bins_.find(bin);
        if (it != bins_.end() and it->second.score >= score) return false;
        bins_[bin] = {score, std::move(jpeg)};
        return true;
    }
    const std::map<int, View>& bins() const { return bins_; }

private:
    std::map<int, View> bins_;
};
}   // namespace rc::thumbs
