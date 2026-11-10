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
inline constexpr int kViewBins = 12;     // the view atlas's bins (common/view_atlas), so a thumbnail and a wedge are the same direction

/// The bin of the camera's direction as seen from the object, in the object's own frame (its yaw).
/// Bin i covers [-pi + i*w, -pi + (i+1)*w), w = 2pi/n_bins, as ViewAtlas::bin_of: 0 rad, the object's front, is the edge between n/2-1 and n/2.
inline int view_bin(float cam_x, float cam_y, float obj_x, float obj_y, float obj_yaw, int n_bins)
{
    const float pi = std::numbers::pi_v<float>;
    const float two_pi = 2.f * pi;
    float a = std::fmod(std::atan2(cam_y - obj_y, cam_x - obj_x) - obj_yaw + pi, two_pi);
    if (a < 0.f) a += two_pi;
    return std::min(static_cast<int>(a / two_pi * static_cast<float>(n_bins)), n_bins - 1);
}

/// The direction at the centre of a bin, in radians from the object's front.
inline float bin_centre_rad(int bin, int n_bins)
{
    const float pi = std::numbers::pi_v<float>;
    return -pi + (static_cast<float>(bin) + .5f) * 2.f * pi / static_cast<float>(n_bins);
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
