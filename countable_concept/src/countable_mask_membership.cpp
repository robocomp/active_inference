#include "countable_mask_membership.h"

#include <algorithm>
#include <cmath>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace rc {

MaskMembership MaskMembership::build(std::span<const Eigen::Vector2f> pixels)
{
    MaskMembership mm;
    if (pixels.empty())
        return mm;

    // Stride q: per row, the distinct columns sorted; each column's nearest-neighbour gap; median over all.
    std::map<int, std::vector<float>> rows;
    float min_c = pixels[0].x(), max_c = min_c, min_r = pixels[0].y(), max_r = min_r;
    for (const auto& p : pixels)
    {
        rows[static_cast<int>(std::lround(p.y()))].push_back(p.x());
        min_c = std::min(min_c, p.x()); max_c = std::max(max_c, p.x());
        min_r = std::min(min_r, p.y()); max_r = std::max(max_r, p.y());
    }
    std::vector<float> gaps;
    for (auto& [r, cs] : rows)
    {
        std::ranges::sort(cs);
        const auto last = std::ranges::unique(cs, [](float a, float b) { return std::abs(a - b) < 1e-3f; });
        cs.erase(last.begin(), last.end());
        for (std::size_t i = 0; i < cs.size(); ++i)
        {
            float g = 1e30f;
            if (i > 0)            g = std::min(g, cs[i] - cs[i - 1]);
            if (i + 1 < cs.size()) g = std::min(g, cs[i + 1] - cs[i]);
            if (g < 1e29f) gaps.push_back(g);
        }
    }
    float q = 1.f;
    if (not gaps.empty())
    {
        std::ranges::nth_element(gaps, gaps.begin() + gaps.size() / 2);
        q = std::max(gaps[gaps.size() / 2], 1e-3f);
    }

    mm.q_ = q; mm.c0_ = min_c; mm.r0_ = min_r;
    const int nw = static_cast<int>(std::lround((max_c - min_c) / q)) + 1;
    const int nh = static_cast<int>(std::lround((max_r - min_r) / q)) + 1;
    mm.w_ = nw + 2; mm.h_ = nh + 2;

    cv::Mat inside = cv::Mat::zeros(mm.h_, mm.w_, CV_8UC1);
    for (const auto& p : pixels)
    {
        const int u = static_cast<int>(std::lround((p.x() - min_c) / q)) + 1;
        const int v = static_cast<int>(std::lround((p.y() - min_r) / q)) + 1;
        inside.at<std::uint8_t>(v, u) = 255;
    }
    cv::Mat outside;
    cv::bitwise_not(inside, outside);
    cv::Mat d_in, d_out;
    cv::distanceTransform(inside,  d_in,  cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);   // to the nearest non-mask cell
    cv::distanceTransform(outside, d_out, cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);   // to the nearest mask cell
    mm.s_.assign(static_cast<std::size_t>(mm.w_) * mm.h_, 0.f);
    for (int v = 0; v < mm.h_; ++v)
        for (int u = 0; u < mm.w_; ++u)
        {
            const bool in = inside.at<std::uint8_t>(v, u) != 0;
            // A mask cell at distance 1 from the outside reads +q/2; the first cell outside reads −q/2.
            mm.s_[static_cast<std::size_t>(v) * mm.w_ + u] =
                in ? d_in.at<float>(v, u) * q - 0.5f * q : -(d_out.at<float>(v, u) * q - 0.5f * q);
        }
    mm.empty_ = false;
    return mm;
}

float MaskMembership::signed_distance_px(float col, float row) const
{
    if (empty_)
        return -1e9f;
    const float u = (col - c0_) / q_ + 1.f, v = (row - r0_) / q_ + 1.f;
    const float uc = std::clamp(u, 0.f, static_cast<float>(w_ - 1));
    const float vc = std::clamp(v, 0.f, static_cast<float>(h_ - 1));
    const int u0 = std::min(static_cast<int>(std::floor(uc)), w_ - 2 >= 0 ? w_ - 2 : 0);
    const int v0 = std::min(static_cast<int>(std::floor(vc)), h_ - 2 >= 0 ? h_ - 2 : 0);
    const float fu = uc - u0, fv = vc - v0;
    const auto at = [&](int a, int b) { return s_[static_cast<std::size_t>(b) * w_ + a]; };
    const float s = (1 - fu) * (1 - fv) * at(u0, v0) + fu * (1 - fv) * at(u0 + 1, v0)
                  + (1 - fu) * fv * at(u0, v0 + 1) + fu * fv * at(u0 + 1, v0 + 1);
    const float dist_px = std::hypot(u - uc, v - vc) * q_;   // 0 inside the raster
    return s - dist_px;
}

std::function<float(const Eigen::Vector3f&)> make_membership(Projector proj, const MaskMembership& mm, float f_px,
                                                             float sigma_range_m, float sigma_chain_m,
                                                             float mot_pos_var_m2, float pi_mask)
{
    return [proj = std::move(proj), mm, f_px, sigma_range_m, sigma_chain_m, mot_pos_var_m2,
            pi_mask](const Eigen::Vector3f& p_room) -> float
    {
        const auto cr = proj(p_room);
        if (not cr.has_value())
            return 0.f;
        const float sigma = projection_sigma_px(f_px, cr->z(), sigma_range_m, sigma_chain_m, mot_pos_var_m2,
                                                mm.stride_px());
        return pi_mask * membership_weight(mm.signed_distance_px(cr->x(), cr->y()), sigma);
    };
}

}  // namespace rc
