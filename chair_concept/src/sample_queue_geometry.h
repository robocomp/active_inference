/*
 * sample_queue_geometry.h — chair's object-specific geometry policy for the shared SampleQueue.
 *
 * Specialises rc::SampleQueueGeometry<ChairModel> with the box geometry (yaw-aware; faces = ±x/±y/top).
 * Bodies are chair's original sample_queue primitives, moved verbatim. Include this wherever
 * SampleQueue<ChairModel> methods are instantiated (chair_instance.h pulls it in).
 */

#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Dense>

#include "../../common/sample_queue/sample_queue.h"
#include "chair_model.h"

namespace rc {

template <>
struct SampleQueueGeometry<ChairModel>
{
    static int bin_index(const Eigen::Vector3f& p, const ChairState& s, const SampleQueueParams& params)
    {
        const int na = params.num_angle_bins;
        const int nz = params.num_z_bins;

        const float angle = std::atan2(p.y() - s.cy, p.x() - s.cx);
        int ab = static_cast<int>((angle + std::numbers::pi_v<float>) / (2.0f * std::numbers::pi_v<float>) * na) % na;
        if (ab < 0) ab += na;

        int zb = static_cast<int>(p.z() / params.z_bin_size);
        zb = std::clamp(zb, 0, nz - 1);

        return ab * nz + zb;
    }

    static float edge_score(const Eigen::Vector3f& p, const ChairModel& model, const SampleQueueParams& params)
    {
        const auto& s = model.state();
        const float half_w = s.seat_w * 0.5f;
        const float half_h = s.seat_d * 0.5f;
        const float half_t = ChairModel::SEAT_THICKNESS * 0.5f;
        const float top_cz = s.cz + s.seat_h - half_t;

        const float cos_t = std::cos(-s.yaw);
        const float sin_t = std::sin(-s.yaw);
        const float px = p.x() - s.cx;
        const float py = p.y() - s.cy;
        const float lx = px * cos_t - py * sin_t;
        const float ly = px * sin_t + py * cos_t;
        const float lz = p.z();

        const float th = params.edge_proximity_threshold;

        const float dist_x = std::abs(std::abs(lx) - half_w);
        const float dist_y = std::abs(std::abs(ly) - half_h);
        const float dist_z = std::abs(lz - top_cz);

        const float close_x = (dist_x < th) ? 1.0f : 0.0f;
        const float close_y = (dist_y < th) ? 1.0f : 0.0f;
        const float close_z = (dist_z < half_t + th) ? 1.0f : 0.0f;

        const float faces_close = close_x + close_y + close_z;

        float score = std::max(0.0f, faces_close - 1.0f) / 2.0f;

        const float min_dist = std::min(dist_x, dist_y);
        score += std::exp(-min_dist / 0.02f) * 0.2f;

        return std::min(score, 1.0f);
    }

    static Eigen::Vector3f to_local_frame(const Eigen::Vector3f& p, const ChairState& s)
    {
        const float cos_t = std::cos(-s.yaw);
        const float sin_t = std::sin(-s.yaw);
        const float px = p.x() - s.cx;
        const float py = p.y() - s.cy;
        return {
            px * cos_t - py * sin_t,
            px * sin_t + py * cos_t,
            p.z()
        };
    }

    static std::array<float, 6> face_coverage(const ChairModel& model,
                                              const std::vector<SamplePoint>& pts,
                                              const SampleQueueParams& params)
    {
        // Faces indexed as: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z(top), 5=-z(bottom)
        std::array<float, 6> coverage{};
        coverage.fill(0.0f);

        if (pts.empty())
            return coverage;

        const auto& s    = model.state();
        const float hw   = s.seat_w * 0.5f;
        const float hh   = s.seat_d * 0.5f;
        const float eps  = 0.05f;
        const float cos_t = std::cos(-s.yaw);
        const float sin_t = std::sin(-s.yaw);

        for (const auto& sp : pts)
        {
            const float px = sp.position.x() - s.cx;
            const float py = sp.position.y() - s.cy;
            const float lx = px * cos_t - py * sin_t;
            const float ly = px * sin_t + py * cos_t;

            const float w_i = anchor_weight(sp, params);

            if (std::abs(lx - hw)  < eps) coverage[0] += w_i;  // +x face
            if (std::abs(lx + hw)  < eps) coverage[1] += w_i;  // -x face
            if (std::abs(ly - hh)  < eps) coverage[2] += w_i;  // +y face
            if (std::abs(ly + hh)  < eps) coverage[3] += w_i;  // -y face
            if (std::abs(sp.position.z() - (s.cz + s.seat_h)) < eps) coverage[4] += w_i;  // +z seat top
        }

        return coverage;
    }
};

}  // namespace rc
