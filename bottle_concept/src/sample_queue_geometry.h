/*
 * sample_queue_geometry.h — bottle's object-specific geometry policy for the shared SampleQueue.
 *
 * Specialises rc::SampleQueueGeometry<BottleModel> with the cylinder geometry (no yaw; rim = lateral
 * surface + caps). Bodies are the bottle's original sample_queue primitives, moved verbatim. Include
 * this wherever SampleQueue<BottleModel> methods are instantiated (bottle_instance.h pulls it in).
 */

#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Dense>

#include "../../common/sample_queue/sample_queue.h"
#include "bottle_model.h"

namespace rc {

template <>
struct SampleQueueGeometry<BottleModel>
{
    static int bin_index(const Eigen::Vector3f& p, const BottleState& s, const SampleQueueParams& params)
    {
        const int na = params.num_angle_bins;
        const int nz = params.num_z_bins;

        const float angle = std::atan2(p.y() - s.cy, p.x() - s.cx);
        int ab = static_cast<int>((angle + std::numbers::pi_v<float>) / (2.0f * std::numbers::pi_v<float>) * na) % na;
        if (ab < 0) ab += na;

        // Height measured from the cylinder base so bins span the bottle, not the floor.
        const float z_rel = p.z() - (s.cz - s.height * 0.5f);
        int zb = static_cast<int>(z_rel / params.z_bin_size);
        zb = std::clamp(zb, 0, nz - 1);

        return ab * nz + zb;
    }

    static float edge_score(const Eigen::Vector3f& p, const BottleModel& model, const SampleQueueParams& params)
    {
        const auto& s = model.state();
        const float lx = p.x() - s.cx;
        const float ly = p.y() - s.cy;
        const float lz = p.z() - s.cz;

        const float r_xy   = std::sqrt(lx * lx + ly * ly);
        const float d_side = std::abs(r_xy - s.radius);                 // to lateral surface
        const float d_cap  = std::abs(std::abs(lz) - s.height * 0.5f);  // to nearer cap plane

        const float th = params.edge_proximity_threshold;
        const bool close_side = d_side < th;
        const bool close_cap  = d_cap  < th;

        float score = (close_side and close_cap) ? 1.0f
                    : (close_side or close_cap)   ? 0.4f
                                                  : 0.0f;
        score += std::exp(-std::min(d_side, d_cap) / 0.005f) * 0.2f;

        return std::min(score, 1.0f);
    }

    static Eigen::Vector3f to_local_frame(const Eigen::Vector3f& p, const BottleState& s)
    {
        // A cylinder is rotationally symmetric: no yaw, just translate to its centre.
        return {p.x() - s.cx, p.y() - s.cy, p.z() - s.cz};
    }

    static std::array<float, 6> face_coverage(const BottleModel& model,
                                              const std::vector<SamplePoint>& pts,
                                              const SampleQueueParams& params)
    {
        // Buckets: 0=+x, 1=-x, 2=+y, 3=-y, 4=top cap, 5=bottom cap (bottle frame).
        std::array<float, 6> coverage{};
        coverage.fill(0.0f);

        if (pts.empty())
            return coverage;

        const auto& s   = model.state();
        const float hh  = s.height * 0.5f;
        const float eps = 0.02f;

        for (const auto& sp : pts)
        {
            const float lx = sp.position.x() - s.cx;
            const float ly = sp.position.y() - s.cy;
            const float lz = sp.position.z() - s.cz;
            const float w_i = anchor_weight(sp, params);

            if (std::abs(lx) >= std::abs(ly))
                coverage[lx > 0.0f ? 0 : 1] += w_i;
            else
                coverage[ly > 0.0f ? 2 : 3] += w_i;

            if (lz >  hh - eps) coverage[4] += w_i;
            if (lz < -hh + eps) coverage[5] += w_i;
        }

        return coverage;
    }
};

}  // namespace rc
