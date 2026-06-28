/*
 * sample_queue_geometry.h — table's object-specific geometry policy for the shared SampleQueue.
 *
 * Specialises rc::SampleQueueGeometry<TableModel> with the box geometry (yaw-aware; faces = ±x/±y/top).
 * Bodies are table's original sample_queue primitives, moved verbatim. Include this wherever
 * SampleQueue<TableModel> methods are instantiated (table_instance.h pulls it in).
 */

#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Dense>

#include "../../common/sample_queue/sample_queue.h"
#include "table_model.h"

namespace rc {

template <>
struct SampleQueueGeometry<TableModel>
{
    // A table top is a flat rectangle, so anchors must distribute over its (x,y) FOOTPRINT — especially
    // the edges that define w/h. Bin by a yaw-aware LOCAL xy-grid, NOT by angle/z: the old atan2 policy
    // (a cylinder leftover) lumped centre+edge points into one angular sector and collapsed the single-
    // height top into one z-bin, so MaxPerBin under-sampled the rectangle's extent and the shape never
    // got pinned. num_angle_bins → x-grid divisions, num_z_bins → y-grid divisions (raise NumZBins for
    // finer y). The grid spans a fixed ±kHalfExtent so points BEYOND an undersized model still land in
    // their own cells — letting the edges be sampled and the box expand to them.
    static int bin_index(const Eigen::Vector3f& p, const TableState& s, const SampleQueueParams& params)
    {
        const int nx = std::max(1, params.num_angle_bins);
        const int ny = std::max(1, params.num_z_bins);
        constexpr float kHalfExtent = 1.0f;   // m: full grid resolution out to 2 m across; larger clamps to edge cells

        const float cos_t = std::cos(-s.yaw), sin_t = std::sin(-s.yaw);
        const float px = p.x() - s.cx, py = p.y() - s.cy;
        const float lx = px * cos_t - py * sin_t;
        const float ly = px * sin_t + py * cos_t;

        const float u = (lx + kHalfExtent) / (2.0f * kHalfExtent);   // → [0,1] across the grid
        const float v = (ly + kHalfExtent) / (2.0f * kHalfExtent);
        const int xb = std::clamp(static_cast<int>(u * nx), 0, nx - 1);
        const int yb = std::clamp(static_cast<int>(v * ny), 0, ny - 1);
        return xb * ny + yb;
    }

    static float edge_score(const Eigen::Vector3f& p, const TableModel& model, const SampleQueueParams& params)
    {
        const auto& s = model.state();
        const float half_w = s.w * 0.5f;
        const float half_h = s.h * 0.5f;
        const float half_t = TableModel::TOP_THICKNESS * 0.5f;
        const float top_cz = s.table_height - half_t;

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

    static Eigen::Vector3f to_local_frame(const Eigen::Vector3f& p, const TableState& s)
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

    static std::array<float, 6> face_coverage(const TableModel& model,
                                              const std::vector<SamplePoint>& pts,
                                              const SampleQueueParams& params)
    {
        // Faces indexed as: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z(top), 5=-z(bottom)
        std::array<float, 6> coverage{};
        coverage.fill(0.0f);

        if (pts.empty())
            return coverage;

        const auto& s    = model.state();
        const float hw   = s.w * 0.5f;
        const float hh   = s.h * 0.5f;
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
            if (std::abs(sp.position.z() - s.table_height) < eps) coverage[4] += w_i;  // +z top
        }

        return coverage;
    }
};

}  // namespace rc
