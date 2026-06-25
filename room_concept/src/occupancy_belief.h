/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#pragma once

// OccupancyBelief — a non-saturating epistemic state for room_concept.
//
// The room *shape* is a fixed prior (read from SVG), so pose uncertainty is fully
// resolvable and the pose-FIM epistemic term self-extinguishes once localized.
// This adds the one hidden state that genuinely PERSISTS: the room's *interior*
// occupancy/visibility. Every in-room cell starts unknown (p=0.5, 1 bit). Each
// cycle the robot's actual 360° visibility (within a known room, determined by
// geometry) is folded in — observed cells lose entropy. The planner scores a
// candidate vantage by the *expected entropy it would resolve* (next-best-view /
// occupancy mutual information): a term that → 0 only when the room is COVERED,
// not when the pose is known. This is the principled replacement for the IoR
// heuristic. Upgrade path: feed the real lidar into observe_from() (ray-trace
// free along the beam / occupied at the endpoint) to also discover objects.

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace rc
{

class OccupancyBelief
{
public:
    struct Params
    {
        float cell_size     = 0.5f;   // grid resolution (m)
        float sensor_range  = 8.0f;   // lidar range used for visibility (m)
        int   n_rays        = 64;     // simulated 360° scan density
        float free_logodds  = 0.9f;   // |Δ log-odds| toward "known" per cell observation
        float logodds_clamp = 5.0f;   // saturation (|l| ≤ clamp ⇒ entropy floor)
    };

    /// Circle-approximated occluder (an object/obstacle): rays stop here, casting a
    /// shadow of persistently-unobserved cells the robot must move around to resolve.
    struct Obstacle { Eigen::Vector2f center{0.f, 0.f}; float radius = 0.f; };

    void configure(const Eigen::Vector2f& room_min, const Eigen::Vector2f& room_max,
                   const std::vector<Eigen::Vector2f>& polygon, const Params& p);
    [[nodiscard]] bool configured() const { return configured_; }

    void set_obstacles(std::vector<Obstacle> obs) { obstacles_ = std::move(obs); }

    /// Fold a real robot pose's 360° visibility into the belief (cells it saw → known).
    void observe_from(const Eigen::Vector2f& robot_pos);

    /// Expected entropy (bits) a 360° scan from `cand` would resolve = Σ current
    /// entropy over the cells it would newly observe. → 0 when nothing new is visible.
    [[nodiscard]] float expected_info_gain(const Eigen::Vector2f& cand) const;

    [[nodiscard]] float total_entropy() const;   // Σ entropy over in-room cells (bits)
    [[nodiscard]] float entropy_at(const Eigen::Vector2f& pos) const;   // O(1) cell lookup

    struct EntropyCell { Eigen::Vector2f center; float entropy; };
    [[nodiscard]] std::vector<EntropyCell> entropy_field() const;   // for the viewer
    [[nodiscard]] float cell_size() const { return params_.cell_size; }

private:
    template <class F> void march_visible(const Eigen::Vector2f& p, F&& f) const;
    [[nodiscard]] int index_of(const Eigen::Vector2f& pos) const;
    [[nodiscard]] Eigen::Vector2f cell_center(int idx) const;
    /// Distance along ray (o,d unit) to the nearest wall or obstacle, capped at sensor_range.
    [[nodiscard]] float ray_block_distance(const Eigen::Vector2f& o, const Eigen::Vector2f& d) const;

    Params params_;
    std::vector<Eigen::Vector2f> polygon_;
    std::vector<Obstacle>        obstacles_;
    std::vector<float>           logodds_;   // occupancy log-odds; 0 = unknown (p=0.5, H=1 bit)
    std::vector<std::uint8_t>    in_room_;
    Eigen::Vector2f origin_{0.f, 0.f};
    int  cols_ = 0, rows_ = 0;
    bool configured_ = false;

    // Per-call dedup so a cell hit by adjacent rays is counted once.
    mutable std::vector<std::uint32_t> visit_stamp_;
    mutable std::uint32_t epoch_ = 0;
};

}  // namespace rc
