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

// LidarPlaneReader — the ONE way every cortex agent reads the zero-copy LiDAR media plane.
//
// robot_concept's lidar3d_dds producers publish each physical LiDAR on its OWN plane, in that
// sensor's DEVICE frame (metres): "helios" (high 360) and "bpearl" (low). The legacy fused
// "lidar3D" plane (already in the robot frame) is bridged by robot_concept only while the
// per-device producers are absent. A DSR sensor node is BOTH the media-descriptor host and the
// name of the frame its points live in, so the device->target transform is just an RT-tree query
// keyed by the plane's node name.
//
// Every agent used to hand-roll the same discover->drain->transform dance against a single hard
// coded "lidar3D" node. This class factors that out so all agents subscribe IDENTICALLY:
//   - preference-ordered planes (e.g. {"helios","bpearl"}) with a fused fallback ("lidar3D"),
//   - lazy, throttled, descriptor-driven discovery (rc::media::make_lidar_subscriber_from_graph),
//   - drain the NEWEST frame per live plane (dedup by source stamp),
//   - transform each plane's points from ITS node frame into `target_frame` via the DSR RT tree
//     (InnerEigenAPI), and merge into one sweep (max source stamp).
//
// Crash-safety (mirrors the sanctioned room_concept::LidarIngestor pattern): construct AFTER the
// graph is loaded; call poll() ONLY from the Operating compute/main thread (it reads the graph);
// reset the reader BEFORE tearing the graph down, while G is alive. Fully dormant (no DDS
// participant) while poll(..., enabled=false), so an OFF feature is a true no-op.

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace DSR { class DSRGraph; class InnerEigenAPI; }

namespace rc::media
{

class LidarSubscriber;

// One merged sweep, transformed into the caller's requested target frame.
struct LidarSweep
{
    std::vector<Eigen::Vector3f> points;                             // merged returns, target frame
    Eigen::Vector3f              origin   = Eigen::Vector3f::Zero();  // sensor centre, target frame
    std::int64_t                 stamp_ms = 0;                       // max source stamp of the merged planes
    // NOTE: `origin` is the sensor centre and is only meaningful when the reader consumes a SINGLE
    // plane (e.g. a first-hit ray factor). With several planes merged, every return no longer shares
    // one origin; multi-plane consumers (obstacle clouds) must not rely on it. `origin` then holds
    // the last contributing plane's centre.
};

class LidarPlaneReader
{
public:
    // `preferred_planes` are DSR sensor node names in preference order; `fused_fallback` is used only
    // while NONE of the preferred planes is live. graph + inner_eigen must outlive this reader.
    LidarPlaneReader(std::shared_ptr<DSR::DSRGraph> graph,
                     DSR::InnerEigenAPI* inner_eigen,
                     std::vector<std::string> preferred_planes,
                     std::string fused_fallback = "lidar3D",
                     std::string stream_key = "lidar");
    ~LidarPlaneReader();
    LidarPlaneReader(const LidarPlaneReader&) = delete;
    LidarPlaneReader& operator=(const LidarPlaneReader&) = delete;

    // Drain + transform + merge. Returns nullopt if nothing fresh this cycle (or disabled).
    //  - target_frame : DSR node whose frame the merged points end up in ("room","body",<robot>,…).
    //  - interpolate  : true ⇒ Interpolated RT at each sweep's stamp (needed for the dynamic
    //                   room<-robot leg when target is the room); false ⇒ Nearest (a robot-fixed
    //                   target only crosses static mount edges, so the stamp is irrelevant).
    //  - enabled      : false ⇒ stay dormant, create no DDS participant, return nullopt.
    std::optional<LidarSweep> poll(const std::string& target_frame,
                                   bool interpolate = true, bool enabled = true);

    // True once at least one plane (preferred or fallback) has a live subscriber.
    [[nodiscard]] bool any_live() const noexcept;

private:
    void ensure_subscribers();
    // Drain one plane's newest frame into `raw` (its own device frame); returns its stamp or <=prev.
    std::int64_t drain_newest(LidarSubscriber& sub, std::int64_t prev_ts,
                              std::vector<Eigen::Vector3f>& raw) const;
    // Append `raw` transformed node_frame->target into `out`; returns the sensor centre in target frame.
    bool append_transformed(const std::string& node_frame, const std::string& target_frame,
                            std::int64_t stamp_ms, bool interpolate,
                            const std::vector<Eigen::Vector3f>& raw,
                            std::vector<Eigen::Vector3f>& out, Eigen::Vector3f& origin_out) const;

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    std::vector<std::string>      preferred_;
    std::string                   fallback_;
    std::string                   stream_key_;

    struct Plane
    {
        std::string                      node;
        std::unique_ptr<LidarSubscriber> sub;
        std::int64_t                     last_ts = 0;
    };
    std::vector<Plane> preferred_planes_;   // one entry per preferred node (sub filled lazily)
    Plane              fallback_plane_;      // fused fallback (sub filled lazily, dropped once preferred live)
    std::chrono::steady_clock::time_point last_discovery_{};
};

}  // namespace rc::media
