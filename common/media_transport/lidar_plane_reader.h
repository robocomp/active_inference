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
// sensor's DEVICE frame (metres): "helios" (high 360) and "bpearl" (low). While robot_concept
// bridges from Ice instead, it publishes onto those SAME nodes. A DSR sensor node is BOTH the
// media-descriptor host and the name of the frame its points live in, so the device->target
// transform is just an RT-tree query keyed by the plane's node name.
//
// Every agent used to hand-roll the same discover->drain->transform dance against a single hard
// coded node. This class factors that out so all agents subscribe IDENTICALLY:
//   - preference-ordered planes (e.g. {"helios","bpearl"}),
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
    std::vector<std::uint8_t>    plane_id;                           // per-point source-plane index (0-based
                                                                     //   over emitted planes: helios=0, bpearl=1)
    // CAPTURE stamp of each plane, indexed by plane_id. `stamp_ms` below is their MAX, so on a
    // rotating robot the older plane's points are up to one full period out of date with respect to
    // it — a consumer that then applies ONE world←robot pose to the whole merged cloud misregisters
    // that plane by ω·(stamp_ms − plane_stamp_ms[k]). Consumers that transform into a WORLD frame
    // should query the pose per plane at these stamps; consumers staying in the robot frame can
    // ignore them (the device→robot mounts are static, which is why the merge is safe at all).
    std::vector<std::int64_t>    plane_stamp_ms;
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
    // `preferred_planes` are DSR sensor node names in preference order.
    // graph + inner_eigen must outlive this reader.
    LidarPlaneReader(std::shared_ptr<DSR::DSRGraph> graph,
                     DSR::InnerEigenAPI* inner_eigen,
                     std::vector<std::string> preferred_planes,
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

    // True once at least one plane has a live subscriber.
    [[nodiscard]] bool any_live() const noexcept;

private:
    struct Plane;   // defined below; declared here so the helpers can take one
    void ensure_subscribers();
    // Say, ONCE per plane, whether its subscriber actually matched a publisher — see the definition.
    void report_discovery(Plane& p);
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
    std::string                   stream_key_;

    struct Plane
    {
        std::string                      node;
        std::unique_ptr<LidarSubscriber> sub;
        std::int64_t                     last_ts = 0;
        // Latest device-frame sweep of THIS plane, kept so a plane that didn't publish a fresh frame this
        // poll still contributes its last points to the merge — otherwise out-of-phase planes (helios vs
        // bpearl) drop in and out and the merged cloud BLINKS. Re-transformed per poll at cache_ts.
        std::vector<Eigen::Vector3f>     raw_cache;
        std::int64_t                     cache_ts = 0;
        // Discovery health of this plane. A created subscriber is NOT a live stream: when the media
        // domain's SHM discovery port is wedged (a participant that died uncleanly can leave it that
        // way mid-session), init() still succeeds and poll() simply never yields a frame — the
        // consumer goes quiet and looks like a code bug. Track when the subscriber came up and
        // whether a publisher was ever matched so the reader can SAY that instead.
        std::chrono::steady_clock::time_point sub_since{};
        bool                                  ever_matched = false;
        bool                                  warned_unmatched = false;
        // A plane whose DSR node does not exist is not this robot's sensor (p3bot.json has helios but
        // no bpearl, shadow.json has both). Discovery skips it silently instead of asking the factory
        // for a descriptor it can never have — otherwise every consumer that names the pair prints a
        // "no descriptor" line once a second for the life of the run. Said ONCE, after a grace, so a
        // node that simply has not synced in yet at t=0 does not read as a missing sensor.
        bool                                  announced_absent = false;
    };
    std::vector<Plane> preferred_planes_;   // one entry per preferred node (sub filled lazily)
    std::chrono::steady_clock::time_point last_discovery_{};
    std::chrono::steady_clock::time_point created_at_{std::chrono::steady_clock::now()};
};

}  // namespace rc::media
