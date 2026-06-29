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

// RoomConfig — the room_concept agent's worker-level configuration, plus a
// single entry point that loads EVERY config block (the agent's own params, the
// RoomConcept localizer params and the EpistemicController/planner params) from
// the ConfigLoader. Keeps the ~150 lines of load_* boilerplate out of
// SpecificWorker::initialize().

#include <string>

#include <QRectF>

class ConfigLoader;

namespace rc
{
class RoomConcept;
class EpistemicController;

struct RoomConfig
{
    float ROBOT_WIDTH  = 0.460f;   // m
    float ROBOT_LENGTH = 0.480f;   // m
    float ROBOT_HEIGHT = 1.6f;     // m, obstacle cloud ceiling

    // Lidar
    std::string LIDAR_NAME            = "lidar3D";
    float MAX_LIDAR_HIGH_RANGE        = 100.f;  // m
    int   LIDAR_LOW_DECIMATION_FACTOR = 1;
    float LIDAR_HIGH_MIN_HEIGHT       = 1.5f;   // m
    float LIDAR_HIGH_MAX_HEIGHT       = 2.0f;   // m
    float LIDAR_HIGH_FLOOR_HEIGHT     = 0.15f;  // m

    // View
    QRectF GRID_MAX_DIM{-5, -5, 10, 10};
    int    MAX_LIDAR_DRAW_POINTS = 500;

    // Localizer
    bool        PREDICTION_EARLY_EXIT = true;
    std::string OptimizerType         = "LBFGS";
    std::string ROOM_LAYOUT_SVG       = "beta_layout.svg";  // config: RoomConcept.RoomLayoutSvg
    float       ODOMETRY_NOISE_FACTOR = 0.0f;

    // DSR stabilization: this many consecutive "stable" frames before creating the
    // room node and re-parenting the robot under it.
    int   STABLE_FRAMES_REQUIRED = 30;
    float STABLE_SDF_MSE_MAX     = 0.06f;
    float STABLE_COV_TT_MAX      = 0.001f;

    // Static-room mode: ADOPT a pre-seeded room/table from the bootstrap graph
    // instead of deleting+recreating it, and do NOT write room pose / robot->room.
    bool  PRESERVE_BOOTSTRAP_ROOM = false;

    // ODOMETRY-DRIVEN COMPLEMENTARY-FILTER PUBLISH (v2). The robot↔room RT edge is published from the
    // ODOMETRY callback (event-driven, at the odometry rate, e.g. 20-30 Hz) — decoupled from the bursty
    // ~15 Hz lidar/compute path. Each odometry sample integrates a SMALL dt step (no catch-up jump) and
    // publishes a monotonic now-stamped block with a GROWING covariance. Lidar corrections are BLENDED
    // into the filter (complementary, gain below) and NOT written to RT directly (no past-stamped block
    // interleaving). This is the smooth replacement for the v1 QTimer predict-publish (which injected
    // 17 m/s spikes via per-correction catch-up + hard snap + block interleaving — see git history).
    // Robot BODY-FRAME velocity (adv=fwd, side=lat, rot=yaw-rate) is published on the robot↔room RT
    // edge (rt_translation_velocity / rt_rotation_euler_xyz_velocity) so the controller reads velocity
    // DIRECTLY instead of differentiating the pose (which would turn lidar-correction jumps into spikes).
    // A diagonal velocity covariance (variances below) goes on rt_se2_covariance_velocity.
    float ROBOT_VEL_COV_ADV            = 0.0025f; // (0.05 m/s)²
    float ROBOT_VEL_COV_SIDE           = 0.0025f; // (0.05 m/s)²
    float ROBOT_VEL_COV_ROT            = 0.01f;   // (0.1 rad/s)²

    bool  PREDICT_PUBLISH_ENABLED      = true;    // PredictPublish.enabled (drives RT from odometry)
    float PREDICT_PUBLISH_MAX_COAST_S  = 1.0f;    // stop publishing if no lidar correction for this long
    float PREDICT_PROCESS_NOISE_XY     = 0.04f;   // (m/√s)² → variance growth m²/s on x,y while coasting
    float PREDICT_PROCESS_NOISE_THETA  = 0.05f;   // (rad/√s)² → variance growth rad²/s on theta
    float PREDICT_MAX_DT_S             = 0.1f;    // clamp per-sample integration dt (a missed/late odom
                                                  // sample must not dead-reckon a big jump); also caps
                                                  // the lidar-lag used to extrapolate the correction
    // Correction is NOT applied as a position jump (that injects velocity spikes the MPPI differentiates
    // into garbage — measured post-blend 3.46 m/s vs pure-odom 0.84). Instead each lidar correction sets
    // a residual that is BLED IN smoothly over the 30 Hz odometry ticks, slew-limited, so it's a small
    // continuous velocity contribution. Per tick: step = min(blend_gain·residual, max_blend_step).
    float PREDICT_BLEND_GAIN           = 0.2f;    // fraction of the residual applied per odometry tick
    float PREDICT_MAX_BLEND_STEP_M     = 0.01f;   // max position bleed per tick (→ ≤0.3 m/s @30Hz)
    float PREDICT_MAX_BLEND_STEP_RAD   = 0.03f;   // max heading bleed per tick
    // Outlier-aware: a residual bigger than these is a relocalization/jump (the gentle slew would take
    // ~30 s to catch up → the pose strands metres from truth). SNAP to it immediately instead — bounds
    // divergence to this threshold; the rare spike on a real relocalization is legitimate. Small
    // residuals (normal drift) are still slewed smoothly.
    float PREDICT_SNAP_THRESH_M        = 0.30f;
    float PREDICT_SNAP_THRESH_RAD      = 0.40f;

    float room_height = 2.4f;  // m, room DSR node attribute

    // Debug bootstrap table hanging from the room node
    bool  BOOTSTRAP_TABLE_ENABLED = true;
    float BOOTSTRAP_TABLE_X       = 0.f;
    float BOOTSTRAP_TABLE_Y       = 0.f;
    float BOOTSTRAP_TABLE_YAW     = 0.f;
    float BOOTSTRAP_TABLE_WIDTH   = 1.5f;
    float BOOTSTRAP_TABLE_DEPTH   = 1.4f;
    float BOOTSTRAP_TABLE_HEIGHT  = 0.74f;

    // Media plane (zero-copy DDS) — RGB (camera window) + LiDAR (LidarIngestor).
    // DDS domain + topics are NOT configured: they are read from the media descriptor
    // JSON the producer authors on the "zed"/"lidar3D" nodes, so the consumer always
    // uses the producer's dedicated domain. Subscribers are created lazily once those
    // nodes + descriptors exist.
    bool        LIDAR_USE_MEDIA   = true;   // false ⇒ DSR graph laser_* only
};

// Load the agent params + RoomConcept params + EpistemicController/planner params,
// and seed the planner's robot footprint. Call once from initialize().
void load_room_config(const ConfigLoader& cl, RoomConfig& p,
                      RoomConcept& room_concept, EpistemicController& epistemic);

}  // namespace rc
