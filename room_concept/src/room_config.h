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
#include <vector>

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
    std::string LIDAR_NAME            = "lidar3D";   // legacy fused plane (already robot-frame, bridged)
    // Per-device high LiDAR plane (lidar3d_dds): points arrive in the DEVICE frame (metres) and
    // must be transformed device->robot via the DSR RT tree. Preferred over LIDAR_NAME when live.
    std::string LIDAR_HELIOS_NAME     = "helios";
    // Destination frame for the device->robot transform (the mount RT edge parent, e.g. body->helios).
    std::string LIDAR_ROBOT_FRAME     = "";   // empty ⇒ auto-derived from the type-"robot" node at init
                                              // (so the SDF optimises in the SAME frame the robot↔room RT
                                              // is published onto). Set explicitly only to override.
    float MAX_LIDAR_HIGH_RANGE        = 100.f;  // m
    int   LIDAR_LOW_DECIMATION_FACTOR = 1;
    float LIDAR_HIGH_MIN_HEIGHT       = 1.5f;   // m
    float LIDAR_HIGH_MAX_HEIGHT       = 2.0f;   // m — upper bound of the high band (excludes the ceiling)
    float LIDAR_HIGH_FLOOR_HEIGHT     = 0.15f;  // m
    // Startup geometry self-check: from the first few sweeps, detect the floor plane (warn if it
    // disagrees with the robot mount geometry -> a mis-set LiDAR mount height) and the ceiling plane
    // (cap the high band at ceiling - margin so only upper-wall points feed the localizer).
    bool  LIDAR_STARTUP_GEOMETRY_CHECK = true;
    int   LIDAR_STARTUP_CHECK_SWEEPS   = 15;    // sweeps to accumulate before running the check
    float LIDAR_FLOOR_TOLERANCE        = 0.06f; // m — warn if the measured floor is off by more
    float LIDAR_CEILING_MARGIN         = 0.15f; // m — keep wall points this far below the ceiling

    // View
    QRectF GRID_MAX_DIM{-5, -5, 10, 10};
    int    MAX_LIDAR_DRAW_POINTS = 500;

    // Localizer
    bool        PREDICTION_EARLY_EXIT = true;
    std::string OptimizerType         = "LBFGS";
    std::string ROOM_LAYOUT_SVG       = "beta_layout.svg";  // config: RoomConcept.RoomLayoutSvg
    // Put the room-frame origin on the layout's geometric (bounding-box) centre instead of wherever
    // the SVG author happened to place (0,0). Only affects plans traced from a corner
    // (apartamento_layout); layouts already drawn about their middle barely move.
    // NOTE when flipping this: the room frame shifts, so etc/last_robot_pose.txt (seed pose, in room
    // coords) and any object RT edges persisted under the DSR `room` node are stale by the offset.
    bool        RECENTER_ROOM_POLYGON = true;   // config: RoomConcept.RecenterRoomPolygon
    float       ODOMETRY_NOISE_FACTOR = 0.0f;

    // Where the IMU comes from: "media" (the DDS media-plane topic the producer advertises) or "dsr"
    // (the legacy imu_* node attributes). Default "dsr" so the switch is one word to revert and both
    // paths can be compared on one run — imu_cover and wg_ratio in the debug log show a mismatch
    // immediately. On hardware only "media" is meaningful: the real IMU component publishes to the
    // topic directly and never writes those attributes, so the consumer code is then identical in
    // both worlds and the sim/real divergence disappears at exactly the point being calibrated.
    std::string IMU_SOURCE = "dsr";   // "dsr" | "media"


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
    // A diagonal velocity covariance (variances below) goes on rt_covariance_velocity.
    float ROBOT_VEL_COV_ADV            = 0.0025f; // (0.05 m/s)²
    float ROBOT_VEL_COV_SIDE           = 0.0025f; // (0.05 m/s)²
    float ROBOT_VEL_COV_ROT            = 0.01f;   // (0.1 rad/s)²

    bool  PUBLISH_AFFORDANCE           = true;    // EpistemicController.PublishAffordance — publish the room
                                                  // exploration affordance. false ⇒ room never offers an
                                                  // affordance (so it can't out-compete object affordances in
                                                  // the controller's EFE selection — e.g. the 360-glance test).
    // ---- Kinematic clamp on the published pose ----
    // Bounds the pose delta between two published RT blocks to what the robot can physically do in the
    // elapsed time. Not a tuning threshold: it is the support of the motion model, so a delta outside
    // it cannot be a report about where the robot went. The un-applied residual is added to the
    // published covariance so the lag is visible rather than hidden.
    // ★ The limits must be the CONSUMER's, not this agent's own beliefs — room_concept's MaxAdvSpeed /
    // MaxRotSpeed (0.9 / 0.75) disagree with what the controller actually commands (0.7 / 0.8) and
    // neither is a superset. These default to the controller's and must be kept tracking it.
    bool  POSE_CLAMP_ENABLED           = true;    // RoomConcept.PoseClampEnabled
    float POSE_CLAMP_V_MAX             = 0.7f;    // m/s   — controller MaxAdvSpeed
    float POSE_CLAMP_W_MAX             = 0.8f;    // rad/s — controller MaxRotSpeed
    // Gap beyond which the clamp is skipped entirely (stream restart, agent stall, relocalization after
    // a long silence). Clamping across a long gap would slew the pose in at v_max for many frames.
    float POSE_CLAMP_MAX_DT_S          = 0.5f;    // s

    // ---- Execution-stall watchdog (RoomSceneGraph::break_execution_stall) ----
    // While the controller holds the afford_room execution claim the planner is idle by design, so
    // a target the controller can never reach parks the whole run. If the robot's closest approach
    // to the claimed target fails to improve by EXEC_STALL_PROGRESS_M for EXEC_STALL_TIMEOUT_S, the
    // claim is released and the target is stamped visited so a different one is selected. This is a
    // LIVENESS guard on the affordance handshake, not a modelling threshold — see the header note.
    // Set EXEC_STALL_TIMEOUT_S = 0 to disable.
    float EXEC_STALL_TIMEOUT_S         = 25.0f;   // EpistemicController.ExecStallTimeout (s)
    float EXEC_STALL_PROGRESS_M        = 0.10f;   // EpistemicController.ExecStallProgress (m)

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

    // Waiting/Operating gate on the LiDAR media plane. Without LiDAR the localizer can never
    // stabilize, so the agent stays in (or falls back to) Waiting instead of sitting in Operating
    // doing nothing. 0 on either knob disables that half of the gate.
    int LIDAR_STALL_TIMEOUT_MS = 3000;   // Operating: no sweep for this long ⇒ back to Waiting
    int LIDAR_WAIT_LOG_PERIOD_MS = 2000; // Waiting: how often to reprint why we are still waiting

    // Camera-overlay object projection: DSR node TYPES whose oriented boxes are projected on
    // the live RGB image (alongside the always-drawn walls). Concept agents now publish all
    // furniture as generic "object" nodes (class in object_subtype), so "object" alone covers
    // tables/chairs/bottles/…. Config Overlay.ObjectTypes overrides this comma-separated list.
    std::vector<std::string> OVERLAY_OBJECT_TYPES = {"object"};

    // ── Object anchors (validated modelled objects as SE(2) pose landmarks for localization) ──
    // OFF by default. Precision-weighted by each object's own belief covariance (no threshold).
    bool  OBJECT_ANCHOR_ENABLE      = false;  // ObjectAnchor.enable
    bool  OBJECT_ANCHOR_OPTIMIZE_LANDMARK = false;  // ObjectAnchor.optimizeLandmark — p_o as a private
                                              // VARIABLE (birth prior = producer's belief) instead of a pin
    // ObjectAnchor.subtypes — which object CLASSES become landmarks. Default {"table"} keeps the
    // historical behaviour; the classes are deliberately selectable one at a time (see the header of
    // ObjectAnchorSource::Config for why table and refrigerator are not interchangeable).
    std::vector<std::string> OBJECT_ANCHOR_SUBTYPES = {"table"};
    float OBJECT_ANCHOR_WEIGHT      = 1.0f;   // ObjectAnchor.weight  (keep < walls)
    float OBJECT_ANCHOR_HUBER       = 3.0f;   // ObjectAnchor.huber   (whitened σ units)
    int   OBJECT_ANCHOR_MAX_SLOTS   = 3;      // ObjectAnchor.maxSlots
    float OBJECT_ANCHOR_MEAS_SIG_XY = 0.05f;  // ObjectAnchor.measSigmaXY  (m)  fallback R_o
    float OBJECT_ANCHOR_MEAS_SIG_YAW= 0.15f;  // ObjectAnchor.measSigmaYaw (rad) fallback R_o
    float OBJECT_ANCHOR_EARLY_EXIT_SIGMA = 2.0f;  // ObjectAnchor.earlyExitSigma — whitened anchor-residual
                                                  // σ-cutoff that forces the optimizer to run (drift catch)
    float OBJECT_ANCHOR_VALIDATE_SIGMA  = 0.10f;  // ObjectAnchor.validateSigma — map-pose σ (m) below which
                                                  // room PINS the table's world pose (breaks the circularity)
    bool  OBJECT_ANCHOR_FRESHNESS_ENABLE    = true;  // ObjectAnchor.freshnessEnable — grow R_o with obs age
    float OBJECT_ANCHOR_FRESHNESS_AGE_SCALE = 3.0f;  // ObjectAnchor.freshnessAgeScale — frames→σ doubles
};

// Load the agent params + RoomConcept params + EpistemicController/planner params,
// and seed the planner's robot footprint. Call once from initialize().
void load_room_config(const ConfigLoader& cl, RoomConfig& p,
                      RoomConcept& room_concept, EpistemicController& epistemic);

}  // namespace rc
