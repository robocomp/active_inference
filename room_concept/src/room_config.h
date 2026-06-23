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

    float room_height = 2.4f;  // m, room DSR node attribute

    // Debug bootstrap table hanging from the room node
    bool  BOOTSTRAP_TABLE_ENABLED = true;
    float BOOTSTRAP_TABLE_X       = 0.f;
    float BOOTSTRAP_TABLE_Y       = 0.f;
    float BOOTSTRAP_TABLE_YAW     = 0.f;
    float BOOTSTRAP_TABLE_WIDTH   = 1.5f;
    float BOOTSTRAP_TABLE_DEPTH   = 1.4f;
    float BOOTSTRAP_TABLE_HEIGHT  = 0.74f;

    // Media plane (zero-copy DDS) — RGB consumer for the camera-projection window
    // and the LiDAR stream consumed by LidarIngestor.
    int         MEDIA_DOMAIN_ID   = 0;
    std::string MEDIA_RGB_TOPIC   = "rc/zed/rgb";
    std::string MEDIA_LIDAR_TOPIC = "rc/lidar3d/points";
    std::string MEDIA_IMU_TOPIC   = "rc/imu/data";
    bool        LIDAR_USE_MEDIA   = true;   // false ⇒ DSR graph laser_* only
};

// Load the agent params + RoomConcept params + EpistemicController/planner params,
// and seed the planner's robot footprint. Call once from initialize().
void load_room_config(const ConfigLoader& cl, RoomConfig& p,
                      RoomConcept& room_concept, EpistemicController& epistemic);

}  // namespace rc
