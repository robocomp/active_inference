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

// RoomViewController — all GUI/visualization state for room_concept.
//
// Owns the docked 2-D layout viewer, the free-energy time-series plot, and the
// camera-projection window (CameraVisualizer, which overlays the room layout on
// the live RGB image fed by the zero-copy media plane). SpecificWorker keeps the
// Qt wiring (it is the QObject) but delegates every per-frame draw + the camera
// media-plane bring-up here. Plain class (no QObject/MOC); the worker connects
// widget signals to lambdas that call these methods.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <QPointF>
#include <QPointer>
#include <QRectF>
#include <Eigen/Geometry>

#include <genericworker.h>            // DSR::DSRGraph / DSR::DSRViewer

#include "room_concept.h"            // rc::RoomConcept (+ UpdateResult)
#include "viewer_2d.h"
#include "timeseries_plot.h"
#include "custom_widget.h"

namespace rc { class EpistemicController; class CameraVisualizer; }

namespace rc
{

class RoomViewController
{
public:
    struct Config
    {
        QRectF      grid_max_dim{-5, -5, 10, 10};
        float       robot_width  = 0.460f;
        float       robot_length = 0.480f;
        int         max_lidar_draw_points = 500;
        bool        has_room_polygon = false;
        // Media plane fallback (used if the graph descriptor on "zed" is absent).
        std::uint32_t media_domain_id = 7;
        std::string   media_rgb_topic = "rc/zed/rgb";
    };

    RoomViewController();           // out-of-line (unique_ptr<CameraVisualizer> incomplete in callers)
    ~RoomViewController();

    // Build the docked widgets + camera window and bring up the RGB media plane.
    void setup(DSR::DSRViewer* default_viewer,
               std::shared_ptr<DSR::DSRGraph> graph,
               const Config& cfg,
               const std::vector<Eigen::Vector2f>& room_polygon,
               rc::RoomConcept* room_concept,
               rc::EpistemicController* epistemic);

    // Accessors so the worker (the QObject) can connect widget signals.
    [[nodiscard]] Custom_widget* widget()  const { return custom_widget_; }
    [[nodiscard]] rc::Viewer2D*  viewer()  const { return viewer_2d_; }

    // Best pose to draw (localized → predicted → identity).
    [[nodiscard]] Eigen::Affine2f best_available_pose(
        const std::optional<rc::RoomConcept::UpdateResult>& loc_res, bool have_loc) const;

    // Per-frame 2-D viewer refresh (lidar, robot, covariance, overlay, corners).
    void update_viewer(const std::optional<rc::RoomConcept::UpdateResult>& loc_res, bool have_loc,
                       const Eigen::Affine2f& pose_for_draw,
                       const std::vector<Eigen::Vector3f>& lidar_for_canvas,
                       const Eigen::Affine2f& loc_pose, bool use_loc);

    // Free-energy / covariance time-series update.
    void update_ui(const std::optional<rc::RoomConcept::UpdateResult>& loc_res);

    // Button actions.
    void show_camera();
    void toggle_lidar_points(bool checked);

    // Mouse-driven pose reset (Shift+Left = translate, Ctrl+Left = rotate); both
    // push a thread-safe CmdSetPose into the localizer.
    void on_robot_moved(QPointF scene_pos);
    void on_robot_rotated(QPointF scene_pos);

private:
    void update_epistemic_overlay();

    Config cfg_;
    rc::RoomConcept*         room_concept_ = nullptr;
    rc::EpistemicController* epistemic_    = nullptr;

    QPointer<Custom_widget>      custom_widget_;
    QPointer<rc::Viewer2D>       viewer_2d_;
    QPointer<rc::TimeSeriesPlot> ts_plot_fe_;
    std::unique_ptr<rc::CameraVisualizer> camera_viz_;
    bool camera_media_plane_initialized_ = false;
};

}  // namespace rc
