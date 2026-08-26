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

// RoomViewer — all GUI/visualization state for room_concept.
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
#include "room_config.h"       // rc::RoomConfig (shared config)
#include "viewer_2d.h"
#include "timeseries_plot.h"
#include "calibration_viewer.h"
#include "custom_widget.h"

namespace rc { class EpistemicController; class CameraVisualizer; }

namespace rc
{

class RoomViewer
{
public:
    // Constructor injection (out-of-line — unique_ptr<CameraVisualizer> incomplete in
    // callers). Builds the layout widget in its OWN top-level window (independent of the
    // DSR graph viewer, so the agent runs with Agent.graph=false) + the camera-projection
    // window, and brings up the RGB media plane. Fully constructed == ready.
    RoomViewer(std::shared_ptr<DSR::DSRGraph> graph,
                       rc::RoomConfig&     params,
                       const std::vector<Eigen::Vector2f>& room_polygon,
                       bool                     has_room_polygon,
                       rc::RoomConcept&         room_concept,
                       rc::EpistemicController& epistemic);
    ~RoomViewer();

    // Accessors so the worker (the QObject) can connect widget signals.
    [[nodiscard]] Custom_widget* widget()  const { return custom_widget_; }
    [[nodiscard]] rc::Viewer2D*  viewer()  const { return viewer_2d_; }

    // Persist the layout window geometry + splitter division to QSettings NOW. The destructor also
    // does this, but SpecificWorker::request_shutdown() hard-exits via std::_Exit (skips all C++
    // destructors), so it must be called explicitly from the shutdown path.
    void persist_window_state() const { save_window_geometry(); }

    // Best pose to draw (localized → predicted → identity).
    [[nodiscard]] Eigen::Affine2f best_available_pose(
        const std::optional<rc::RoomConcept::UpdateResult>& loc_res, bool have_loc) const;

    // Per-frame 2-D viewer refresh (lidar, robot, covariance, overlay, corners).
    void update_viewer(const std::optional<rc::RoomConcept::UpdateResult>& loc_res, bool have_loc,
                       const Eigen::Affine2f& pose_for_draw,
                       const std::vector<Eigen::Vector3f>& lidar_for_canvas,
                       const Eigen::Affine2f& loc_pose, bool use_loc);

    // Draw robot→pinned-landmark sight lines (room-frame landmark positions). `measured` is 1:1 with
    // landmarks_world; the sight line is drawn only for objects being actively measured this frame.
    void draw_landmarks(const std::vector<Eigen::Vector2f>& landmarks_world,
                        const std::vector<char>& measured,
                        const Eigen::Affine2f& robot_pose);

    // Free-energy / covariance time-series update.
    void update_ui(const std::optional<rc::RoomConcept::UpdateResult>& loc_res);

    // Set the RT publish-rate readout shown in the custom widget (call ~1 Hz from the GUI thread).
    void set_rt_rate_text(const QString& text);

    // Room-stabilization indicator (the button-sized label right of the Lidar button). GREEN once the
    // room node exists in the DSR graph — the same condition consumers gate on — RED while the
    // localizer is still accumulating consecutive stable frames (shown as n/required). GUI thread only.
    /// `searching` (a global grid search is running or just finished) outranks the other two states.
    void set_room_stable(bool stable, int stable_frames, int frames_required, bool searching = false);

    // Push one ~1 Hz sample of the RT-publish rate (corrected pose) and optimizer rate to the live
    // rate plot in the lower frame. Predicted poses are no longer published, so there is no pred rate.
    void add_rate_samples(float corr_hz, float opt_hz);

    // Button actions.
    void show_camera();
    /// Open (or raise) the self-calibration window.
    void show_calibration();
    void toggle_lidar_points(bool checked);

    // Mouse-driven pose reset (Shift+Left = translate, Ctrl+Left = rotate); both
    // push a thread-safe CmdSetPose into the localizer.
    void on_robot_moved(QPointF scene_pos);
    void on_robot_rotated(QPointF scene_pos);

private:
    void update_epistemic_overlay();

    // Persist the independent layout window's position/size across runs (QSettings).
    void restore_window_geometry();
    void save_window_geometry() const;

    rc::RoomConfig*     params_       = nullptr;   // shared config
    bool                     has_room_polygon_ = false;
    rc::RoomConcept*         room_concept_ = nullptr;
    rc::EpistemicController* epistemic_    = nullptr;

    QPointer<Custom_widget>      custom_widget_;
    QPointer<QLabel>             rt_rate_label_;   // RT publish-rate readout in the controls row
    QPointer<rc::Viewer2D>       viewer_2d_;
    QPointer<rc::TimeSeriesPlot> ts_plot_fe_;
    QPointer<rc::TimeSeriesPlot> ts_plot_rates_;   // RT-publish + optimizer rates (Hz) over time
    // Integrated odometry as the PREDICTOR consumes it — see get_predictor_delta().
    QPointer<rc::TimeSeriesPlot> ts_plot_conf_;    // localization confidence (raw, 0..1) over time
    // Retained so the GT series can read robot_gt_* off the robot node each update.
    std::shared_ptr<DSR::DSRGraph> graph_;
    // Ground truth vs estimate, x / y / theta on one axis. SIMULATION ONLY: robot_concept writes
    // robot_gt_* on the robot node just while the producer reports simulated, so on real hardware
    // the attributes are absent, the series stay empty and the plot is simply flat.
    QPointer<rc::TimeSeriesPlot> ts_plot_gt_;
    // RGB projection agreement (ImageEdge). Empty and flat unless ImageEdge.enable && shadow.
    QPointer<rc::TimeSeriesPlot> ts_plot_imgedge_;
    std::int64_t last_imgedge_ts_ms_ = 0;   // only append when a NEW cycle produced a number
    QPointer<rc::CalibrationViewer> calib_viewer_;
    // Last state pushed to lbl_room_stable: -1 = never painted, 0 = red, 1 = green, 2 = amber (searching).
    // Tri-state so the very first update always applies a stylesheet, whichever state it reports.
    int room_stable_shown_ = -1;
    std::unique_ptr<rc::CameraVisualizer> camera_viz_;
    bool camera_media_plane_initialized_ = false;
};

}  // namespace rc
