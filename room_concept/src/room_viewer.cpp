/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_viewer.h.
 */

#include "room_viewer.h"

#include <algorithm>
#include <cmath>

#include <QColor>
#include <QVBoxLayout>
#include <QtCore/qdebug.h>

#include "epistemic_controller.h"
#include "camera_visualizer.h"
#include "../../common/media_transport/media_transport.h"   // descriptor_from_graph

namespace rc
{

RoomViewer::~RoomViewer() = default;

RoomViewer::RoomViewer(DSR::DSRViewer* default_viewer,
                                       std::shared_ptr<DSR::DSRGraph> graph,
                                       rc::RoomConfig& params,
                                       const std::vector<Eigen::Vector2f>& room_polygon,
                                       bool has_room_polygon,
                                       rc::RoomConcept& room_concept,
                                       rc::EpistemicController& epistemic)
    : params_(&params), has_room_polygon_(has_room_polygon),
      room_concept_(&room_concept), epistemic_(&epistemic), graph_(graph)
{
    custom_widget_ = new Custom_widget();
    default_viewer->add_custom_widget_to_dock("layout", custom_widget_);
    viewer_2d_ = new rc::Viewer2D(custom_widget_->frame, params_->GRID_MAX_DIM, true);
    viewer_2d_->show();
    viewer_2d_->add_robot(params_->ROBOT_WIDTH, params_->ROBOT_LENGTH, 0.f, 0.f, QColor("blue"));

    // Free-Energy time series in the lower frame of the custom widget.
    if (custom_widget_->frame_series->layout() == nullptr)
    {
        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(2, 2, 2, 2);
        series_layout->setSpacing(2);
        custom_widget_->frame_series->setLayout(series_layout);
    }
    ts_plot_fe_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
    ts_plot_fe_->set_visible_window(60.f);
    ts_plot_fe_->add_series("free_energy", QColor(255, 170, 0), 1.8f, 0);
    ts_plot_fe_->add_series("cov_det_scaled", QColor(0, 190, 255), 1.6f, 0);
    custom_widget_->frame_series->layout()->addWidget(ts_plot_fe_);

    if (has_room_polygon_ && room_polygon.size() >= 3)
        viewer_2d_->draw_room_polygon(room_polygon, false);

    // Camera-projection window: overlays the room layout on the live RGB image.
    camera_viz_ = std::make_unique<rc::CameraVisualizer>(graph, room_polygon, nullptr);

    // Bring up the RGB media plane. The subscriber is created lazily by the camera
    // visualizer once the "zed" node + media descriptor exist, reading the DDS
    // domain/topic straight from that JSON descriptor (no config). This just starts
    // the always-on drain/discovery timer.
    camera_viz_->start_media_plane();
    camera_media_plane_initialized_ = true;
    qInfo() << "[room][camera] RGB media plane discovery started (waits for 'zed' descriptor)";
}

Eigen::Affine2f RoomViewer::best_available_pose(
    const std::optional<rc::RoomConcept::UpdateResult>& loc_res, bool have_loc) const
{
    if (have_loc)
        return loc_res->robot_pose;
    if (room_concept_ && room_concept_->is_initialized())
    {
        const auto s = room_concept_->get_current_state();
        Eigen::Affine2f p = Eigen::Affine2f::Identity();
        p.translation() = Eigen::Vector2f(s[2], s[3]);
        p.linear() = Eigen::Rotation2Df(s[4]).toRotationMatrix();
        return p;
    }
    return Eigen::Affine2f::Identity();
}

void RoomViewer::update_viewer(const std::optional<rc::RoomConcept::UpdateResult>& loc_res, bool have_loc,
                                       const Eigen::Affine2f& pose_for_draw,
                                       const std::vector<Eigen::Vector3f>& lidar_for_canvas,
                                       const Eigen::Affine2f& loc_pose, bool use_loc)
{
    if (!viewer_2d_)
        return;

    viewer_2d_->update_frame({
        .lidar_points     = lidar_for_canvas,
        .display_pose     = pose_for_draw,
        .covariance       = have_loc ? loc_res->covariance : Eigen::Matrix3f::Identity(),
        .max_lidar_points = params_->MAX_LIDAR_DRAW_POINTS,
        .have_loc         = have_loc,
        .is_initialized   = room_concept_ && room_concept_->is_initialized(),
        .has_room_polygon = has_room_polygon_,
        .room_width       = have_loc ? loc_res->state[0] : 0.f,
        .room_length      = have_loc ? loc_res->state[1] : 0.f,
        .loc_pose         = loc_pose,
        .use_loc_pose     = use_loc,
    });

    update_epistemic_overlay();

    if (have_loc && !loc_res->corner_matches.empty())
        viewer_2d_->draw_corners(loc_res->corner_matches, pose_for_draw);
    else
        viewer_2d_->draw_corners({}, pose_for_draw);

    // Draw object/obstacle BBs from the graph (self-throttled to ~1 Hz). Main-thread poll.
    viewer_2d_->refresh_semantic_bboxes(graph_);
}

void RoomViewer::update_epistemic_overlay()
{
    if (!viewer_2d_ || !epistemic_)
        return;

    // Epistemic score grid heatmap overlay (drawn behind lidar/robot by z-order).
    const auto& planner = epistemic_->epistemic_planner();
    const auto& cell_scores = planner.cell_scores();
    std::vector<std::pair<Eigen::Vector2f, float>> score_cells;
    score_cells.reserve(cell_scores.size());
    for (const auto& cell : cell_scores)
        score_cells.emplace_back(cell.center, cell.score);
    viewer_2d_->draw_score_grid(score_cells, planner.cell_size());

    // IoR inhibition overlay: warm red fades out as visited cells recover
    const auto& ior = planner.ior_cells();
    std::vector<std::pair<Eigen::Vector2f, float>> ior_cells;
    ior_cells.reserve(ior.size());
    for (const auto& cell : ior)
        ior_cells.emplace_back(cell.center, cell.freshness);
    viewer_2d_->draw_ior_grid(ior_cells, planner.cell_size());

    const auto& current_target = planner.current_target();
    if (current_target.has_value() && !current_target->rotate_in_place)
    {
        viewer_2d_->draw_selected_grid_cell(current_target->position, planner.cell_size());
        viewer_2d_->update_target_marker(current_target->position.x(),
                                         current_target->position.y(),
                                         true);
    }
    else
    {
        viewer_2d_->draw_selected_grid_cell(std::nullopt, planner.cell_size());
        viewer_2d_->update_target_marker(0.f, 0.f, false);
    }
}

void RoomViewer::update_ui(const std::optional<rc::RoomConcept::UpdateResult>& loc_res)
{
    if (!loc_res.has_value() || !ts_plot_fe_)
        return;
    ts_plot_fe_->add_point("free_energy", loc_res->final_loss);

    const float det_cov = std::max(1e-12f, std::abs(loc_res->covariance.determinant()));
    float det_scaled = -std::log10(det_cov) / 10.f;  // map ~[1..1e-10] to [0..1]
    if (det_scaled < 0.f) det_scaled = 0.f;
    if (det_scaled > 1.f) det_scaled = 1.f;
    ts_plot_fe_->add_point("cov_det_scaled", det_scaled);
}

void RoomViewer::show_camera()
{
    if (camera_viz_)
    {
        camera_viz_->update_frame();
        camera_viz_->show();
        camera_viz_->raise();
        camera_viz_->activateWindow();
    }
}

void RoomViewer::toggle_lidar_points(bool checked)
{
    if (viewer_2d_)
        viewer_2d_->set_lidar_points_visible(checked);
}

void RoomViewer::on_robot_moved(QPointF scene_pos)
{
    // Shift+Left: move robot to clicked position, keep current heading. push_command
    // (thread-safe queue) — never set_robot_pose() directly from the GUI thread while
    // the localization thread may be mid-backward().
    if (!room_concept_)
        return;
    const auto state = room_concept_->get_current_state();
    room_concept_->push_command(rc::RoomConcept::CmdSetPose{
        static_cast<float>(scene_pos.x()), static_cast<float>(scene_pos.y()), state[4]});
}

void RoomViewer::on_robot_rotated(QPointF scene_pos)
{
    // Ctrl+Left: rotate robot to face the clicked point, keep current position.
    if (!room_concept_)
        return;
    const auto state = room_concept_->get_current_state();
    const float rx = state[2];
    const float ry = state[3];
    const float theta = std::atan2(static_cast<float>(scene_pos.y()) - ry,
                                   static_cast<float>(scene_pos.x()) - rx);
    room_concept_->push_command(rc::RoomConcept::CmdSetPose{rx, ry, theta});
}

}  // namespace rc
