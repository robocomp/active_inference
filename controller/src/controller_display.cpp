#include "controller_display.h"

#include <QPushButton>

void ControllerDisplay::initialize(std::unordered_map<std::string, std::shared_ptr<DSR::DSRViewer>> &graph_viewers,
                                   rc::LidarPointBuffer *lidar_buffer,
                                   ManualTargetCallback on_manual_target,
                                   ClearTargetCallback on_clear_target,
                                   FollowToggleCallback on_follow_toggle)
{
    custom_widget_ = std::make_unique<Custom_widget>();
    if (graph_viewers.contains(""))
        graph_viewers.at("")->add_custom_widget_to_dock("controller", custom_widget_.get());
    else if (!graph_viewers.empty())
        graph_viewers.begin()->second->add_custom_widget_to_dock("controller", custom_widget_.get());

    viewer_2d_ = std::make_unique<rc::Viewer2D>(custom_widget_->frame, QRectF(-5.0, -5.0, 10.0, 10.0), true);
    viewer_2d_->add_robot(0.5f, 0.6f, 0.f, 0.f, QColor("Tomato"));
    viewer_2d_->set_lidar_buffer(lidar_buffer);
    viewer_2d_->set_lidar_visible(custom_widget_->lidar_toggle_btn != nullptr
                                 ? custom_widget_->lidar_toggle_btn->isChecked()
                                 : false);
    viewer_2d_->show();

    QObject::connect(viewer_2d_.get(), &rc::Viewer2D::new_mouse_coordinates,
                     custom_widget_.get(),
                     [on_manual_target](const QPointF &point)
                     {
                         if (on_manual_target)
                             on_manual_target(point);
                     });
    QObject::connect(viewer_2d_.get(), &rc::Viewer2D::right_click,
                     custom_widget_.get(),
                     [on_clear_target](const QPointF &)
                     {
                         if (on_clear_target)
                             on_clear_target();
                     });
    QObject::connect(custom_widget_->lidar_toggle_btn, &QPushButton::toggled,
                     custom_widget_.get(),
                     [this](bool checked)
                     {
                         if (viewer_2d_)
                             viewer_2d_->set_lidar_visible(checked);
                     });
    QObject::connect(custom_widget_->follow_toggle_btn, &QPushButton::toggled,
                     custom_widget_.get(),
                     [this, on_follow_toggle](bool checked)
                     {
                         if (custom_widget_ && custom_widget_->follow_toggle_btn)
                             custom_widget_->follow_toggle_btn->setText(checked ? "Stop" : "Start");
                         if (on_follow_toggle)
                             on_follow_toggle(checked);
                     });
    QObject::connect(custom_widget_->mppi_paths_toggle_btn, &QPushButton::toggled,
                     custom_widget_.get(),
                     [this](bool checked)
                     {
                         if (viewer_2d_)
                             viewer_2d_->set_mppi_paths_visible(checked);
                     });

    if (custom_widget_ && custom_widget_->follow_toggle_btn)
        custom_widget_->follow_toggle_btn->setText(custom_widget_->follow_toggle_btn->isChecked() ? "Stop" : "Start");
}

void ControllerDisplay::update(const std::optional<ControllerRobotPose> &robot_pose,
                               const ControllerPolygon &room_polygon,
                               const ControllerPolygon &inner_polygon,
                               const std::optional<ControllerPathPlan> &current_plan,
                               const ControllerPolygons &obstacle_polys,
                               const ControllerPolygons &obstacle_rfe_points,
                               const std::optional<Eigen::Vector2f> &current_target_room,
                               const std::vector<ControllerPolygon> &last_mppi_trajectories,
                               const ControllerPolygon &last_mppi_average_trajectory,
                               int last_best_mppi_trajectory_idx,
                               int last_display_wp_index,
                               int max_lidar_draw_points)
{
    if (!custom_widget_)
        return;

    if (viewer_2d_)
    {
        ControllerPolygon display_path;
        if (current_plan.has_value())
            display_path = current_plan->room_path;

        viewer_2d_->draw_room_polygon(room_polygon);
        viewer_2d_->draw_lidar_points_from_buffer(max_lidar_draw_points);
        viewer_2d_->draw_path({
            .path = std::move(display_path),
            .inner_poly = inner_polygon,
            .graph_nodes = current_plan.has_value() ? current_plan->graph_nodes : ControllerPolygon{},
            .obstacle_polys = obstacle_polys,
            .obstacle_rfe_points = obstacle_rfe_points,
            .candidate_trajectories = last_mppi_trajectories,
            .average_trajectory = last_mppi_average_trajectory,
            .best_trajectory_idx = last_best_mppi_trajectory_idx
        });
        if (!room_view_fitted_ && !room_polygon.empty())
        {
            viewer_2d_->fit_view();
            room_view_fitted_ = true;
        }

        if (current_target_room.has_value())
            viewer_2d_->update_target_marker(current_target_room->x(), current_target_room->y(), true);
        else
            viewer_2d_->update_target_marker(0.f, 0.f, false);

        if (robot_pose.has_value())
            viewer_2d_->update_robot(robot_pose->as_transform());
    }

    if (robot_pose.has_value())
    {
        const float theta_deg = robot_pose->theta * 180.f / static_cast<float>(M_PI);
        custom_widget_->set_pose_text(QStringLiteral("x %1 m   y %2 m   th %3 deg")
                                          .arg(robot_pose->pos.x(), 0, 'f', 2)
                                          .arg(robot_pose->pos.y(), 0, 'f', 2)
                                          .arg(theta_deg, 0, 'f', 1));
    }
    else
    {
        custom_widget_->set_pose_text(QStringLiteral("Waiting for robot pose"));
    }
}

void ControllerDisplay::set_command_text(const QString &text)
{
    if (custom_widget_)
        custom_widget_->set_cmd_vel_text(text);
}

void ControllerDisplay::clear_robot_trajectory()
{
    if (viewer_2d_)
        viewer_2d_->clear_robot_trajectory();
}