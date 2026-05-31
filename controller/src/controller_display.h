#pragma once

#include <genericworker.h>

#include <functional>
#include <memory>

#include "controller_runtime_types.h"
#include "custom_widget.h"
#include "viewer_2d.h"

class ControllerDisplay
{
public:
    using ManualTargetCallback = std::function<void(const QPointF &)>;
    using ClearTargetCallback = std::function<void()>;
    using FollowToggleCallback = std::function<void(bool)>;

    void initialize(std::unordered_map<std::string, std::shared_ptr<DSR::DSRViewer>> &graph_viewers,
                    rc::LidarPointBuffer *lidar_buffer,
                    ManualTargetCallback on_manual_target,
                    ClearTargetCallback on_clear_target,
                    FollowToggleCallback on_follow_toggle);

    Custom_widget *widget() const { return custom_widget_.get(); }

    void update(const std::optional<ControllerRobotPose> &robot_pose,
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
                int max_lidar_draw_points);

    void set_command_text(const QString &text);
    void clear_robot_trajectory();

private:
    std::unique_ptr<Custom_widget> custom_widget_;
    std::unique_ptr<rc::Viewer2D> viewer_2d_;
    bool room_view_fitted_ = false;
};