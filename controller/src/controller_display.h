#pragma once

#include <genericworker.h>

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <QString>

#include "controller_runtime_types.h"
#include "custom_widget.h"
#include "viewer_2d.h"

class ControllerDisplay
{
public:
    using ManualTargetCallback = std::function<void(const QPointF &)>;
    using ClearTargetCallback = std::function<void()>;
    using FollowToggleCallback = std::function<void(bool)>;

    // Creates the planner GUI as its OWN top-level window (not docked into the DSR graph viewer),
    // so the agent runs with Agent.graph=false. Mirrors room_concept's RoomViewer.
    void initialize(rc::LidarPointBuffer *lidar_buffer,
                    ManualTargetCallback on_manual_target,
                    ClearTargetCallback on_clear_target,
                    FollowToggleCallback on_follow_toggle);

    // Persist the standalone window's geometry (call on shutdown, GUI thread).
    void save_window_geometry() const;

    Custom_widget *widget() const { return custom_widget_.get(); }

    // Staging API — safe to call from any thread. These only copy data into an
    // internal snapshot; no Qt objects are touched here.
    void update(const std::optional<ControllerRobotPose> &robot_pose,
                const ControllerPolygon &room_polygon,
                const ControllerPolygon &inner_polygon,
                const std::optional<ControllerPathPlan> &current_plan,
                const ControllerObstacleVisuals &obstacle_polys,
                const ControllerPolygons &obstacle_rfe_points,
                const std::optional<Eigen::Vector2f> &current_target_room,
                const std::vector<ControllerPolygon> &last_mppi_trajectories,
                const ControllerPolygon &last_mppi_average_trajectory,
                int last_best_mppi_trajectory_idx,
                int last_display_wp_index,
                int max_lidar_draw_points,
                const std::optional<Eigen::Affine2f> &lidar_correction = std::nullopt);

    void set_command_text(const QString &text);
    void set_selected_affordance_text(const QString &text);
    // One sample per evaluated affordance for the EFE panel below the 2D view. Plots TWO lines per
    // affordance: the selection score (gain − λ·dist, solid) and the raw gain (ΔH, lighter) — so the
    // vertical gap between them is λ·dist. Thread-safe (the plot buffers under its own mutex).
    struct AffordanceEfeSample { std::string name; float gain = 0.f; float score = 0.f; };
    void update_affordance_efe(const std::vector<AffordanceEfeSample> &samples);
    void clear_robot_trajectory();

    // Presentation — MUST be called on the GUI thread only. Reads the latest
    // staged snapshot and performs all Qt scene drawing.
    void present();

private:
    struct DisplaySnapshot
    {
        std::optional<ControllerRobotPose> robot_pose;
        ControllerPolygon room_polygon;
        ControllerPolygon inner_polygon;
        std::optional<ControllerPathPlan> current_plan;
        ControllerObstacleVisuals obstacle_polys;
        ControllerPolygons obstacle_rfe_points;
        std::optional<Eigen::Vector2f> current_target_room;
        std::vector<ControllerPolygon> last_mppi_trajectories;
        ControllerPolygon last_mppi_average_trajectory;
        int last_best_mppi_trajectory_idx = -1;
        int last_display_wp_index = 0;
        int max_lidar_draw_points = 0;
        std::optional<Eigen::Affine2f> lidar_correction;   // room(now)←room(scan) overlay dead-reckoning
        bool valid = false;

        QString command_text;
        bool command_text_pending = false;
        QString selected_affordance_text;
        bool selected_affordance_text_pending = false;
        bool clear_trajectory_pending = false;
    };

    void restore_window_geometry();

    std::unique_ptr<Custom_widget> custom_widget_;
    std::unique_ptr<rc::Viewer2D> viewer_2d_;
    bool room_view_fitted_ = false;
    std::unordered_set<std::string> efe_series_known_;   // plot series already registered
    std::size_t efe_color_next_ = 0;                     // next palette colour for a new affordance

    mutable std::mutex snapshot_mutex_;
    DisplaySnapshot snapshot_;
};