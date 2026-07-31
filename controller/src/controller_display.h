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
#include "controller_mission_panel.h"
#include "custom_widget.h"
#include "viewer_2d.h"

class ControllerDisplay
{
public:
    // Everything the GUI can ask the worker to do. Bundled rather than passed positionally: the list grew
    // past the point where a caller could get the order right by reading the call site.
    struct Callbacks
    {
        std::function<void(const QPointF &)> on_manual_target;   // left click in the 2D view
        std::function<void()>                on_clear_target;    // Ctrl+right click
        // A mission waypoint was dragged to a new place (index, room x, room y).
        std::function<void(int, float, float)> on_waypoint_moved;
        // Everything mission-shaped is the panel's own vocabulary; the display just forwards it.
        rc::MissionPanel::Callbacks          mission;
    };

    // Creates the planner GUI as its OWN top-level window (not docked into the DSR graph viewer),
    // so the agent runs with Agent.graph=false. Mirrors room_concept's RoomViewer.
    void initialize(rc::LidarPointBuffer *lidar_buffer, Callbacks callbacks);

    // Repopulate the mission dropdown (after load, or after a recording is saved). Thread-safe: STAGED,
     // then applied on the GUI thread in present(). It is called from the control thread (a save finishing),
     // and touching a QComboBox from there is undefined behaviour, not merely untidy.
    void set_mission_list(const std::vector<std::string> &names, const std::string &selected);
    // Stage the mission readout + waypoint overlay. Thread-safe, like update().
    void set_mission_state(const rc::MissionPanel::View &view,
                           const std::vector<Eigen::Vector2f> &waypoints,
                           int current_index);
    // GUI thread. True if a mission is running, so a click can ask before cancelling it.
    bool mission_running() const;
    bool mission_recording() const;
    bool confirm_mission_supersede();

    // Persist the standalone window's geometry (call on shutdown, GUI thread).
    void save_window_geometry() const;

    Custom_widget *widget() const { return custom_widget_.get(); }

    // Staging API — safe to call from any thread. These only copy data into an
    // internal snapshot; no Qt objects are touched here.
    void update(const std::optional<ControllerRobotPose> &robot_pose,
                const ControllerPolygon &room_polygon,
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
    // Stage the stuck-recovery indicator state (thread-safe; applied on the GUI thread in present()).
    void set_stuck_active(bool active);
    // Push the live arrival state to the toolbar readout. Called every cycle; the widget dedups.
    void set_goal_distance(std::optional<float> dist_m, std::optional<float> yaw_err_rad, bool aligning);
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
        bool stuck_active = false;   // stuck-recovery indicator (pushed every cycle; widget dedups)
        // Remaining distance to target, shown beside the MPPI-paths button. nullopt = no active plan.
        // goal_yaw_err_rad is nullopt when the target carries no commanded facing yaw.
        std::optional<float> goal_dist_m;
        std::optional<float> goal_yaw_err_rad;
        bool goal_aligning = false;
        // Mission overlay + readout.
        rc::MissionPanel::View mission_view;
        std::vector<Eigen::Vector2f> mission_waypoints;
        int mission_index = -1;
        std::vector<std::string> mission_names;
        std::string mission_selected;
        bool mission_list_pending = false;
    };

    void restore_window_geometry();

    std::unique_ptr<Custom_widget> custom_widget_;
    std::unique_ptr<rc::MissionPanel> mission_panel_;
    std::unique_ptr<rc::Viewer2D> viewer_2d_;
    bool room_view_fitted_ = false;
    std::unordered_set<std::string> efe_series_known_;   // plot series already registered
    std::size_t efe_color_next_ = 0;                     // next palette colour for a new affordance

    mutable std::mutex snapshot_mutex_;
    DisplaySnapshot snapshot_;
};