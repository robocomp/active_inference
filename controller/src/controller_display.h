#pragma once

#include <atomic>

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
#include "controller_affordance_panel.h"
#include "controller_camera_masks.h"
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
        // Skip pressed in the affordance window: abandon the running affordance, take the next.
        std::function<void()>                on_skip_affordance;
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
    /// Control-loop rate, shown in the window title. Written from the CONTROL thread once a second and
    /// read on the GUI thread when the title is composed — two independent scalars, so plain atomics are
    /// enough and no snapshot plumbing is needed. worst_ms is the tail, which is what a stall feels like;
    /// the mean alone hid a loop whose median was healthy while its tail ran past a second.
    void set_control_rate(float hz, float worst_ms)
    {
        control_hz_.store(hz, std::memory_order_relaxed);
        control_worst_ms_.store(worst_ms, std::memory_order_relaxed);
    }

    void set_mission_state(const rc::MissionPanel::View &view,
                           const std::vector<Eigen::Vector2f> &waypoints,
                           int current_index);
    // Adopt the plain tracker's L into the Stick <-> Loose slider. GUI thread, call once after the
    // config is loaded so the control never shows a value the robot is not using.
    void set_plain_l(float metres);

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

    void set_command_values(float adv_mm_s, float side_mm_s, float rot_rps);
    void set_command_text(const QString &text);   // alerts only (LiDAR stall / recovery)
    void set_selected_affordance(const QString &current, const QString &previous);
    // Stage the stuck-recovery indicator state (thread-safe; applied on the GUI thread in present()).
    void set_stuck_active(bool active);
    // Push the live arrival state to the toolbar readout. Called every cycle; the widget dedups.
    void set_goal_distance(std::optional<float> dist_m, std::optional<float> yaw_err_rad, bool aligning);
    void set_session_totals(float metres, float seconds);   // top row: metres and time since startup
    void set_affordance_execution(const rc::AffordanceExecution &v);   // the affordance-program window
    // One sample per evaluated affordance for the EFE panel below the 2D view. Plots TWO lines per
    // affordance: the selection score (gain − λ·dist, solid) and the raw gain (ΔH, lighter) — so the
    // vertical gap between them is λ·dist. Thread-safe (the plot buffers under its own mutex).
    // `eligible` is the protocol state's verdict: is this affordance actually IN the contest (Offered or
    // Executing) or merely present in the graph? The plot draws neg_efe, the exact quantity selection
    // maximises, so among eligible candidates the highest line ALWAYS wins — which means a line that
    // stays highest and is never chosen is not losing, it is not competing. Plotting the two alike made
    // that indistinguishable, and a viewer that cannot tell "lost" from "not entered" is worse than none.
    struct AffordanceEfeSample
    {
        std::string name;
        float gain = 0.f;
        float score = 0.f;
        bool  eligible = true;
        std::string state;   // Offered / Executing / Completed / Missing / Invalid
    };
    void update_affordance_efe(const std::vector<AffordanceEfeSample> &samples);
    // The commanded velocities, for the smoothness trace. Fed from the motion commander's OUTPUT loop
    // (every command actually emitted, ~40 Hz), not from the GUI tick: smoothness is a property of the
    // signal the base receives, and sampling it at the redraw rate would alias the jitter worth seeing.
    // adv in m/s, rot in rad/s. Thread-safe; the plot buffers under its own mutex and the series are
    // registered on the GUI thread at construction.
    void update_velocity_trace(float adv_mps, float rot_rps);
    // ── THE UNCERTAINTY THAT IS SLOWING THE ROBOT, ON THE SAME AXES ──────────────────────────────
    // sigma_xy is metres and the plot's axis is m/s, so it is handed over ALREADY MAPPED onto that axis
    // by the control thread (which is where the knees live): the STOP knee maps to max_adv, so the line
    // touching the top of the velocity band means "at or past PoseXYStdStop, throttle floored". Clamped
    // there, because TimeSeriesPlot auto-scales Y and an out-of-range series would squash the velocity
    // traces it exists to be compared against.
    // NEGATIVE = the limiter saw no covariance at all. That is genuinely UNDEFINED rather than zero, so
    // the trace breaks (add_point treats non-finite as a gap marker) instead of drawing a confident 0.
    void set_uncertainty_trace_value(float sigma_on_speed_axis);
    // The camera frame with the YOLO silhouettes on it, for the affordance panel. Composed on the
    // control thread and handed over whole (see controller_camera_masks.h) — the QImage inside is
    // already private to this snapshot, so nothing shares a pixel buffer across the thread boundary.
    void set_camera_masks(const rc::CameraMasksView &view);
    // Is the affordance window open? Read from the CONTROL thread to decide whether composing a camera
    // frame is worth anything, so it is an atomic mirror written on the GUI thread — never the widget's
    // own isVisible(), which may not be queried off the GUI thread.
    [[nodiscard]] bool affordance_panel_visible() const
    { return affordance_panel_visible_.load(std::memory_order_relaxed); }
    void clear_robot_trajectory();

    // Presentation — MUST be called on the GUI thread only. Reads the latest
    // staged snapshot and performs all Qt scene drawing.
    void present();

private:
    std::atomic<bool>  affordance_panel_visible_{false};
    std::atomic<float> control_hz_{0.f};
    std::atomic<float> control_worst_ms_{0.f};
    // Written by the control thread, read by the output thread that feeds the velocity plot. An atomic
    // rather than a plain float because those are two different threads and the profile path next door
    // already made exactly this handoff atomic for the same reason.
    std::atomic<float> unc_trace_{-1.f};

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
        float cmd_adv_mm_s = 0.f, cmd_side_mm_s = 0.f, cmd_rot_rps = 0.f;
        bool cmd_values_pending = false;
        QString affordance_current, affordance_previous;
        bool selected_affordance_text_pending = false;
        bool clear_trajectory_pending = false;
        bool stuck_active = false;   // stuck-recovery indicator (pushed every cycle; widget dedups)
        // Remaining distance to target, shown beside the MPPI-paths button. nullopt = no active plan.
        // goal_yaw_err_rad is nullopt when the target carries no commanded facing yaw.
        std::optional<float> goal_dist_m;
        std::optional<float> goal_yaw_err_rad;
        bool goal_aligning = false;
        float session_distance_m = 0.f;
        float session_elapsed_s = 0.f;
        rc::AffordanceExecution affordance;
        rc::CameraMasksView camera_masks;
        bool camera_masks_pending = false;
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
    std::unique_ptr<rc::AffordancePanel> affordance_panel_;
    std::unique_ptr<rc::Viewer2D> viewer_2d_;
    bool room_view_fitted_ = false;
    std::unordered_set<std::string> efe_series_known_;   // plot series already registered
    std::size_t efe_color_next_ = 0;                     // next palette colour for a new affordance

    mutable std::mutex snapshot_mutex_;
    DisplaySnapshot snapshot_;
};