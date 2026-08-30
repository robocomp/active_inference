#include "controller_display.h"

#include <QByteArray>
#include <limits>
#include <QComboBox>
#include <QInputDialog>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>

namespace
{
constexpr auto kWinSettingsOrg = "RoboComp";
constexpr auto kWinSettingsApp = "controller";
constexpr auto kWinSettingsKey = "ControllerPlannerWindow_geometry";
}  // namespace

void ControllerDisplay::restore_window_geometry()
{
    if (!custom_widget_)
        return;
    QSettings settings(kWinSettingsOrg, kWinSettingsApp);
    const QByteArray geom = settings.value(kWinSettingsKey).toByteArray();
    if (!geom.isEmpty())
        custom_widget_->restoreGeometry(geom);
    else
        custom_widget_->resize(820, 600);
}

void ControllerDisplay::save_window_geometry() const
{
    if (!custom_widget_)
        return;
    QSettings settings(kWinSettingsOrg, kWinSettingsApp);
    settings.setValue(kWinSettingsKey, custom_widget_->saveGeometry());
    settings.sync();
}

void ControllerDisplay::initialize(rc::LidarPointBuffer *lidar_buffer, Callbacks callbacks)
{
    const auto on_manual_target = callbacks.on_manual_target;
    const auto on_clear_target = callbacks.on_clear_target;

    // Own top-level window (parent == nullptr), NOT docked into the DSR graph viewer. This decouples
    // the planner GUI from Agent.graph, so the agent runs with graph=false (no DSRViewer). Mirrors
    // room_concept's RoomViewer.
    custom_widget_ = std::make_unique<Custom_widget>();
    custom_widget_->setWindowTitle(QStringLiteral("controller — planner"));
    restore_window_geometry();
    custom_widget_->show();

    // No axis from the base viewer: it is pinned to the centre of this initial view rect, not to the
    // room. The only axis drawn is the one on the room polygon centroid (see Viewer2D::draw_room_polygon).
    viewer_2d_ = std::make_unique<rc::Viewer2D>(custom_widget_->frame, QRectF(-5.0, -5.0, 10.0, 10.0), false);
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
    QObject::connect(viewer_2d_.get(), &rc::Viewer2D::mission_waypoint_moved,
                     custom_widget_.get(),
                     [cb = callbacks.on_waypoint_moved](int index, const QPointF &p)
                     {
                         if (cb) cb(index, static_cast<float>(p.x()), static_cast<float>(p.y()));
                     });
    QObject::connect(viewer_2d_.get(), &rc::Viewer2D::mission_waypoint_inserted,
                     custom_widget_.get(),
                     [cb = callbacks.on_waypoint_inserted](const QPointF &p)
                     {
                         if (cb) cb(static_cast<float>(p.x()), static_cast<float>(p.y()));
                     });
    QObject::connect(viewer_2d_.get(), &rc::Viewer2D::mission_waypoint_removed,
                     custom_widget_.get(),
                     [cb = callbacks.on_waypoint_removed](int index)
                     {
                         if (cb) cb(index);
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
    QObject::connect(custom_widget_->mppi_paths_toggle_btn, &QPushButton::toggled,
                     custom_widget_.get(),
                     [this](bool checked)
                     {
                         if (viewer_2d_)
                             viewer_2d_->set_mppi_paths_visible(checked);
                     });

    mission_panel_ = std::make_unique<rc::MissionPanel>(custom_widget_.get(), callbacks.mission);
    // The affordance program window. Constructed hidden and fed every cycle; clicking the affordance
    // name in the toolbar shows it. It is a QDialog with the Tool flag, so it floats over the 2D view
    // without taking focus from it — this is meant to be watched WHILE driving.
    affordance_panel_ = std::make_unique<rc::AffordancePanel>(custom_widget_.get());
    // The button runs on the GUI thread; the session it acts on lives on the control thread. The
    // callback the worker installs is an enqueue, never a direct call — same rule as every other
    // control the panels own.
    affordance_panel_->set_skip_callback(callbacks.on_skip_affordance);
    custom_widget_->set_affordance_clicked([this]
    {
        if (affordance_panel_ == nullptr) return;
        affordance_panel_->setVisible(not affordance_panel_->isVisible());
        if (affordance_panel_->isVisible()) affordance_panel_->raise();
        affordance_panel_visible_.store(affordance_panel_->isVisible(), std::memory_order_relaxed);
    });
    custom_widget_->attach_mission_panel(mission_panel_.get());

}

void ControllerDisplay::set_plain_l(float metres)
{
    if (mission_panel_) mission_panel_->set_plain_l(metres);
}

void ControllerDisplay::set_mission_list(const std::vector<std::string> &names, const std::string &selected)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.mission_names = names;
    snapshot_.mission_selected = selected;
    snapshot_.mission_list_pending = true;
}

bool ControllerDisplay::mission_running() const
{ return mission_panel_ && mission_panel_->mission_running(); }

bool ControllerDisplay::mission_recording() const
{ return mission_panel_ && mission_panel_->recording(); }

bool ControllerDisplay::confirm_mission_supersede()
{ return mission_panel_ && mission_panel_->confirm_supersede(); }

void ControllerDisplay::set_mission_state(const rc::MissionPanel::View &view,
                                          const std::vector<Eigen::Vector2f> &waypoints,
                                          int current_index)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.mission_view = view;
    snapshot_.mission_waypoints = waypoints;
    snapshot_.mission_index = current_index;
}

void ControllerDisplay::update(const std::optional<ControllerRobotPose> &robot_pose,
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
                               const std::optional<Eigen::Affine2f> &lidar_correction)
{
    // Staging only — copy into the snapshot, no Qt access. Safe from any thread.
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.robot_pose = robot_pose;
    snapshot_.room_polygon = room_polygon;
    snapshot_.current_plan = current_plan;
    snapshot_.obstacle_polys = obstacle_polys;
    snapshot_.obstacle_rfe_points = obstacle_rfe_points;
    snapshot_.current_target_room = current_target_room;
    snapshot_.last_mppi_trajectories = last_mppi_trajectories;
    snapshot_.last_mppi_average_trajectory = last_mppi_average_trajectory;
    snapshot_.last_best_mppi_trajectory_idx = last_best_mppi_trajectory_idx;
    snapshot_.last_display_wp_index = last_display_wp_index;
    snapshot_.max_lidar_draw_points = max_lidar_draw_points;
    snapshot_.lidar_correction = lidar_correction;
    snapshot_.valid = true;
}

void ControllerDisplay::set_command_values(float adv_mm_s, float side_mm_s, float rot_rps)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.cmd_adv_mm_s = adv_mm_s;
    snapshot_.cmd_side_mm_s = side_mm_s;
    snapshot_.cmd_rot_rps = rot_rps;
    snapshot_.cmd_values_pending = true;
}

void ControllerDisplay::set_command_text(const QString &text)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.command_text = text;
    snapshot_.command_text_pending = true;
}

void ControllerDisplay::set_orient_overlay(float x, float y, float target_yaw, float current_yaw,
                                          bool visible)
{
    std::scoped_lock lock(snapshot_mutex_);
    snapshot_.orient_x = x;
    snapshot_.orient_y = y;
    snapshot_.orient_target_yaw = target_yaw;
    snapshot_.orient_current_yaw = current_yaw;
    snapshot_.orient_visible = visible;
}

void ControllerDisplay::set_affordance_execution(const rc::AffordanceExecution &v)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.affordance = v;
}

void ControllerDisplay::set_camera_masks(const rc::CameraMasksView &view)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.camera_masks = view;
    snapshot_.camera_masks_pending = true;
}

void ControllerDisplay::set_selected_affordance(const QString &current, const QString &previous)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.affordance_current = current;
    snapshot_.affordance_previous = previous;
    snapshot_.selected_affordance_text_pending = true;
}

void ControllerDisplay::set_stuck_active(bool active)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.stuck_active = active;
}

void ControllerDisplay::set_session_totals(float metres, float seconds)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.session_distance_m = metres;
    snapshot_.session_elapsed_s = seconds;
}

void ControllerDisplay::set_goal_distance(std::optional<float> dist_m, std::optional<float> yaw_err_rad,
                                          bool aligning)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.goal_dist_m = dist_m;
    snapshot_.goal_yaw_err_rad = yaw_err_rad;
    snapshot_.goal_aligning = aligning;
}

// Running cross-track error. This replaced a four-series J plot: J's smoothness terms are properties
// of the ROUTE (re-planned against a live grid each run, so a 119 mm waypoint repair moved jerk/m by
// 38%), while rms repeats to cv 2.3% and is what the tracker itself controls.
void ControllerDisplay::update_velocity_trace(float adv_mps, float rot_rps)
{
    auto *plot = custom_widget_ ? custom_widget_->mission_j_plot : nullptr;
    if (plot == nullptr) return;
    // Series are registered in Custom_widget's constructor, on the GUI thread; add_point() is
    // mutex-guarded and documented callable from any thread, which is what lets this be fed straight
    // from the motion commander's output loop instead of at the redraw rate.
    const auto ok = [](float v) { return std::isfinite(v) ? v : 0.f; };
    plot->add_point("adv", ok(adv_mps));
    plot->add_point("rot", ok(rot_rps));
    // Already mapped onto the speed axis by the control thread — see set_uncertainty_trace_value.
    // A negative value means "no covariance reached the limiter", which is undefined rather than zero:
    // NaN makes the line BREAK there, which is what the plot's gap marker is for.
    const float unc = unc_trace_.load(std::memory_order_relaxed);
    plot->add_point("sigma", unc >= 0.f ? unc : std::numeric_limits<float>::quiet_NaN());
    // We are commanding, so the measured-fallback pair has nothing to say: gap it rather than let it
    // hold its last value across the hand-over, which would draw a line through a stretch it did not
    // describe.
    const float gap = std::numeric_limits<float>::quiet_NaN();
    plot->add_point("adv_meas", gap);
    plot->add_point("rot_meas", gap);
    vel_local_fed_.store(true, std::memory_order_relaxed);
}

void ControllerDisplay::update_velocity_trace_external(float ref_adv_mps, float ref_rot_rps,
                                                      bool ref_fresh,
                                                      float meas_adv_mps, float meas_rot_rps)
{
    auto *plot = custom_widget_ ? custom_widget_->mission_j_plot : nullptr;
    if (plot == nullptr) return;
    // Our own output loop spoke since the last control cycle — it is the higher-rate and more faithful
    // source, so leave the trace to it. The exchange also arms the next window.
    if (vel_local_fed_.exchange(false, std::memory_order_relaxed)) return;

    const float gap = std::numeric_limits<float>::quiet_NaN();
    const auto ok = [](float v) { return std::isfinite(v) ? v : 0.f; };

    // A published command, from something that is not us. Drawn in the commanded series, because that
    // is what it is.
    plot->add_point("adv", ref_fresh ? ok(ref_adv_mps) : gap);
    plot->add_point("rot", ref_fresh ? ok(ref_rot_rps) : gap);
    // Nobody is publishing a command, so the only witness left is what the base actually DID. Its own
    // series, so "measured" is never mistaken for "commanded".
    plot->add_point("adv_meas", ref_fresh ? gap : ok(meas_adv_mps));
    plot->add_point("rot_meas", ref_fresh ? gap : ok(meas_rot_rps));
    // Whoever is commanding, it is not through OUR uncertainty limiter, so the sigma line has nothing
    // to say about this stretch. A gap, not a zero — zero would read as "perfectly localised".
    plot->add_point("sigma", gap);
}

void ControllerDisplay::set_uncertainty_trace_value(float sigma_on_speed_axis)
{
    unc_trace_.store(sigma_on_speed_axis, std::memory_order_relaxed);
}

void ControllerDisplay::set_lidar_rate_hz(float hz)
{
    lidar_hz_.store(hz, std::memory_order_relaxed);
    if (custom_widget_) custom_widget_->set_lidar_hz(hz);
}

void ControllerDisplay::update_affordance_efe(const std::vector<AffordanceEfeSample> &samples)
{
    auto *plot = custom_widget_ ? custom_widget_->affordance_efe_plot : nullptr;
    if (plot == nullptr)
        return;

    // Distinct, stable colours per affordance (assigned in first-seen order).
    static const QColor kPalette[] = {QColor("Tomato"),     QColor("SteelBlue"), QColor("MediumSeaGreen"),
                                      QColor("Goldenrod"),   QColor("MediumPurple"), QColor("Teal"),
                                      QColor("OrangeRed"),   QColor("SlateGray")};
    constexpr std::size_t kPaletteSize = sizeof(kPalette) / sizeof(kPalette[0]);

    // Drop series whose affordance is no longer among the candidates (object removed from the graph),
    // so the plot tracks the live scene instead of accumulating stale lines.
    std::unordered_set<std::string> current;
    current.reserve(samples.size());
    for (const auto &s : samples)
        current.insert(s.name);
    for (auto it = efe_series_known_.begin(); it != efe_series_known_.end();)
    {
        if (current.find(*it) == current.end())
        {
            plot->remove_series(*it);
            it = efe_series_known_.erase(it);
        }
        else
            ++it;
    }

    // One line per affordance: the selection score (gain − λ·dist + hysteresis) — the EXACT value
    // selection maximises, so the highest line among the eligible ones is the one that gets chosen.
    // ★AN INELIGIBLE AFFORDANCE PLOTS A GAP, NOT A VALUE. Its score is still computable and still high,
    // but it is not in the contest — a Completed or Invalid node is skipped by every selection branch —
    // and drawing it as a continuous line said "this keeps winning on merit and keeps being passed
    // over", which is a bug report about the selector rather than what it is: a node whose owning agent
    // is not offering it. The break in the line is the answer.
    for (const auto &s : samples)
    {
        if (efe_series_known_.find(s.name) == efe_series_known_.end())
        {
            plot->add_series(s.name, kPalette[efe_color_next_ % kPaletteSize], 1.8f);
            ++efe_color_next_;
            efe_series_known_.insert(s.name);
        }
        plot->add_point(s.name, s.eligible ? s.score : std::numeric_limits<float>::quiet_NaN());
    }
}

void ControllerDisplay::clear_robot_trajectory()
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.clear_trajectory_pending = true;
}

void ControllerDisplay::present()
{
    // GUI thread only. Consume the latest staged snapshot and draw it.
    DisplaySnapshot snap;
    {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        snap = snapshot_;
        snapshot_.command_text_pending = false;
        snapshot_.cmd_values_pending = false;
        snapshot_.selected_affordance_text_pending = false;
        snapshot_.clear_trajectory_pending = false;
        snapshot_.mission_list_pending = false;
    }

    if (!custom_widget_)
        return;

    if (snap.clear_trajectory_pending && viewer_2d_)
        viewer_2d_->clear_robot_trajectory();

    if (snap.cmd_values_pending)
        custom_widget_->set_cmd_vel(snap.cmd_adv_mm_s, snap.cmd_side_mm_s, snap.cmd_rot_rps);

    if (snap.command_text_pending)
        custom_widget_->set_cmd_vel_text(snap.command_text);

    if (snap.selected_affordance_text_pending)
        custom_widget_->set_selected_affordance(snap.affordance_current, snap.affordance_previous);
    // Re-published every frame, not only on the toggle: the window can also be closed by its own
    // title-bar button, which never runs the toggle lambda and would leave the mirror stuck on true.
    if (affordance_panel_)
        affordance_panel_visible_.store(affordance_panel_->isVisible(), std::memory_order_relaxed);
    if (affordance_panel_ and affordance_panel_->isVisible())
    {
        affordance_panel_->update_view(snap.affordance);
        affordance_panel_->set_camera_view(snap.camera_masks);
    }

    custom_widget_->set_stuck_active(snap.stuck_active);   // widget dedups same-state calls
    custom_widget_->set_goal_distance(snap.goal_dist_m, snap.goal_yaw_err_rad, snap.goal_aligning);
    custom_widget_->set_session(snap.session_distance_m, snap.session_elapsed_s);
    if (mission_panel_)
    {
        if (snap.mission_list_pending)
            mission_panel_->set_missions(snap.mission_names, snap.mission_selected);
        mission_panel_->apply(snap.mission_view);
    }
    if (viewer_2d_)
        viewer_2d_->draw_mission(snap.mission_waypoints, snap.mission_index, snap.mission_view.recording,
                                 // Draggable whenever nothing is being measured: editing the route under a
                                 // running mission would invalidate the run without saying so.
                                 not snap.mission_view.running);
    // Mission status rides in the WINDOW TITLE. As a stretchy label in the mission row it forced the whole
    // window wider than the 2D view needs; the title bar is free real estate and always visible.
    // Control rate rides here too. It is the number that says whether the loop is keeping its deadline,
    // and it was only ever visible on stdout — which is exactly where nobody looks while driving. Shown
    // as rate plus the WORST period in the last window, because the mean stays healthy through a stall.
    const float hz = control_hz_.load(std::memory_order_relaxed);
    const float worst = control_worst_ms_.load(std::memory_order_relaxed);
    QString rate;
    if (hz > 0.f)
        rate = QStringLiteral("%1 Hz").arg(hz, 0, 'f', 1)
             + (worst > 0.f ? QStringLiteral(" (worst %1 ms)").arg(worst, 0, 'f', 0) : QString());
    QString t = QStringLiteral("controller — planner");
    if (!rate.isEmpty()) t += QStringLiteral(" · ") + rate;
    if (!snap.mission_view.status.empty())
        t += QStringLiteral(" · ") + QString::fromStdString(snap.mission_view.status);
    if (custom_widget_->windowTitle() != t)
        custom_widget_->setWindowTitle(t);

    if (!snap.valid || !viewer_2d_)
        return;

    ControllerPolygon display_path;
    if (snap.current_plan.has_value())
        display_path = snap.current_plan->room_path;

    viewer_2d_->draw_room_polygon(snap.room_polygon);
    viewer_2d_->set_lidar_draw_correction(snap.lidar_correction.value_or(Eigen::Affine2f::Identity()));
    viewer_2d_->draw_lidar_points_from_buffer(snap.max_lidar_draw_points);
    viewer_2d_->draw_path({
        .path = std::move(display_path),
        // The plan's turning points, drawn as dots — but ONLY when the plan IS a set of turning points.
        // A continuous route is thousands of 5 cm samples and dotting each one is noise, not information.
        .waypoints = (snap.current_plan.has_value() and snap.current_plan->room_path.size() <= 64)
                         ? snap.current_plan->room_path : ControllerPolygon{},
        .obstacle_polys = snap.obstacle_polys,
        .obstacle_rfe_points = snap.obstacle_rfe_points,
        .candidate_trajectories = snap.last_mppi_trajectories,
        .average_trajectory = snap.last_mppi_average_trajectory,
        .best_trajectory_idx = snap.last_best_mppi_trajectory_idx
    });
    if (!room_view_fitted_ && !snap.room_polygon.empty())
    {
        viewer_2d_->fit_view();
        room_view_fitted_ = true;
    }

    if (snap.current_target_room.has_value())
        viewer_2d_->update_target_marker(snap.current_target_room->x(), snap.current_target_room->y(), true);
    else
        viewer_2d_->update_target_marker(0.f, 0.f, false);

    viewer_2d_->update_orient_overlay(snap.orient_x, snap.orient_y, snap.orient_target_yaw,
                                      snap.orient_current_yaw, snap.orient_visible);

    if (snap.robot_pose.has_value())
        viewer_2d_->update_robot(snap.robot_pose->as_transform());
}