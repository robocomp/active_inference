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
                               const ControllerObstacleVisuals &obstacle_polys,
                               const ControllerPolygons &obstacle_rfe_points,
                               const std::optional<Eigen::Vector2f> &current_target_room,
                               const std::vector<ControllerPolygon> &last_mppi_trajectories,
                               const ControllerPolygon &last_mppi_average_trajectory,
                               int last_best_mppi_trajectory_idx,
                               int last_display_wp_index,
                               int max_lidar_draw_points)
{
    // Staging only — copy into the snapshot, no Qt access. Safe from any thread.
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.robot_pose = robot_pose;
    snapshot_.room_polygon = room_polygon;
    snapshot_.inner_polygon = inner_polygon;
    snapshot_.current_plan = current_plan;
    snapshot_.obstacle_polys = obstacle_polys;
    snapshot_.obstacle_rfe_points = obstacle_rfe_points;
    snapshot_.current_target_room = current_target_room;
    snapshot_.last_mppi_trajectories = last_mppi_trajectories;
    snapshot_.last_mppi_average_trajectory = last_mppi_average_trajectory;
    snapshot_.last_best_mppi_trajectory_idx = last_best_mppi_trajectory_idx;
    snapshot_.last_display_wp_index = last_display_wp_index;
    snapshot_.max_lidar_draw_points = max_lidar_draw_points;
    snapshot_.valid = true;
}

void ControllerDisplay::set_command_text(const QString &text)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.command_text = text;
    snapshot_.command_text_pending = true;
}

void ControllerDisplay::set_selected_affordance_text(const QString &text)
{
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.selected_affordance_text = text;
    snapshot_.selected_affordance_text_pending = true;
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

    // One line per affordance: the selection score (gain − λ·dist) — the value used to choose.
    for (const auto &s : samples)
    {
        if (efe_series_known_.find(s.name) == efe_series_known_.end())
        {
            plot->add_series(s.name, kPalette[efe_color_next_ % kPaletteSize], 1.8f);
            ++efe_color_next_;
            efe_series_known_.insert(s.name);
        }
        plot->add_point(s.name, s.score);
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
        snapshot_.selected_affordance_text_pending = false;
        snapshot_.clear_trajectory_pending = false;
    }

    if (!custom_widget_)
        return;

    if (snap.clear_trajectory_pending && viewer_2d_)
        viewer_2d_->clear_robot_trajectory();

    if (snap.command_text_pending)
        custom_widget_->set_cmd_vel_text(snap.command_text);

    if (snap.selected_affordance_text_pending)
        custom_widget_->set_selected_affordance_text(snap.selected_affordance_text);

    if (!snap.valid || !viewer_2d_)
        return;

    ControllerPolygon display_path;
    if (snap.current_plan.has_value())
        display_path = snap.current_plan->room_path;

    viewer_2d_->draw_room_polygon(snap.room_polygon);
    viewer_2d_->draw_lidar_points_from_buffer(snap.max_lidar_draw_points);
    viewer_2d_->draw_path({
        .path = std::move(display_path),
        .inner_poly = snap.inner_polygon,
        .graph_nodes = snap.current_plan.has_value() ? snap.current_plan->graph_nodes : ControllerPolygon{},
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

    if (snap.robot_pose.has_value())
        viewer_2d_->update_robot(snap.robot_pose->as_transform());
}