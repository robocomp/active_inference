#ifndef CONTROLLER_VIEWER_2D_H
#define CONTROLLER_VIEWER_2D_H

#include <QObject>
#include <QColor>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QGraphicsPolygonItem>
#include <QGraphicsItem>
#include <QPointF>
#include <QRectF>
#include <QTransform>
#include <QWidget>

#include <Eigen/Dense>

#include "controller_runtime_types.h"
#include "lidar_buffer_types.h"

#include <optional>
#include <vector>

class AbstractGraphicViewer;

namespace rc
{

class Viewer2D : public QObject
{
    Q_OBJECT
    public:
    struct PathDrawData
    {
        ControllerPolygon path;
        ControllerPolygon waypoints;      // the plan's turning points, drawn as dots
        ControllerObstacleVisuals obstacle_polys;
        ControllerPolygons obstacle_rfe_points;
        ControllerPolygons candidate_trajectories;
        ControllerPolygon average_trajectory;
        int best_trajectory_idx = -1;
    };

    explicit Viewer2D(QWidget *parent, const QRectF &grid_dim, bool show_axis = true);
    ~Viewer2D() override;

    QWidget *get_widget() const;
    void add_robot(float width, float length, float offset, float rotation, QColor color);
    void show();
    void fit_to_scene(const QRectF &rect);
    void fit_view(float margin_ratio = 0.05f);

    void update_robot(const Eigen::Affine2f &robot_pose);
    void draw_room_polygon(const std::vector<Eigen::Vector2f> &verts);
    void set_lidar_buffer(LidarPointBuffer *buffer);
    void set_lidar_visible(bool visible);
    void set_mppi_paths_visible(bool visible);
    void set_lidar_draw_correction(const Eigen::Affine2f &correction) { lidar_draw_correction_ = correction; }
    void draw_lidar_points_from_buffer(int max_points);
    void draw_path(const PathDrawData &data);
    void clear_path_items();
    // Mission waypoints: numbered markers plus the tour's own connecting line, so a recorded route is
    // visible while you record it and its progress is visible while it runs.
    void draw_mission(const std::vector<Eigen::Vector2f> &waypoints, int current_index, bool recording,
                      bool draggable);
    void clear_mission_items();
    void clear_robot_trajectory();
    void update_target_marker(float x, float y, bool visible);

Q_SIGNALS:
    void new_mouse_coordinates(QPointF);
    void right_click(QPointF);
    // A mission waypoint was dragged to a new place. Index into the waypoint list last passed to
    // draw_mission(); position in room coordinates.
    void mission_waypoint_moved(int index, QPointF room_pos);

private:
    bool eventFilter(QObject *watched, QEvent *event) override;

    AbstractGraphicViewer *agv_ = nullptr;
    QGraphicsPolygonItem *polygon_item_ = nullptr;
    LidarPointBuffer *lidar_buffer_ = nullptr;
    Eigen::Affine2f lidar_draw_correction_ = Eigen::Affine2f::Identity();   // overlay dead-reckoning (room→room)
    bool lidar_visible_ = true;
    bool mppi_paths_visible_ = false;
    std::vector<QGraphicsEllipseItem *> lidar_items_;
    std::vector<QGraphicsItem *> room_axis_items_;
    std::vector<QGraphicsItem *> path_draw_items_;
    std::vector<QGraphicsItem *> mission_items_;
    // Drag-to-edit state. The waypoints are cached so a right-press can be tested against them BEFORE the
    // base viewer turns the press into a pan.
    std::vector<Eigen::Vector2f> mission_wps_;
    bool mission_draggable_ = false;
    int  mission_index_ = -1;
    bool mission_recording_ = false;
    int  drag_index_ = -1;
    QGraphicsEllipseItem *target_marker_ = nullptr;
    std::vector<QGraphicsLineItem *> robot_traj_items_;
    std::optional<Eigen::Vector2f> last_robot_pos_;

    void clear_polygon_item(QGraphicsPolygonItem *&item);
    void clear_lidar_items();
    void clear_room_axis_items();
    // Draw the mission from the LOCAL cache. Separate from draw_mission() because a drag in flight
    // must survive the per-cycle redraw that present() issues from the control thread's snapshot.
    void render_mission();
};

} // namespace rc

#endif