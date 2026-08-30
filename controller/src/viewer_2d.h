#ifndef CONTROLLER_VIEWER_2D_H
#define CONTROLLER_VIEWER_2D_H

#include <QObject>
#include <QColor>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QGraphicsPolygonItem>
#include <QGraphicsPathItem>
#include <QGraphicsItem>
#include <QPointF>
#include <QRectF>
#include <QTransform>
#include <QWidget>

#include <Eigen/Dense>

#include "controller_runtime_types.h"
#include "lidar_buffer_types.h"

#include <optional>
#include <string>
#include <unordered_map>
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
    // ── AN ORIENT HAS NOTHING TO DRAW AS A PLACE ────────────────────────────────────────────────
    // A Reach shows up on this view as a marker somewhere else and a path leading to it. An Orient is
    // published AT the robot's own pose and turns on the spot, so the marker lands under the body and
    // there is no path: the view shows nothing at all, which is exactly what it looked like while the
    // calibration pivot was running. What an Orient asks for is a BEARING, so that is what gets drawn
    // -- the asked heading as a ray from the body, and the arc still to be turned between where the
    // robot points now and where it was asked to point.
    void update_orient_overlay(float x, float y, float target_yaw, float current_yaw, bool visible);

Q_SIGNALS:
    void new_mouse_coordinates(QPointF);
    void right_click(QPointF);
    // A mission waypoint was dragged to a new place. Index into the waypoint list last passed to
    // draw_mission(); position in room coordinates.
    void mission_waypoint_moved(int index, QPointF room_pos);
    // A right CLICK (press and release without panning) on empty canvas while the mission is editable:
    // insert a waypoint here, into whichever leg passes nearest. The receiver decides the index.
    void mission_waypoint_inserted(QPointF room_pos);
    // Ctrl + right click ON a waypoint: drop it. Index into the list last passed to draw_mission().
    void mission_waypoint_removed(int index);

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
    // ★TELLING A CLICK FROM A PAN. Right-drag on empty canvas is how the view PANS, and panning is how
    // you reach the part of the room you are about to edit — so inserting on right-button PRESS would
    // have traded the pan gesture for the insert one. Instead the press is remembered and the insert
    // fires on RELEASE, only if the pointer never left a few pixels of where it went down. A pan moves
    // far more than that, so the two gestures separate cleanly and neither is lost.
    // Pixels, not metres, on purpose: this measures a hand holding a mouse still, not a distance in
    // the room, and it must not change meaning when the view zooms.
    static constexpr int kClickSlopPx = 4;
    bool    pending_insert_ = false;
    QPoint  press_pos_px_;
    QGraphicsEllipseItem *target_marker_ = nullptr;
    QGraphicsLineItem    *orient_ray_ = nullptr;      // the bearing the producer asked for
    QGraphicsPathItem    *orient_arc_ = nullptr;      // what is still to be turned
    std::vector<QGraphicsLineItem *> robot_traj_items_;
    std::optional<Eigen::Vector2f> last_robot_pos_;

    // ── WHICH HUE THIS OBJECT GETS, AND WHY IT NEVER CHANGES ─────────────────────────────────────
    // Slots are handed out in FIRST-SEEN order and never taken back, keyed on the object's stable
    // identity (ControllerObstacleVisual::color_key = the DSR node name). Two properties follow, and
    // both matter more than which colour any one object gets:
    //   an object keeps its colour for the whole session, so you can track it across frames; and
    //   an object appearing or vanishing REPAINTS NOTHING ELSE — colour follows the entity, never its
    //   rank in a list that reshuffles. Hashing the name would give the second but not stable
    //   separation: five objects could collide into two hues by luck. A registry cannot.
    // The map only ever grows, by one small string per distinct object the run has seen.
    std::unordered_map<std::string, int> object_color_slots_;
    int next_object_color_slot_ = 0;
    int object_color_slot(const std::string &key);

    void clear_polygon_item(QGraphicsPolygonItem *&item);
    void clear_lidar_items();
    void clear_room_axis_items();
    // Draw the mission from the LOCAL cache. Separate from draw_mission() because a drag in flight
    // must survive the per-cycle redraw that present() issues from the control thread's snapshot.
    void render_mission();
};

} // namespace rc

#endif