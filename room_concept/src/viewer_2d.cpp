#include "viewer_2d.h"

#include <abstract_graphic_viewer/abstract_graphic_viewer.h>

#include <QPen>
#include <QBrush>
#include <QPolygonF>
#include <QLayout>
#include <QDateTime>
#include <QtMath>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>

namespace rc {

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────
Viewer2D::Viewer2D(QWidget* parent, const QRectF& grid_dim, bool show_axis)
{
    agv_ = new AbstractGraphicViewer(parent, grid_dim, show_axis);

    // Forward all AGV signals as Viewer2D signals
    connect(agv_, &AbstractGraphicViewer::robot_moved,
            this, &Viewer2D::robot_moved);
    connect(agv_, &AbstractGraphicViewer::robot_rotate,
            this, &Viewer2D::robot_rotate);
    connect(agv_, &AbstractGraphicViewer::robot_dragging,
            this, &Viewer2D::robot_dragging);
    connect(agv_, &AbstractGraphicViewer::robot_drag_end,
            this, &Viewer2D::robot_drag_end);
    connect(agv_, &AbstractGraphicViewer::new_mouse_coordinates,
            this, &Viewer2D::new_mouse_coordinates);
    connect(agv_, &AbstractGraphicViewer::right_click,
            this, &Viewer2D::right_click);
}

// ─────────────────────────────────────────────────────────────────────────────
// Widget / view management
// ─────────────────────────────────────────────────────────────────────────────
QWidget* Viewer2D::get_widget() const { return agv_; }

void Viewer2D::add_robot(float w, float l, float offset, float rotation, QColor color)
{
    agv_->add_robot(w, l, offset, rotation, color);
}

void Viewer2D::show() { agv_->show(); }

QTransform Viewer2D::transform() const { return agv_->transform(); }

void Viewer2D::set_transform(const QTransform& t) { agv_->setTransform(t); }

void Viewer2D::fit_to_scene(const QRectF& r) { agv_->fitToScene(r); }

void Viewer2D::fit_view(float margin_ratio)
{
    QRectF bounds;
    if (polygon_item_ != nullptr)
        bounds = polygon_item_->boundingRect().translated(polygon_item_->pos());
    else if (estimated_room_item_ != nullptr)
        bounds = estimated_room_item_->boundingRect().translated(estimated_room_item_->pos());
    else
        bounds = agv_->scene.itemsBoundingRect();

    if (!bounds.isValid() || bounds.isEmpty())
        return;

    const qreal dx = std::max(0.1, bounds.width() * std::max(0.f, margin_ratio));
    const qreal dy = std::max(0.1, bounds.height() * std::max(0.f, margin_ratio));
    agv_->fitToScene(bounds.adjusted(-dx, -dy, dx, dy));
}

void Viewer2D::fit_to_robot_and_points(const Eigen::Affine2f& robot_pose,
                                       const std::vector<Eigen::Vector3f>& lidar_points,
                                       float fit_radius,
                                       float margin_ratio)
{
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    const Eigen::Vector2f robot_center = robot_pose.translation();
    min_x = std::min(min_x, robot_center.x());
    max_x = std::max(max_x, robot_center.x());
    min_y = std::min(min_y, robot_center.y());
    max_y = std::max(max_y, robot_center.y());

    const float clamped_radius = std::max(1.0f, fit_radius);
    const float fit_radius_sq = clamped_radius * clamped_radius;
    for (const auto& p : lidar_points)
    {
        const Eigen::Vector2f pr = p.head<2>();
        const Eigen::Vector2f pw = robot_pose.linear() * pr + robot_center;
        const Eigen::Vector2f d = pw - robot_center;
        if (d.squaredNorm() > fit_radius_sq)
            continue;

        min_x = std::min(min_x, pw.x());
        max_x = std::max(max_x, pw.x());
        min_y = std::min(min_y, pw.y());
        max_y = std::max(max_y, pw.y());
    }

    const float width = std::max(2.0f, max_x - min_x);
    const float height = std::max(2.0f, max_y - min_y);
    const float mr = std::max(0.f, margin_ratio);
    const float margin_x = std::max(0.3f, width * mr);
    const float margin_y = std::max(0.3f, height * mr);

    fit_to_scene(QRectF(min_x - margin_x,
                        min_y - margin_y,
                        width + 2.f * margin_x,
                        height + 2.f * margin_y));
}

void Viewer2D::center_on(float x, float y) { agv_->centerOn(x, y); }

// ─────────────────────────────────────────────────────────────────────────────
// Robot
// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::update_robot(const Eigen::Affine2f& robot_pose)
{
    const auto t = robot_pose.translation();
    const float angle_rad = std::atan2(robot_pose.linear()(1, 0), robot_pose.linear()(0, 0));
    agv_->robot_poly()->setPos(t.x(), t.y());
    agv_->robot_poly()->setRotation(qRadiansToDegrees(angle_rad));
}

// ─────────────────────────────────────────────────────────────────────────────
// Covariance ellipse
// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::update_covariance_ellipse(float cx, float cy,
                                         float rx, float ry,
                                         float angle_deg)
{
    if (cov_ellipse_item_ == nullptr)
    {
        cov_ellipse_item_ = agv_->scene.addEllipse(
            -rx, -ry, 2.f * rx, 2.f * ry,
            QPen(QColor(40, 120, 220), 0.03),
            QBrush(QColor(40, 120, 220, 70)));
        cov_ellipse_item_->setZValue(100);
    }
    else
    {
        cov_ellipse_item_->setRect(-rx, -ry, 2.f * rx, 2.f * ry);
    }
    cov_ellipse_item_->setVisible(true);
    cov_ellipse_item_->setPos(cx, cy);
    cov_ellipse_item_->setRotation(angle_deg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Estimated room rect
// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::update_estimated_room_rect(float width, float length, bool has_polygon)
{
    if (has_polygon)
    {
        if (estimated_room_item_ != nullptr)
        {
            agv_->scene.removeItem(estimated_room_item_);
            delete estimated_room_item_;
            estimated_room_item_ = nullptr;
        }
        return;
    }

    const QRectF room_rect(-width / 2.f, -length / 2.f, width, length);
    if (estimated_room_item_ == nullptr)
    {
        estimated_room_item_ = agv_->scene.addRect(
            room_rect, QPen(Qt::magenta, 0.05), QBrush(Qt::NoBrush));
        estimated_room_item_->setZValue(2);
    }
    else
    {
        estimated_room_item_->setRect(room_rect);
    }

    update_room_axes(room_rect);

    fit_view();
}

void Viewer2D::update_room_axes(const QRectF& room_bounds)
{
    QRectF bounds = room_bounds;
    if (!bounds.isValid() || bounds.isEmpty())
        bounds = QRectF(-1.0, -1.0, 2.0, 2.0);

    const qreal axis_length = std::clamp<qreal>(0.05 * std::max(bounds.width(), bounds.height()), 0.16, 0.55);
    const qreal label_axis_fraction = 0.58;
    const QPen x_pen(QColor(220, 70, 70), 0.04);
    const QPen y_pen(QColor(60, 170, 90), 0.04);
    QFont label_font;
    label_font.setPointSizeF(8.0);

    if (room_axis_x_item_ == nullptr)
    {
        room_axis_x_item_ = agv_->scene.addLine(QLineF(0.0, 0.0, axis_length, 0.0), x_pen);
        room_axis_x_item_->setZValue(7);
    }
    else
    {
        room_axis_x_item_->setLine(QLineF(0.0, 0.0, axis_length, 0.0));
        room_axis_x_item_->setPen(x_pen);
    }

    if (room_axis_y_item_ == nullptr)
    {
        room_axis_y_item_ = agv_->scene.addLine(QLineF(0.0, 0.0, 0.0, axis_length), y_pen);
        room_axis_y_item_->setZValue(7);
    }
    else
    {
        room_axis_y_item_->setLine(QLineF(0.0, 0.0, 0.0, axis_length));
        room_axis_y_item_->setPen(y_pen);
    }

    if (room_axis_x_label_ == nullptr)
    {
        room_axis_x_label_ = agv_->scene.addText("X");
        room_axis_x_label_->setDefaultTextColor(x_pen.color());
        room_axis_x_label_->setZValue(8);
        room_axis_x_label_->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
    }
    room_axis_x_label_->setFont(label_font);
    {
        const QRectF label_rect = room_axis_x_label_->boundingRect();
        room_axis_x_label_->setPos(label_axis_fraction * axis_length - label_rect.width() * 0.5,
                                   -label_rect.height() * 0.85);
    }

    if (room_axis_y_label_ == nullptr)
    {
        room_axis_y_label_ = agv_->scene.addText("Y");
        room_axis_y_label_->setDefaultTextColor(y_pen.color());
        room_axis_y_label_->setZValue(8);
        room_axis_y_label_->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
    }
    room_axis_y_label_->setFont(label_font);
    {
        const QRectF label_rect = room_axis_y_label_->boundingRect();
        room_axis_y_label_->setPos(-label_rect.width() - 2.0,
                                   label_axis_fraction * axis_length - label_rect.height() * 0.5);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Room polygon outline
// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::draw_room_polygon(const std::vector<Eigen::Vector2f>& verts, bool is_capturing)
{
    if (verts.size() < 2)
        return;

    if (polygon_item_ != nullptr)
    {
        agv_->scene.removeItem(polygon_item_);
        delete polygon_item_;
        polygon_item_ = nullptr;
    }
    if (polygon_fill_item_ != nullptr)
    {
        agv_->scene.removeItem(polygon_fill_item_);
        delete polygon_fill_item_;
        polygon_fill_item_ = nullptr;
    }

    QPolygonF poly;
    for (const auto& v : verts)
        poly << QPointF(v.x(), v.y());
    if (!is_capturing && verts.size() >= 3)
        poly << QPointF(verts.front().x(), verts.front().y());

    // Warm, light orange/brown floor fill — covers ONLY the interior of the (closed) room polygon,
    // drawn behind every overlay (negative z) so lidar/robot/objects stay on top.
    if (!is_capturing && verts.size() >= 3)
    {
        polygon_fill_item_ = agv_->scene.addPolygon(poly, QPen(Qt::NoPen), QBrush(QColor(240, 219, 195)));
        polygon_fill_item_->setZValue(-10);
    }

    QPen pen(is_capturing ? Qt::yellow : Qt::magenta,
             is_capturing ? 0.08 : 0.15);
    polygon_item_ = agv_->scene.addPolygon(poly, pen, QBrush(Qt::NoBrush));
    polygon_item_->setZValue(8);

    update_room_axes(poly.boundingRect());

    if (!is_capturing)
        fit_view();
}

void Viewer2D::draw_lidar_points(const std::vector<Eigen::Vector3f>& points_high,
                                 const std::vector<Eigen::Vector3f>& points_low,
                                 const Eigen::Affine2f& robot_pose,
                                 int max_points_high)
{
    if (!lidar_points_visible_)
    {
        for (auto *item : lidar_pool_high_)
            item->setVisible(false);
        for (auto *item : lidar_pool_low_)
            item->setVisible(false);
        return;
    }

    auto draw_layer = [&](const std::vector<Eigen::Vector3f>& points,
                          std::vector<QGraphicsEllipseItem*>& pool,
                          const QColor& color,
                          int max_points)
    {
        static const qreal radius_px = 2.5;
        const QRectF ellipse_rect(-radius_px, -radius_px, 2 * radius_px, 2 * radius_px);
        QPen pen(color);
        pen.setWidthF(0.0);
        pen.setCosmetic(true);
        QBrush brush(color);

        const int clamped_max_points = std::max(1, max_points);
        const int stride = std::max(1, static_cast<int>(points.size() / clamped_max_points));
        const size_t num_draw = (points.size() + stride - 1) / stride;

        while (pool.size() > num_draw)
        {
            auto* p = pool.back();
            agv_->scene.removeItem(p);
            delete p;
            pool.pop_back();
        }

        size_t idx = 0;
        for (size_t i = 0; i < points.size() && idx < num_draw; i += stride, ++idx)
        {
            const Eigen::Vector2f pr = points[i].head<2>();
            const Eigen::Vector2f pw = robot_pose.linear() * pr + robot_pose.translation();

            if (idx < pool.size())
            {
                pool[idx]->setPos(pw.x(), pw.y());
                pool[idx]->setVisible(true);
            }
            else
            {
                auto* item = agv_->scene.addEllipse(ellipse_rect, pen, brush);
                item->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
                item->setPos(pw.x(), pw.y());
                item->setZValue(5);
                pool.push_back(item);
            }
        }
    };

    draw_layer(points_high, lidar_pool_high_, QColor("Green"), max_points_high);
    draw_layer(points_low, lidar_pool_low_, QColor("Cyan"), max_points_high / 2);
}

void Viewer2D::set_lidar_points_visible(bool visible)
{
    lidar_points_visible_ = visible;
    if (visible)
        return;

    for (auto *item : lidar_pool_high_)
        item->setVisible(false);
    for (auto *item : lidar_pool_low_)
        item->setVisible(false);
}

bool Viewer2D::lidar_points_visible() const
{
    return lidar_points_visible_;
}


// ─────────────────────────────────────────────────────────────────────────────
// Composite per-frame update
// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::update_frame(const FrameData& fd)
{
    const Eigen::Affine2f& lidar_pose = fd.use_loc_pose ? fd.loc_pose : fd.display_pose;
    draw_lidar_points(fd.lidar_points, {}, lidar_pose, fd.max_lidar_points);

    if (fd.have_loc || fd.is_initialized)
        update_robot(fd.display_pose);

    // 1-sigma translation covariance ellipse aligned with the robot axis.
    if (fd.have_loc)
    {
        const float theta = std::atan2(fd.display_pose.linear()(1, 0), fd.display_pose.linear()(0, 0));
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        Eigen::Matrix2f R;
        R << c, -s,
             s,  c;

        const Eigen::Matrix2f cov_xy_world = fd.covariance.topLeftCorner<2, 2>();
        const Eigen::Matrix2f cov_xy_robot = R.transpose() * cov_xy_world * R;

        const float sigma_x = std::sqrt(std::max(1e-9f, cov_xy_robot(0, 0))) * 2.0f;
        const float sigma_y = std::sqrt(std::max(1e-9f, cov_xy_robot(1, 1))) * 2.0f;
        const auto t = fd.display_pose.translation();
        update_covariance_ellipse(t.x(), t.y(), sigma_x, sigma_y, qRadiansToDegrees(theta));
    }
    else if (cov_ellipse_item_ != nullptr)
    {
        cov_ellipse_item_->setVisible(false);
    }

    if (fd.have_loc && !fd.has_room_polygon)
        update_estimated_room_rect(fd.room_width, fd.room_length, false);

}

// ─────────────────────────────────────────────────────────────────────────────
// Path
// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::draw_path(const PathDrawData& data)
{
    clear_path_items();

    if (data.path.size() < 2)
        return;

    // Original polygon vertex dots (green)
    for (const auto& v : data.orig_poly_verts)
    {
        constexpr float r = 0.1f;
        auto* dot = agv_->scene.addEllipse(
            -r, -r, 2.f * r, 2.f * r,
            QPen(QColor(0, 200, 0), 0.02),
            QBrush(QColor(0, 200, 0, 80)));
        dot->setPos(v.x(), v.y());
        dot->setZValue(17);
        path_draw_items_.push_back(dot);
    }

    // Inner (shrunken) polygon outline
    if (navigable_poly_item_ != nullptr)
    {
        agv_->scene.removeItem(navigable_poly_item_);
        delete navigable_poly_item_;
        navigable_poly_item_ = nullptr;
    }
    if (data.inner_poly.size() >= 3)
    {
        QPolygonF qpoly;
        for (const auto& v : data.inner_poly)
            qpoly << QPointF(v.x(), v.y());
        qpoly << QPointF(data.inner_poly.front().x(), data.inner_poly.front().y());

        navigable_poly_item_ = agv_->scene.addPolygon(
            qpoly,
            QPen(QColor(255, 140, 0, 200), 0.03, Qt::DashLine),
            Qt::NoBrush);
        navigable_poly_item_->setZValue(19);
    }

    // Expanded obstacle outlines
    for (auto* item : obstacle_expanded_items_)
    {
        agv_->scene.removeItem(item);
        delete item;
    }
    obstacle_expanded_items_.clear();

    for (const auto& obs : data.expanded_obstacles)
    {
        if (obs.size() < 3)
            continue;

        QPolygonF qpoly;
        for (const auto& v : obs)
            qpoly << QPointF(v.x(), v.y());
        qpoly << QPointF(obs.front().x(), obs.front().y());

        auto* obs_item = agv_->scene.addPolygon(
            qpoly,
            QPen(QColor(255, 140, 0, 200), 0.03, Qt::DashLine),
            Qt::NoBrush);
        obs_item->setZValue(19);
        obstacle_expanded_items_.push_back(obs_item);
    }

    // Navigable polygon vertex dots (yellow)
    for (const auto& v : data.nav_poly)
    {
        constexpr float r = 0.08f;
        auto* dot = agv_->scene.addEllipse(
            -r, -r, 2.f * r, 2.f * r,
            QPen(QColor(255, 255, 0, 200), 0.01),
            QBrush(QColor(255, 255, 0, 120)));
        dot->setPos(v.x(), v.y());
        dot->setZValue(18);
        path_draw_items_.push_back(dot);
    }

    // Path line segments
    const QPen path_pen(QColor(100, 255, 100), 0.04);
    for (std::size_t i = 0; i + 1 < data.path.size(); ++i)
    {
        auto* line = agv_->scene.addLine(
            data.path[i].x(),     data.path[i].y(),
            data.path[i+1].x(),   data.path[i+1].y(),
            path_pen);
        line->setZValue(20);
        path_draw_items_.push_back(line);
    }

    // Intermediate waypoint dots (cyan)
    const QBrush wp_brush(QColor(0, 220, 220));
    for (std::size_t i = 1; i + 1 < data.path.size(); ++i)
    {
        constexpr float r = 0.06f;
        auto* dot = agv_->scene.addEllipse(-r, -r, 2.f * r, 2.f * r, Qt::NoPen, wp_brush);
        dot->setPos(data.path[i].x(), data.path[i].y());
        dot->setZValue(21);
        path_draw_items_.push_back(dot);
    }

    // Goal / target marker
    const auto& goal = data.path.back();
    if (target_marker_ == nullptr)
    {
        constexpr float tr = 0.12f;
        target_marker_ = agv_->scene.addEllipse(
            -tr, -tr, 2.f * tr, 2.f * tr,
            QPen(QColor(255, 50, 50), 0.03),
            QBrush(QColor(255, 50, 50, 120)));
        target_marker_->setZValue(22);
    }
    target_marker_->setPos(goal.x(), goal.y());
    target_marker_->setVisible(true);
}

void Viewer2D::clear_path_items()
{
    for (auto* item : path_draw_items_)
    {
        agv_->scene.removeItem(item);
        delete item;
    }
    path_draw_items_.clear();

    for (auto* item : obstacle_expanded_items_)
    {
        agv_->scene.removeItem(item);
        delete item;
    }
    obstacle_expanded_items_.clear();

    if (target_marker_ != nullptr)
        target_marker_->setVisible(false);
}

void Viewer2D::update_target_marker(float x, float y, bool visible)
{
    if (epistemic_target_item_ == nullptr)
    {
        constexpr float r = 0.10f;
        epistemic_target_item_ = agv_->scene.addEllipse(
            -r, -r, 2.f * r, 2.f * r,
            QPen(QColor(50, 200, 50), 0.03),
            QBrush(QColor(50, 200, 50, 140)));
        epistemic_target_item_->setZValue(25);
    }
    epistemic_target_item_->setPos(x, y);
    epistemic_target_item_->setVisible(visible);
}

// ─────────────────────────────────────────────────────────────────────────────
// Trajectory overlay — draw the selected arc as a polyline with dots
// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::draw_trajectory(const std::vector<Eigen::Vector3f>& states)
{
    const size_t n = states.size();
    const size_t n_seg = (n > 1) ? n - 1 : 0;

    // Resize line pool
    while (traj_line_items_.size() < n_seg)
    {
        auto* item = agv_->scene.addLine(0, 0, 0, 0,
            QPen(QColor(255, 50, 255), 0.07));
        item->setZValue(35);
        traj_line_items_.push_back(item);
    }
    for (size_t i = 0; i < traj_line_items_.size(); ++i)
        traj_line_items_[i]->setVisible(i < n_seg);

    // Resize dot pool
    while (traj_dot_items_.size() < n)
    {
        constexpr float r = 0.08f;
        auto* item = agv_->scene.addEllipse(-r, -r, 2*r, 2*r,
            QPen(Qt::NoPen), QBrush(QColor(255, 50, 255, 240)));
        item->setZValue(36);
        traj_dot_items_.push_back(item);
    }
    for (size_t i = 0; i < traj_dot_items_.size(); ++i)
        traj_dot_items_[i]->setVisible(i < n);

    // Position items
    for (size_t i = 0; i < n; ++i)
    {
        traj_dot_items_[i]->setPos(states[i][0], states[i][1]);
        if (i > 0)
            traj_line_items_[i - 1]->setLine(
                states[i - 1][0], states[i - 1][1],
                states[i][0], states[i][1]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::draw_all_trajectories(
    const std::vector<std::vector<Eigen::Vector3f>>& candidates,
    const std::vector<Eigen::Vector3f>& best)
{
    // Count total candidate segments
    size_t total_segs = 0;
    for (const auto& c : candidates)
        if (c.size() > 1) total_segs += c.size() - 1;

    // Resize candidate line pool
    while (cand_line_items_.size() < total_segs)
    {
        auto* item = agv_->scene.addLine(0, 0, 0, 0,
            QPen(QColor(180, 130, 255, 70), 0.025));
        item->setZValue(33);
        cand_line_items_.push_back(item);
    }
    for (size_t i = 0; i < cand_line_items_.size(); ++i)
        cand_line_items_[i]->setVisible(i < total_segs);

    // Position candidate line segments
    size_t idx = 0;
    for (const auto& c : candidates)
    {
        for (size_t i = 1; i < c.size(); ++i)
        {
            cand_line_items_[idx]->setLine(
                c[i - 1][0], c[i - 1][1],
                c[i][0], c[i][1]);
            ++idx;
        }
    }

    // Draw the best trajectory on top (thick, opaque)
    draw_trajectory(best);
}

// ─────────────────────────────────────────────────────────────────────────────
// Corner detection markers
// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::draw_rgb_corners(const std::vector<rc::TriplePoint>& points,
                                const Eigen::Affine2f& robot_pose)
{
    // Only points that actually got a depth reading have a room position; range_m < 0 is the
    // "no depth" marker and p_room_meas is left at the origin, which must not be drawn as a corner
    // sitting at the room origin.
    std::vector<const rc::TriplePoint*> shown;
    shown.reserve(points.size());
    // ── Only corners that are actually MATCHED to the model get drawn ────────────────────────────
    // Three conditions, and the last two are new: the canvas is a picture of what the agent believes
    // it is looking at, and a marker for a corner the agent does not believe is a claim it is not
    // making. This is a DISPLAY gate and nothing more — the loss still sees every crossing, weighted
    // by exactly these quantities, because down-weighting inside a mixture and refusing to draw are
    // different jobs and only one of them is inference.
    //   1. a room position at all;
    //   2. the occlusion prior does not disbelieve it (pi_vis) — every polygon vertex projects
    //      somewhere in a 360 image, including the ones behind this room's own walls;
    //   3. the crossing sits within 3 sigma of the model vertex it claims to be, in ITS OWN
    //      covariance. An edge search that locked onto a different structure returns a confident
    //      sub-pixel position that is nowhere near the prediction, and this is what says so.
    int n_occluded = 0, n_unmatched = 0;
    for (const auto& t : points)
    {
        // Gate on the ROOM POSITION, not on range_m. range_m used to come only from the ZED depth
        // stream, so every corner from the panorama was silently dropped here — the markers simply
        // never appeared and nothing said why. p_room_meas is now filled by a ray-plane
        // intersection at the corner's known height, which needs no depth and works for both.
        if (not (t.p_room_meas.allFinite() and t.p_room_meas.squaredNorm() > 1e-12f)) continue;
        // Conditions 2 and 3 now live in rc::visible_and_matched (image_edge_types.h), so the camera
        // overlays apply the SAME rule rather than a second copy of it. Called as its two halves
        // here, and only here, so the log below can say WHICH half withheld a corner — their
        // conjunction is exactly rc::visible_and_matched(t).
        if (not rc::visible_for_display(t)) { ++n_occluded; continue; }
        if (not rc::matched_to_model(t))    { ++n_unmatched; continue; }
        shown.push_back(&t);
    }
    // Silence and "nothing to show" look identical on a canvas, so say what was withheld.
    if (n_occluded + n_unmatched > 0 and points.size() > 0)
    {
        static std::int64_t last_ms = 0;
        const auto now = QDateTime::currentMSecsSinceEpoch();
        if (now - last_ms > 5000)
        {
            last_ms = now;
            qInfo() << "[canvas] RGB corners" << static_cast<int>(shown.size()) << "of"
                    << static_cast<int>(points.size()) << "drawn —" << n_occluded
                    << "behind a wall," << n_unmatched << "not matched to their model vertex";
        }
    }
    const size_t n = shown.size();

    auto resize_pool = [&](auto& pool, size_t count, auto make_item)
    {
        while (pool.size() < count) pool.push_back(make_item());
        for (size_t i = 0; i < pool.size(); ++i) pool[i]->setVisible(i < count);
    };

    // MEASURED corner — solid ORANGE square. Square and orange both deliberate, and now part of one
    // grammar the whole canvas obeys (see the pool declarations in viewer_2d.h): SQUARE = RGB,
    // CIRCLE = LiDAR; FILLED = measured, HOLLOW = model; warm hues = RGB, blues = LiDAR. Magenta is
    // the OBJECT ANCHOR pair further down, which is why these are not magenta.
    resize_pool(rgb_corner_items_, n, [&]() {
        constexpr float r = 0.22f;
        auto* item = agv_->scene.addRect(-r, -r, 2*r, 2*r,
            QPen(QColor(255, 150, 0), 0.03), QBrush(QColor(255, 150, 0, 180)));
        item->setZValue(31);      // above the LiDAR corners, being the smaller marker
        return item;
    });

    // MODEL vertex — hollow DARK BROWN square, the RGB counterpart of the LiDAR's hollow dark-blue
    // circle. The residual line below already ran to this point, but nothing marked it, so the line
    // ended in mid-air and the pair "where the image says / where the model says" could not be read
    // off the canvas the way the LiDAR pair can. Drawn LARGER than the measured square, the same
    // ordering the LiDAR pair uses (0.35 hollow around 0.30 filled), so the measurement sits inside
    // its prediction rather than beside it.
    resize_pool(rgb_model_items_, n, [&]() {
        constexpr float r = 0.27f;
        auto* item = agv_->scene.addRect(-r, -r, 2*r, 2*r,
            QPen(QColor(101, 67, 33), 0.05), QBrush(Qt::NoBrush));
        item->setZValue(30);      // under the measured square, over the LiDAR markers
        return item;
    });

    // Line from the MODEL vertex to where the image says the corner is. This is the residual drawn
    // at true scale — the quantity the whole mount calibration is about, in metres on the canvas.
    // ★ This pool is the RESIDUAL, not a sight line: it spans p_room (the model vertex) → p_room_meas
    //   (the crossing). The robot→corner line below is a SECOND pool, kept separate so neither
    //   meaning is quietly overwritten by the other.
    resize_pool(rgb_corner_line_items_, n, [&]() {
        auto* item = agv_->scene.addLine(0, 0, 0, 0, QPen(QColor(255, 150, 0, 120), 0.02));
        item->setZValue(28);
        return item;
    });

    // Sight line robot → measured RGB corner, the counterpart of corner_robot_line_items_ for the
    // LiDAR corners, and drawn on the SAME visual scale: SOLID, with the width carrying the corner's
    // own positional 1-sigma under the identical law draw_corners uses — same expression, same
    // constants, same metres. What separates the two channels is HUE alone: these are the RGB term's
    // orange, the LiDAR ones ramp green→red with their sigma. Sharing the width law is the whole
    // point — a fat line must mean the same size of doubt in both channels, which it could not while
    // one of them was a fixed thin dot-dash encoding nothing.
    // ★ The width needs a METRIC covariance and cov_uv is in pixels, so the pen is set per corner
    //   below from TriplePoint::cov_room — cov_uv propagated through the ray-plane intersection
    //   itself (SpecificWorker::place_triple_points_in_room). Where that propagation did not run the
    //   line falls back to DOTTED at the base width, this canvas's existing idiom for "no number is
    //   encoded here", rather than drawing the hairline a zero covariance would otherwise imply.
    resize_pool(rgb_corner_robot_line_items_, n, [&]() {
        auto* item = agv_->scene.addLine(0, 0, 0, 0, QPen(QColor(255, 150, 0, 110), 0.025));
        item->setZValue(26);   // under the LiDAR sight lines (27), which carry the localiser's weight
        return item;
    });

    const Eigen::Vector2f robot_xy = robot_pose.translation();

    for (size_t i = 0; i < n; ++i)
    {
        const auto& t = *shown[i];
        rgb_corner_items_[i]->setPos(t.p_room_meas.x(), t.p_room_meas.y());
        rgb_model_items_[i]->setPos(t.p_room.x(), t.p_room.y());
        rgb_corner_line_items_[i]->setLine(t.p_room.x(), t.p_room.y(),
                                           t.p_room_meas.x(), t.p_room_meas.y());
        rgb_corner_robot_line_items_[i]->setLine(robot_xy.x(), robot_xy.y(),
                                                 t.p_room_meas.x(), t.p_room_meas.y());
        // ── The marker carries the model's own belief in it, on two independent axes ─────────────
        // CONDITIONING: a vertex seen edge-on has a poorly defined crossing, and its marker should
        // not look as authoritative as a well-conditioned one.
        // VISIBILITY: pi_vis is the occlusion prior the mixture ALREADY applied to this corner's
        // samples. Every polygon vertex projects somewhere in a 360 image, including the ones behind
        // this room's own walls — a non-convex plan hides half of itself from any single viewpoint —
        // so drawing them all identically painted a room with transparent walls. The loss was never
        // fooled (an occluded corner's samples are downweighted, its w collapses and cov_uv blows
        // up), but the CANVAS was, and a display that hides the model's doubt is how a term ends up
        // trusted more than it earned.
        // Everything reaching here is past the visibility and match gates above, so the remaining
        // fade is a matter of degree, not of belief: how well conditioned the crossing is, and how
        // clear its line of sight was.
        const double q_cond = std::clamp(2.0 / std::max(1.0, static_cast<double>(t.cond)), 0.25, 1.0);
        const double vis    = std::clamp(static_cast<double>(t.pi_vis), 0.0, 1.0);
        const double q      = q_cond * vis;
        rgb_corner_items_[i]->setOpacity(q);
        rgb_model_items_[i]->setOpacity(q);
        rgb_corner_line_items_[i]->setOpacity(q);
        rgb_corner_robot_line_items_[i]->setOpacity(q);

        // ── Sight-line width = this corner's own 1-sigma, on the LiDAR channel's scale ───────────
        // The same law draw_corners applies to a LiDAR corner: the characteristic 1-sigma length is
        // (det cov)^{1/4}, the geometric mean of the two sigma axes in metres, and the width is
        // linear in it over the same clamp. cov_room is metric by construction, so a fat orange line
        // and a fat green one stand for the same size of doubt — which is the only reason to share
        // an encoding at all.
        const Eigen::Matrix2f Cr = t.cov_room;
        const float det_room = Cr(0, 0) * Cr(1, 1) - Cr(0, 1) * Cr(1, 0);
        if (std::isfinite(det_room) and det_room > 0.f)
        {
            const float sigma_scale = std::pow(std::max(det_room, 1e-12f), 0.25f);
            const float width = std::clamp(0.03f + 0.9f * sigma_scale, 0.03f, 0.9f);
            rgb_corner_robot_line_items_[i]->setPen(QPen(QColor(255, 150, 0, 110), width));
        }
        else
        {   // No metric covariance reached us. Say that, rather than draw the hairline a zero
            // covariance would produce and let it read as a perfectly located corner.
            QPen pen(QColor(255, 150, 0, 110), 0.025);
            pen.setStyle(Qt::DotLine);
            rgb_corner_robot_line_items_[i]->setPen(pen);
        }
    }
}

void Viewer2D::draw_corners(const std::vector<rc::CornerDetector::CornerMatch>& matches,
                             const Eigen::Affine2f& robot_pose)
{
    // ── Only corners that are actually MATCHED to a model vertex get drawn ───────────────────────
    // ONE decision, and it is NOT taken here: rc::CornerDetector::matched_for_display (corner_detector.h)
    // is the single rule, so this canvas and the camera overlays cannot drift apart. A marker for a
    // corner the agent does not believe is a claim it is not making, and this canvas is read as
    // evidence of what the localiser is using. DISPLAY only: the loss still sees every match,
    // weighted by exactly the same quantities.
    //
    // The counters below only ATTRIBUTE a rejection the predicate has already made — they decide
    // nothing, and they mirror the predicate's own two conditions in its own order:
    //   • no model vertex / non-finite detection  → n_unmatched;
    //   • assoc_prob below a coin flip            → n_ambiguous (the detection is the best of several
    //     guesses rather than a match to the vertex it claims).
    // There is no chi2 reason to report: the association gate bounds assoc_chi2_val upstream
    // (see the note on matched_for_display), so a residual test here could never withhold a marker.
    std::vector<const rc::CornerDetector::CornerMatch*> shown;
    shown.reserve(matches.size());
    int n_ambiguous = 0, n_unmatched = 0;
    for (const auto& m : matches)
    {
        if (rc::CornerDetector::matched_for_display(m)) { shown.push_back(&m); continue; }
        if (m.model_index < 0 or not m.detected.allFinite()) ++n_unmatched;
        else                                                 ++n_ambiguous;
    }
    // Silence and "nothing to show" look identical on a canvas, so say what was withheld.
    if (n_ambiguous + n_unmatched > 0 and not matches.empty())
    {
        static std::int64_t last_ms = 0;
        const auto now = QDateTime::currentMSecsSinceEpoch();
        if (now - last_ms > 5000)
        {
            last_ms = now;
            qInfo() << "[canvas] LiDAR corners" << static_cast<int>(shown.size()) << "of"
                    << static_cast<int>(matches.size()) << "drawn —" << n_ambiguous
                    << "ambiguous association," << n_unmatched << "not matched to a model vertex";
        }
    }
    const size_t n = shown.size();

    // Helper: ensure a pool has exactly `count` items, hiding extras
    auto resize_pool = [&](auto& pool, size_t count, auto make_item)
    {
        while (pool.size() < count)
            pool.push_back(make_item());
        for (size_t i = 0; i < pool.size(); ++i)
            pool[i]->setVisible(i < count);
    };

    // Detected corners — solid cyan dots in world frame
    resize_pool(corner_detected_items_, n, [&]() {
        constexpr float r = 0.30f;
        auto* item = agv_->scene.addEllipse(-r, -r, 2*r, 2*r,
            QPen(QColor(0, 220, 255), 0.03), QBrush(QColor(0, 220, 255, 200)));
        item->setZValue(30);
        return item;
    });

    // MODEL corners — hollow DARK BLUE circles in world frame. Blue, not orange: hue names the
    // CHANNEL on this canvas, and orange belongs to the RGB pair. An orange ring around a cyan dot
    // put two channels' colours on one measurement and read as a cross-sensor pairing that is not
    // what it is — this ring is the LiDAR's own model vertex, so it stays inside the LiDAR's blues.
    resize_pool(corner_predicted_items_, n, [&]() {
        constexpr float r = 0.35f;
        auto* item = agv_->scene.addEllipse(-r, -r, 2*r, 2*r,
            QPen(QColor(20, 60, 170), 0.06), QBrush(Qt::NoBrush));
        item->setZValue(29);
        return item;
    });

    // Residual line, model → detected. Follows its own channel for the same reason the ring does.
    resize_pool(corner_line_items_, n, [&]() {
        QPen pen(QColor(20, 60, 170, 200), 0.03);
        auto* item = agv_->scene.addLine(0, 0, 0, 0, pen);
        item->setZValue(28);
        return item;
    });

    // Lines connecting robot → detected corner (sight lines). Width + colour encode the detection
    // covariance determinant (uncertainty); the pen is re-styled per corner below.
    resize_pool(corner_robot_line_items_, n, [&]() {
        auto* item = agv_->scene.addLine(0, 0, 0, 0, QPen(QColor(0, 0, 139), 0.08));
        item->setZValue(27);
        return item;
    });

    const Eigen::Matrix2f R = robot_pose.linear();
    const Eigen::Vector2f t = robot_pose.translation();

    for (size_t i = 0; i < n; ++i)
    {
        const auto& m = *shown[i];

        // Detected: transform from robot frame to world using display pose
        const Eigen::Vector2f det_world = R * m.detected + t;
        // Predicted: use known model world position (exact, no lag)
        const Eigen::Vector2f pred_world = m.model_world;

        corner_detected_items_[i]->setPos(det_world.x(), det_world.y());
        corner_predicted_items_[i]->setPos(pred_world.x(), pred_world.y());
        corner_line_items_[i]->setLine(pred_world.x(), pred_world.y(),
                                        det_world.x(), det_world.y());
        corner_robot_line_items_[i]->setLine(t.x(), t.y(),
                                             det_world.x(), det_world.y());

        // ── Covariance-determinant encoding ────────────────────────────────────────────────────
        // cov = Λ_det⁻¹. det(cov) = 1/det(Λ_det). A shallow (rank-1) corner has det(Λ)→0 ⇒ det(cov)→∞,
        // so invert via the eigenvalues with a precision floor and cap the covariance eigenvalues.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(m.information);
        Eigen::Vector2f lam = es.eigenvalues();                 // Λ_det eigenvalues (precision, 1/m²)
        constexpr float kPrecFloor = 1e-3f;                     // → σ ≤ ~31 m for an unconstrained axis
        const float cov0 = 1.f / std::max(lam(0), kPrecFloor);  // covariance eigenvalues (m²)
        const float cov1 = 1.f / std::max(lam(1), kPrecFloor);
        const float det_cov = cov0 * cov1;                      // m⁴  (= 1/det(Λ_det))
        // Characteristic 1σ length = (det cov)^{1/4} = geometric-mean of the two σ axes (metres).
        const float sigma_scale = std::pow(std::max(det_cov, 1e-12f), 0.25f);

        // Line width grows with uncertainty (thin = confident, fat = uncertain), clamped to sane px-in-m.
        const float width = std::clamp(0.03f + 0.9f * sigma_scale, 0.03f, 0.9f);
        // Colour: green (confident, small σ) → red (uncertain, large σ) over ~[0.02, 0.6] m.
        const float u = std::clamp((sigma_scale - 0.02f) / (0.60f - 0.02f), 0.f, 1.f);
        QColor col(static_cast<int>(255 * u), static_cast<int>(200 * (1.f - u)), 60, 220);

        // A RETIRED corner (information yield never materialised — see CornerDetector's yield rule) is
        // still detected and still shown, because a landmark that silently disappears reads as a broken
        // detector. It must not read as a live one either: draw it dim, thin and DASHED so it is
        // obvious at a glance that it is being watched but is not voting.
        if (m.suppressed)
        {
            col = QColor(150, 150, 150, 110);
            QPen pen(col, 0.03f);
            pen.setStyle(Qt::DashLine);
            corner_robot_line_items_[i]->setPen(pen);
        }
        else
            corner_robot_line_items_[i]->setPen(QPen(col, width));

        corner_detected_items_[i]->setOpacity(m.suppressed ? 0.25 : 1.0);
        corner_predicted_items_[i]->setOpacity(m.suppressed ? 0.25 : 1.0);
        corner_line_items_[i]->setOpacity(m.suppressed ? 0.25 : 1.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Landmark sight lines — robot → each pinned-object landmark (room frame)
// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::draw_landmark_lines(const std::vector<Eigen::Vector2f>& landmarks_world,
                                   const std::vector<char>& measured,
                                   const Eigen::Vector2f& robot_xy)
{
    const size_t n = landmarks_world.size();

    auto resize_pool = [&](auto& pool, size_t count, auto make_item)
    {
        while (pool.size() < count)
            pool.push_back(make_item());
        for (size_t i = 0; i < pool.size(); ++i)
            pool[i]->setVisible(i < count);
    };

    // Object "being measured" this frame? Missing/short flag ⇒ treat as measured (draw the line).
    const auto is_measured = [&](size_t i) { return i >= measured.size() or measured[i] != 0; };

    // Dark-blue thick line from robot to each pinned landmark
    resize_pool(landmark_line_items_, n, [&]() {
        QPen pen(QColor(0, 0, 139), 0.08);   // dark blue, thick
        auto* item = agv_->scene.addLine(0, 0, 0, 0, pen);
        item->setZValue(27);
        return item;
    });

    // Dark-blue marker at each landmark position
    resize_pool(landmark_marker_items_, n, [&]() {
        constexpr float r = 0.20f;
        auto* item = agv_->scene.addEllipse(-r, -r, 2*r, 2*r,
            QPen(QColor(0, 0, 139), 0.03), QBrush(QColor(0, 0, 139, 180)));
        item->setZValue(31);
        return item;
    });

    for (size_t i = 0; i < n; ++i)
    {
        const auto& lw = landmarks_world[i];
        landmark_line_items_[i]->setLine(robot_xy.x(), robot_xy.y(), lw.x(), lw.y());
        landmark_line_items_[i]->setVisible(is_measured(i));   // sight line only while measured
        landmark_marker_items_[i]->setPos(lw.x(), lw.y());     // marker always shown
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wall-SLAM overlay — segments, wall landmarks, derived corners, derived polygon
// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::draw_wall_map(const std::vector<rc::wallseg::WallSegment>& segments,
                             const std::vector<rc::wallmap::WallLandmark>& walls,
                             const rc::wallmap::Polygon& polygon, bool map_ready,
                             const Eigen::Affine2f& robot_pose)
{
    auto resize_pool = [&](auto& pool, size_t count, auto make_item)
    {
        while (pool.size() < count) pool.push_back(make_item());
        for (size_t i = 0; i < pool.size(); ++i) pool[i]->setVisible(i < count);
    };

    // This frame's segments: thin cyan, in the room frame through the pose.
    resize_pool(wall_seg_items_, segments.size(), [&]() {
        auto* item = agv_->scene.addLine(0, 0, 0, 0, QPen(QColor(0, 200, 220), 0.05));
        item->setZValue(26);
        return item;
    });
    for (size_t i = 0; i < segments.size(); ++i)
    {
        const Eigen::Vector2f a = robot_pose * segments[i].p0;
        const Eigen::Vector2f b = robot_pose * segments[i].p1;
        wall_seg_items_[i]->setLine(a.x(), a.y(), b.x(), b.y());
    }

    // Wall landmarks: thick lines over their observed extent; colour by Manhattan class, grey when
    // class-less (a chamfer, or born before θ₀ existed).
    static const QColor kClass[4] = {QColor(200, 40, 40), QColor(40, 160, 40), QColor(40, 80, 220), QColor(220, 140, 0)};
    resize_pool(wall_line_items_, walls.size(), [&]() {
        auto* item = agv_->scene.addLine(0, 0, 0, 0, QPen(Qt::gray, 0.12));
        item->setZValue(28);
        return item;
    });
    resize_pool(wall_label_items_, walls.size(), [&]() {
        auto* item = agv_->scene.addSimpleText("");
        item->setZValue(35);
        QFont f = item->font(); f.setPointSizeF(0.35); item->setFont(f);
        return item;
    });
    for (size_t i = 0; i < walls.size(); ++i)
    {
        const auto& w = walls[i];
        const Eigen::Vector2f n = w.normal(), t = w.tangent();
        const float s0 = w.has_extent ? w.s_min : -1.f, s1 = w.has_extent ? w.s_max : 1.f;
        const Eigen::Vector2f a = n * w.d + t * s0, b = n * w.d + t * s1;
        wall_line_items_[i]->setLine(a.x(), a.y(), b.x(), b.y());
        wall_line_items_[i]->setPen(QPen(w.k >= 0 ? kClass[w.k] : QColor(120, 120, 120), 0.12));
        const Eigen::Vector2f mid = (a + b) * 0.5f + n * 0.25f;
        wall_label_items_[i]->setText(QString("w%1 k=%2%3").arg(static_cast<qulonglong>(w.id))
                                          .arg(w.k).arg(w.room_factor_dF > 4.6f ? QString(" dF=%1").arg(w.room_factor_dF, 0, 'f', 1) : ""));
        wall_label_items_[i]->setPos(mid.x(), mid.y());
        wall_label_items_[i]->setTransform(QTransform::fromScale(1, -1));   // scene y is up
    }

    // Derived corners: filled when observed meeting, hollow when inferred by intersection.
    resize_pool(wall_corner_items_, polygon.corners.size(), [&]() {
        constexpr float r = 0.12f;
        auto* item = agv_->scene.addEllipse(-r, -r, 2 * r, 2 * r, QPen(QColor(200, 0, 200), 0.04), QBrush(Qt::NoBrush));
        item->setZValue(32);
        return item;
    });
    for (size_t i = 0; i < polygon.corners.size(); ++i)
    {
        const auto& c = polygon.corners[i];
        wall_corner_items_[i]->setPos(c.p.x(), c.p.y());
        wall_corner_items_[i]->setBrush(c.inferred ? QBrush(Qt::NoBrush) : QBrush(QColor(200, 0, 200, 160)));
    }

    // Derived polygon: dashed magenta until published, solid after.
    if (wall_poly_item_ == nullptr)
    {
        wall_poly_item_ = agv_->scene.addPolygon(QPolygonF(), QPen(Qt::magenta, 0.10), QBrush(Qt::NoBrush));
        wall_poly_item_->setZValue(9);
    }
    if (polygon.closed and polygon.verts.size() >= 3)
    {
        QPolygonF poly;
        for (const auto& v : polygon.verts) poly << QPointF(v.x(), v.y());
        poly << QPointF(polygon.verts.front().x(), polygon.verts.front().y());
        wall_poly_item_->setPolygon(poly);
        QPen pen(Qt::magenta, 0.10);
        pen.setStyle(map_ready ? Qt::SolidLine : Qt::DashLine);
        wall_poly_item_->setPen(pen);
        wall_poly_item_->setVisible(true);
    }
    else
        wall_poly_item_->setVisible(false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Score grid overlay — soft-toned coloured rectangles per cell
// ─────────────────────────────────────────────────────────────────────────────
void Viewer2D::draw_score_grid(const std::vector<std::pair<Eigen::Vector2f, float>>& cells,
                               float cell_size)
{
    // Find max score for normalisation
    float max_score = 1e-9f;
    for (const auto& [center, score] : cells)
        max_score = std::max(max_score, score);

    const std::size_t n = cells.size();

    // Grow pool if needed
    while (score_grid_items_.size() < n)
    {
        auto* item = agv_->scene.addRect(0, 0, cell_size, cell_size);
        item->setZValue(-5);          // behind everything else
        item->setPen(Qt::NoPen);
        score_grid_items_.push_back(item);
    }

    // Update visible items
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& [center, score] = cells[i];
        const float t = score / max_score;             // 0..1

        // Interpolate:  low = pale blue (cool)  →  high = warm orange/red
        const int r = static_cast<int>(30 + 225 * t);
        const int g = static_cast<int>(120 - 60 * t);
        const int b = static_cast<int>(200 * (1.f - t));
        const int a = static_cast<int>(40 + 60 * t);  // soft transparency

        auto* item = score_grid_items_[i];
        item->setRect(center.x() - cell_size * 0.5f,
                      center.y() - cell_size * 0.5f,
                      cell_size, cell_size);
        item->setBrush(QColor(r, g, b, a));
        item->setVisible(true);
    }

    // Hide excess items
    for (std::size_t i = n; i < score_grid_items_.size(); ++i)
        score_grid_items_[i]->setVisible(false);
}

void Viewer2D::draw_ior_grid(const std::vector<std::pair<Eigen::Vector2f, float>>& cells,
                              float cell_size)
{
    const std::size_t n = cells.size();

    while (ior_grid_items_.size() < n)
    {
        auto* item = agv_->scene.addRect(0, 0, cell_size, cell_size);
        item->setZValue(-4);   // above score grid (z=-5), below lidar/robot
        item->setPen(Qt::NoPen);
        ior_grid_items_.push_back(item);
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& [center, freshness] = cells[i];
        // Never visited / fully stale (f=0) → dark exploration fog
        // Just visited                (f=1) → bright warm light
        // Smooth gradient encodes coverage: the path the robot has taken lights up
        const float f = freshness;  // 0..1
        // Never visited (f=0) → dark blue fog;  just visited (f=1) → bright sky-blue
        const int r = static_cast<int>( 10 +  70 * f);  //  10.. 80
        const int g = static_cast<int>( 15 + 185 * f);  //  15..200
        const int b = static_cast<int>( 25 + 230 * f);  //  25..255
        const int a = static_cast<int>( 60 + 140 * f);  //  60..200
        auto* item = ior_grid_items_[i];
        item->setRect(center.x() - cell_size * 0.5f,
                      center.y() - cell_size * 0.5f,
                      cell_size, cell_size);
        item->setBrush(QColor(r, g, b, a));
        item->setVisible(true);
    }

    for (std::size_t i = n; i < ior_grid_items_.size(); ++i)
        ior_grid_items_[i]->setVisible(false);
}

void Viewer2D::draw_selected_grid_cell(const std::optional<Eigen::Vector2f>& center,
                                       float cell_size)
{
    if (!center.has_value())
    {
        if (selected_grid_cell_item_ != nullptr)
            selected_grid_cell_item_->setVisible(false);
        return;
    }

    if (selected_grid_cell_item_ == nullptr)
    {
        selected_grid_cell_item_ = agv_->scene.addRect(0, 0, cell_size, cell_size);
        selected_grid_cell_item_->setPen(QPen(QColor(255, 255, 80, 230), 0.06));
        selected_grid_cell_item_->setBrush(QBrush(QColor(255, 255, 80, 40)));
        selected_grid_cell_item_->setZValue(26);
    }

    selected_grid_cell_item_->setRect(center->x() - cell_size * 0.5f,
                                      center->y() - cell_size * 0.5f,
                                      cell_size, cell_size);
    selected_grid_cell_item_->setVisible(true);
}

namespace
{
// Furniture concept nodes are now generic type()=="object"; the class is carried in object_subtype (name
// prefix table_*, chair_*, … unchanged). Return that class, or "" for a generic/unclassified object.
std::string object_node_class(DSR::DSRGraph& G, const DSR::Node& n)
{
    if (const auto s = G.get_attrib_by_name<object_subtype_att>(n); s.has_value() and not s.value().empty())
        return s.value();
    for (std::string_view p : {"table", "chair", "bottle", "cabinet", "refrigerator"})
        if (std::string_view(n.name()).starts_with(p))
            return std::string(p);
    return {};
}
}  // namespace

void Viewer2D::refresh_semantic_bboxes(const std::shared_ptr<DSR::DSRGraph>& graph)
{
    // Poll DSR nodes (object/obstacle) on the main thread and draw their oriented BBs. Furniture is now
    // generic "object" (class in object_subtype); tables get their own coloured layer via a class filter.
    // Cadence is driven by the overlay's QTimer (no internal throttle).
    auto clear_map = [this](std::unordered_map<std::uint64_t, QGraphicsPolygonItem*>& items)
    {
        for (auto& [id, item] : items)
        {
            Q_UNUSED(id);
            if (item != nullptr)
            {
                agv_->scene.removeItem(item);
                delete item;
            }
        }
        items.clear();
    };

    if (!graph)
    {
        clear_map(object_bbox_items_);
        clear_map(obstacle_bbox_items_);
        clear_map(table_bbox_items_);
        return;
    }

    const auto rt_api = graph->get_rt_api();
    if (!rt_api)
        return;

    // want_class: "" ⇒ draw only GENERIC objects (no recognised furniture class), so classed nodes
    // (tables, …) don't double-draw under the generic layer; non-empty ⇒ draw only that class.
    auto refresh_type = [this, &graph, &rt_api](const std::string& type,
                                                 std::string_view want_class,
                                                 std::unordered_map<std::uint64_t, QGraphicsPolygonItem*>& items,
                                                 const QColor& stroke,
                                                 const QColor& fill,
                                                 qreal z)
    {
        std::unordered_set<std::uint64_t> seen;
        for (const auto& node : graph->get_nodes_by_type(type))
        {
            const std::string cls = object_node_class(*graph, node);
            if (want_class.empty()) { if (not cls.empty()) continue; }   // generic objects only
            else if (cls != want_class)               continue;         // a specific class layer

            const auto w_opt = graph->get_attrib_by_name<width_m_att>(node);
            const auto d_opt = graph->get_attrib_by_name<depth_m_att>(node);
            if (!w_opt.has_value() || !d_opt.has_value())
                continue;

            const float width = w_opt.value();
            const float depth = d_opt.value();
            if (width <= 0.f || depth <= 0.f)
                continue;

            const auto rt_opt = rt_api->get_RT_pose_from_parent(node);
            if (!rt_opt.has_value())
                continue;

            const Eigen::Vector3d tr = rt_opt->translation();
            const float yaw = static_cast<float>(std::atan2(rt_opt->linear()(1, 0), rt_opt->linear()(0, 0)));
            const float c = std::cos(yaw);
            const float s = std::sin(yaw);
            const float hw = 0.5f * width;
            const float hd = 0.5f * depth;

            auto rot_tr = [c, s, tr](float lx, float ly)
            {
                return QPointF(tr.x() + c * lx - s * ly,
                               tr.y() + s * lx + c * ly);
            };

            QPolygonF poly;
            poly << rot_tr(-hw, -hd)
                 << rot_tr( hw, -hd)
                 << rot_tr( hw,  hd)
                 << rot_tr(-hw,  hd)
                 << rot_tr(-hw, -hd);

            auto it = items.find(node.id());
            if (it == items.end())
            {
                auto* item = agv_->scene.addPolygon(poly,
                                                    QPen(stroke, 0.05),
                                                    QBrush(fill));
                item->setOpacity(1.0);
                item->setZValue(z);
                items.emplace(node.id(), item);
            }
            else
            {
                it->second->setPolygon(poly);
                it->second->setOpacity(1.0);
                it->second->setVisible(true);
            }

            seen.insert(node.id());
        }

        for (auto it = items.begin(); it != items.end(); )
        {
            if (!seen.contains(it->first))
            {
                if (it->second != nullptr)
                {
                    agv_->scene.removeItem(it->second);
                    delete it->second;
                }
                it = items.erase(it);
            }
            else
            {
                ++it;
            }
        }
    };

    // Generic objects: blue. Obstacles: green. Tables: orange. Translucent fill so overlaps stay readable.
    // All furniture is type "object" now (class in object_subtype); the table layer filters that class out
    // of the generic-object pass so tables draw once, in orange.
    refresh_type("object",   "",      object_bbox_items_,   QColor(0, 160, 255), QColor(0, 160, 255, 60), 26);
    refresh_type("obstacle", "",      obstacle_bbox_items_, QColor(0, 200, 80),  QColor(0, 200, 80, 60),  27);
    refresh_type("object",   "table", table_bbox_items_,    QColor(255, 140, 0), QColor(255, 140, 0, 60), 28);
}

void Viewer2D::start_semantic_bbox_overlay(std::shared_ptr<DSR::DSRGraph> graph, int period_ms)
{
    semantic_graph_ = std::move(graph);
    if (semantic_bbox_timer_ == nullptr)
    {
        // Parented to this viewer (a QObject living on the GUI thread), so timeout()
        // is delivered on the GUI thread — the only thread allowed to touch the scene
        // and the safe thread for the DSR reads. No locks needed.
        semantic_bbox_timer_ = new QTimer(this);
        connect(semantic_bbox_timer_, &QTimer::timeout, this,
                [this]() { refresh_semantic_bboxes(semantic_graph_); });
    }
    semantic_bbox_timer_->start(period_ms);
}


// ─────────────────────────────────────────────────────────────────────────────────────────────
//  Object-anchor overlay (fridge, …)
//
//  Four things, because three of them are only meaningful together:
//    p_o   the PINNED map anchor — fixed the moment the object's map σ dropped below validateSigma
//          and the localizer was sustained-stable. It never moves again.
//    z_o   this frame's observation, carried from the robot frame to world by the DISPLAY pose. It
//          is the raw camera-frame mask centroid, so it moves with the robot's belief, not with p_o.
//    the sight line robot→z_o, and the residual z_o→p_o. That residual is exactly what the landmark
//          factor is minimising: if it grows, the anchor is fighting the SDF, which is the failure
//          a wrong pin produces and the reason to be able to SEE it rather than infer it from a CSV.
//
//  The covariance is drawn as an oriented 1σ ellipse of S = Λ⁻¹, not as a scalar. The producer
//  publishes an ANISOTROPIC R_o whose loose axis lies along the viewing ray, so the ellipse's
//  elongation and direction are the whole point — a determinant would average that away.
// ─────────────────────────────────────────────────────────────────────────────────────────────
void Viewer2D::draw_object_anchors(const std::vector<rc::ObjectAnchorObs>& anchors,
                                    const Eigen::Affine2f& robot_pose,
                                    const std::map<std::uint64_t, Eigen::Vector2f>& landmarks)
{
    const size_t n = anchors.size();

    auto resize_pool = [&](auto& pool, size_t count, auto make_item)
    {
        while (pool.size() < count)
            pool.push_back(make_item());
        for (size_t i = 0; i < pool.size(); ++i)
            pool[i]->setVisible(i < count);
    };

    // Pinned map anchor p_o — hollow magenta square-ish ring, deliberately unlike the corner markers.
    resize_pool(anchor_pin_items_, n, [&]() {
        constexpr float r = 0.22f;
        auto* item = agv_->scene.addEllipse(-r, -r, 2*r, 2*r,
            QPen(QColor(255, 0, 200), 0.07), QBrush(Qt::NoBrush));
        item->setZValue(33);
        return item;
    });

    // Observation z_o — solid magenta dot.
    resize_pool(anchor_obs_items_, n, [&]() {
        constexpr float r = 0.12f;
        auto* item = agv_->scene.addEllipse(-r, -r, 2*r, 2*r,
            QPen(QColor(255, 0, 200), 0.03), QBrush(QColor(255, 0, 200, 190)));
        item->setZValue(34);
        return item;
    });

    // 1σ innovation ellipse at z_o (sized/rotated per anchor below).
    resize_pool(anchor_cov_items_, n, [&]() {
        auto* item = agv_->scene.addEllipse(-1, -1, 2, 2,
            QPen(QColor(255, 0, 200, 160), 0.04), QBrush(QColor(255, 0, 200, 40)));
        item->setZValue(32);
        return item;
    });

    // Sight line robot → z_o.
    resize_pool(anchor_sight_items_, n, [&]() {
        auto* item = agv_->scene.addLine(0, 0, 0, 0, QPen(QColor(255, 0, 200, 170), 0.05));
        item->setZValue(31);
        return item;
    });

    // Residual z_o → p_o, dashed: the quantity the factor pulls on.
    resize_pool(anchor_resid_items_, n, [&]() {
        QPen pen(QColor(255, 220, 0, 230), 0.06);
        pen.setStyle(Qt::DashLine);
        auto* item = agv_->scene.addLine(0, 0, 0, 0, pen);
        item->setZValue(35);
        return item;
    });

    resize_pool(anchor_text_items_, n, [&]() {
        auto* item = agv_->scene.addText("");
        item->setZValue(36);
        item->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
        QFont f = item->font(); f.setPointSizeF(7.0); item->setFont(f);
        return item;
    });

    // Room's private landmark estimate — white ring — and a link to the producer's published pose.
    // Room never writes this back, so the two drift apart freely; that gap is exactly what a private
    // estimate buys over a merge, and it is worth one more marker to be able to see it.
    resize_pool(anchor_lm_items_, n, [&]() {
        constexpr float r = 0.16f;
        auto* item = agv_->scene.addEllipse(-r, -r, 2*r, 2*r,
            QPen(QColor(255, 255, 255), 0.05), QBrush(Qt::NoBrush));
        item->setZValue(37);
        return item;
    });
    resize_pool(anchor_lm_link_items_, n, [&]() {
        QPen pen(QColor(255, 255, 255, 150), 0.03);
        pen.setStyle(Qt::DotLine);
        auto* item = agv_->scene.addLine(0, 0, 0, 0, pen);
        item->setZValue(36);
        return item;
    });

    const Eigen::Matrix2f R = robot_pose.linear();
    const Eigen::Vector2f t = robot_pose.translation();

    for (size_t i = 0; i < n; ++i)
    {
        const auto& a = anchors[i];
        if (const auto lit = landmarks.find(a.node_id); lit != landmarks.end())
        {
            anchor_lm_items_[i]->setVisible(true);
            anchor_lm_link_items_[i]->setVisible(true);
            anchor_lm_items_[i]->setPos(lit->second.x(), lit->second.y());
            anchor_lm_link_items_[i]->setLine(lit->second.x(), lit->second.y(),
                                              a.pose_world.x(), a.pose_world.y());
        }
        else
        {
            anchor_lm_items_[i]->setVisible(false);
            anchor_lm_link_items_[i]->setVisible(false);
        }
        const Eigen::Vector2f p_o = a.pose_world.head<2>();              // pinned, world
        const Eigen::Vector2f z_w = R * a.obs_robot.head<2>() + t;       // observation → world

        anchor_pin_items_[i]->setPos(p_o.x(), p_o.y());
        anchor_obs_items_[i]->setPos(z_w.x(), z_w.y());
        anchor_sight_items_[i]->setLine(t.x(), t.y(), z_w.x(), z_w.y());
        anchor_resid_items_[i]->setLine(z_w.x(), z_w.y(), p_o.x(), p_o.y());

        // S = Λ⁻¹ on the POSITION block. Λ is built as (Σ_o ⊕ R_o)⁻¹, and for a position-only landmark
        // its yaw row/col are zero, so only the 2×2 is meaningful. A near-singular Λ (an axis the fit
        // does not constrain) would invert to an enormous axis — floor the precision instead of letting
        // the ellipse escape the canvas, the same way draw_corners does.
        const Eigen::Matrix2f info = a.information.topLeftCorner<2, 2>();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(info);
        constexpr float kPrecFloor = 1e-3f;
        // Eigen returns eigenvalues in ASCENDING order, and these are PRECISIONS: the smallest one is
        // the largest sigma. s_major therefore comes from eigenvalue(0), not (1).
        const float s_major = std::sqrt(1.f / std::max(es.eigenvalues()(0), kPrecFloor));   // 1σ, metres
        const float s_minor = std::sqrt(1.f / std::max(es.eigenvalues()(1), kPrecFloor));
        // Λ lives in the ROBOT frame -- the residual it weights is r = z_o − R(−θ)(p_o − t), which is a
        // robot-frame quantity. The canvas is the ROOM frame. So the ellipse has to be carried over:
        // S_world = R·S_robot·Rᵀ, which leaves the eigenvalues alone and rotates the eigenvectors by R.
        // Drawing it without that rotation is a yaw-sized error, and at the yaw in fridge_2.png it put
        // the loose axis across the ray instead of along it -- the exact opposite of what R_o encodes.
        const Eigen::Vector2f v_robot = es.eigenvectors().col(0);       // eigvals ascending ⇒ col(0) is
                                                                       // the SMALLEST precision = the
                                                                       // LARGEST σ = the loose axis
        const Eigen::Vector2f v_world = R * v_robot;
        const float ang_deg = static_cast<float>(std::atan2(v_world.y(), v_world.x()) * 180.0 / M_PI);

        // ── Drawn size vs REPORTED size ────────────────────────────────────────────────────────
        // An anchor whose observation has gone stale is muted ON PURPOSE: freshness-as-precision
        // inflates R_o by (1+age/ageScale)^2, so sigma reaches metres within a second of the object
        // leaving view and Lambda -> 0. That is the model behaving correctly. Drawing it literally is
        // not: a 5 m ellipse buries the entire canvas, and silently clamping it makes the LABEL lie
        // about the number it is displaying.
        //
        // So the drawn ellipse is capped and the label always reports the TRUE sigma, flagged when the
        // drawing saturated. The cap is a DISPLAY cap -- it touches nothing the factor uses.
        constexpr float kDrawCapM = 1.2f;
        const bool saturated = (s_major > kDrawCapM);
        const float a_major = std::clamp(s_major, 0.01f, kDrawCapM);
        const float a_minor = std::clamp(s_minor, 0.01f, kDrawCapM);
        auto* ell = anchor_cov_items_[i];
        ell->setRect(-a_major, -a_minor, 2 * a_major, 2 * a_minor);
        ell->setRotation(ang_deg);
        ell->setPos(z_w.x(), z_w.y());

        // A muted anchor is still SHOWN -- a landmark that vanishes reads as a broken producer -- but it
        // must not read as a live one either. Same idiom as a retired corner: dim, thin, dashed, no fill.
        if (saturated)
        {
            QPen pen(QColor(150, 110, 150, 90), 0.03f);
            pen.setStyle(Qt::DashLine);
            ell->setPen(pen);
            ell->setBrush(Qt::NoBrush);
        }
        else
        {
            ell->setPen(QPen(QColor(255, 0, 200, 160), 0.04f));
            ell->setBrush(QBrush(QColor(255, 0, 200, 40)));
        }
        const double dim = saturated ? 0.28 : 1.0;
        anchor_obs_items_[i]->setOpacity(dim);
        anchor_sight_items_[i]->setOpacity(dim);
        // The residual line goes with it: while the anchor is muted, z_o is a PHANTOM (a stale
        // robot-frame observation paired with the CURRENT robot pose), so the gap it draws is not a
        // residual anything is pulling on.
        anchor_resid_items_[i]->setOpacity(saturated ? 0.20 : 1.0);

        // Label: the residual (what the factor pulls on) and the 1sigma axes, given SEPARATELY rather
        // than as a determinant -- "loose along the ray, tight across it" is the shape of this
        // measurement and the thing that explains a weak pull. Age is shown because a stale observation
        // and an uncertain map both produce a big ellipse and are completely different problems.
        const float resid_mm = (z_w - p_o).norm() * 1000.f;
        QString label = QStringLiteral("%1  r=%2mm  σ=%3×%4mm")
                .arg(QString::fromStdString(a.type))
                .arg(resid_mm, 0, 'f', 0)
                .arg(s_major * 1000.f, 0, 'f', 0)   // TRUE sigma, not the clamped drawing
                .arg(s_minor * 1000.f, 0, 'f', 0);
        if (a.obs_age > 0)
            label += QStringLiteral("  age=%1").arg(a.obs_age);
        if (saturated)
            label += QStringLiteral("  MUTED");
        anchor_text_items_[i]->setPlainText(label);
        anchor_text_items_[i]->setDefaultTextColor(saturated ? QColor(160, 130, 160)
                                                             : QColor(255, 120, 220));
        anchor_text_items_[i]->setPos(0.5f * (z_w.x() + p_o.x()), 0.5f * (z_w.y() + p_o.y()));
    }
}

} // namespace rc
