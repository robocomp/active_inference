#include "viewer_2d.h"

#include <abstract_graphic_viewer/abstract_graphic_viewer.h>

#include <QBrush>
#include <QMouseEvent>
#include <QPen>
#include <QPolygonF>

#include <algorithm>
#include <cmath>

namespace rc
{

class ControllerGraphicViewer : public AbstractGraphicViewer
{
public:
    ControllerGraphicViewer(QWidget *parent, const QRectF &dim, bool draw_axis)
        : AbstractGraphicViewer(parent, dim, draw_axis)
    {
    }

protected:
    void mousePressEvent(QMouseEvent *event) override
    {
        if (event->button() == Qt::LeftButton and event->modifiers() == Qt::NoModifier)
        {
            const QPointF cursor_in_scene = this->mapToScene(event->position().toPoint());
            emit new_mouse_coordinates(cursor_in_scene);
            event->accept();
            return;
        }

        AbstractGraphicViewer::mousePressEvent(event);
    }
};

Viewer2D::Viewer2D(QWidget *parent, const QRectF &grid_dim, bool show_axis)
{
    agv_ = new ControllerGraphicViewer(parent, grid_dim, show_axis);

    connect(agv_, &AbstractGraphicViewer::new_mouse_coordinates,
            this, &Viewer2D::new_mouse_coordinates);
    connect(agv_, &AbstractGraphicViewer::right_click,
            this, &Viewer2D::right_click);
}

Viewer2D::~Viewer2D()
{
    clear_lidar_items();
    clear_path_items();
    clear_robot_trajectory();
    clear_polygon_item(inner_polygon_item_);
    clear_polygon_item(polygon_item_);
}

QWidget *Viewer2D::get_widget() const
{
    return agv_;
}

void Viewer2D::add_robot(float width, float length, float offset, float rotation, QColor color)
{
    agv_->add_robot(width, length, offset, rotation, color);
}

void Viewer2D::show()
{
    agv_->show();
}

void Viewer2D::fit_to_scene(const QRectF &rect)
{
    agv_->fitToScene(rect);
}

void Viewer2D::fit_view(float margin_ratio)
{
    QRectF bounds;
    if (polygon_item_ != nullptr)
        bounds = polygon_item_->boundingRect().translated(polygon_item_->pos());
    else
        bounds = agv_->scene.itemsBoundingRect();

    if (!bounds.isValid() or bounds.isEmpty())
        return;

    const qreal dx = std::max(0.1, bounds.width() * std::max(0.f, margin_ratio));
    const qreal dy = std::max(0.1, bounds.height() * std::max(0.f, margin_ratio));
    agv_->fitToScene(bounds.adjusted(-dx, -dy, dx, dy));
}

void Viewer2D::update_robot(const Eigen::Affine2f &robot_pose)
{
    const auto translation = robot_pose.translation();

    if (last_robot_pos_.has_value())
    {
        const Eigen::Vector2f current_pos = translation;
        const float distance = (current_pos - *last_robot_pos_).norm();
        if (distance > 0.01f && distance < 1.5f)
        {
            const QPen traj_pen(QColor(255, 87, 34), 0.04);
            auto *segment = agv_->scene.addLine(last_robot_pos_->x(),
                                                last_robot_pos_->y(),
                                                current_pos.x(),
                                                current_pos.y(),
                                                traj_pen);
            segment->setZValue(17);
            robot_traj_items_.push_back(segment);

            constexpr std::size_t max_traj_segments = 4000;
            while (robot_traj_items_.size() > max_traj_segments)
            {
                auto *old = robot_traj_items_.front();
                agv_->scene.removeItem(old);
                delete old;
                robot_traj_items_.erase(robot_traj_items_.begin());
            }
        }
    }

    last_robot_pos_ = Eigen::Vector2f(translation.x(), translation.y());
    const float angle_rad = std::atan2(robot_pose.linear()(1, 0), robot_pose.linear()(0, 0));
    agv_->robot_poly()->setPos(translation.x(), translation.y());
    agv_->robot_poly()->setRotation(qRadiansToDegrees(angle_rad));
}

void Viewer2D::clear_polygon_item(QGraphicsPolygonItem *&item)
{
    if (item == nullptr)
        return;

    agv_->scene.removeItem(item);
    delete item;
    item = nullptr;
}

void Viewer2D::draw_room_polygon(const std::vector<Eigen::Vector2f> &verts)
{
    clear_polygon_item(polygon_item_);

    if (verts.size() < 2)
        return;

    QPolygonF poly;
    for (const auto &vertex : verts)
        poly << QPointF(vertex.x(), vertex.y());
    if (verts.size() >= 3)
        poly << QPointF(verts.front().x(), verts.front().y());

    polygon_item_ = agv_->scene.addPolygon(poly,
                                           QPen(QColor(67, 87, 100), 0.06),
                                           QBrush(QColor(219, 227, 231, 80)));
    polygon_item_->setZValue(8);
}

void Viewer2D::set_lidar_buffer(LidarPointBuffer *buffer)
{
    lidar_buffer_ = buffer;
}

void Viewer2D::set_lidar_visible(bool visible)
{
    lidar_visible_ = visible;
    if (!visible)
        clear_lidar_items();
}

void Viewer2D::set_mppi_paths_visible(bool visible)
{
    mppi_paths_visible_ = visible;
}

void Viewer2D::clear_lidar_items()
{
    for (auto *item : lidar_items_)
    {
        agv_->scene.removeItem(item);
        delete item;
    }
    lidar_items_.clear();
}

void Viewer2D::clear_robot_trajectory()
{
    for (auto *item : robot_traj_items_)
    {
        agv_->scene.removeItem(item);
        delete item;
    }
    robot_traj_items_.clear();
    last_robot_pos_.reset();
}

void Viewer2D::draw_lidar_points_from_buffer(int max_points)
{
    if (!lidar_visible_)
        return;
    if (lidar_buffer_ == nullptr)
        return;

    const auto [cloud_opt] = lidar_buffer_->read_last();
    if (!cloud_opt.has_value())
    {
        clear_lidar_items();
        return;
    }

    const auto &[xs, ys, zs] = cloud_opt.value();
    Q_UNUSED(zs)

    const std::size_t count = std::min(xs.size(), ys.size());
    if (count == 0)
    {
        clear_lidar_items();
        return;
    }

    const int clamped_max_points = std::max(1, max_points);
    const std::size_t stride = std::max<std::size_t>(1, count / static_cast<std::size_t>(clamped_max_points));
    const std::size_t draw_count = (count + stride - 1) / stride;

    while (lidar_items_.size() > draw_count)
    {
        auto *item = lidar_items_.back();
        agv_->scene.removeItem(item);
        delete item;
        lidar_items_.pop_back();
    }

    static const QRectF ellipse_rect(-1.5, -1.5, 3.0, 3.0);
    QPen pen(QColor("ForestGreen"));
    pen.setWidthF(0.0);
    pen.setCosmetic(true);
    QBrush brush(QColor("ForestGreen"));

    std::size_t draw_index = 0;
    for (std::size_t point_index = 0; point_index < count and draw_index < draw_count; point_index += stride, ++draw_index)
    {
        if (draw_index < lidar_items_.size())
        {
            lidar_items_[draw_index]->setPos(xs[point_index], ys[point_index]);
            lidar_items_[draw_index]->setVisible(true);
        }
        else
        {
            auto *item = agv_->scene.addEllipse(ellipse_rect, pen, brush);
            item->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
            item->setPos(xs[point_index], ys[point_index]);
            item->setZValue(5);
            lidar_items_.push_back(item);
        }
    }
}

void Viewer2D::draw_path(const PathDrawData &data)
{
    clear_path_items();
    clear_polygon_item(inner_polygon_item_);

    if (mppi_paths_visible_ && !data.candidate_trajectories.empty())
    {
        const int total = static_cast<int>(data.candidate_trajectories.size());
        for (int index = 0; index < total; ++index)
        {
            const auto &trajectory = data.candidate_trajectories[index];
            if (trajectory.size() < 2)
                continue;

            const bool is_best = index == data.best_trajectory_idx;
            QColor color = is_best
                ? QColor(255, 99, 71, 230)
                : QColor::fromHsv((205 + (index * 37) % 120) % 360, 190, 230, 90);

            QPen pen(color, is_best ? 0.07 : 0.035);
            pen.setCosmetic(false);
            for (std::size_t point_index = 0; point_index + 1 < trajectory.size(); ++point_index)
            {
                auto *line = agv_->scene.addLine(trajectory[point_index].x(), trajectory[point_index].y(),
                                                 trajectory[point_index + 1].x(), trajectory[point_index + 1].y(),
                                                 pen);
                line->setZValue(is_best ? 24 : 16);
                path_draw_items_.push_back(line);
            }
        }
    }

    if (data.inner_poly.size() >= 3)
    {
        QPolygonF qpoly;
        for (const auto &vertex : data.inner_poly)
            qpoly << QPointF(vertex.x(), vertex.y());
        qpoly << QPointF(data.inner_poly.front().x(), data.inner_poly.front().y());

        QPen inner_pen(QColor(255, 220, 0));
        inner_pen.setWidthF(3.0);
        inner_pen.setCosmetic(true);
        inner_pen.setStyle(Qt::DashLine);

        inner_polygon_item_ = agv_->scene.addPolygon(
            qpoly,
            inner_pen,
            Qt::NoBrush);
        inner_polygon_item_->setZValue(30);
    }

    if (!data.graph_nodes.empty())
    {
        constexpr float radius = 0.04f;
        const QBrush graph_brush(QColor(0, 145, 199, 180));
        for (const auto &point : data.graph_nodes)
        {
            auto *dot = agv_->scene.addEllipse(-radius, -radius, 2.f * radius, 2.f * radius,
                                               Qt::NoPen, graph_brush);
            dot->setPos(point.x(), point.y());
            dot->setZValue(19);
            path_draw_items_.push_back(dot);
        }
    }

    if (!data.obstacle_polys.empty())
    {
        const QPen obstacle_pen(QColor(120, 73, 32), 0.04);
        const QBrush obstacle_brush(QColor(181, 119, 58, 150));
        const QBrush obstacle_center_brush(QColor(255, 0, 128, 220));
        for (const auto &obstacle : data.obstacle_polys)
        {
            if (obstacle.size() < 3)
                continue;

            QPolygonF qpoly;
            for (const auto &vertex : obstacle)
                qpoly << QPointF(vertex.x(), vertex.y());
            qpoly << QPointF(obstacle.front().x(), obstacle.front().y());

            auto *polygon = agv_->scene.addPolygon(qpoly, obstacle_pen, obstacle_brush);
            polygon->setZValue(18);
            path_draw_items_.push_back(polygon);

            QPointF center;
            for (const auto &vertex : obstacle)
                center += QPointF(vertex.x(), vertex.y());
            center /= obstacle.size();
            auto *dot = agv_->scene.addEllipse(-0.06, -0.06, 0.12, 0.12, Qt::NoPen, obstacle_center_brush);
            dot->setPos(center);
            dot->setZValue(19);
            path_draw_items_.push_back(dot);
        }
    }

    if (data.path.empty())
        return;

    const QPen path_pen(QColor(56, 114, 219), 0.08);
    for (std::size_t index = 0; index + 1 < data.path.size(); ++index)
    {
        auto *line = agv_->scene.addLine(data.path[index].x(), data.path[index].y(),
                                         data.path[index + 1].x(), data.path[index + 1].y(),
                                         path_pen);
        line->setZValue(20);
        path_draw_items_.push_back(line);
    }

    for (const auto &point : data.path)
    {
        constexpr float radius = 0.06f;
        auto *dot = agv_->scene.addEllipse(-radius, -radius, 2.f * radius, 2.f * radius,
                                           Qt::NoPen, QBrush(QColor(56, 114, 219)));
        dot->setPos(point.x(), point.y());
        dot->setZValue(21);
        path_draw_items_.push_back(dot);
    }

}

void Viewer2D::clear_path_items()
{
    for (auto *item : path_draw_items_)
    {
        agv_->scene.removeItem(item);
        delete item;
    }
    path_draw_items_.clear();
}

void Viewer2D::update_target_marker(float x, float y, bool visible)
{
    if (target_marker_ == nullptr)
    {
        constexpr float radius = 0.12f;
        target_marker_ = agv_->scene.addEllipse(-radius, -radius, 2.f * radius, 2.f * radius,
                                                QPen(QColor(255, 50, 50), 0.03),
                                                QBrush(QColor(255, 50, 50, 120)));
        target_marker_->setZValue(22);
    }

    target_marker_->setPos(x, y);
    target_marker_->setVisible(visible);
}

} // namespace rc