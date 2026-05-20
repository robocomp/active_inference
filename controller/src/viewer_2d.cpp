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
    clear_path_items();
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

void Viewer2D::draw_path(const PathDrawData &data)
{
    clear_path_items();
    clear_polygon_item(inner_polygon_item_);

    if (data.inner_poly.size() >= 3)
    {
        QPolygonF qpoly;
        for (const auto &vertex : data.inner_poly)
            qpoly << QPointF(vertex.x(), vertex.y());
        qpoly << QPointF(data.inner_poly.front().x(), data.inner_poly.front().y());

        inner_polygon_item_ = agv_->scene.addPolygon(
            qpoly,
            QPen(QColor(201, 131, 55), 0.04, Qt::DashLine),
            Qt::NoBrush);
        inner_polygon_item_->setZValue(19);
    }

    if (data.path.empty())
        return;

    const QPen path_pen(QColor(56, 114, 219), 0.06);
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

    const auto &goal = data.path.back();
    update_target_marker(goal.x(), goal.y(), true);
}

void Viewer2D::clear_path_items()
{
    for (auto *item : path_draw_items_)
    {
        agv_->scene.removeItem(item);
        delete item;
    }
    path_draw_items_.clear();

    if (target_marker_ != nullptr)
        target_marker_->setVisible(false);
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