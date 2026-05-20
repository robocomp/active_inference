#ifndef CONTROLLER_VIEWER_2D_H
#define CONTROLLER_VIEWER_2D_H

#include <QObject>
#include <QColor>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QGraphicsPolygonItem>
#include <QPointF>
#include <QRectF>
#include <QTransform>
#include <QWidget>

#include <Eigen/Dense>

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
        std::vector<Eigen::Vector2f> path;
        std::vector<Eigen::Vector2f> inner_poly;
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
    void draw_path(const PathDrawData &data);
    void clear_path_items();
    void update_target_marker(float x, float y, bool visible);

Q_SIGNALS:
    void new_mouse_coordinates(QPointF);
    void right_click(QPointF);

private:
    AbstractGraphicViewer *agv_ = nullptr;
    QGraphicsPolygonItem *polygon_item_ = nullptr;
    QGraphicsPolygonItem *inner_polygon_item_ = nullptr;
    std::vector<QGraphicsItem *> path_draw_items_;
    QGraphicsEllipseItem *target_marker_ = nullptr;

    void clear_polygon_item(QGraphicsPolygonItem *&item);
};

} // namespace rc

#endif