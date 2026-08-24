#include "viewer_2d.h"

#include <abstract_graphic_viewer/abstract_graphic_viewer.h>

#include <QBrush>
#include <QFont>
#include <QMouseEvent>
#include <QPen>
#include <QPainterPath>
#include <QPolygonF>

#include <algorithm>
#include <cmath>

namespace rc
{

namespace
{
struct ObstaclePalette
{
    QColor pen;
    QColor fill;
    QColor center;
    QColor edge;
};

// ── ONE HUE PER OBJECT ───────────────────────────────────────────────────────────────────────────
// Every concept-agent object used to be drawn the same green, so a canvas with a table, four chairs, a
// bottle and a fridge on it was one undifferentiated green mass and the only thing separating them was
// a two-character label. Identity is a CATEGORICAL quantity; it gets categorical hues.
//
// WHY THESE SEVEN AND NOT MORE. The obstacle KINDS keep their existing colours — orange for anything
// unmodelled, crimson for residual_concept's occupancy — and those are filled footprints too, so an
// object hue that lands near either destroys the kind encoding to buy the object encoding. The seven
// below are the largest set that clears, WITH those two reserved colours in the set, the all-pairs
// separation gates on a light surface: worst pair ΔE 8.1 under simulated colour-vision deficiency
// (protan/deutan) and 16.2 with normal vision, in OKLab×100. ALL pairs, not adjacent ones: two
// footprints can end up side by side anywhere on a map, so there is no ordering to lean on.
// Verified with the dataviz skill's validator, not by eye.
// One slot (the lime) sits below 3:1 contrast against the light background; every footprint carries a
// direct label in its own hue, which is the secondary encoding that requirement asks for.
// PAST SEVEN OBJECTS THE SLOTS REPEAT. That is a deliberate wrap, not a generated hue: an eighth
// invented colour would have no separation guarantee at all, and the label still tells them apart.
constexpr int kObjectHueCount = 7;
QColor object_hue(int slot)
{
    static const QColor kHues[kObjectHueCount] = {
        QColor(0x3a, 0x95, 0xcd),   // blue
        QColor(0x00, 0x8d, 0x65),   // teal-green
        QColor(0xed, 0x3b, 0x87),   // magenta
        QColor(0x7d, 0x4b, 0x92),   // plum
        QColor(0x89, 0xb3, 0x1d),   // lime
        QColor(0x23, 0x51, 0xde),   // indigo
        QColor(0xa3, 0x61, 0xfb)};  // violet
    // Slots come from a registry that only ever counts up from 0, so no negative normalisation is needed.
    return kHues[slot % kObjectHueCount];
}

// The four roles, derived from ONE hue so they cannot drift apart: the pen and the centre dot carry the
// validated colour at full opacity (they are what the eye actually matches on), the fill is the same hue
// dropped to a wash so overlapping footprints stay readable, and the edge is a darkened form of it.
ObstaclePalette palette_from_hue(const QColor &hue)
{
    QColor fill = hue;   fill.setAlpha(110);
    QColor centre = hue; centre.setAlpha(235);
    QColor edge = hue.darker(160); edge.setAlpha(240);
    return {hue, fill, centre, edge};
}

ObstaclePalette obstacle_palette(ControllerObstacleKind kind, int object_slot = 0)
{
    switch (kind)
    {
        case ControllerObstacleKind::Object:
        // Model objects interpreted by a concept agent (table/cylinder/chair/object) — one hue each,
        // assigned per OBJECT IDENTITY by Viewer2D::object_color_slot.
        return palette_from_hue(object_hue(object_slot));
        case ControllerObstacleKind::Temporary:
        // The controller's OWN temporary lidar obstacles (unexplained returns) — orange.
        return {QColor(194, 103, 25),
            QColor(245, 158, 11, 150),
            QColor(251, 146, 60, 230),
            QColor(154, 52, 18, 240)};
        case ControllerObstacleKind::Obstacle:
        // Graph "obstacle" nodes — also unmodelled, rendered orange like the temporary ones.
        return {QColor(194, 103, 25),
            QColor(245, 158, 11, 150),
            QColor(251, 146, 60, 230),
            QColor(154, 52, 18, 240)};
        case ControllerObstacleKind::GridOccupancy:
        // residual_concept occupancy-grid hulls = what the controller plans around as OCCUPIED.
        // Semi-transparent crimson FILL so occupied area is shaded and free space stays clear.
        return {QColor(150, 20, 40, 220),
            QColor(220, 40, 60, 70),
            QColor(150, 20, 40, 120),
            QColor(150, 20, 40, 120)};
        default:
        return {QColor(194, 103, 25),
            QColor(245, 158, 11, 150),
            QColor(251, 146, 60, 230),
            QColor(154, 52, 18, 240)};
    }
}

Eigen::Vector2f polygon_centroid(const std::vector<Eigen::Vector2f> &verts)
{
    if (verts.empty())
        return Eigen::Vector2f::Zero();

    float signed_area_twice = 0.f;
    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    for (std::size_t index = 0; index < verts.size(); ++index)
    {
        const auto &from = verts[index];
        const auto &to = verts[(index + 1) % verts.size()];
        const float cross = from.x() * to.y() - to.x() * from.y();
        signed_area_twice += cross;
        centroid += (from + to) * cross;
    }

    if (std::abs(signed_area_twice) < 1e-5f)
    {
        for (const auto &vertex : verts)
            centroid += vertex;
        return centroid / static_cast<float>(verts.size());
    }

    return centroid / (3.f * signed_area_twice);
}
}

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
    // Mouse events reach the VIEWPORT, not the view, so the filter goes there.
    agv_->viewport()->installEventFilter(this);
}

Viewer2D::~Viewer2D()
{
    clear_lidar_items();
    clear_path_items();
    clear_room_axis_items();
    clear_robot_trajectory();
    clear_mission_items();
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

// See the declaration for why this is a registry and not a hash of the name. An object with no
// identity to key on (older producers, or a visual built without one) all share slot 0 rather than
// each grabbing a fresh colour — one shared "unidentified" hue is honest; seven of them would assert
// distinctions the data does not contain.
int Viewer2D::object_color_slot(const std::string &key)
{
    if (key.empty()) return 0;
    const auto [it, inserted] = object_color_slots_.try_emplace(key, next_object_color_slot_);
    if (inserted) ++next_object_color_slot_;
    return it->second;
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
    clear_room_axis_items();

    if (verts.size() < 2)
        return;

    QPolygonF poly;
    for (const auto &vertex : verts)
        poly << QPointF(vertex.x(), vertex.y());
    if (verts.size() >= 3)
        poly << QPointF(verts.front().x(), verts.front().y());

    // 12 cm: the room outline is the frame everything else is read against, so it should read as the
    // boundary rather than as one more line among the route, the trajectories and the obstacle hulls.
    polygon_item_ = agv_->scene.addPolygon(poly,
                                           QPen(QColor(67, 87, 100), 0.12),
                                           QBrush(QColor(219, 227, 231, 80)));
    polygon_item_->setZValue(8);

    if (verts.size() >= 3)
    {
        const Eigen::Vector2f center = polygon_centroid(verts);
        Eigen::Vector2f min_corner = verts.front();
        Eigen::Vector2f max_corner = verts.front();
        for (const auto &vertex : verts)
        {
            min_corner = min_corner.cwiseMin(vertex);
            max_corner = max_corner.cwiseMax(vertex);
        }

        const float axis_length = std::clamp(0.07f * (max_corner - min_corner).minCoeff(), 0.25f, 0.5f);
        QPen x_pen(QColor(214, 64, 69), 0.04);
        x_pen.setCosmetic(false);
        QPen y_pen(QColor(44, 146, 88), 0.04);
        y_pen.setCosmetic(false);

        auto *x_axis = agv_->scene.addLine(center.x(), center.y(), center.x() + axis_length, center.y(), x_pen);
        auto *y_axis = agv_->scene.addLine(center.x(), center.y(), center.x(), center.y() + axis_length, y_pen);
        x_axis->setZValue(9);
        y_axis->setZValue(9);
        room_axis_items_.push_back(x_axis);
        room_axis_items_.push_back(y_axis);

        auto *origin = agv_->scene.addEllipse(-0.03, -0.03, 0.06, 0.06,
                                              Qt::NoPen, QBrush(QColor(38, 50, 56, 220)));
        origin->setPos(center.x(), center.y());
        origin->setZValue(9.1);
        room_axis_items_.push_back(origin);

        QFont axis_font;
        axis_font.setPointSizeF(9.0);
        auto *x_label = agv_->scene.addSimpleText("x", axis_font);
        x_label->setBrush(QBrush(QColor(214, 64, 69)));
        x_label->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
        x_label->setPos(center.x() + axis_length + 0.04f, center.y() - 0.02f);
        x_label->setZValue(9.2);
        room_axis_items_.push_back(x_label);

        auto *y_label = agv_->scene.addSimpleText("y", axis_font);
        y_label->setBrush(QBrush(QColor(44, 146, 88)));
        y_label->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
        y_label->setPos(center.x() - 0.02f, center.y() + axis_length + 0.04f);
        y_label->setZValue(9.2);
        room_axis_items_.push_back(y_label);
    }
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

void Viewer2D::clear_room_axis_items()
{
    for (auto *item : room_axis_items_)
    {
        agv_->scene.removeItem(item);
        delete item;
    }
    room_axis_items_.clear();
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
    // Draw only returns that are plausibly ABOVE the floor. The buffer's own z-band starts at 0.20 m, which is
    // enough to reject a head-on floor return but NOT a grazing one: helios sits ~1.1 m up and reads the floor
    // 13-17 cm high, rising with range, so its far floor returns clear 0.20 m and land in the buffer. Drawn as
    // filled ForestGreen dots they carpet the floor and hide everything the overlay exists to show.
    // This is DISPLAY ONLY — the obstacle/ESDF paths are untouched, so nothing about what the robot avoids
    // changes. kDrawMinZ is deliberately above the band rather than equal to it, for exactly that grazing margin.
    constexpr float kDrawMinZ = 0.30f;

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

    // Dead-reckoning correction: re-anchor the (robot-attached) cloud from the scan-time pose to
    // "now". Identity when overlay extrapolation is off, so this is a no-op in that case.
    const bool correct = !lidar_draw_correction_.isApprox(Eigen::Affine2f::Identity());

    std::size_t draw_index = 0;
    for (std::size_t point_index = 0; point_index < count && draw_index < draw_count; point_index += stride)
    {
        if (point_index < zs.size() and zs[point_index] < kDrawMinZ) continue;   // floor carpet — display only
        float px = xs[point_index];
        float py = ys[point_index];
        if (correct)
        {
            const Eigen::Vector2f c = lidar_draw_correction_ * Eigen::Vector2f(px, py);
            px = c.x();
            py = c.y();
        }
        if (draw_index < lidar_items_.size())
        {
            lidar_items_[draw_index]->setPos(px, py);
            lidar_items_[draw_index]->setVisible(true);
        }
        else
        {
            auto *item = agv_->scene.addEllipse(ellipse_rect, pen, brush);
            item->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
            item->setPos(px, py);
            item->setZValue(5);
            lidar_items_.push_back(item);
        }

        ++draw_index;
    }

    while (draw_index < lidar_items_.size())
    {
        lidar_items_[draw_index]->setVisible(false);
        ++draw_index;
    }
}

void Viewer2D::draw_path(const PathDrawData &data)
{
    clear_path_items();

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

    if (mppi_paths_visible_ && data.average_trajectory.size() >= 2)
    {
        QPen average_pen(QColor(0, 170, 255, 230), 0.09);
        average_pen.setCosmetic(false);
        average_pen.setStyle(Qt::DashLine);
        for (std::size_t point_index = 0; point_index + 1 < data.average_trajectory.size(); ++point_index)
        {
            auto *line = agv_->scene.addLine(data.average_trajectory[point_index].x(), data.average_trajectory[point_index].y(),
                                             data.average_trajectory[point_index + 1].x(), data.average_trajectory[point_index + 1].y(),
                                             average_pen);
            line->setZValue(23);
            path_draw_items_.push_back(line);
        }
    }

    if (!data.waypoints.empty())
    {
        constexpr float radius = 0.04f;
        const QBrush graph_brush(QColor(0, 145, 199, 180));
        for (const auto &point : data.waypoints)
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
        for (const auto &obstacle_visual : data.obstacle_polys)
        {
            const auto &obstacle = obstacle_visual.polygon;
            if (obstacle.size() < 3)
                continue;

            // No kind test here: obstacle_palette ignores the slot for every non-Object kind, and only
            // objects carry a color_key — object_color_slot("") returns 0 without touching the registry.
            const auto palette = obstacle_palette(obstacle_visual.kind,
                                                  object_color_slot(obstacle_visual.color_key));
            const bool is_grid_cell = obstacle_visual.kind == ControllerObstacleKind::GridOccupancy;
            // Grid cells tile edge-to-edge — use a thin hairline pen so they read as one filled region
            // (a 0.085 m border per 0.35 m cell would drown the fill). Real obstacles keep the bold edge.
            QPen obstacle_pen(palette.pen, is_grid_cell ? 0.0 : 0.085);
            obstacle_pen.setCosmetic(is_grid_cell);
            const QBrush obstacle_brush(palette.fill);
            const QBrush obstacle_center_brush(palette.center);
            const QBrush obstacle_edge_brush(palette.edge);

            // The centre is needed by the round branch as well as by the marker + label below.
            QPointF center;
            for (const auto &vertex : obstacle)
                center += QPointF(vertex.x(), vertex.y());
            center /= obstacle.size();

            if (obstacle_visual.round and obstacle_visual.round_radius_m > 0.f)
            {
                // A ROUND table is drawn as the disc it is. The box would draw four corners and an
                // orientation that the round belief does not contain — it has no yaw at all — so the
                // rectangle was showing structure that had never been inferred. See the note on
                // ControllerObstacleVisual::round: this is the DISPLAY only; the planner still avoids
                // the circumscribing polygon.
                const qreal r = obstacle_visual.round_radius_m;
                auto *disc = agv_->scene.addEllipse(QRectF(center.x() - r, center.y() - r, 2 * r, 2 * r),
                                                    obstacle_pen, obstacle_brush);
                disc->setZValue(18);
                path_draw_items_.push_back(disc);
            }
            else
            {
                QPolygonF qpoly;
                for (const auto &vertex : obstacle)
                    qpoly << QPointF(vertex.x(), vertex.y());
                qpoly << QPointF(obstacle.front().x(), obstacle.front().y());

                auto *polygon = agv_->scene.addPolygon(qpoly, obstacle_pen, obstacle_brush);
                polygon->setZValue(18);
                path_draw_items_.push_back(polygon);
            }

            // ── 2-sigma "known-ness" halo (DISPLAY ONLY) ──────────────────────────────────────────
            // A dashed ring at 2*sigma_pos around the footprint: how well this object's POSITION is
            // known, straight from the rt_covariance its concept agent publishes. Nine agents publish
            // one and nothing used to read any of them.
            //
            // ★It is NOT what the planner avoids, and it is drawn so it cannot be mistaken for that:
            // no fill, dashed, cosmetic pen, under the footprint. Same discipline as the round-table
            // disc above — display may show what the belief contains; only the polygon is planned
            // against. (The uncertainty-derived clearance is a separate, currently disabled, setting.)
            //
            // WATCH IT GROW: an agent inflates Sigma while it is not measuring, so the halo swells
            // whenever the robot looks away and should shrink on a fixation. A halo that keeps growing
            // is the visible form of "peripheral evidence is not bounding this belief".
            if (obstacle_visual.sigma_pos_m > 0.f
                and obstacle_visual.kind != ControllerObstacleKind::GridOccupancy)
            {
                // ★ITS OWN COLOUR, NOT THE OBJECT'S. The ring used to take palette.pen, so on a model
                // object it came out the same green as the footprint it surrounds — two different
                // quantities (where the object IS, how well that is KNOWN) in one colour, which is
                // exactly the confusion the "not what the planner avoids" note above is trying to
                // prevent. One dedicated magenta for every kind instead: uncertainty reads as a layer
                // rather than as a property of whichever agent happened to publish the object.
                // ★Hue picked to be unused in this scene — the greens (34,139,58), oranges
                // (194,103,25), crimsons (150,20,40), blues (0,145,199 / 56,114,219) and the teal
                // (22,160,133) are all taken, and the one purple (142,68,173) is the mission-recording
                // waypoint, far enough round the wheel and only present while recording.
                const qreal r = 2.0 * obstacle_visual.sigma_pos_m;   // 2 sigma ~ 95%
                QColor hc(224, 64, 191); hc.setAlpha(150);
                QPen halo_pen(hc, 0.0);
                halo_pen.setCosmetic(true);
                halo_pen.setStyle(Qt::DashLine);
                auto *halo = agv_->scene.addEllipse(QRectF(center.x() - r, center.y() - r, 2 * r, 2 * r),
                                                    halo_pen, Qt::NoBrush);
                halo->setZValue(17);   // BELOW the footprint (18) — context, not the thing itself
                halo->setToolTip(QStringLiteral("%1: position sigma %2 m (2σ ring)")
                                     .arg(QString::fromStdString(obstacle_visual.label))
                                     .arg(obstacle_visual.sigma_pos_m, 0, 'f', 3));
                path_draw_items_.push_back(halo);
            }

            // Grid-occupancy hulls read as a filled region only — skip the dense contour dots + centre
            // marker (they are the "small points" that made occupied/free space unreadable).
            if (obstacle_visual.kind == ControllerObstacleKind::GridOccupancy)
                continue;

            // The per-edge contour dots are gone. They added nothing the filled polygon + its pen did not
            // already show, and they were expensive: one QGraphicsEllipseItem every 0.12 m of perimeter,
            // rebuilt every frame for every obstacle. That is hundreds of scene items per redraw, which is a
            // real contributor to the viewer's jank. The outline is the polygon's own pen.
            auto *dot = agv_->scene.addEllipse(-0.06, -0.06, 0.12, 0.12, Qt::NoPen, obstacle_center_brush);
            dot->setPos(center);
            dot->setZValue(19);
            path_draw_items_.push_back(dot);

            if (!obstacle_visual.label.empty())
            {
                QFont label_font;
                label_font.setPointSizeF(8.0);
                label_font.setBold(true);
                auto *label = agv_->scene.addSimpleText(QString::fromStdString(obstacle_visual.label), label_font);
                // ★INK, NOT THE OBJECT'S HUE. The tag is what tells two footprints apart when their
                // colours are hard to separate — colour-vision deficiency, a projector, a printout —
                // so it is the one thing that must stay readable in every case. Drawn in the object's
                // own hue it inherited that hue's contrast against the background, which for the
                // lighter slots is under 3:1: the label went faint exactly where it was needed most.
                // Identity is carried by the footprint beside it; the text carries the name.
                label->setBrush(QBrush(QColor(28, 37, 42)));
                label->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
                label->setPos(center.x() + 0.06, center.y() - 0.06);
                label->setZValue(20);
                path_draw_items_.push_back(label);
            }
        }
    }

    if (!data.obstacle_rfe_points.empty())
    {
        QPen rfe_pen(QColor(110, 92, 0, 230), 0.018);
        rfe_pen.setCosmetic(false);
        const QBrush rfe_brush(QColor(255, 225, 40, 245));
        for (const auto &obstacle_points : data.obstacle_rfe_points)
        {
            for (const auto &point : obstacle_points)
            {
                auto *dot = agv_->scene.addEllipse(-0.045, -0.045, 0.09, 0.09, rfe_pen, rfe_brush);
                dot->setPos(point.x(), point.y());
                dot->setZValue(19.5);
                path_draw_items_.push_back(dot);
            }
        }
    }

    if (data.path.empty())
        return;

    // ONE item for the whole path, not one per segment. This used to add a line AND a 12 cm dot per
    // point, which is fine for the handful of turning points the leg planner produced and hopeless for a
    // continuous route: 3300 samples at 5 cm became ~10 000 QGraphicsItems rebuilt every frame, drawn as
    // a solid blob of overlapping dots that did not look like a route at all.
    QPainterPath painter_path(QPointF(data.path.front().x(), data.path.front().y()));
    for (std::size_t index = 1; index < data.path.size(); ++index)
        painter_path.lineTo(data.path[index].x(), data.path[index].y());
    auto *route_item = agv_->scene.addPath(painter_path, QPen(QColor(56, 114, 219), 0.06), Qt::NoBrush);
    route_item->setZValue(20);
    path_draw_items_.push_back(route_item);

    // Vertex dots only when there are few enough for one to MEAN something. On a densely sampled curve
    // every sample would carry a dot and the marks would say nothing the line does not already say.
    if (data.path.size() <= 64)
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

void Viewer2D::clear_mission_items()
{
    for (auto *item : mission_items_)
    {
        agv_->scene.removeItem(item);
        delete item;
    }
    mission_items_.clear();
}

void Viewer2D::draw_mission(const std::vector<Eigen::Vector2f> &waypoints, int current_index, bool recording,
                            bool draggable)
{
    // A drag in flight OWNS the waypoint positions. present() redraws every cycle from the control thread's
    // snapshot, which cannot yet contain the point being dragged — rendering that snapshot mid-drag snapped
    // the marker back to its old place between mouse moves, which is what made dragging feel like it only
    // jumped, or did nothing at all for a point whose move was overwritten before it was ever painted.
    if (drag_index_ < 0)
    {
        mission_wps_ = waypoints;
        mission_draggable_ = draggable;
    }
    mission_index_ = current_index;
    mission_recording_ = recording;
    render_mission();
}

void Viewer2D::render_mission()
{
    clear_mission_items();
    if (mission_wps_.empty())
        return;

    // Violet while recording, teal while running — the two states look nothing alike, because clicking in
    // the view means "append a waypoint" in one and "drive there now" in the other.
    const QColor base = mission_recording_ ? QColor(142, 68, 173) : QColor(22, 160, 133);

    QPen link_pen(base.lighter(120), 0.03);
    link_pen.setCosmetic(false);
    link_pen.setStyle(Qt::DotLine);
    for (std::size_t i = 0; i + 1 < mission_wps_.size(); ++i)
    {
        auto *line = agv_->scene.addLine(mission_wps_[i].x(), mission_wps_[i].y(),
                                         mission_wps_[i + 1].x(), mission_wps_[i + 1].y(), link_pen);
        line->setZValue(17);
        mission_items_.push_back(line);
    }

    constexpr float radius = 0.10f;
    for (std::size_t i = 0; i < mission_wps_.size(); ++i)
    {
        const bool is_current = static_cast<int>(i) == mission_index_;
        const bool is_dragged = static_cast<int>(i) == drag_index_;
        auto *dot = agv_->scene.addEllipse(-radius, -radius, 2.f * radius, 2.f * radius,
                                           QPen(is_dragged ? QColor(255, 255, 255) : base.darker(140), 0.02),
                                           QBrush(is_current ? QColor(241, 196, 15) : base));
        dot->setPos(mission_wps_[i].x(), mission_wps_[i].y());
        dot->setZValue(is_current or is_dragged ? 29 : 27);
        mission_items_.push_back(dot);

        auto *label = agv_->scene.addSimpleText(QString::number(i + 1));
        label->setBrush(QBrush(Qt::white));
        QFont f = label->font();
        f.setPointSizeF(6.0);
        label->setFont(f);
        // The scene y axis points down for text; flip so the numbers read the right way up in room frame.
        label->setTransform(QTransform().scale(0.02, -0.02));
        const QRectF br = label->boundingRect();
        label->setPos(mission_wps_[i].x() - 0.02 * br.width() * 0.5,
                      mission_wps_[i].y() + 0.02 * br.height() * 0.5);
        label->setZValue(30);
        mission_items_.push_back(label);
    }
}

bool Viewer2D::eventFilter(QObject *watched, QEvent *event)
{
    const auto index_valid = [this](int i) { return i >= 0 and i < static_cast<int>(mission_wps_.size()); };
    // Right-drag a waypoint to move it. The base viewer binds a plain right-press to PANNING, so this
    // filter consumes the press ONLY when it lands on a waypoint — panning still works everywhere else,
    // which matters because panning is how you reach the parts of the room you are about to edit.
    if (agv_ == nullptr or watched != agv_->viewport())
        return QObject::eventFilter(watched, event);

    // Pick radius in METRES, not pixels: the scene is in metres and the view zooms, so a pixel radius
    // would grab a different amount of world at every zoom level.
    constexpr float pick_radius_m = 0.18f;

    const auto scene_pos = [this](QMouseEvent *me)
    { return agv_->mapToScene(me->position().toPoint()); };

    if (event->type() == QEvent::MouseButtonPress)
    {
        auto *me = static_cast<QMouseEvent *>(event);
        if (me->button() != Qt::RightButton or me->modifiers() != Qt::NoModifier or not mission_draggable_)
            return false;
        const QPointF p = scene_pos(me);
        // COINCIDENT WAYPOINTS ARE NORMAL — a hand-clicked tour that returns to the same corner stacks
        // several on one spot. Ties go to the HIGHEST index, i.e. the one drawn last and therefore on top,
        // so what you grab is what you can see. A strict `<` here silently grabbed the lowest index
        // instead: clicking the visible marker moved a different, buried waypoint, which reads as "this
        // point won't move" and leaves the edit apparently unsaved.
        int best = -1;
        float best_d2 = pick_radius_m * pick_radius_m;
        for (std::size_t i = 0; i < mission_wps_.size(); ++i)
        {
            const float dx = static_cast<float>(p.x()) - mission_wps_[i].x();
            const float dy = static_cast<float>(p.y()) - mission_wps_[i].y();
            if (const float d2 = dx * dx + dy * dy; d2 <= best_d2) { best_d2 = d2; best = static_cast<int>(i); }
        }
        if (best < 0)
            return false;                    // not on a waypoint → let the base viewer pan
        drag_index_ = best;
        agv_->viewport()->setCursor(Qt::ClosedHandCursor);
        return true;
    }

    if (event->type() == QEvent::MouseMove and drag_index_ >= 0)
    {
        auto *me = static_cast<QMouseEvent *>(event);
        const QPointF p = scene_pos(me);
        if (drag_index_ < static_cast<int>(mission_wps_.size()))
            mission_wps_[drag_index_] = {static_cast<float>(p.x()), static_cast<float>(p.y())};
        render_mission();   // follows the cursor immediately, no round trip through the control thread
        return true;
    }

    if (event->type() == QEvent::MouseButtonRelease and drag_index_ >= 0)
    {
        auto *me = static_cast<QMouseEvent *>(event);
        if (me->button() != Qt::RightButton)
            return false;
        const QPointF p = scene_pos(me);
        if (index_valid(drag_index_))
            mission_wps_[drag_index_] = {static_cast<float>(p.x()), static_cast<float>(p.y())};
        const int index = drag_index_;
        drag_index_ = -1;
        agv_->viewport()->unsetCursor();
        render_mission();
        emit mission_waypoint_moved(index, p);
        return true;
    }

    return QObject::eventFilter(watched, event);
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

void Viewer2D::update_orient_overlay(float x, float y, float target_yaw, float current_yaw,
                                     bool visible)
{
    // Long enough to read against the room, short enough not to reach across it. The ray is a
    // DIRECTION, not a distance — nothing about an Orient says how far away anything is.
    constexpr float kRayM = 1.20f;
    constexpr float kArcM = 0.85f;

    if (orient_ray_ == nullptr)
    {
        orient_ray_ = agv_->scene.addLine(0, 0, 0, 0, QPen(QColor(0, 170, 255), 0.04,
                                                           Qt::SolidLine, Qt::RoundCap));
        orient_ray_->setZValue(23);
        orient_arc_ = agv_->scene.addPath(QPainterPath(), QPen(QColor(0, 170, 255, 170), 0.05),
                                          QBrush(Qt::NoBrush));
        orient_arc_->setZValue(23);
    }

    orient_ray_->setVisible(visible);
    orient_arc_->setVisible(visible);
    if (not visible) return;

    orient_ray_->setLine(x, y, x + kRayM * std::cos(target_yaw), y + kRayM * std::sin(target_yaw));

    // The arc runs from where the robot points NOW to where it was asked to point, the short way —
    // which is the way the executor turns, so the drawing and the motion cannot disagree about which
    // direction the robot is about to go. It shrinks to nothing as the turn completes, so the picture
    // empties out exactly when the affordance is satisfied.
    const double sweep = std::atan2(std::sin(target_yaw - current_yaw),
                                    std::cos(target_yaw - current_yaw));
    QPainterPath arc;
    const QRectF box(x - kArcM, y - kArcM, 2.0 * kArcM, 2.0 * kArcM);
    // ★QT'S ARC ANGLES ARE THE NEGATION OF A WORLD YAW, AND THE VIEW'S FLIP DOES NOT UNDO IT. QPainterPath
    // places an angle θ at (cx + r·cosθ, cy − r·sinθ): the minus is Qt assuming a y-DOWN painter and
    // wanting positive angles to LOOK counter-clockwise there. Our scene coordinates are world metres
    // with y up (AbstractGraphicViewer's scale(1,-1) is applied by the VIEW, after the path is built),
    // so a world yaw ψ has to be handed over as −ψ, and the sweep with it. Passing them through
    // unnegated draws the arc mirrored about the x axis and sweeping away from the target.
    const double start_qt = -current_yaw * 180.0 / M_PI;
    const double sweep_qt = -sweep * 180.0 / M_PI;
    // arcMoveTo first: arcTo alone would draw a straight line in from wherever the path currently is.
    arc.arcMoveTo(box, start_qt);
    arc.arcTo(box, start_qt, sweep_qt);
    orient_arc_->setPath(arc);
}

} // namespace rc