/*
 * topology_view.cpp — see topology_view.h.
 */
#include "topology_view.h"
#include "room_forget.h"

#include <QBrush>
#include <QGraphicsEllipseItem>
#include <QGraphicsPathItem>
#include <QGraphicsPolygonItem>
#include <QApplication>
#include <QGraphicsSimpleTextItem>
#include <QMenu>
#include <QMouseEvent>
#include <QPainterPath>
#include <QPen>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <format>
#include <functional>
#include <ranges>

namespace
{
// QGraphicsItem::data key carrying the memory room name: what a click resolves to.
constexpr int kRoomKey = 0;

// World +y is up; the scene's +y is down.
QPointF to_scene(const Eigen::Vector2d &p) { return {p.x(), -p.y()}; }

QColor room_fill(const std::string &name, bool current)
{
    const int hue = static_cast<int>(std::hash<std::string>{}(name) % 360);
    return QColor::fromHsv(hue, current ? 90 : 50, 235, current ? 210 : 150);
}

QPen cosmetic(const QColor &c, double px, Qt::PenStyle style = Qt::SolidLine)
{
    QPen pen(c, px, style, Qt::RoundCap, Qt::RoundJoin);
    pen.setCosmetic(true);   // widths in pixels, whatever the zoom
    return pen;
}

/// A marker of fixed screen size (it ignores the view's zoom) centred on `at`.
template <typename Item>
Item *marker(QGraphicsScene &scene, Item *item, const QPointF &at)
{
    item->setFlag(QGraphicsItem::ItemIgnoresTransformations);
    item->setPos(at);
    scene.addItem(item);
    return item;
}

void label(QGraphicsScene &scene, const QString &text, const QPointF &at, const QPointF &offset_px, int z)
{
    auto *t = marker(scene, new QGraphicsSimpleTextItem(text), at);
    t->setTransform(QTransform::fromTranslate(offset_px.x(), offset_px.y()));
    t->setZValue(z);
}
}   // namespace

TopologyView::TopologyView(std::shared_ptr<DSR::DSRGraph> mem, RobotSource robot, QWidget *parent)
    : QGraphicsView(parent), mem_(std::move(mem)), robot_(std::move(robot))
{
    setScene(&scene_);
    setRenderHint(QPainter::Antialiasing);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setBackgroundBrush(QColor(250, 250, 248));
    setToolTip({});
}

void TopologyView::rebuild()
{
    if (not mem_) return;
    const auto robot = robot_ ? robot_() : std::nullopt;
    std::map<std::string, std::vector<Eigen::Vector2d>> live_outlines;
    if (robot.has_value() and not robot->live_outline.empty()) live_outlines[robot->mem_room] = robot->live_outline;
    const auto map = ltsm::topo::build(*mem_, anchor_, live_outlines, optimise_ ? &cache_ : nullptr);
    if (cache_.fresh)
    {
        cache_.fresh = false;
        if (on_solve_) on_solve_(cache_);
    }
    const auto current = robot.has_value() ? map.room_index(robot->mem_room) : std::nullopt;

    scene_.clear();
    // What the auto-fit frames: the GEOMETRY only. The fixed-size markers ignore the view transform, so
    // itemsBoundingRect() would count an 18-px disc as 18 m.
    content_ = QRectF();
    const auto grow = [this](const Eigen::Vector2d &p) { content_ |= QRectF(to_scene(p), QSizeF(1e-3, 1e-3)); };
    for (const auto &r : map.rooms) { grow(r.centre); for (const auto &p : r.outline) grow(p); }
    for (const auto &d : map.doors) { grow(d.a); grow(d.b); }

    // ── Rooms: shaded outline (back), node at the centroid (front) ─────────────────────────────
    for (const auto &[i, r] : std::views::enumerate(map.rooms))
    {
        const bool is_current = current == static_cast<std::size_t>(i);
        // Where the spanning tree alone would put it: faint and dashed, behind, so the MAP's correction is visible.
        if (map.optimised and not r.outline_raw.empty())
        {
            QPolygonF poly;
            for (const auto &p : r.outline_raw) poly << to_scene(p);
            scene_.addPolygon(poly, cosmetic(QColor(150, 150, 150, 160), 1.0, Qt::DashLine))->setZValue(-1);
        }
        // 1-σ position ellipse of the room's frame origin (Laplace covariance; none for the gauge room).
        if (const Eigen::Matrix2d C = r.cov.topLeftCorner<2, 2>(); C.trace() > 0)
        {
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(C);
            const double a = std::sqrt(std::max(es.eigenvalues()(1), 0.0)), b = std::sqrt(std::max(es.eigenvalues()(0), 0.0));
            const Eigen::Vector2d v = es.eigenvectors().col(1);
            auto *e = scene_.addEllipse(-a, -b, 2 * a, 2 * b, cosmetic(QColor(40, 90, 200, 200), 1.0));
            e->setPos(to_scene(r.world.translation().head<2>()));
            e->setRotation(-std::atan2(v.y(), v.x()) * 180.0 / M_PI);   // scene y is down
            e->setZValue(3);
        }
        if (not r.outline.empty())
        {
            QPolygonF poly;
            for (const auto &p : r.outline) poly << to_scene(p);
            auto *item = scene_.addPolygon(poly, is_current ? cosmetic(QColor(40, 90, 200), 3.0)
                                                            : cosmetic(QColor(120, 120, 120), 1.0),
                                           QBrush(room_fill(r.name, is_current)));
            item->setZValue(0);
            item->setToolTip(QString::fromStdString(r.p_manhattan >= 0
                ? std::format("{}\nP(square to the building) = {:.2f}", r.name, r.p_manhattan) : r.name));
            item->setData(kRoomKey, QString::fromStdString(r.name));
        }
        auto *node = marker(scene_, new QGraphicsEllipseItem(-9, -9, 18, 18), to_scene(r.centre));
        node->setBrush(is_current ? QColor(40, 90, 200) : QColor(60, 60, 60));
        node->setPen(cosmetic(Qt::white, 2.0));
        node->setZValue(4);
        node->setToolTip(QString::fromStdString(r.name));
        node->setData(kRoomKey, QString::fromStdString(r.name));
        label(scene_, QString::fromStdString(r.name), to_scene(r.centre), {12, -8}, 5);
    }

    // ── Doors: the aperture as a thick segment on the outline ──────────────────────────────────
    for (const auto &d : map.doors)
    {
        auto *seg = scene_.addLine(QLineF(to_scene(d.a), to_scene(d.b)), cosmetic(QColor(150, 75, 20), 6.0));
        seg->setZValue(2);
        seg->setToolTip(QString::fromStdString(d.name));
        // Inside ITS OWN room. A passage's two faces are one hole, so they share a centre: labels at a common
        // offset would stack and hide one face. Stepping each toward its room's centre puts the two names on
        // either side of the wall.
        const Eigen::Vector2d inward = map.rooms[d.room].centre - d.centre;
        const double step = std::min(0.35, 0.5 * inward.norm());
        const Eigen::Vector2d at = inward.norm() > 0 ? Eigen::Vector2d(d.centre + step * inward.normalized()) : d.centre;
        label(scene_, QString::fromStdString(d.name), to_scene(at), {-20, -6}, 5);
    }

    // ── Passages: room → door → room, weight and opacity = P(passable) ─────────────────────────
    for (const auto &p : map.passages)
    {
        const double pp = p.p_pass();
        const auto tip = QString::fromStdString(std::format("{}\nP(pass) = {:.2f}   (α {:.1f}, β {:.1f})\ncrossings {}{}",
            p.name, pp, p.alpha, p.beta, p.crossings,
            p.loop ? std::format("\nloop closure {:.3f} m", p.closure_m) : std::string{})
            + (p.seam_d2 >= 0 ? std::format("\nseam Mahalanobis² {:.2f}, P(right) {:.3f}  ({} of 2 faces state a covariance)",
                                            p.seam_d2, p.p_inlier, p.stated_faces) : std::string{}));
        // A seam the MAP explains as wrong (P(right) < ½) is drawn dashed red: memory holds it, the layout ignores it.
        const bool disowned = p.p_inlier >= 0 and p.p_inlier < 0.5;
        const QColor colour = not p.seam_ok ? QColor(140, 140, 140)
                            : disowned      ? QColor(200, 40, 40)
                                            : QColor(30, 130, 60, static_cast<int>(80 + 175 * pp));
        const QPen pen = p.seam_ok and not disowned ? cosmetic(colour, 1.5 + 4.5 * pp) : cosmetic(colour, 1.5, Qt::DashLine);

        // One half per side, each through ITS OWN door face: on a tree passage they meet at the same point; on a
        // loop the gap between them is the closure error, drawn rather than hidden.
        const auto half = [&](const std::optional<std::size_t> &room, const std::optional<std::size_t> &door)
        {
            if (not room.has_value() or not door.has_value()) return;
            QPainterPath path(to_scene(map.rooms[*room].centre));
            path.lineTo(to_scene(map.doors[*door].centre));
            auto *item = scene_.addPath(path, pen);
            item->setZValue(3);
            item->setToolTip(tip);
        };
        half(p.room_a, p.door_a);
        half(p.room_b, p.door_b);
        if (p.loop and p.door_a.has_value() and p.door_b.has_value())
            scene_.addLine(QLineF(to_scene(map.doors[*p.door_a].centre), to_scene(map.doors[*p.door_b].centre)),
                           cosmetic(QColor(200, 40, 40), 2.0, Qt::DashLine))->setZValue(3);

        // The passage node, on the door it goes through.
        const auto door = p.door_a.has_value() ? p.door_a : p.door_b;
        if (not door.has_value()) continue;
        QPolygonF diamond;
        diamond << QPointF(0, -7) << QPointF(7, 0) << QPointF(0, 7) << QPointF(-7, 0);
        auto *node = marker(scene_, new QGraphicsPolygonItem(diamond), to_scene(map.doors[*door].centre));
        node->setBrush(p.seam_ok ? QColor(30, 130, 60) : QColor(140, 140, 140));
        node->setPen(cosmetic(Qt::white, 1.5));
        node->setZValue(4);
        node->setToolTip(tip);
    }

    // ── The global frame: x (red) and y (green) axes at the anchor room's origin ───────────────
    if (map.anchor.has_value())
    {
        const auto &w = map.rooms[*map.anchor].world;
        const auto at = [&](double x, double y) { return to_scene((w * Eigen::Vector3d(x, y, 0)).head<2>()); };
        scene_.addLine(QLineF(at(0, 0), at(1.0, 0)), cosmetic(QColor(210, 40, 40), 2.5))->setZValue(5);
        scene_.addLine(QLineF(at(0, 0), at(0, 1.0)), cosmetic(QColor(40, 160, 40), 2.5))->setZValue(5);
    }

    // ── Robot: a triangle pointing along its +y, in the current room ───────────────────────────
    if (current.has_value() and robot->pose.has_value())
    {
        const Mat::RTMat t = map.rooms[*current].world * robot->pose.value();
        const auto at = [&](double x, double y) { return to_scene((t * Eigen::Vector3d(x, y, 0)).head<2>()); };
        QPolygonF tri;
        tri << at(0, 0.35) << at(0.22, -0.2) << at(-0.22, -0.2);
        auto *item = scene_.addPolygon(tri, cosmetic(Qt::white, 1.5), QBrush(QColor(220, 60, 30)));
        item->setZValue(6);
        item->setToolTip(QStringLiteral("robot"));
    }

    if (map.rooms.empty())
        label(scene_, QStringLiteral("long-term memory holds no room yet"), {0, 0}, {0, 0}, 1);
    else if (map.optimised)
    {
        const auto &c = cache_;
        label(scene_, QString::fromStdString(std::format(
            "MAP layout · solve #{} ({:.0f} ms, {} it{})\nF {:.2f} → {:.2f}   seam {:.2f}  square {:.2f}  overlap {:.2f}  wall {:.2f}{}",
            c.solves, c.ms, c.iterations, c.converged ? "" : ", NOT converged", c.before.total(), c.after.total(),
            c.after.seam, c.after.manhattan, c.after.overlap, c.after.wall,
            c.building_axis.has_value() ? std::format("\nbuilding axis {:.1f}°", c.building_axis.value() * 180.0 / M_PI)
                                        : std::string{})),
            content_.topLeft(), {0, -40}, 7);
    }

    if (not user_framed_) fit();
}

void TopologyView::fit()
{
    if (content_.isEmpty()) return;
    const QRectF box = content_;
    const double m = 0.08 * std::max({box.width(), box.height(), 1.0});
    setSceneRect(box.adjusted(-m, -m, m, m));
    fitInView(sceneRect(), Qt::KeepAspectRatio);
}

void TopologyView::wheelEvent(QWheelEvent *event)
{
    user_framed_ = true;
    const double f = std::pow(1.0015, event->angleDelta().y());
    scale(f, f);
    event->accept();
}

void TopologyView::mouseMoveEvent(QMouseEvent *event)
{
    if (event->buttons() & Qt::LeftButton)
    {
        user_framed_ = true;
        // Let the drag reach past the graph's edge (a sceneRect pinned to the items dead-stops it).
        setSceneRect(sceneRect().united(mapToScene(viewport()->rect()).boundingRect()));
    }
    QGraphicsView::mouseMoveEvent(event);
}

void TopologyView::mousePressEvent(QMouseEvent *event)
{
    press_pos_ = event->pos();
    QGraphicsView::mousePressEvent(event);
}

void TopologyView::mouseReleaseEvent(QMouseEvent *event)
{
    QGraphicsView::mouseReleaseEvent(event);
    // A click, not the end of a pan: the pointer stayed within the platform's drag distance.
    if (event->button() != Qt::LeftButton or not forget_
        or (event->pos() - press_pos_).manhattanLength() >= QApplication::startDragDistance())
        return;
    for (auto *item : items(event->pos()))
        if (const auto room = item->data(kRoomKey).toString(); not room.isEmpty())
        {
            room_menu(room.toStdString(), event->globalPosition().toPoint());
            return;
        }
}

void TopologyView::room_menu(const std::string &room, const QPoint &global)
{
    const auto plan = ltsm::plan_forget(*mem_, room);
    if (not plan.has_value()) return;
    const auto blocked = blocker_ ? blocker_(room) : std::nullopt;

    QMenu menu(this);
    menu.addSection(QString::fromStdString(room));
    auto *forget = menu.addAction(QString::fromStdString(blocked.has_value()
        ? "Delete room -- not allowed: " + blocked.value()
        : std::format("Delete room ({} node(s) below it, {} passage(s))", plan->subtree.size(), plan->passages.size())));
    forget->setEnabled(not blocked.has_value());
    if (menu.exec(global) == forget)
    {
        forget_(room);
        rebuild();
    }
}

void TopologyView::mouseDoubleClickEvent(QMouseEvent *event)
{
    user_framed_ = false;
    fit();
    QGraphicsView::mouseDoubleClickEvent(event);
}

void TopologyView::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);
    if (not user_framed_) fit();
}
