/*
 * topology_view.h — the "Topology" dock of the ltsm viewer: long-term memory drawn as a MAP.
 *
 * The stock graph view draws every memory node (floors, walls, doors, RT edges), which buries the structure this
 * agent exists to hold. This view draws only that structure, placed by the rooms' own geometry
 * (topology_layout.h): each room a shaded outline with its node at the centroid, each door a glyph on the
 * outline, each passage an edge running room → door → room whose weight is the pass-Beta mean. The room the
 * live `current` edge points at is highlighted and the robot drawn in it. Display only: it writes no graph.
 */
#pragma once

#include "topology_layout.h"

#include <QGraphicsScene>
#include <QGraphicsView>

#include <functional>
#include <memory>
#include <optional>
#include <string>

class TopologyView : public QGraphicsView
{
public:
    /// Where the robot is: the MEMORY name of the room `current` points at, T(room <- robot), and that room's
    /// outline as the LIVE node holds it now (memory's copy is a snapshot taken at seed/eviction).
    struct Robot
    {
        std::string mem_room;
        std::optional<Mat::RTMat> pose;
        std::vector<Eigen::Vector2d> live_outline;
    };
    using RobotSource = std::function<std::optional<Robot>()>;

    TopologyView(std::shared_ptr<DSR::DSRGraph> mem, RobotSource robot, QWidget *parent = nullptr);

    /// Left click (no drag) on a room opens a menu to forget it. `blocker` returns why a room may NOT be forgotten
    /// (the robot is in it), nullopt when it may; `forget` deletes it. Unset ⇒ no menu.
    using Blocker = std::function<std::optional<std::string>(const std::string &mem_room)>;
    using Forget  = std::function<void(const std::string &mem_room)>;
    void set_forget(Blocker blocker, Forget forget) { blocker_ = std::move(blocker); forget_ = std::move(forget); }

    /// The room whose frame is the map's global frame (a live name, e.g. "room_1"; see topo::build).
    void set_anchor(std::string anchor) { anchor_ = std::move(anchor); }

    /// Draw the MAP layout (layout_optimizer.h) instead of the spanning tree. It is solved only when memory's structure
    /// changes (a room or a resolvable passage appears or goes); `on_solve` hears each new solve, to log and persist it.
    using OnSolve = std::function<void(const ltsm::topo::LayoutCache &)>;
    void set_optimise(bool on, OnSolve on_solve = {}) { optimise_ = on; on_solve_ = std::move(on_solve); }
    /// Warm start: poses from the last run's sidecar become the first solve's initial guess (its key is empty, so it
    /// is re-solved against the graph as it is now).
    void seed_layout(ltsm::topo::LayoutCache warm) { cache_ = std::move(warm); cache_.key.clear(); cache_.fresh = false; }
    [[nodiscard]] const ltsm::topo::LayoutCache &layout() const { return cache_; }
    [[nodiscard]] bool optimising() const { return optimise_; }

    /// Re-read the memory graph and redraw. Main thread only.
    void rebuild();

protected:
    void wheelEvent(QWheelEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;   ///< a CLICK (not a drag) on a room: the forget menu
    void mouseDoubleClickEvent(QMouseEvent *event) override;   ///< hands the framing back to auto-fit
    void resizeEvent(QResizeEvent *event) override;

private:
    std::shared_ptr<DSR::DSRGraph> mem_;
    RobotSource robot_;
    QGraphicsScene scene_;
    std::string anchor_ = "room_1";
    Blocker blocker_;
    Forget forget_;
    QPoint press_pos_;
    bool optimise_ = false;
    ltsm::topo::LayoutCache cache_;
    OnSolve on_solve_;
    void room_menu(const std::string &room, const QPoint &global);
    QRectF content_;             ///< scene box of the rooms and doors (what auto-fit frames)
    bool user_framed_ = false;   ///< auto-fit until the user zooms or pans, as the stock graph view does
    void fit();
};
