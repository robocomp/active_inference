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

    /// The room whose frame is the map's global frame (a live name, e.g. "room_1"; see topo::build).
    void set_anchor(std::string anchor) { anchor_ = std::move(anchor); }

    /// Re-read the memory graph and redraw. Main thread only.
    void rebuild();

protected:
    void wheelEvent(QWheelEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;   ///< hands the framing back to auto-fit
    void resizeEvent(QResizeEvent *event) override;

private:
    std::shared_ptr<DSR::DSRGraph> mem_;
    RobotSource robot_;
    QGraphicsScene scene_;
    std::string anchor_ = "room_1";
    QRectF content_;             ///< scene box of the rooms and doors (what auto-fit frames)
    bool user_framed_ = false;   ///< auto-fit until the user zooms or pans, as the stock graph view does
    void fit();
};
