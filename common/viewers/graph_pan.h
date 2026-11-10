/*
 * common/viewers/graph_pan.h — a left-drag pans the cortex graph viewer, on a quiet graph too.
 *
 * The viewer pans by moving its scrollbars, so the scene rect is the pan limit. Each show shrinks that rect to the graph
 * plus a small margin, leaving next to no scroll range, and only the next refit pads it again. robot_concept's graph changes
 * many times a second, so its refit comes at once. A memory graph can stay still for minutes and is left unpannable.
 * keep_pannable pads the rect after each show as GraphViewer::update_scene_rect does (private there), leaving zoom and framing alone.
 */
#pragma once

#include <dsr/gui/viewers/graph_viewer/graph_viewer.h>

#include <QEvent>
#include <QObject>
#include <QTimer>

#include <algorithm>
#include <cmath>

namespace rc::viewers
{
inline void pad_scene_rect(DSR::GraphViewer* gv)
{
    const QRectF items = gv->scene.itemsBoundingRect();
    if (items.isNull()) return;
    const QPointF centre = gv->mapToScene(gv->viewport()->rect().center());
    const qreal sx = std::max(std::abs(gv->transform().m11()), 1e-9);
    const qreal sy = std::max(std::abs(gv->transform().m22()), 1e-9);
    const qreal pad_x = std::max(items.width(), gv->viewport()->width() / sx);
    const qreal pad_y = std::max(items.height(), gv->viewport()->height() / sy);
    gv->scene.setSceneRect(items.adjusted(-pad_x, -pad_y, pad_x, pad_y));
    gv->centerOn(centre);
}

class PanAfterShow : public QObject
{
public:
    explicit PanAfterShow(DSR::GraphViewer* gv) : QObject(gv), gv_(gv) { gv->installEventFilter(this); }

    bool eventFilter(QObject* obj, QEvent* ev) override
    {
        static constexpr int kAfterShowMs = 100;   // UI timing: lets the show's own refit finish first
        if (obj == gv_ and ev->type() == QEvent::Show)
            QTimer::singleShot(kAfterShowMs, gv_, [gv = gv_] { pad_scene_rect(gv); });
        return false;
    }

private:
    DSR::GraphViewer* gv_;
};

inline void keep_pannable(DSR::GraphViewer* gv)
{
    new PanAfterShow(gv);   // parented to the viewer
    if (gv->isVisible()) pad_scene_rect(gv);
}
}   // namespace rc::viewers
