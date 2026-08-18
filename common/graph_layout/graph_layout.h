#pragma once
/*
 * common/graph_layout/graph_layout.h — re-run the DSR graph-view layout after the graph's shape changed.
 *
 * Six agents carried this verbatim (two of them byte-identical, the rest differing only in comment wording).
 * It is pure viewer plumbing — no belief, no per-object anything — which is why one copy is enough.
 *
 * ★IT RUNS THE LAYOUT TWICE, AND THAT IS THE WHOLE TRICK. A node/edge change reaches the viewer through
 * QUEUED signals, so at the moment an agent notices it created a node, the viewer has not necessarily
 * processed the corresponding update yet: laying out immediately arranges the graph as it was a moment ago,
 * and the new node lands wherever the default placement puts it. Running once now and once more via a queued
 * invocation means the second pass happens AFTER the viewer has drained those pending signals. Cheap, and it
 * is the difference between a coherent view and a stray node in the corner.
 *
 * MAIN-THREAD ONLY (Qt widgets).
 */

#include <memory>
#include <string>
#include <unordered_map>

#include <QMetaObject>
#include <QWidget>

#include <dsr/gui/dsr_gui.h>

namespace rc::gui
{

// `viewers` is GenericWorker::graph_viewers. The unnamed ("") entry is the main viewer every agent creates.
inline void trigger_layout_twopi(
    const std::unordered_map<std::string, std::shared_ptr<DSR::DSRViewer>>& viewers)
{
    const auto it = viewers.find("");
    if (it == viewers.end() or not it->second)
        return;

    QWidget* graph_widget = it->second->get_widget(DSR::DSRViewer::view::graph);
    auto* graph_viewer = qobject_cast<DSR::GraphViewer*>(graph_widget);
    if (not graph_viewer)
        return;

    // Now AND once queued — see the header note on why one pass is not enough.
    graph_viewer->compute_layout("twopi");
    QMetaObject::invokeMethod(graph_viewer,
                              [graph_viewer] { graph_viewer->compute_layout("twopi"); },
                              Qt::QueuedConnection);
}

}  // namespace rc::gui
