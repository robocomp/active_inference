#pragma once
#include <utility>
#include <exception>
#include <QDebug>            // qWarning() with << streaming

namespace rc
{
// Wrap DSRGraph::update_node so an (id,name) collision or deleted-node throw can never
// cross a Qt-slot or Ice-servant boundary. DSRGraph::update_node throws a std::runtime_error
// ("Cannot update node in G, id and name must be unique" / "is deleted") — see cortex
// dsr_api.cpp update_node. An uncaught throw on the Qt main thread, an Ice-servant callback
// (FullPoseEstimationPub_newFullPose) or a raw reader thread unwinds through non-C++-aware
// frames and segfaults the process. Swallow it, log the offending node once, return false.
// The forwarding reference preserves each call site's lvalue-copy vs rvalue-move semantics
// (rvalue callers still move — no deep blob copy under the graph mutex).
template <class Graph, class NodeT>
bool safe_update_node(Graph& g, NodeT&& n) noexcept
{
    try
    {
        return g.update_node(std::forward<NodeT>(n));
    }
    catch (const std::exception& e)
    {
        qWarning() << "[graph] update_node rejected (id/name collision or deleted node):" << e.what();
        return false;
    }
    catch (...)
    {
        qWarning() << "[graph] update_node rejected: unknown exception";
        return false;
    }
}
} // namespace rc
