/*
 * layout_sidecar.h — the MAP layout of a memory generation, on disk beside it.
 *
 * MemoryStore writes each generation of the memory graph as ltsm_<stamp>.json; the layout solved for that graph goes
 * to ltsm_<stamp>_layout.json. It is a REPORT and a WARM START, never a source of truth: at start-up its poses seed
 * the first solve's initial guess (so a restart does not reshuffle a settled map), and its structure key is
 * discarded — the key is recomputed from the graph, so a sidecar that no longer matches can only cost iterations.
 * JSON through QJsonDocument, which is locale-independent (the machines run es_ES: never strtod a data file).
 */
#pragma once

#include "topology_layout.h"

#include <QByteArray>

#include <filesystem>
#include <optional>

namespace ltsm::topo
{
QByteArray layout_to_json(const LayoutCache &cache);

/// The poses (and the report fields) of a sidecar, key left EMPTY so the next build() solves. nullopt if unreadable.
std::optional<LayoutCache> layout_from_json(const std::filesystem::path &path);
}   // namespace ltsm::topo
