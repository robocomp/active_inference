/*
 * em_graph.h — episodic_agent's WRITE adapter: the episodic-memory DSR graph on domain 4 (2026-10-17).
 *
 * Layout (read back by common/episodic/em_query_dsr.h load_store):
 *   - one node per record: type `metaconcept`, name = record id, attr em_record = to_json(Record) compact;
 *     a `has` edge parent→child (a mission, or a record whose parent node is not in the graph, hangs from root);
 *   - one node per Stats: type `metaconcept`, name `stats_<level>_<kind>[_<author>]_<identity or "any">` (author for
 *     missions only), attr em_stats, under root.
 * Upsert = update the attribute when a node of that name exists, insert it otherwise.
 *
 * Viewer placement (pos_x / pos_y, re-laid out whenever a node is inserted or removed): the experiences are a
 * TIMELINE — root at the origin, the records hanging from it in time order (id "<YYYYMMDD>-<seq>") left to right on
 * root's row, so the root→record edges overlap into one axis — and each experience's tree HANGS DOWN below it as an
 * indented list (one row per record, preorder, indented by depth). The stats nodes are a column left of root.
 */
#pragma once

#include "../../common/episodic/em_schema.h"

#include <memory>
#include <string>

#include <dsr/api/dsr_api.h>

namespace rc::em
{

class EmGraph
{
public:
    explicit EmGraph(std::shared_ptr<DSR::DSRGraph> em);

    bool upsert(const Record& r);
    bool upsert(const Stats& st);
    /// Deletes the record node of that id (and with it its edges); its children are re-hung under root first.
    /// False when there is none.
    bool remove(const std::string& record_id);

    static std::string stats_name(const Stats& st);

private:
    std::shared_ptr<DSR::DSRGraph> G_;

    /// Insert a metaconcept node `name` hanging (has edge) from `parent_id`, or return the existing one.
    std::optional<DSR::Node> ensure_node(const std::string& name, std::uint64_t parent_id);
    /// The viewer placement above, written only where a node's position changed.
    void relayout();
};

}   // namespace rc::em
