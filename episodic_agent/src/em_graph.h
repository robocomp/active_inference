/*
 * em_graph.h — episodic_agent's WRITE adapter: the episodic-memory DSR graph on domain 4 (2026-10-17).
 *
 * Layout (read back by common/episodic/em_query_dsr.h load_store):
 *   - one node per record: type `metaconcept`, name = record id, attr em_record = to_json(Record) compact;
 *     a `has` edge parent→child (a mission, or a record whose parent node is not in the graph, hangs from root);
 *   - one node per Stats: type `metaconcept`, name `stats_<level>_<kind>[_<author>]_<identity or "any">` (author for
 *     missions only), attr em_stats, under root.
 * Upsert = update the attribute when a node of that name exists, insert it otherwise.
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
    int placed_ = 0;   // viewer placement only: a fan of new nodes so they do not all land on one spot

    /// Insert a metaconcept node `name` hanging (has edge) from `parent_id`, or return the existing one.
    std::optional<DSR::Node> ensure_node(const std::string& name, std::uint64_t parent_id);
};

}   // namespace rc::em
