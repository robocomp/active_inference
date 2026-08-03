/*
 * table_existence.h — evidence-based instance REMOVAL for table_concept (extracted from SpecificWorker).
 *
 * Owns the per-cycle existence update. For each tracked table it carves the LiDAR sweep(s) against the
 * footprint (top slab + legs; the high "helios" plane plus the optional low "bpearl" plane, each from its own
 * origin) and projects the tabletop silhouette against the YOLO foreground, integrates the per-instance
 * existence log-odds (common/existence_belief.h), and deletes the tables whose volume is demonstrably empty
 * once the removal decision holds for ExistenceRemoveFrames consecutive evidence cycles (debounce).
 *
 * Discipline: OCCUPANCY confirms (holds L up), ABSENCE removes, OCCLUSION / out-of-FoV HOLDs (never a false
 * removal of an unseen table). Reads the fitter's instances + the LiDAR ingestor's sweeps; deletes nodes via
 * the DSR graph. The worker calls this only while TableModel.ExistenceRemovalEnabled. Plain class (no Q_OBJECT).
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include <dsr/api/dsr_api.h>

#include "table_config.h"      // rc::TableConfig

namespace rc {

class TableFitter;             // owns the instances (+ silhouette existence)
struct TableInstance;          // one tracked table (referenced by the on_remove sink below)
class TableLidarIngestor;      // stages the per-plane room-frame sweeps
struct EvidenceGlobals;        // dashboard/evidence_monitor.h — removal counters

class TableExistence
{
public:
    TableExistence(std::shared_ptr<DSR::DSRGraph> graph, const TableConfig& cfg)
        : G_(std::move(graph)), cfg_(cfg) {}

    // Integrate each existence channel on its own sensor clock (silhouette on a fresh mask frame, LiDAR carve
    // on a fresh sweep) and delete the demonstrably-empty tables. Removed ids are forgotten in the fitter and
    // deleted from the graph; removal counters are accrued into ev_g. No-op when no channel has fresh evidence.
    // on_remove (optional) is invoked for each doomed instance BEFORE it is forgotten/deleted, so the caller
    // can record the death while the existence-channel state that justified it is still readable. Used for the
    // shadow-mode phantom log (CONCEPT_AGENT_LIFECYCLE.md §4.2) — RECORDING ONLY, it must never influence the
    // removal decision, which is why it is a sink and not a veto.
    void update_and_remove(TableFitter& fitter, TableLidarIngestor* lidar,
                           bool fresh_masks, bool fresh_sweep, EvidenceGlobals& ev_g,
                           const std::function<void(std::uint64_t, const TableInstance&)>& on_remove = {});

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    const TableConfig&             cfg_;
};

}  // namespace rc
