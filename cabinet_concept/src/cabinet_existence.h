/*
 * cabinet_existence.h — evidence-based instance REMOVAL for cabinet_concept (extracted from SpecificWorker).
 *
 * Owns the per-cycle existence update. For each tracked cabinet it carves the LiDAR sweep(s) against the
 * footprint (top slab + legs; the high "helios" plane plus the optional low "bpearl" plane, each from its own
 * origin) and projects the tabletop silhouette against the YOLO foreground, integrates the per-instance
 * existence log-odds (common/existence_belief.h), and deletes the cabinets whose volume is demonstrably empty
 * once the removal decision holds for ExistenceRemoveFrames consecutive evidence cycles (debounce).
 *
 * Discipline: OCCUPANCY confirms (holds L up), ABSENCE removes, OCCLUSION / out-of-FoV HOLDs (never a false
 * removal of an unseen cabinet). Reads the fitter's instances + the LiDAR ingestor's sweeps; deletes nodes via
 * the DSR graph. The worker calls this only while CabinetModel.ExistenceRemovalEnabled. Plain class (no Q_OBJECT).
 */

#pragma once

#include <cstdint>
#include "../../common/exclusion/exclusion.h"   // rc::exclusion:: (SHARED)
#include <vector>
#include <functional>
#include <memory>

#include <dsr/api/dsr_api.h>

#include "cabinet_config.h"      // rc::CabinetConfig

namespace rc {

class CabinetFitter;
struct CabinetInstance;   // one tracked cabinet (referenced by the on_remove sink below)             // owns the instances (+ silhouette existence)
class ConceptLidarIngestor;      // stages the per-plane room-frame sweeps
struct EvidenceGlobals;        // dashboard/evidence_monitor.h — removal counters

class CabinetExistence
{
public:
    // The other concepts' standing claims on room space, refreshed by the caller once per cycle
    // (one graph walk shared with the birth path). SHARED: a JUNIOR instance's occupancy is
    // discounted by how much of it a SENIOR object already explains — common/exclusion.
    void set_foreign_claims(const std::vector<rc::exclusion::Claim>* c) { claims_ = c; }

    CabinetExistence(std::shared_ptr<DSR::DSRGraph> graph, const CabinetConfig& cfg)
        : G_(std::move(graph)), cfg_(cfg) {}

    // Integrate each existence channel on its own sensor clock (silhouette on a fresh mask frame, LiDAR carve
    // on a fresh sweep) and delete the demonstrably-empty cabinets. Removed ids are forgotten in the fitter and
    // deleted from the graph; removal counters are accrued into ev_g. No-op when no channel has fresh evidence.
    void update_and_remove(CabinetFitter& fitter, ConceptLidarIngestor* lidar,
                           bool fresh_masks, bool fresh_sweep, EvidenceGlobals& ev_g,
                           // on_remove (optional) fires for each doomed instance BEFORE teardown, so the caller can
                           // record the death while the existence state that justified it is still readable. Shadow-mode
                           // phantom log (CONCEPT_AGENT_LIFECYCLE.md §4.2) — a SINK, never a veto: it cannot alter removal.
                           const std::function<void(std::uint64_t, const CabinetInstance&)>& on_remove = {});

private:
    const std::vector<rc::exclusion::Claim>* claims_ = nullptr;
    std::shared_ptr<DSR::DSRGraph> G_;
    const CabinetConfig&             cfg_;
};

}  // namespace rc
