/*
 * hood_existence.h — evidence-based instance REMOVAL for hood_concept (extracted from SpecificWorker).
 *
 * Owns the per-cycle existence update. For each tracked hood it carves the LiDAR sweep(s) against the
 * footprint (top slab + legs; the high "helios" plane plus the optional low "bpearl" plane, each from its own
 * origin) and projects the hoodtop silhouette against the YOLO foreground, integrates the per-instance
 * existence log-odds (common/existence_belief.h), and deletes the hoods whose volume is demonstrably empty
 * once the removal decision holds for ExistenceRemoveFrames consecutive evidence cycles (debounce).
 *
 * Discipline: OCCUPANCY confirms (holds L up), ABSENCE removes, OCCLUSION / out-of-FoV HOLDs (never a false
 * removal of an unseen hood). Reads the fitter's instances + the LiDAR ingestor's sweeps; deletes nodes via
 * the DSR graph. The worker calls this only while HoodModel.ExistenceRemovalEnabled. Plain class (no Q_OBJECT).
 */

#pragma once

#include "../../common/detectability/detectability.h"   // rc::detect::DetectorEnvelope

#include <cstdint>
#include <functional>
#include <memory>

#include <dsr/api/dsr_api.h>

#include "hood_config.h"      // rc::HoodConfig

namespace rc {

class HoodFitter;
struct HoodInstance;   // one tracked hood (referenced by the on_remove sink below)             // owns the instances (+ silhouette existence)
class ConceptLidarIngestor;      // stages the per-plane room-frame sweeps
struct EvidenceGlobals;        // dashboard/evidence_monitor.h — removal counters

class HoodExistence
{
public:
    // The detector's operating envelope (min/max projected fill). Set once from config; shared with the
    // epistemic planner so the viewpoint we ASK for and the absence we BELIEVE use the same model.
    void set_detector_envelope(const rc::detect::DetectorEnvelope& e) { det_env_ = e; }

    HoodExistence(std::shared_ptr<DSR::DSRGraph> graph, const HoodConfig& cfg)
        : G_(std::move(graph)), cfg_(cfg) {}

    // Integrate each existence channel on its own sensor clock (silhouette on a fresh mask frame, LiDAR carve
    // on a fresh sweep) and delete the demonstrably-empty hoods. Removed ids are forgotten in the fitter and
    // deleted from the graph; removal counters are accrued into ev_g. No-op when no channel has fresh evidence.
    void update_and_remove(HoodFitter& fitter, ConceptLidarIngestor* lidar,
                           bool fresh_masks, bool fresh_sweep, EvidenceGlobals& ev_g,
                           // on_remove (optional) fires for each doomed instance BEFORE teardown, so the caller can
                           // record the death while the existence state that justified it is still readable. Shadow-mode
                           // phantom log (CONCEPT_AGENT_LIFECYCLE.md §4.2) — a SINK, never a veto: it cannot alter removal.
                           const std::function<void(std::uint64_t, const HoodInstance&)>& on_remove = {});

private:
    rc::detect::DetectorEnvelope det_env_{};

    std::shared_ptr<DSR::DSRGraph> G_;
    const HoodConfig&             cfg_;
};

}  // namespace rc
