/*
 * bed_existence.h — evidence-based instance REMOVAL for bed_concept (extracted from SpecificWorker).
 *
 * Owns the per-cycle existence update. For each tracked bed it carves the LiDAR sweep(s) against the
 * footprint (top slab + legs; the high "helios" plane plus the optional low "bpearl" plane, each from its own
 * origin) and projects the bedtop silhouette against the YOLO foreground, integrates the per-instance
 * existence log-odds (common/existence_belief.h), and deletes the beds whose volume is demonstrably empty
 * once the removal decision holds for ExistenceRemoveFrames consecutive evidence cycles (debounce).
 *
 * Discipline: OCCUPANCY confirms (holds L up), ABSENCE removes, OCCLUSION / out-of-FoV HOLDs (never a false
 * removal of an unseen bed). Reads the fitter's instances + the LiDAR ingestor's sweeps; deletes nodes via
 * the DSR graph. The worker calls this only while BedModel.ExistenceRemovalEnabled. Plain class (no Q_OBJECT).
 */

#pragma once

#include "../../common/detectability/detectability.h"   // rc::detect::DetectorEnvelope
#include "../../common/exclusion/exclusion.h"           // rc::exclusion:: (SHARED no-two-objects rule)

#include <cstdint>
#include <functional>
#include <vector>
#include <memory>

#include <opencv2/core.hpp>

#include <dsr/api/dsr_api.h>

#include "bed_config.h"      // rc::BedConfig

namespace rc {

class BedFitter;
struct BedInstance;   // one tracked bed (referenced by the on_remove sink below)             // owns the instances (+ silhouette existence)
class ConceptLidarIngestor;      // stages the per-plane room-frame sweeps
struct EvidenceGlobals;        // dashboard/evidence_monitor.h — removal counters

class BedExistence
{
public:
    // The detector's operating envelope (min/max projected fill). Set once from config; shared with the
    // epistemic planner so the viewpoint we ASK for and the absence we BELIEVE use the same model.
    void set_detector_envelope(const rc::detect::DetectorEnvelope& e) { det_env_ = e; }

    // ── The classifier-free CONTOUR channel's inputs ────────────────────────────────────────────────
    // The ZED frames for THIS cycle, set by the worker before update_and_remove. Both optional: with no
    // RGB the gradient half stays silent, with no depth the metric half does, and with neither the whole
    // channel abstains — which is not the same as it refuting, and the code below keeps them distinct.
    //
    // ⚠BORROWED, NOT OWNED, and only for the duration of the call. The ingestors outlive it and this all
    // runs on the main thread; a cv::Mat handed across a thread boundary would have to be DEEP-COPIED
    // (a cv::Mat copy is a refcounted shallow handle — see CLAUDE.md). Nothing here writes to them.
    void set_camera_frames(const cv::Mat* rgb, const cv::Mat* depth, std::uint64_t stamp_ms, bool fresh)
    { rgb_frame_ = rgb; depth_frame_ = depth; frame_stamp_ms_ = stamp_ms; frames_fresh_ = fresh; }

    // The other concepts' standing claims on room space, refreshed by the caller once per cycle (one graph
    // walk shared with the birth path). SHARED policy: a junior instance's occupancy is discounted by how
    // much of it a SENIOR object already explains — see common/exclusion/exclusion.h.
    void set_foreign_claims(const std::vector<rc::exclusion::Claim>* c) { claims_ = c; }

    BedExistence(std::shared_ptr<DSR::DSRGraph> graph, const BedConfig& cfg)
        : G_(std::move(graph)), cfg_(cfg) {}

    // Integrate each existence channel on its own sensor clock (silhouette on a fresh mask frame, LiDAR carve
    // on a fresh sweep) and delete the demonstrably-empty beds. Removed ids are forgotten in the fitter and
    // deleted from the graph; removal counters are accrued into ev_g. No-op when no channel has fresh evidence.
    void update_and_remove(BedFitter& fitter, ConceptLidarIngestor* lidar,
                           bool fresh_masks, bool fresh_sweep, EvidenceGlobals& ev_g,
                           // on_remove (optional) fires for each doomed instance BEFORE teardown, so the caller can
                           // record the death while the existence state that justified it is still readable. Shadow-mode
                           // phantom log (CONCEPT_AGENT_LIFECYCLE.md §4.2) — a SINK, never a veto: it cannot alter removal.
                           const std::function<void(std::uint64_t, const BedInstance&)>& on_remove = {});

private:
    rc::detect::DetectorEnvelope det_env_{};
    const std::vector<rc::exclusion::Claim>* claims_ = nullptr;

    const cv::Mat* rgb_frame_       = nullptr;   // borrowed, this cycle only — see set_camera_frames
    const cv::Mat* depth_frame_     = nullptr;   // CV_32F metres, NaN = no return
    std::uint64_t  frame_stamp_ms_  = 0;
    bool           frames_fresh_    = false;

    std::shared_ptr<DSR::DSRGraph> G_;
    const BedConfig&             cfg_;
};

}  // namespace rc
