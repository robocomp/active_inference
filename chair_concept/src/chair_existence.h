/*
 * chair_existence.h — chair_concept's EXISTENCE channel (the shared rc::exist policy). Extracted from SpecificWorker.
 *
 * Mirrors bottle_existence.h / table_existence.h. Once per fresh MASK frame (the sensor clock, never the compute
 * clock) it integrates every chair's evidence into its existence log-odds and decides removal through the SHARED
 * rc::exist::decide_removal. Three channels, each reduced to (p_vis, log-ratio):
 *
 *   · ROOM-CONTAINMENT PRIOR — a centre outside the room polygon draws OutOfRoomGain every frame, at p_vis = 1.
 *   · SUPPORT MASS (mask) — a won mask is scored by how much of the expected silhouette the model EXPLAINS
 *     (support adequacy × (1 − clutter)); a lost one is absence weighted by staleness × ZED detectability ×
 *     remaining line of sight. ★Denominated in Existence.EvidenceGain nats, not log(pd/pc).
 *   · CONTOUR (classifier-free; common/contour_edge) — RGB edge support + depth support of the projected outline,
 *     worth at most one chair observation (g·tanh), so it can HOLD a chair YOLO went blind on but not keep a
 *     phantom alive against a working detector.
 *
 * Never deletes anything itself: it hands back each condemned instance through `on_remove` and the worker owns
 * the affordance → instance → node teardown order, exactly as for a tracker DEATH. Main thread. Plain class.
 */

#pragma once

#include <cstdint>
#include <functional>

#include <opencv2/core.hpp>

#include "chair_config.h"
#include "chair_instance.h"
#include "../../common/mask_ingestor/mask_ingestor.h"

namespace rc {

class ChairFitter;

class ChairExistence
{
public:
    explicit ChairExistence(const ChairConfig& cfg) : cfg_(cfg) {}

    // This cycle's sensor inputs, gathered by the worker (which owns the ingestors).
    struct Inputs
    {
        const MaskIngestor::MasksPacket* pkt = nullptr;   // the mask clock; nothing happens on a stale frame
        const cv::Mat* rgb   = nullptr;                   // latest ZED BGR frame, or nullptr (contour RGB half)
        std::uint64_t  rgb_stamp_ms = 0;
        const cv::Mat* depth = nullptr;                   // latest ZED depth (CV_32F m), or nullptr (contour depth half)
        std::uint64_t  depth_stamp_ms = 0;
    };

    // Integrate all channels into every instance's existence log-odds and hand back the ones condemned by the
    // shared removal decision. `on_remove` runs after the integration pass, once per condemned instance.
    void update_and_remove(ChairFitter& fitter, const Inputs& in,
                           const std::function<void(std::uint64_t, ChairInstance&)>& on_remove);

private:
    const ChairConfig& cfg_;
    int   last_mask_frame_ = -1;      // mask frame_id last integrated (sensor-clock guard)
    float support_scale_   = 0.0f;    // expected-support scale C = E[npts·range²] (0 = seed from cfg)
    // Capture stamps of the camera frames the contour channel last scored — scoring the same RGB/depth frame
    // twice would charge one look twice.
    std::uint64_t contour_last_stamp_ms_       = 0;
    std::uint64_t contour_last_depth_stamp_ms_ = 0;
};

}  // namespace rc
