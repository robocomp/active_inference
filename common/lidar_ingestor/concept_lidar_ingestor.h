/*
 * common/lidar_ingestor/concept_lidar_ingestor.h  —  lidar3D media-plane consumer for the concept agents. SHARED.
 *
 * Brings up the shared zero-copy LidarFrame subscribers (the same descriptor-driven factory every agent uses) and,
 * once per compute() cycle, stages the newest sweep of each plane transformed into the ROOM frame plus that
 * plane's sensor origin — exactly what the shared LiDAR first-hit factor needs (see
 * common/ai_belief/lidar_ray_factor.h). Any belief exposing the sdf_prim / sdf_jacobian / n_prims hooks consumes
 * it unchanged, which is why nothing here is object-specific.
 *
 * SHARED because it was five near-identical copies (bottle/cabinet/hood/refrigerator/table). Four of them were
 * BYTE-identical once the object noun was normalised; bottle's was the pre-bpearl, pre-free-space version — a
 * capability its siblings had grown and it had not. Per the UNION rule this module carries the full capability
 * and the agent declares which gates it can actually honour (see LidarGates).
 *
 * Crash-safety (mirrors room_concept::LidarIngestor, the sanctioned pattern):
 *  - The subscribers are created LAZILY inside poll() (called from the Operating compute/main thread), never in a
 *    ctor or a free-running thread, and only once the "lidar3D" node + media descriptor exist. Discovery is
 *    self-throttled to ~1 Hz.
 *  - pump() reads the DSR graph (inner_eigen room<-lidar3D) — it MUST be called on the main thread.
 *  - The owner (SpecificWorker) must reset this ingestor BEFORE tearing the graph down, while G is alive.
 *  - It stays entirely dormant (no DDS participant) while the corresponding gate is 0, so each feature is a true
 *    no-op when off.
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

namespace rc::media { class LidarPlaneReader; }

namespace rc
{

// The agent's live feature gates, read fresh on every pump() (config is reloadable, so this must NOT be cached).
// A gate of 0 keeps that plane's subscriber dormant — no DDS participant is ever created for it.
//
// An agent whose belief has no free-space or bpearl factor passes 0 for those and simply never wakes them. That
// is a DECLARATION, not a limitation of this module: the capability is here the day the belief grows the factor.
struct LidarGates
{
    float helios_precision     = 0.0f;   // per-ray range precision on the high 360 plane (0 = OFF)
    float free_space_precision = 0.0f;   // free-space/carve factor — also feeds off the helios sweep (0 = OFF)
    float bpearl_precision     = 0.0f;   // per-ray range precision on the low bpearl dome (0 = OFF)
};

class ConceptLidarIngestor
{
public:
    ConceptLidarIngestor(std::shared_ptr<DSR::DSRGraph> graph, DSR::InnerEigenAPI* inner_eigen,
                         std::function<LidarGates()> gates);
    ~ConceptLidarIngestor();

    // Poll BOTH planes (helios primary, bpearl if its gate > 0); on a NEW sweep, transform each into the room
    // frame at its OWN capture stamp and stage it. Returns true iff a fresh sweep was staged for EITHER plane
    // this call. Per-plane freshness via helios_fresh()/bpearl_fresh(). Main-thread only (reads the graph).
    bool pump();

    const std::vector<Eigen::Vector3f>& sweep_room()  const { return sweep_room_; }
    const Eigen::Vector3f&              origin_room() const { return origin_room_; }
    bool                                helios_fresh() const { return helios_fresh_; }
    // Low bpearl plane (separate per-device ray-set; own origin). Only pumped while its gate > 0.
    const std::vector<Eigen::Vector3f>& sweep_bpearl_room()  const { return sweep_bpearl_room_; }
    const Eigen::Vector3f&              origin_bpearl_room() const { return origin_bpearl_room_; }
    bool                                bpearl_fresh() const { return bpearl_fresh_; }

private:
    std::shared_ptr<DSR::DSRGraph>               G_;
    DSR::InnerEigenAPI*                          inner_eigen_ = nullptr;
    std::function<LidarGates()>                  gates_;
    // Per-device media-plane readers: the high "helios" 360 plane and the low "bpearl" plane, each a SINGLE
    // plane transformed straight to the ROOM frame — a single plane per reader keeps the first-hit ray factor's
    // origin well-defined (see lidar_ray_factor.h). They are consumed as SEPARATE ray-sets, never merged.
    std::unique_ptr<rc::media::LidarPlaneReader> reader_;
    std::unique_ptr<rc::media::LidarPlaneReader> reader_bpearl_;

    std::vector<Eigen::Vector3f> sweep_room_;                              // latest helios sweep, ROOM frame
    Eigen::Vector3f              origin_room_ = Eigen::Vector3f::Zero();   // helios sensor centre, ROOM frame
    bool                         helios_fresh_ = false;
    std::vector<Eigen::Vector3f> sweep_bpearl_room_;                       // latest bpearl sweep, ROOM frame
    Eigen::Vector3f              origin_bpearl_room_ = Eigen::Vector3f::Zero();
    bool                         bpearl_fresh_ = false;
};

}  // namespace rc
