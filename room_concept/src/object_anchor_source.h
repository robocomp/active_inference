/*
 *  object_anchor_source.h — the DSR-graph side of the object-anchor subsystem.
 *
 *  Walks the shared graph and turns each VALIDATED modelled object into an
 *  ObjectAnchorObs (plain data, see object_anchor_types.h).  Extensible by design: a
 *  Reader is registered per node type, so adding chairs/fridges/… later is one
 *  register_type() call — the localizer factor code never changes.
 *
 *  Validation is NOT a threshold here: every object is read, and its innovation
 *  precision Λ folds in the object's own belief covariance Σ_o.  A fresh, uncertain
 *  object therefore lands with Λ ≈ 0 and contributes nothing to localization; a
 *  converged one pulls hard.  (A near-square table with unresolved yaw carries a huge
 *  vyaw from its mode-entropy, so its orientation channel self-mutes the same way.)
 *
 *  Pose + covariance are read through DSR::InnerGaussianAPI (the covariance analog of
 *  InnerEigenAPI): get_transformation_gaussian(dest, orig) returns the object's pose AND
 *  a chain-propagated covariance in `dest`.  We query it in the object's PARENT (room)
 *  frame — the room→object edge covariance already folds the localization chain
 *  (table_concept), so recomposing through room→robot would double-count the robot pose.
 *
 *  MAIN THREAD ONLY: InnerGaussianAPI owns an RT_API whose ts==0 path hits the InnerEigen
 *  unlocked cache — safe single-threaded per instance with a Qt event loop (CLAUDE.md).
 *  Run gather() on the main thread and hand the plain-data vector to the localizer
 *  (RoomConcept::set_object_anchors).
 *
 *  No <torch/torch.h> here — DSR only.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <genericworker.h>          // DSR API + generated attribute tags
#include <dsr/api/dsr_inner_gaussian_api.h>

#include "object_anchor_types.h"

namespace rc
{
    class ObjectAnchorSource
    {
    public:
        struct Config
        {
            bool  enable         = false;
            // Which object CLASSES become landmarks, matched against object_subtype (or the node-name
            // prefix, which is the same discriminator every furniture agent uses). Default {"table"}
            // preserves the historical behaviour. This is a list and not a bool because the classes are
            // not interchangeable: a table's yaw is mode-ambiguous and its map pin was implicated in the
            // 07-13 delocalizations, while a refrigerator is a position-only, unambiguous landmark. Being
            // able to admit one without the other is the point.
            std::vector<std::string> subtypes = {"table"};
            // Fallback per-frame measurement noise R_o used when the producing concept does not
            // publish its own robot-frame covariance (obj_obs_robot_cov) on the node.
            float meas_sigma_xy  = 0.05f;   // metres
            float meas_sigma_yaw = 0.15f;   // radians
            // PIN threshold: once a table's map-pose σ drops below this (m), room snapshots ("pins") its
            // world pose and uses that FIXED value as p_o thereafter — breaking the circularity where p_o,
            // re-derived from the robot pose each frame, always agrees with room and carries no information.
            float validate_sigma = 0.10f;   // metres
            // Guard: NEW pins may only be created this frame when the ROOM localization is confident +
            // stable (set by RoomSceneGraph from the localizer's sustained good-fit state). A pin captured
            // mid-flip/mid-delocalization is worse than no anchor — it fights the correct pose forever.
            // Existing pins are always honoured; this only gates fresh captures.
            bool  allow_pin = true;
            // When true, room treats p_o as a VARIABLE it optimises privately (birth prior = the
            // producer's own belief) instead of pinning it. Mutually exclusive with the pin. GN-only:
            // the autograd backends have no landmark variables. Default off.
            bool  optimize_landmark = false;
            // ── Freshness as precision ──────────────────────────────────────────────────────────────
            // obj_obs_robot PERSISTS in the graph between actual sightings (the producer only rewrites it
            // when a mask is assigned; `table_detection_alive` stays true for ~40 frames). A stale in-front
            // observation paired with the moved robot pose is a PHANTOM landmark (confirmed 07-13). Instead
            // of a hard freshness gate, grow the measurement covariance R_o with the observation's age
            // (table_frames_since_detection): R_o ← R_o·(1 + age/age_scale)² ⇒ a stale obs lands with Λ≈0
            // and pulls nothing; a fresh (age 0) one keeps full weight. No threshold — smooth decay.
            bool  freshness_enable    = true;
            float freshness_age_scale = 3.0f;   // frames of staleness at which σ roughly doubles
        };

        /// A Reader fills `out` from `node`; returns false to skip it (e.g. no measurement yet).
        using Reader = std::function<bool(const DSR::Node&         node,
                                          DSR::DSRGraph&           G,
                                          DSR::InnerGaussianAPI&   gauss,
                                          const Config&            cfg,
                                          ObjectAnchorObs&         out)>;

        ObjectAnchorSource();                     // installs the built-in furniture reader (see Config::subtypes)

        void register_type(const std::string& node_type, Reader reader);

        /// Collect anchors for every registered type.  Empty when disabled.  MAIN THREAD ONLY.
        std::vector<ObjectAnchorObs> gather(DSR::DSRGraph& G, DSR::InnerGaussianAPI& gauss,
                                            const Config& cfg) const;

    private:
        std::map<std::string, Reader> readers_;
        // Pinned world poses (per table node id): the FIXED landmark p_o, snapshotted the first time the
        // table's map σ falls below Config::validate_sigma. Held thereafter (mutable: gather() is const).
        mutable std::map<std::uint64_t, Eigen::Vector3f> pinned_pose_;
    };
} // namespace rc
