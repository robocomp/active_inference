// door_core.h — DoorCore: door_concept's per-cycle DECISION logic with no DSR / Qt / Ice / media dependency, driven by
// one DoorCycleInput per compute() cycle (docs/superpowers/specs/2026-10-21-door-core-replay-design.md).
//
// ★WHAT IS IN IT NOW (Task 4 of .superpowers/sdd/2026-10-21-door-core-replay): the TRACKER slice — merge of duplicate
// instances, the shared InstanceTracker's birth/association bookkeeping, the evidence association (Pass 1 / Pass 2),
// and the birth decision with its four suppressions (outside the room, not door-height, explained by a live door's
// swept arc or aperture, claimed by a live door) and re-acquisition by ghost or stored identity. Moved, not rewritten:
// specificworker.cpp's run_instance_tracker / merge_overlapping_instances, plus the two pieces of DoorFitter the
// birth rule leans on (frame_admissible and the ego-motion it reads; point_in_room).
//
// ★WHAT IS NOT IN IT YET, AND HOW IT GETS IN. Existence (Task 5), identity / room following / carry / restore / look
// (Task 6), affordances and the anchor (Task 7) — and the FIT itself (observe / run_inference / estimate_phi: the
// instance beliefs). The tracker reads the state those slices own (which room, the room polygon, each door's fitted
// pose and covariance, ghosts, identities) and acts on it (retire a merged door, hand a slice to a door, create and
// adopt a newborn). It does both ONLY through TrackerWorld below: live, SpecificWorker implements it over its DSR-bound
// instances; in the replay tool, a world rebuilt from the recorded decision log does. As each slice moves in, its part
// of TrackerWorld becomes core state and the interface shrinks.
//
// ★ONE CALL, IN THE ORDER THE OLD CODE RAN. step() = merge → associate → TrackerWorld::after_association() (live: the
// semantic/image pumps and the existence update, which ran between association and birth) → births. Effects on the
// world are made SYNCHRONOUSLY at the point the old code made them (a newborn is created and adopted before the next
// birth is judged: a second birth in the same cycle is tested against the first one's swept arc), and decisions reach
// the sink at the same point, so the DDEC stream keeps its order (IdentityUpsert written by the old identity code
// inside adopt_birth still lands between REACQUIRE and BIRTH).
// ★NODE IDS. A door node gets its DSR id when the adapter inserts it. The core asks for it through create_door() and
// gets the answer in the same call — the ruling's NodeIdAssigned, delivered synchronously, because birth is ATOMIC in
// this agent (create → adopt → own the birth slice in one cycle; decoupling those steps was the 403-birth churn the
// birth block's own comment records). The replay answers create_door() with the id the recorded run got.
#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <Eigen/Dense>
#include "door_cycle_io.h"
#include "sensing.h"
#include "../door_config.h"   // rc::DoorConfig (plain struct)
#include "../door_model.h"    // rc::DoorState
#include "../door_belief.h"   // rc::DoorBelief (association_nll)
#include "../door_geometry.h" // rc::door::Aperture / LeafPose
#include "../../../common/instance_tracker/instance_tracker.h"   // rc::InstanceTracker (SHARED, Eigen-only)
#include "../../../common/exclusion/claims.h"                    // rc::exclusion::Claim (SHARED, pure)

namespace rc::door::core
{

// ═══ THE WORLD THE TRACKER SLICE READS AND ACTS ON (state owned by slices not yet in the core) ════════════════════

/// One live door instance as the tracker sees it — exactly the fields run_instance_tracker read off rc::DoorInstance.
struct TrackedDoor
{
    std::uint64_t id = 0;
    std::string   name;
    rc::DoorState state{};                 // model.state(): leaf centre (cx,cy), aperture (ap_*), w, thickness
    rc::door::Aperture aperture{};
    rc::door::LeafPose leaf_pose{};
    float phi_est = 0.0f;
    bool  is_bearing_hypothesis = false;
    bool  ai2_initialized = false;
    const rc::DoorBelief* belief = nullptr;   // the AI2 belief when initialised; null ⇒ nothing to score association with
    float chain_cov_xx = 0.0f, chain_cov_yy = 0.0f;
    bool  roi_valid = false;
    int   matched_frames = 0;
    int   assigned_mask_idx = -1;
};

/// A removed door remembered in RAM (SpecificWorker::DoorGhost) — what re-acquisition needs to know about it.
struct GhostMatch
{
    std::string name;
    Vec2f xy = Vec2f::Zero();
    int   lived_cycles = 0;
    std::size_t handle = 0;   // the world's own reference to it (reacquire() resumes ITS belief)
};

/// A stored identity row (etc/door_identities.csv) matched by place in the current room.
struct IdentityMatch
{
    std::string name;
    Vec2f xy = Vec2f::Zero();
};

/// What the tracker asks the world to create (DoorSceneGraph::create_instance_from_detection's arguments).
struct BirthRequest
{
    Vec3f centroid = Vec3f::Zero();        // room frame
    std::uint64_t room_id = 0;
    std::string preferred_name;            // a ghost's or an identity's name; empty = allocate
    std::vector<std::string> reserved;     // names a ghost or a stored identity still holds
    int slice = -1;
};

/// The per-cycle numbers the dashboard and the remembered-door look read (ev_g_, look_new_frame_, look_dets_xy_).
struct TrackerCycleInfo
{
    int  mask_frame_id = -1;
    int  total_slices = 0;
    int  class_dets = 0;                   // "door" detections offered to the tracker
    int  assigned = 0;                     // of them, assigned to a track by the InstanceTracker
    int  tracker_births = 0;               // InstanceTracker promotions (before the suppressions)
    int  tracker_deaths = 0;
    int  merges = 0;
    bool new_frame = false;                // a NEW mask frame arrived this cycle (birth evidence counts it once)
    std::vector<Vec2f> dets_xy;            // this frame's door detections (room frame)
};

class TrackerWorld
{
public:
    virtual ~TrackerWorld() = default;
    // ── read: room following (Task 6), the fit, ghosts + identities (Tasks 5/6) ──
    [[nodiscard]] virtual std::uint64_t room_id() const = 0;
    /// The room-containment prior's polygon as the fit holds it (empty = none loaded: no prior).
    [[nodiscard]] virtual std::vector<Vec2f> room_polygon() const = 0;
    /// The live instances, in the agent's own iteration order (ties in the association resolve by it).
    [[nodiscard]] virtual std::vector<TrackedDoor> doors() const = 0;
    [[nodiscard]] virtual std::optional<GhostMatch> match_ghost(const Vec2f& xy) const = 0;
    [[nodiscard]] virtual std::optional<IdentityMatch> match_identity(const Vec2f& xy) const = 0;
    [[nodiscard]] virtual std::vector<std::string> reserved_names() const = 0;
    // ── act ──
    virtual void retire(std::uint64_t id) = 0;                           // a merged duplicate: affordances + fit + node
    virtual void set_assigned_mask(std::uint64_t id, int slice) = 0;     // -1 = none this cycle
    /// The old code between association and births (live: semantic/RGB/depth pumps, actuation UI, detect probe,
    /// update_existence_beliefs — which may retire doors). The cycle's tracker numbers are final at this point.
    virtual void after_association(const TrackerCycleInfo& info) = 0;
    /// Create the door node; 0 = the insert failed (nothing else happens for that birth).
    virtual std::uint64_t create_door(const BirthRequest& r) = 0;
    virtual void reacquire(std::uint64_t new_id, const GhostMatch& g) = 0;   // resume the ghost's belief
    virtual void forget_ghost(const Vec2f& xy) = 0;                        // consume the ghost(s) at that place
    /// Materialise the newborn's instance NOW, hand it its birth slice, note its identity; its name, or nullopt if the
    /// node could not be read back (no instance this cycle).
    virtual std::optional<std::string> adopt_birth(std::uint64_t id, int slice, const Vec3f& centroid) = 0;
};

// ═══ THE CORE ════════════════════════════════════════════════════════════════════════════════════════════════════

struct DoorCoreConfig
{
    rc::DoorConfig door{};   // the agent's whole config (door_config_fields.h reads it; the replay rebuilds it)
};

/// Where decisions and diagnostic rows go AS THEY ARE MADE (live: the DDEC log + door_events.csv, stdout, the dets CSV).
struct DoorCoreSinks
{
    std::function<void(const rc::door::DoorDecision&)> decision;
    std::function<void(const DiagRow&)> diag;   // stream "stdout" (line ends in '\n') or a CSV file name
};

class DoorCore
{
public:
    explicit DoorCore(const DoorCoreConfig& cfg);

    void connect(TrackerWorld* world, DoorCoreSinks sinks = {}) { world_ = world; sinks_ = std::move(sinks); }

    /// Take the cycle's media (mask packet, bpearl sweep) into the views. step() calls it; call it ALONE on a cycle
    /// whose decision step does not run (no room) — see sensing.h.
    void observe_media(const DoorCycleInput& in);

    /// One cycle of the slices the core owns. Task 4: the tracker slice. Requires connect().
    DoorCycleOutput step(const DoorCycleInput& in);

    [[nodiscard]] const TrackerCycleInfo& tracker_info() const { return info_; }
    [[nodiscard]] const EgoMotion& ego() const { return ego_; }
    [[nodiscard]] const MaskSource& masks() const { return masks_; }
    [[nodiscard]] const std::vector<rc::exclusion::Claim>& claims() const { return claims_; }

    // ── pure rules shared with the adapter's DoorFitter (ONE implementation) ──
    /// May this frame move geometry — and therefore create it? (moved from DoorFitter::frame_admissible)
    [[nodiscard]] static bool frame_admissible(const MaskSlice& sl, const EgoMotion& ego, const rc::DoorConfig& cfg);
    /// The other concepts' claims on room space, from the recorded object nodes (rc::exclusion::foreign_claims's rule).
    [[nodiscard]] static std::vector<rc::exclusion::Claim> claims_from(const std::vector<ObjectInfo>& objects);

private:
    // (not "emit": Qt #defines it away in every TU that includes Qt before this header)
    void record(rc::door::DoorDecisionKind k, std::uint64_t id, float x, float y, std::string note, std::string name,
              DoorCycleOutput& out);
    void diag(std::string stream, std::string line, DoorCycleOutput& out);
    void merge(DoorCycleOutput& out);
    void associate(const DoorCycleInput& in, DoorCycleOutput& out);
    void births(const DoorCycleInput& in, DoorCycleOutput& out);

    DoorCoreConfig cfg_;
    TrackerWorld*  world_ = nullptr;
    DoorCoreSinks  sinks_;

    MaskSource masks_;
    LeafSweep  leaf_sweep_;
    EgoMotion  ego_;
    std::vector<rc::exclusion::Claim> claims_;

    rc::InstanceTracker tracker_;
    long last_birth_mask_frame_ = -1;
    int  dbg_cycles_ = 0;     // the "[tracker] instances=…" print throttle
    int  dets_csv_rows_ = 0;  // etc/door_dets_log.csv's cycle column (one per tracker call)
    std::uint64_t create_seq_ = 0;

    // this cycle
    std::vector<rc::DetectionView> dets_;
    rc::TrackerResult res_;
    TrackerCycleInfo info_;
};

/// The name a newborn door node gets (DoorSceneGraph::next_door_name's rule, moved here so the replay names a door
/// exactly as the live agent does): `preferred_name` if no node in `graph_door_names` (every "door_*" object node in
/// the graph, live) already wears it, else the LOWEST free door_<N> (N ≥ 1) — free of both the graph's names and the
/// `reserved` ones (ghosts, stored identities).
[[nodiscard]] std::string allocate_door_name(std::string_view preferred_name, const std::vector<std::string>& graph_door_names,
                                             std::span<const std::string> reserved);

/// point_in_room's rule (DoorFitter delegates here): inside the polygon, or outside by no more than margin_m. No
/// polygon (< 3 vertices) ⇒ true — an unknown room imposes no prior.
[[nodiscard]] bool point_in_polygon(const std::vector<Vec2f>& poly, const Vec2f& q, float margin_m);

}  // namespace rc::door::core
