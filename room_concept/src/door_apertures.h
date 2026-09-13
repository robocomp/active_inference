#pragma once

// ── OPEN DOORS ARE HOLES IN THE ROOM, AND THE SCAN GOES THROUGH THEM ─────────────────────────────
// A LiDAR beam that passes through an open doorway terminates on something in the NEXT room. That
// return is a real measurement of the world and a false measurement of THIS room: it sits far outside
// the layout, so the SDF residual counts it as error, the pose fit is dragged toward explaining it,
// and the wall segmenter may try to grow a wall out of it. Opening a door should not make the
// localiser think its room got worse.
//
// ★ THE DOOR'S OWN BELIEF DECIDES, NOT A THRESHOLD. door_concept publishes `door_open_prob` on each
//   door node — P(clear width >= robot passage width), already marginalised over the leaf-angle
//   posterior rather than read off its argmax. A beam crossing an aperture is weighted
//
//        w = 1 - p_open
//
//   so a closed door leaves the point at FULL weight (a closed door IS a surface at the wall line and
//   its returns are legitimate wall evidence), a fully open one contributes nothing, and a half-open
//   one degrades smoothly. There is no gate anywhere and nothing to tune: the continuous quantity the
//   door agent already infers is the continuous quantity the weight uses.
//   A door that has never been measured publishes the unmeasured sentinel; that means "no evidence",
//   which weighs 1. Discounting returns because we know nothing about a door would throw away real
//   wall evidence — the failure would be silent and would look like a worse sensor.
//
// ★ GEOMETRY, from door_concept's own contract (door_geometry.h): the door NODE's RT pose is the
//   APERTURE, not the leaf — "nothing here moves when the leaf swings", which is exactly the
//   invariant that lets the RT edge key on it — and its yaw is the wall tangent. So the opening is
//   the segment  centre +- (width_m / 2) * (cos yaw, sin yaw).
//
// ⚠ THREADING, CHECKED AGAINST THE INSTALLED CORTEX HEADER ON 2026-09-13 — NOT against this repo's
//   CLAUDE.md, which still describes the older contract and is now wrong on this point.
//   dsr_inner_eigen_api.h carries `mutable std::mutex cache_mutex` guarding the ts == 0 transform
//   cache, so that path is no longer a corruption hazard and a SINGLE instance may be called from
//   several threads. The lock is held in two short sections and never across the tree walk, so two
//   threads can both miss and both compute the same transform: duplicated work, no corruption.
//   The rule that remains is about OWNERSHIP, not about which thread calls: the invalidation slots
//   are Qt::QueuedConnection and fire on the thread that owns the instance, so use the one owned by
//   the main thread. An instance created on a raw std::thread has no event loop, never receives an
//   invalidation, and serves stale transforms for ever — which trades a crash for a silent
//   correctness bug, the worse of the two.

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DSR { class DSRGraph; class InnerEigenAPI; }

namespace rc
{
    /// One doorway, as a segment in the ROOM frame, with the door's own belief that it is passable.
    struct DoorAperture
    {
        Eigen::Vector2f a{0.f, 0.f};   ///< one edge of the opening
        Eigen::Vector2f b{0.f, 0.f};   ///< the other
        float           p_open = 0.f;  ///< door_open_prob; 0 when unmeasured (= no discount)
        std::uint64_t   id     = 0;    ///< the door node, for logging
        std::string     name;
    };

    /// ── THE TRANSFORM IS RARE; THE ATTRIBUTE READ IS NOT ────────────────────────────────────────
    /// Two facts decide the shape of this class, and together they make the expensive half almost
    /// never run:
    ///   · door_open_prob and width_m are ordinary attribute reads — serialised by DSRGraph's own
    ///     shared_mutex, safe from any thread, no cache involved.
    ///   · the APERTURE DOES NOT MOVE WHEN THE LEAF SWINGS (door_concept's invariant). A door's
    ///     segment is therefore constant for as long as its wall is, so it is resolved ONCE, the
    ///     first time that door is open, and reused from then on.
    /// So the per-cycle cost of a room whose doors are shut is a handful of attribute reads and
    /// nothing else: no transform walk, no cache touched, nothing to hand across a thread boundary.
    /// The chain is only walked when a door's own belief says it is open and we have no segment for
    /// it yet — which happens once per door per run, not once per cycle.
    class DoorApertures
    {
    public:
        /// Refresh from the graph. Safe from any thread PROVIDED `inner` is the instance owned by the
        /// main thread (see the threading note above); it is called from compute() here because that
        /// is where that instance lives, not because the call site must be there.
        /// Returns the apertures currently worth discounting (p_open > 0). All doors shut ⇒ empty,
        /// and empty costs the scan nothing at all.
        std::vector<DoorAperture> refresh(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner,
                                          const std::string& room_frame);

        /// How many chain walks have actually been performed since construction. Expected to settle
        /// at "one per door that has ever been open" — if it climbs with time, the cache is not
        /// holding and that is a bug worth seeing rather than a cost worth paying.
        long resolves() const { return resolves_; }

        /// Why a door did NOT become an aperture. Written to tmp/door_filter.csv once per refresh,
        /// one row per door node, because this pipeline has four places to fail silently — the node
        /// is not found, its subtype is not "door", its belief says shut, or the room<-door chain
        /// does not resolve — and from the outside all four look identical: the scan is not filtered.
        struct Tally
        {
            int objects = 0;        ///< object nodes seen
            int named_door = 0;     ///< ... named door_*
            int subtype_ok = 0;     ///< ... with object_subtype == "door"
            int believed_open = 0;  ///< ... whose door_open_prob is in (0,1]
            int no_width = 0;       ///< ... rejected for a missing/absurd width_m
            int no_transform = 0;   ///< ... whose room<-door chain did not resolve
            int emitted = 0;        ///< ... that became an aperture this cycle
        };
        const Tally& tally() const { return tally_; }

        /// The weight this scan point keeps, given where the beam came from. Room frame, both.
        /// 1 when no aperture lies between them; the product of (1 - p_open) over those that do, so
        /// two open doors in line discount once each rather than cancelling.
        static float weight(const std::vector<DoorAperture>& apertures,
                            const Eigen::Vector2f& beam_origin, const Eigen::Vector2f& hit);

    private:
        /// Door node id -> its aperture segment in the room frame, resolved once. Nothing here
        /// expires on a timer: a segment is dropped when its door leaves the graph, and re-resolved
        /// if the door's wall is re-parented (which changes the node's RT edge, not this cache — see
        /// refresh() for the one case that forces a re-walk).
        std::unordered_map<std::uint64_t, std::pair<Eigen::Vector2f, Eigen::Vector2f>> cache_;
        long resolves_ = 0;
        Tally tally_;
    };
}   // namespace rc
