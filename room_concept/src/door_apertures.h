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
        float           p_open = -1.f; ///< door_open_prob as published; -1 = unmeasured (logging only)
        /// π = P(the aperture holds a SURFACE: the leaf, or the door shut) — the prior of the per-beam mixture
        /// (weight()). 1 − p_open when door_concept has measured the door, ½ when it has not.
        float           prior_closed = 1.f;
        /// How far a return from that surface may lie from the aperture line: LiDAR noise ⊕ half the door depth.
        float           half_depth = 0.02f;
        /// δ (m) ≥ 0: how far the aperture line (the door node = the leaf's line) lies BEHIND the room's own
        /// boundary at the doorway. 0 until seat_in_room() runs; then a/b are the MOUTH (the aperture projected onto
        /// the room polygon's edge) and the leaf sits δ past it. See seat_in_room().
        float           recess = 0.f;
        /// Unit normal of the MOUTH pointing OUT of the room (toward the next room); zero until seat_in_room()
        /// finds the room's edge at the doorway. Lets weight() tell which side of the doorway a return ended on.
        Eigen::Vector2f n_out{0.f, 0.f};
        std::uint64_t   id     = 0;    ///< the door node, for logging
        std::string     name;
    };

    /// ── THE TRANSFORM IS RARE; THE ATTRIBUTE READ IS NOT ────────────────────────────────────────
    /// Two facts decide the shape of this class, and together they make the expensive half almost
    /// never run:
    ///   · door_open_prob and width_m are ordinary attribute reads — serialised by DSRGraph's own
    ///     shared_mutex, safe from any thread, no cache involved.
    ///   · the APERTURE DOES NOT MOVE WHEN THE LEAF SWINGS (door_concept's invariant). A door's
    ///     segment is therefore constant for as long as its wall is, so it is resolved ONCE and
    ///     reused from then on.
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
        /// Returns EVERY door with a resolvable aperture (2026-10-12: an unmeasured door is no longer treated as
        /// shut; the per-beam mixture decides from each return's range). No doors ⇒ empty, which costs nothing.
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
        /// ★ PER-BEAM MIXTURE (2026-10-12). A beam that crosses the aperture segment and returns d metres past
        ///   it is explained either by a SURFACE in the aperture (the leaf, a shut door: d ~ N(0, σ), σ² =
        ///   sigma_obs² + half_depth²) or by the NEXT ROOM (d uniform over kBeyondExtentM). Its weight is the
        ///   posterior of the surface:   w = π·N(d) / (π·N(d) + (1 − π)·U).
        ///   A return 2 m beyond the doorway gets w ≈ 0 even when door_concept has not measured the door (π = ½);
        ///   a return at the doorway line keeps w ≈ 1 unless the door is believed wide open. No threshold: the
        ///   crossing test is exact geometry and the rest is a likelihood ratio.
        /// ★ THE ROOM'S BOUNDARY AT A DOORWAY IS NOT THE DOOR'S LINE (2026-10-12). door_concept puts the door node at
        ///   the leaf's line — the WALL CENTRE in the apartment world — while the room polygon the SDF scores against
        ///   draws its wall at the room-side FACE. Measured live (crossing try15, room_2, door shut): the face is at
        ///   y = 4.84, the aperture at y = 4.62, and 8.9 % of every scan landed in between (header/frame/leaf returns
        ///   at y 4.64–4.72, 12–20 cm outside the polygon) — 85 % of the scan's Σ sdf². None of them CROSSED the
        ///   aperture segment, so every one kept weight 1. This moves the crossing test to the MOUTH — the aperture
        ///   projected onto the polygon edge nearest to it — and records δ = the distance from that edge to the leaf
        ///   line, so weight() can say how far past the room's boundary the surface hypothesis actually sits.
        ///   Polygon in the SAME frame as the aperture. No polygon / degenerate edge ⇒ the aperture is left as is.
        static void seat_in_room(DoorAperture& ap, const std::vector<Eigen::Vector2f>& polygon);

        static float weight(const std::vector<DoorAperture>& apertures,
                            const Eigen::Vector2f& beam_origin, const Eigen::Vector2f& hit, float sigma_obs);
        /// The "next room" hypothesis spreads a return uniformly over this many metres past the doorway. The
        /// weight depends on it only through log(L) (the crossover moves ~1 cm per doubling), so any value of the
        /// order of the sensor range will do.
        static constexpr float kBeyondExtentM = 20.f;

    private:
        /// Door node id -> its aperture segment in the room frame, and the door's OWN RT edge (parent → door:
        /// translation ++ rotation) it was resolved from. Nothing here expires on a timer: a segment is dropped when its
        /// door leaves the graph, and RE-WALKED when door_concept rewrites that edge (Task R2: it used to be walked once
        /// per node and reused for the node's whole life, so a door the producer later re-fitted stayed where it was first
        /// read — live rtD1i, door_5 read vertical 1.1 s after each restore and frozen there for the visit).
        struct Cached { std::pair<Eigen::Vector2f, Eigen::Vector2f> seg; std::vector<float> rt; };
        std::unordered_map<std::uint64_t, Cached> cache_;
        long resolves_ = 0;
        Tally tally_;
        /// ★ OCCLUSION HOLDS (2026-10-18). The apertures emitted last refresh, and those whose door node has since
        /// left the graph. A deleted node is not evidence that the opening closed (CONCEPT_AGENT_LIFECYCLE: absence
        /// removes, occlusion HOLDS) — door_concept deletes its nodes on every restart. Held until a door node
        /// reappears on the same doorway or the room frame changes; see refresh().
        std::unordered_map<std::uint64_t, DoorAperture> last_emitted_, held_;
        std::string frame_;
    };
}   // namespace rc
