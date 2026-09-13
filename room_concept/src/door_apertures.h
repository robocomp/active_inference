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
// ⚠ READ ON THE MAIN THREAD ONLY. read_from_graph() resolves the room<-door chain through
//   InnerEigenAPI with timestamp 0, whose cache is unlocked and single-threaded per instance
//   (CLAUDE.md). The result is a small immutable snapshot handed to the localiser thread, exactly as
//   the object anchors are.

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <string>
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

    class DoorApertures
    {
    public:
        /// MAIN THREAD ONLY (ts == 0 inner_eigen). Returns every door of the current room as an
        /// aperture segment; an empty result simply means no door is known, and every weight is 1.
        static std::vector<DoorAperture> read_from_graph(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner,
                                                         const std::string& room_frame);

        /// The weight this scan point keeps, given where the beam came from. Room frame, both.
        /// 1 when no aperture lies between them; the product of (1 - p_open) over those that do, so
        /// two open doors in line discount once each rather than cancelling.
        static float weight(const std::vector<DoorAperture>& apertures,
                            const Eigen::Vector2f& beam_origin, const Eigen::Vector2f& hit);
    };
}   // namespace rc
