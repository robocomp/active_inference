#pragma once
// ─────────────────────────────────────────────────────────────────────────────────────────────
//  THE BOX LAYOUT CHANNEL — the redesign of room_boxes.h wired into the live agent
//
//  Deliberately a SIDECAR. It consumes exactly what the bench consumes (a robot-frame scan, a
//  map-frame pose, that pose's covariance) and produces exactly what the bench scores (a polygon).
//  It reaches into nothing: no window, no solver, no wall map, no re-anchor. The existing wall-SLAM
//  layout keeps running untouched beside it, so turning `BoxLayout` on can change WHAT IS PUBLISHED
//  and nothing else, and turning it off restores the shipped behaviour exactly.
//
//  ★ WHY A SIDECAR AND NOT A REPLACEMENT. The live agent currently holds a 6.00 x 4.00 rectangle to
//    2-3 mm while driving ([[rectangle-survives-driving-four-fixes]], d307c84). That result was
//    expensive and is not being staked on an estimator whose ladder has open rows. Running both and
//    publishing one is how the comparison gets made on the robot instead of being argued about.
//
//  ⚠ THE BENCH AND THE AGENT MUST SHARE THE SAME SOURCE FILE, and they do — room_boxes.cpp is in
//    both targets. This is not a style preference: on 2026-09-19 the bench passed while the agent
//    failed for six minutes purely because bin/room_concept was older than the fix, and "the bench
//    proves something about the agent" is only true when there is one implementation.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#include <Eigen/Dense>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "room_boxes.h"

namespace rc::boxch
{
    struct Params
    {
        bool  enabled      = false;   ///< publish the box polygon instead of the wall-map one
        int   every_frames = 40;      ///< how often structure is re-examined (COMPUTE, not a bar)
        float sensor_sigma = 0.02f;   ///< PHYSICAL: LiDAR range noise perpendicular to a wall
        float sigma_flat   = 0.010f;  ///< PHYSICAL: wall flatness / line scatter between views
        float cell         = 0.10f;   ///< COMPUTE: voxel + clustering resolution
        int   min_cluster  = 12;      ///< COMPUTE: smallest cluster worth proposing
        float z_min        = 0.20f;   ///< PHYSICAL: floor rejection, in the robot's own frame
        float z_max        = 1.60f;   ///< PHYSICAL: below the ceiling — see
                                      ///  [[lidar-high-band-must-not-reach-ceiling]], where ceiling
                                      ///  returns in a 2-D wall SDF stopped a room stabilising while
                                      ///  the pose looked perfectly healthy.
    };

    /// The channel's own state. One instance lives in RoomConcept; nothing else may touch it.
    class Channel
    {
    public:
        void configure(const Params& p) { p_ = p; }
        const Params& params() const { return p_; }
        bool enabled() const { return p_.enabled; }

        /// Fold one scan in. `pts_robot` is the robot-frame scan, `pose` the map-frame robot pose,
        /// `cov` that pose's 3x3 covariance. Cheap: a voxel update per point, no solve.
        /// `seg_phi` / `seg_len` are the agent's OWN wall segments for this scan: normal angle in
        /// the robot frame, and extent in metres. They carry the Manhattan vote.
        /// ⚠ THE VOTE MUST NOT READ CONSECUTIVE POINT DIFFERENCES. init_from_scan() does exactly
        /// that — it is the right estimator for a synthesised scan, which arrives in azimuthal
        /// order — but the agent's returns come through the media plane and a z-band filter, so
        /// consecutive entries are chords between unrelated returns. Measured live: the vote
        /// converged to +56.18 deg over 680 frames with 0.275 deg of drift, on a room whose true
        /// angle is ~0. ★ Averaging cannot rescue a BIASED estimator, only a noisy one, and a
        /// tight confidence interval around a wrong number is the most expensive kind of wrong.
        void observe(const std::vector<Eigen::Vector3f>& pts_robot,
                     const Eigen::Vector3f& pose, const Eigen::Matrix3f& cov,
                     const std::vector<float>& seg_phi, const std::vector<float>& seg_len);

        /// ── THE MAP FRAME MOVED ──────────────────────────────────────────────────────────────
        /// One-shot gauge change p' = R(-rot)(p - c), the same one RoomConcept::reanchor_map_frame
        /// applies to the window poses, the model tensors, the covariance and the wall map.
        /// ⚠ THE SIDECAR LIVES IN THE MAP FRAME AND MUST MOVE WITH IT. Its voxel map accumulates
        /// over the whole session, so a re-anchor that skips it leaves every past return stale by
        /// a rigid transform while new ones arrive in the new frame — TWO OFFSET COPIES OF THE
        /// SAME ROOM in one cloud. Measured live: 32.28% of cells more than 3 sigma outside the
        /// region and 31.89% more than 3 sigma inside, a near-symmetric split that no real room
        /// geometry produces, followed by extrusions chasing the ghost copy.
        /// ★★★ THE BENCH CANNOT REPRODUCE THIS. WS_BOXES runs odometry-only with no wall map, so
        /// it never re-anchors; the event does not exist there. That is the whole answer to "how
        /// can it work in the bench and not here" — the bench had no such event to get wrong.
        void reanchor(const Eigen::Vector2f& c, float rot);

        /// Re-examine the structure (and create the layout on the first call that can).
        /// Call once per frame; it self-throttles to `every_frames`.
        /// Returns true if the layout changed.
        bool step();

        /// The current layout, empty until the first scan has been folded in.
        const rc::boxes::Layout& layout() const { return L_; }
        /// The outer boundary, CCW, IN THE AGENT'S MAP FRAME, or empty if there is no layout yet.
        std::vector<Eigen::Vector2f> polygon() const;
        /// The layout's own rectilinear frame, as a rotation from the agent's map frame.
        float yaw() const { return yaw_; }

        // ── what the log needs to say whether this is working ──────────────────────────────────
        int   boxes()     const { return static_cast<int>(L_.boxes.size()); }
        int   proposed()  const { return n_proposed_; }
        int   admitted()  const { return n_admitted_; }
        int   removed()   const { return n_removed_; }
        float rms()       const { return rms_; }
        float last_dL()   const { return last_dL_; }
        std::size_t cells() const { return vmap_.size(); }
        /// Open the per-step log. Path is etc/box_layout.csv; grading SIZE needs the published
        /// polygon, and a canvas is not an instrument.
        void open_csv(const std::string& path);
        /// Largest 1-sigma pose uncertainty folded in on the last scan — the term whose absence
        /// let odometry drift buy structure at 449161 nats.
        float last_sigma() const { return last_sigma_; }
        /// Fraction of fused cells lying more than 3 sigma OUTSIDE / INSIDE the region, and the
        /// rms restricted to the cells that are neither. ★ A single rms cannot distinguish "the
        /// box is the wrong size" from "a tail of returns is somewhere the box will never reach":
        /// a run fitting to 0.108 m reported rms 0.337 m, and those two are only reconcilable if
        /// a minority of cells are very far off. Splitting the residual is what makes that
        /// testable instead of arguable.
        float frac_out() const { return frac_out_; }
        float frac_in()  const { return frac_in_; }
        float rms_core() const { return rms_core_; }

    private:
        struct Vox { Eigen::Vector2d acc{0.0, 0.0}; double w = 0.0; float smin = 1e9f; };

        /// ── FREE SPACE: THE CELLS THE BEAMS PASSED THROUGH ──────────────────────────────────
        /// ★★★★★ THE LAYOUT IS BUILT FROM THE INSIDE OUT, NOT THE OUTSIDE IN.
        /// The first design seeded one box = the interval HULL of the returns and let growth carve
        /// it down. That is right when a room is nearly convex — a rectangle with columns — and
        /// catastrophically wrong for an apartment, where the hull is a big rectangle and MOST OF
        /// IT IS NOT THE APARTMENT. Every return then sits deep inside the hull, reads as
        /// unexplained matter, and the estimator carves: measured on an 8-box apartment,
        /// **74 boxes, 290 vertices against 32, IoU 0.195**.
        /// The region the robot has actually traversed is the honest starting point, and it is
        /// something the sensor MEASURES rather than something the hull ASSUMES: a cell a beam
        /// passed through is free, and free space is the room. This is why the box-decomposition
        /// literature builds from occupancy rather than from a bounding volume.
        std::map<std::pair<int, int>, int> free_;
        /// The robot's own cell trail — always free, and the seed's guaranteed starting point.
        std::pair<int, int> last_cell_{0, 0};
        bool have_cell_ = false;

        /// Largest axis-aligned box of free cells containing `seed`, grown greedily one row or
        /// column at a time. No threshold: a side stops when the next line is not free.
        rc::boxes::Box grow_free_box(const std::pair<int, int>& seed) const;

        /// ── THE LAYOUT IS A BOX COVER OF THE OBSERVED FREE SPACE ────────────────────────────
        /// ★★★★★ Greedy local edits do not converge on apartment topology. Seeded from a hull and
        /// asked to carve, an 8-box apartment became 74 boxes at IoU 0.195; seeded from free space
        /// it became 17; with carves forbidden where the beams had swept, the churn simply moved
        /// into extrusions and it became 65. Each fix was right and none of them addressed the
        /// shape of the problem: an apartment is not a room with defects, it is a UNION, and it
        /// should be CONSTRUCTED as one rather than whittled toward one.
        ///
        /// So the layout is built directly: repeatedly take the largest axis-aligned rectangle of
        /// free cells not yet covered, and stop when the next one does not pay its own description
        /// length. That is the same MDL that governs grow(), applied to construction instead of to
        /// repair, and it is what the box-decomposition literature does — build from occupancy,
        /// not from a bounding volume. Returns then only REFINE the offsets, which is the job they
        /// are actually good at.
        bool rebuild_from_free();

        Params p_;
        rc::boxes::Layout L_;
        /// ⚠ THE LAYOUT LIVES IN ITS OWN RECTILINEAR FRAME, ROTATED BY `yaw_` FROM THE AGENT'S MAP
        /// FRAME. init_from_scan() builds its hull along the axes the quadrupled-angle vote
        /// recovered, so the box it returns is NOT expressed in the frame of the points handed to
        /// it. The bench absorbs this by carrying the robot pose in the box frame; this channel
        /// must do the same explicitly, because the agent's map frame belongs to every other
        /// consumer and may not be rotated under them.
        ///   in : map -> box   is R(-yaw_)
        ///   out: box -> map   is R(+yaw_)
        /// Getting this wrong fits an axis-aligned box to a cloud that is rotated relative to it:
        /// measured live in Webots as rms 0.379 m (19x the sensor noise) and a 6.00 x 4.00 room
        /// published as 4.221 x 5.684.
        float yaw_ = 0.f;
        /// Running circular mean of the per-frame Manhattan vote, accumulated on the QUADRUPLED
        /// angle so that the four indistinguishable quarter-turns land on the same direction and
        /// average instead of cancelling. This is the whole reason the gauge can be estimated at
        /// all: at 4*theta a square's symmetry is the identity.
        double yaw4_cos_ = 0.0, yaw4_sin_ = 0.0;
        long   yaw_votes_ = 0;
        bool   qInfo_first_ = false;
        std::map<std::pair<int, int>, Vox> vmap_;
        std::vector<rc::boxes::CloudPoint> cloud_;
        std::vector<Eigen::Vector2f> init_scan_;
        std::uint64_t frames_ = 0;
        int n_proposed_ = 0, n_admitted_ = 0, n_removed_ = 0;
        float rms_ = 0.f, last_dL_ = 0.f, last_sigma_ = 0.f;
        float frac_out_ = 0.f, frac_in_ = 0.f, rms_core_ = 0.f;
        std::ofstream csv_;

        void fuse();
    };
}   // namespace rc::boxch
