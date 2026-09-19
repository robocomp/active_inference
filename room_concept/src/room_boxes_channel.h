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
