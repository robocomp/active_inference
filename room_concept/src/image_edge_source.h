/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#pragma once
#include "object_anchor_types.h"

/*
 *  image_edge_source.h — turns one grayscale frame + the room model into ImageEdgeObs.
 *
 *  It enumerates the model's STRUCTURAL contours in the room frame, projects them through the
 *  camera at the CURRENT pose estimate, searches the image along each projected contour's normal
 *  for the intensity edge, and reports the signed sub-pixel displacement with a derived precision.
 *
 *  ★★★ THE CIRCULARITY TRAP — the single fatal error this class exists to avoid.
 *
 *  CameraVisualizer::predicted_camera_from_room() builds camera_T_room by reading room<-robot FROM
 *  THE GRAPH and dead-reckoning it forward. That transform IS this agent's own published output.
 *  It is the right thing for an overlay and it is why the overlay "fits really good" — but a factor
 *  built on it would have r == 0 at the published pose BY CONSTRUCTION. The term would carry exactly
 *  zero information while every diagnostic looked perfect (small loss, centred residuals), and once
 *  it drove the pose it would weld the estimate to wherever it already was, including to a wrong
 *  pose after a symmetry flip.
 *
 *  So: the ONLY thing taken from the graph is the STATIC camera<-robot extrinsic, read once at
 *  bring-up by CameraIngestor. room<-robot is the pose handed in by the caller for the search, and
 *  it is the VARIABLE the factor differentiates. Nothing in this file may call
 *  get_transformation_matrix for room<-robot. If you are about to add such a call, you are about to
 *  reintroduce the bug.
 *
 *  ★ The normal search runs ONCE PER FRAME here, at the pre-solve pose — NOT inside linearize().
 *    uv_meas must be a constant of the solve, or LM's accept/reject test tracks a moving target. If
 *    a large correction walks the prediction out of the searched window, the mixture responsibility
 *    collapses and the sample mutes itself, which is the correct behaviour.
 */

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "image_edge_types.h"

namespace rc
{
    class ImageEdgeSource
    {
    public:
        struct Config
        {
            bool  use_wall_corners   = true;
            bool  use_floor_junction = false;
            bool  use_wall_ceiling   = false;
            float sample_spacing_m   = 0.10f;
            float search_sigmas      = 3.0f;
            int   max_search_px      = 64;
            float mount_pitch_sigma  = 0.0035f;
            float mount_height_sigma = 0.010f;
            float mount_yaw_sigma    = 0.0035f;
            float wall_position_sigma = 0.015f;   ///< m — see IMAGE_EDGE_NUISANCES column [4]
            float room_height        = 2.5f;
        };

        /// Per-frame counters for the CSV. Every one of these answers a question the plan
        /// pre-registered; none of them is decoration.
        struct Stats
        {
            int   n_contours    = 0;
            int   n_projected   = 0;   ///< samples whose prediction landed in the image
            int   n_visible     = 0;   ///< ... and survived the occlusion prior
            int   n_searched    = 0;   ///< ... and produced a finite-sigma edge
            int   n_occluded    = 0;
            int   n_clamped     = 0;   ///< search half-length hit the compute bound
            float med_sigma_px  = 0.f;
            float med_search_px = 0.f;
            float sigma_i       = 0.f;
            /// Triple points formed, and how many of those the occlusion prior disbelieves.
            /// ★ Reported separately from n_occluded (which counts SAMPLES): with drive = true
            ///   the corners are what moves the pose, so "how many corners, and how many of them
            ///   are behind a wall" is the number that says whether the term is being fed.
            int   n_triple      = 0;
            int   n_triple_occl = 0;
        };

        ImageEdgeSource() = default;

        void set_config(const Config& c) { cfg_ = c; }
        [[nodiscard]] const Config& config() const noexcept { return cfg_; }

        /// The room polygon (room frame, ordered). Taken from the MODEL (the SVG source of truth),
        /// never from the DSR `room` node — that mirror is written by this agent, so reading it back
        /// would be a second, quieter circularity.
        /// Objects that may stand between the camera and a wall. Occlusion by them enters as a soft
        /// visibility PRIOR on each sample, never as a cull — see the pi_vis block in the .cpp.
        void set_object_anchors(std::vector<ObjectAnchorObs> a) { anchors_ = std::move(a); }

        void set_room_polygon(std::vector<Eigen::Vector2f> poly) { polygon_ = std::move(poly); }

        /// Extract one frame's evidence.
        ///   frame       : grayscale + measured sigma_i
        ///   model       : the reduced projection model (from CameraIngestor)
        ///   cam_R/t_robot: the STATIC extrinsic (from CameraIngestor)
        ///   pose        : room<-robot [x, y, theta] at which to PREDICT and SEARCH (the pre-solve pose)
        ///   pose_cov    : 3x3 posterior covariance; sets the search window. Pass Zero if unknown.
        ///   body_twist  : [vx, vy, omega] in the robot frame, for the dt nuisance column. Zero is safe.
        ///   dt_ms       : image stamp - slot stamp
        [[nodiscard]] ImageEdgeObs extract(const GrayFrame& frame,
                                           const CameraModel& model,
                                           const Eigen::Matrix3f& cam_R_robot,
                                           const Eigen::Vector3f& cam_t_robot,
                                           const Eigen::Vector3f& pose,
                                           const Eigen::Matrix3f& pose_cov,
                                           const Eigen::Vector3f& body_twist,
                                           std::int64_t dt_ms,
                                           Stats* stats = nullptr) const;

    private:
        Config cfg_;
        std::vector<ObjectAnchorObs> anchors_;
        std::vector<Eigen::Vector2f> polygon_;
    };
}  // namespace rc
