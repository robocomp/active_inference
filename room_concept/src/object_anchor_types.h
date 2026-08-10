/*
 *  object_anchor_types.h — plain data type shared across the object-anchor subsystem.
 *
 *  An "object anchor" turns a VALIDATED modelled object (a table, later a chair, a
 *  fridge, …) into an SE(2) pose landmark for the room localizer.  As the room fills
 *  with confidently-localised objects, these landmarks add interior factors to the
 *  robot↔room free energy and the pose covariance tightens — without any threshold:
 *  a fresh, uncertain object has a huge covariance ⇒ its factor weight ≈ 0 (see
 *  object_anchor_source.h), a converged one pulls hard.
 *
 *  This header is deliberately free of BOTH <torch/torch.h> and the DSR headers so it
 *  can be the interface that crosses the localizer-thread / graph-thread boundary and
 *  so the torch factor TU and the DSR reader TU never have to include each other
 *  (avoids the torch/TBB/Qt include-order clashes noted in CLAUDE.md).
 *
 *  Frames / symbols:
 *    pose_world  = p_o = (x, y, yaw) of the object in the ROOM frame  — the stable map anchor
 *    obs_robot   = z_o = (x, y, yaw) of the object in the ROBOT frame — this frame's measurement
 *    information = Λ   = (Σ_o ⊕ R_o)^{-1}, the innovation precision (map cov Σ_o from the
 *                        object's own belief, plus this-frame measurement cov R_o).  If
 *                        has_orientation is false the yaw row/col of Λ are zeroed and the
 *                        yaw residual is ignored (symmetric object → position-only landmark).
 */
#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <string>

namespace rc
{
    struct ObjectAnchorObs
    {
        std::string     type;                                 // "table", … (registry key / logging)
        std::uint64_t   node_id = 0;
        Eigen::Vector3f pose_world = Eigen::Vector3f::Zero();  // room frame  (x, y, yaw)
        Eigen::Vector3f obs_robot  = Eigen::Vector3f::Zero();  // robot frame (x, y, yaw)
        Eigen::Matrix3f information = Eigen::Matrix3f::Zero();  // Λ = S^{-1}; yaw row/col zeroed if !has_orientation
        // ── For the VARIABLE-landmark path (room optimises p_o instead of pinning it) ────────────────
        // `information` folds Σ_o ⊕ R_o together, which is right only while p_o is a CONSTANT: the map's
        // uncertainty has to be paid for somewhere. Once p_o is a state variable, its uncertainty lives
        // in the state and folding Σ_o into the measurement weight double-counts it. So the two are also
        // published apart: the observation is weighted by R_o⁻¹ alone, and Σ_o becomes the landmark's
        // BIRTH PRIOR. Both are in the same frames as their combined form (R_o robot, Σ_o room).
        Eigen::Matrix3f meas_information = Eigen::Matrix3f::Zero();   // R_o^{-1}
        Eigen::Matrix3f map_cov          = Eigen::Matrix3f::Zero();   // Σ_o (room frame)
        bool            has_orientation = true;
        // Frames since the producer last actually saw the object (its own counter). The consumer already
        // uses it to inflate R_o (freshness as precision); it is carried here so a VIEWER can tell a muted
        // anchor apart from an uncertain one — "stale observation" and "uncertain map" both end up as a
        // big ellipse but are entirely different problems.
        int             obs_age = 0;
        float           map_pos_sigma = 0.f;                   // σ of the map pose Σ_o (m) — the "how confident
                                                               // is p_o" signal used to decide when to PIN it
    };
} // namespace rc
