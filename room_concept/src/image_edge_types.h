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

/*
 *  image_edge_types.h — plain data exchanged between the RGB edge subsystem and the localizer.
 *
 *  Deliberately free of <torch/torch.h>, DSR and Qt, for the same reason object_anchor_types.h is:
 *  this struct crosses a thread boundary (ingest -> compute) and enters both solver backends, so it
 *  must be cheap to copy, have value semantics, and drag no framework state behind it.
 *
 *  ★ The frame payload is std::vector<uint8_t>, NOT cv::Mat. A cv::Mat copy is a refcounted shallow
 *    handle; sharing one buffer across threads where either side writes is a heap smash whose victim
 *    surfaces later as an unrelated crash (CLAUDE.md). A vector has value semantics, so the hazard
 *    cannot arise by construction.
 */

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace rc
{
    /// One grayscale frame plus everything the measurement needs that only the producer knows.
    /// `sigma_i` is MEASURED per frame (Immerkaer), never configured — see image_edge_ops.h.
    struct GrayFrame
    {
        std::vector<std::uint8_t> gray;      ///< row-major, `width * height` bytes
        int           width   = 0;
        int           height  = 0;
        std::uint64_t stamp   = 0;           ///< producer capture stamp (ms)
        float         sigma_i = 0.f;         ///< per-frame intensity noise sigma (grey levels)
        bool          valid   = false;

        [[nodiscard]] bool ok() const noexcept
        { return valid and width > 0 and height > 0
                 and gray.size() == static_cast<std::size_t>(width) * static_cast<std::size_t>(height); }
    };

    /// A projection model reduced to plain numbers, so the torch mirror can reproject WITHOUT DSR.
    ///
    /// ★ Filled by calibrating against DSR::CameraAPI::project() itself, never by copying constants
    ///   out of a config. For the 360 models `azimuth_sign` / `azimuth_offset` are PRIVATE in
    ///   CameraAPI with no getters, so they are RECOVERED by probing project() with known directions
    ///   and then VERIFIED against it on a test set. Guessing the sign would flip the yaw Jacobian
    ///   and produce a solver that converges on the ZED and diverges only on the Ricoh.
    struct CameraModel
    {
        enum class Kind : std::uint8_t { Pinhole = 0, Equirect = 1, Cylindrical = 2 };
        Kind  kind = Kind::Pinhole;
        float fx = 0.f, fy = 0.f;      ///< pinhole focal lengths (px)
        float cx = 0.f, cy = 0.f;      ///< principal point (px) — image centre, everywhere in this codebase
        float width = 0.f, height = 0.f;
        float azimuth_sign = 1.f;      ///< 360 column convention (mirror)
        float azimuth_offset = 0.f;    ///< 360 seam zero (rad)
        bool  valid = false;
    };

    /// Which structural line a sample came from. The distinction is not cosmetic: a FloorWall sample
    /// carries RANGE and is exposed to the camera pitch/height nuisances (delta_d = theta_p * d^2 / h),
    /// while a WallCorner sample carries BEARING on a horizontal normal and is nearly immune to both.
    /// They are staged separately for exactly that reason.
    enum class ContourClass : std::uint8_t { FloorWall = 0, WallCorner = 1, WallCeiling = 2 };

    /// Number of shared (common-mode) nuisances modelled per contour segment.
    /// Columns: [0] mount pitch, [1] mount height, [2] mount yaw, [3] image/lidar dt,
    ///          [4] THIS CONTOUR'S OWN POSITION IN THE MAP.
    ///
    /// ★ WHY [4] EXISTS. Columns 0-3 are all GLOBAL to the frame: one pitch, one height, one yaw,
    ///   one time offset, shared by every contour. So a wall that is simply in the wrong place in the
    ///   map had nowhere to go except into the residual, and N samples along it counted as N
    ///   independent measurements of a position they all share.
    ///   mount_pooled_solve()'s own header predicted this before it was measured: "every sample taken
    ///   against one wall shares whatever is wrong with THAT wall — its position in the map, the floor
    ///   height beneath it, the residual pose error there."
    ///
    /// ★ MEASURED 2026-08-28, and it is a systematic, not noise. Per-vertex MEDIAN residuals spread
    ///   1.739 px in u and 0.813 px in v across six polygon vertices, against a fixed-pose
    ///   repeatability of 0.019 px and 0.093 px — 90x and 9x. Each corner sits consistently
    ///   somewhere the map does not say it is, frame after frame.
    ///   ★★ A LARGER sigma CANNOT reach this and would be the wrong response: the Cramer-Rao bound
    ///      already OVER-states the random noise (0.692 px stated against 0.019 px measured in u), so
    ///      inflating it only weakens the term while leaving the bias exactly where it was. The cure
    ///      has to be a model term, which is what this column is.
    inline constexpr int IMAGE_EDGE_NUISANCES = 5;

    /// One normal-search measurement on one projected contour sample.
    ///
    /// The residual is SCALAR and along the image-space contour normal. Only the normal component is
    /// observable (aperture problem); a 2-D residual would fabricate a tangential constraint that the
    /// image does not contain.
    struct ImageEdgeSample
    {
        Eigen::Vector3f p_room  = Eigen::Vector3f::Zero();  ///< model point, ROOM frame (a constant of the solve)
        Eigen::Vector2f n_hat   = Eigen::Vector2f::Zero();  ///< unit image-space contour normal at the prediction
        Eigen::Vector2f uv_meas = Eigen::Vector2f::Zero();  ///< sub-pixel edge found by the normal search
        float sigma_px  = 0.f;      ///< Cramer-Rao sigma of the edge location (px). Flat wall -> huge -> self-mutes.
        float pi_vis    = 1.f;      ///< predicted-visibility prior in [0,1] (occlusion enters HERE, not as a cull)
        float search_L  = 0.f;      ///< half-length actually searched (px) — the uniform outlier component's width
        /// Per-nuisance sensitivity, ALREADY contracted as n_hat^T * P * (.) — i.e. px per unit nuisance.
        Eigen::Matrix<float, IMAGE_EDGE_NUISANCES, 1> h = Eigen::Matrix<float, IMAGE_EDGE_NUISANCES, 1>::Zero();
    };

    /// All samples belonging to ONE contour segment. The grouping is what makes the common-mode
    /// marginalisation correct: the nuisances are shared WITHIN a segment, so the Woodbury correction
    /// is applied per segment, not globally and not per sample.
    struct ImageEdgeSegment
    {
        ContourClass class_id = ContourClass::WallCorner;
        /// Polygon vertex this contour belongs to: for WallCorner the vertex itself, for FloorWall
        /// and WallCeiling the edge's FIRST vertex (so edge i runs from vertex i to i+1). Carried so
        /// the triple-point detector can pair a vertical corner with the floor junctions that meet
        /// it; without it a segment is anonymous and the pairing would have to be re-derived from
        /// sample geometry.
        int vertex = -1;
        std::vector<ImageEdgeSample> samples;
    };

    /// The FLOOR-WALL-WALL triple point: where two walls and the floor meet, i.e. a room polygon
    /// vertex at z = 0, and the bottom end of a vertical wall corner.
    ///
    /// ★ WHY IT IS WORTH HAVING. Every existing sample is a SCALAR along its contour's normal —
    ///   1-D by necessity, because a 2-D residual on a straight edge would fabricate a tangential
    ///   constraint the image does not contain (the aperture problem; see ImageEdgeSample). A
    ///   triple point has no aperture problem: it is a genuine 0-D feature with two independent
    ///   components. And it is the SAME physical (x, y) the LiDAR corner detector reports — the
    ///   LiDAR sees the wall-wall intersection at sensor height, the image sees it at floor height —
    ///   so the association between the two sensors is exact in the plane, not approximate.
    ///
    /// ★ IT COSTS NO NEW IMAGE PROCESSING. The two lines through it are already measured: the
    ///   WallCorner segment's weighted mean residual displaces the vertical edge along its
    ///   (horizontal) normal, the FloorWall segment's does the same for the floor line along its
    ///   (vertical) normal. Two scalars, two normals, one 2x2 solve. Fixing each line's DIRECTION
    ///   from the model and fitting only its offset is deliberate: the direction is far better known
    ///   than the offset, and letting it float would trade a well-posed problem for an ill-posed one.
    ///
    /// ★ ITS CONDITIONING IS THE ORTHOGONALITY OF THE TWO CLASSES. A wall corner's normal is
    ///   horizontal and a floor junction's is vertical, so the 2x2 is near-identity — u comes from
    ///   the class that carries bearing (yaw), v from the class that carries range (pitch/height).
    ///   Each component is measured by the contour best able to measure it, which is not an accident
    ///   of this construction but the reason to prefer it.
    struct TriplePoint
    {
        int             vertex  = -1;
        /// Which horizontal contour was intersected with the vertical wall corner. FloorWall gives
        /// the corner at floor level, WallCeiling the one at ceiling level.
        ///
        /// ★ THE CEILING ONE IS THE LESS OCCLUDED FEATURE, and on a panorama it is the better one:
        ///   furniture, people and clutter sit on the FLOOR, so a floor corner is exactly where a
        ///   room is most often blocked, while nothing stands between a camera and the join of two
        ///   walls at the ceiling. The floor corner's compensating advantage — that its depth can be
        ///   read from the ZED — does not apply to a 360 model anyway.
        ContourClass    from    = ContourClass::FloorWall;
        Eigen::Vector3f p_room  = Eigen::Vector3f::Zero(); ///< (vx, vy, 0), a constant of the solve
        Eigen::Vector2f uv_pred = Eigen::Vector2f::Zero(); ///< where the model puts it
        Eigen::Vector2f uv_meas = Eigen::Vector2f::Zero(); ///< where the two fitted lines cross
        Eigen::Matrix2f cov_uv  = Eigen::Matrix2f::Zero(); ///< propagated from the two offset sigmas
        /// The occlusion prior the mixture already applied to this corner's own samples, carried
        /// out so a CONSUMER can see it. Weighted mean of pi_vis over the two segments that formed
        /// the crossing; 1 = clear line of sight, 0.02 = a wall of this very room stands in the way.
        /// ★ It exists because the 2-D canvas was drawing every predicted corner identically, so a
        ///   room whose far side is hidden behind its own walls looked like a room with transparent
        ///   walls. The LOSS was already right — an occluded corner's samples are downweighted, its
        ///   w collapses and cov_uv blows up — but nothing SAID so, and a display that hides the
        ///   model's own doubt is how a term gets trusted more than it deserves.
        float           pi_vis  = 1.f;
        /// The crossing's own residual in pixels, uv_meas − uv_pred, taken where the wrap is known
        /// so a corner beside the panorama seam does not look like a 4000 px miss. With cov_uv this
        /// gives the corner's normalised distance from the model, which is what "matched" means.
        Eigen::Vector2f resid_px = Eigen::Vector2f::Zero();
        /// ★ THREE QUANTITIES, NOT ONE, SO THE DEPTH CONVENTION CAN CHECK ITSELF. `depth_raw` is
        ///   what the plane published; `range_m` is |p_cam_meas| under the assumption that the raw
        ///   value is the FORWARD coordinate (the ZED SDK's convention, and what cortex's
        ///   get_xyz_from_rgbd_points encodes). If the producer were instead reporting range along
        ///   the ray, depth_raw would exceed the model's forward distance by 1/cos(angle off axis)
        ///   — a several-percent excess GROWING toward the image edge, which is a signature no
        ///   single logged number could show. This data reaches us through the Webots bridge and
        ///   the convention has NOT been verified, so it is logged, not assumed.
        float           depth_raw   = -1.f;  ///< as published; < 0 = not available (never 0)
        Eigen::Vector3f p_cam_meas  = Eigen::Vector3f::Zero();
        /// The measured corner in the ROOM frame — p_cam_meas carried through the extrinsic and the
        /// pose. Display-only, and pose-dependent by construction: it is where the corner WAS while
        /// the robot believed it was at that pose, so it is a picture of the residual, not evidence.
        /// Zero when there was no depth (range_m < 0), which is why the drawing checks range_m.
        Eigen::Vector3f p_room_meas = Eigen::Vector3f::Zero();
        float           range_m     = -1.f;
        float           range_sigma = -1.f;  ///< < 0 until a depth sigma is MEASURED, not guessed
        int   n_corner = 0;         ///< weighted-effective samples behind the vertical line
        int   n_floor  = 0;         ///< ... and behind the floor line
        float cond     = 0.f;       ///< of the 2x2; ~1 when the two normals are orthogonal
    };

    /// One frame's worth of edge evidence, attached to the window slot whose stamp is nearest.
    struct ImageEdgeObs
    {
        std::vector<ImageEdgeSegment> segments;
        std::vector<TriplePoint>      triple_points;
        std::int64_t                  depth_stamp_ms = 0;  ///< capture time of the depth frame used
        std::uint64_t frame_stamp = 0;      ///< image capture stamp (ms)
        std::int64_t  dt_to_slot_ms = 0;    ///< image stamp - slot stamp; feeds nuisance column [3]
        float         sigma_i = 0.f;        ///< carried through for the CSV
        /// zed_T_robot, read ONCE on the main thread at bring-up. This is the ONLY transform the
        /// factor may take from the graph — room<-robot is the STATE VARIABLE. See image_edge_source.h.
        Eigen::Matrix3f cam_R_robot = Eigen::Matrix3f::Identity();
        Eigen::Vector3f cam_t_robot = Eigen::Vector3f::Zero();
        /// The projection model, carried WITH the evidence so the factor needs no DSR and no Input
        /// plumbing. Plain numbers, verified against CameraAPI::project() at bring-up.
        CameraModel     cam;

        [[nodiscard]] std::size_t sample_count() const noexcept
        {
            std::size_t n = 0;
            for (const auto& s : segments) n += s.samples.size();
            return n;
        }
        [[nodiscard]] bool empty() const noexcept { return sample_count() == 0; }
    };

}  // namespace rc
