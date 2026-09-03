/*
 *  Copyright (C) 2026 by Pablo Bustos
 *
 *  This program is free software: you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software Foundation, either version 3 or
 *  any later version. See <http://www.gnu.org/licenses/>.
 */
#pragma once

/**
 *  STAGE 2 of the online mount self-calibration: the camera against the LiDAR, with NO POSE IN IT.
 *
 *  ── The one idea ────────────────────────────────────────────────────────────────────────────────
 *  A room corner is seen by both sensors and it is the SAME physical (x, y): the LiDAR finds the
 *  wall-wall intersection at sensor height, the image finds the floor-wall-wall triple point at floor
 *  height. `CornerMatch::model_index` and `TriplePoint::vertex` are both indices into the ORIGINAL
 *  polygon vertex list, so the association is EXACT — a lookup, not a nearest-neighbour search with
 *  a gate. There is no misassociation failure mode to defend against, which matters: "pose jumps =
 *  MISASSOC" is already on record for corners in this codebase.
 *
 *  ── Why it is pose-free, and why that is the whole point ────────────────────────────────────────
 *  `CornerMatch::detected` is in the ROBOT frame. So the LiDAR's corner can be pushed through the
 *  camera<-robot extrinsic ALONE and predicted in the image without the localiser's pose appearing
 *  anywhere:
 *
 *      uv_lidar = project( cam_R_robot * p_robot + cam_t_robot )
 *      r        = uv_image - uv_lidar
 *
 *  Stage 1's residual was image-minus-MODEL, which carries the pose error: a heading error of dtheta
 *  enters as fx*dtheta and is indistinguishable from a boresight yaw within one view. That is exactly
 *  why measuring the boresight needed the Webots supervisor, and why the stage-1 yaw estimate still
 *  rests on the assumption that heading error is zero-mean across poses. Here the pose is not in the
 *  expression at all, so the assumption is not needed. What the residual contains is the CAMERA
 *  against the LIDAR — a hand-eye disagreement — plus each sensor's own corner-detection noise.
 *
 *  ── The second, less obvious dividend ───────────────────────────────────────────────────────────
 *  ★ It makes POOLING ACROSS WINDOWS legitimate, and that is what can break the pitch/height ridge.
 *    Stage 1 had to solve per window and report the BETWEEN-window scatter as the honest uncertainty,
 *    because the formal error ran 17-149x too small: consecutive windows disagreed far more than
 *    their own samples did. The dominant reason is that each window carries a DIFFERENT pose error,
 *    which is a per-window nuisance no amount of samples averages away. Remove the pose and the
 *    windows become genuinely comparable draws of one static quantity, so their information may be
 *    summed — and a pool over many poses spans the range diversity that one 5 s window never does.
 *    Pitch and height are told apart only by the fy/d covariate: within a window the visible corners
 *    span maybe 2-6 m, giving corr(1, fy/d) ~ 0.96 and cond ~ 50-100, which is what was measured.
 *    Across a drive the same columns span 1-8 m.
 *  ★ This is a CLAIM, not yet a result: it is right only if the pose really was the dominant
 *    between-window nuisance. The test is direct and is why both are logged — if the pooled
 *    between-window scatter here does NOT collapse relative to stage 1's, something else was driving
 *    it and pooling is still illegitimate.
 *
 *  ── What it cannot do ───────────────────────────────────────────────────────────────────────────
 *  It measures the camera RELATIVE TO THE LIDAR. A yaw error common to both mounts is invisible here
 *  and stays with the localiser's own gauge. That is the correct target anyway — the point is to make
 *  the two terms agree about the same room — but the number must not be reported as an absolute
 *  camera boresight. Naming it wrongly is the mistake this file exists partly to avoid repeating.
 */

#include <Eigen/Dense>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <locale>
#include <map>
#include <string>
#include <vector>

#include "corner_detector.h"
#include "image_edge_ops.h"
#include "image_edge_types.h"

namespace rc::mount
{
    /// One corner seen by both sensors, with everything needed to form the residual.
    struct PairObs
    {
        int             vertex = -1;
        Eigen::Vector2f uv_image = Eigen::Vector2f::Zero();   ///< the triple point's measurement
        Eigen::Vector2f uv_lidar = Eigen::Vector2f::Zero();   ///< the LiDAR corner through the extrinsic
        Eigen::Vector2f r        = Eigen::Vector2f::Zero();   ///< uv_image - uv_lidar
        Eigen::Matrix2f cov      = Eigen::Matrix2f::Identity();
        Eigen::Matrix<float, 2, IMAGE_EDGE_NUISANCES> J =
            Eigen::Matrix<float, 2, IMAGE_EDGE_NUISANCES>::Zero();  ///< d(uv) / d(nuisance), prior-scaled
        /// d(uv) / d(p_camera) at this point. Kept because the LiDAR half of `cov` is built from it
        /// (`Pxy = (P·cam_R_robot).leftCols<2>()`), and keeping it here is what lets make_pair() and
        /// the offline replay share one projection instead of writing the same algebra twice.
        Eigen::Matrix<float, 2, 3> P = Eigen::Matrix<float, 2, 3>::Zero();
        /// The LiDAR half of `cov` alone — `Pxy·cov_xy·Pxyᵀ`. ★ Logged so an offline replay does not
        /// have to hold the covariance FIXED while the extrinsic moves: `Pxy` depends on the mount,
        /// and with this term separable the replay recovers `cov_xy = Pxy⁻¹·cov_lidar·Pxy⁻ᵀ` and
        /// rebuilds the weight under the injected mount EXACTLY. Without it the replay carries a
        /// stated approximation; with it there is nothing left to state.
        Eigen::Matrix2f cov_lidar = Eigen::Matrix2f::Zero();
        float assoc_prob = 1.f;
        float range_m    = 0.f;
        /// The LiDAR corner in the ROBOT frame, before the extrinsic touched it. ★ Logged so an
        /// extrinsic perturbation can be REPLAYED offline: uv_lidar is a deterministic function of
        /// this and the mount, and the association is exact-by-index so it does not move. That turns
        /// arm 7's four injection legs into four analyses of ONE recorded drive, which also removes
        /// the route variation that swamps between-run comparisons.
        Eigen::Vector3f p_robot = Eigen::Vector3f::Zero();
        bool  ok         = false;
    };

    /// ── The half of a pair that an extrinsic INJECTION moves ────────────────────────────────────
    /// Everything downstream of `p_robot` and the mount: the projection, the residual and J. Split
    /// out of make_pair() so arm 7's offline replay (`tools/mount_replay.cpp`) can perturb the mount
    /// and rebuild the row through THE ESTIMATOR'S OWN CODE rather than a replica that can silently
    /// drift from it — the same reason the bench grades the live path and not a copy of it.
    ///
    /// `cov` is handed in instead of computed here: the live caller builds it from BOTH sensors'
    /// uncertainties (which needs the LiDAR corner's information matrix, a thing a CSV row does not
    /// carry), and a replay reuses the covariance that was recorded at capture time.
    /// ⚠ So a replay holds `cov` FIXED while the extrinsic moves, and the LiDAR half of it does
    ///   depend on the extrinsic through `Pxy = (P·cam_R_robot).leftCols<2>()`. `P` is left in the
    ///   PairObs so a replay can compute how far that approximation reaches instead of asserting it
    ///   is small (VALIDATION_THREE_DEVICE_CORNERS §2b).
    inline PairObs make_pair_from(int vertex, float assoc_prob,
                                  const Eigen::Vector3f& p_robot,
                                  const Eigen::Vector2f& uv_image,
                                  const Eigen::Matrix2f& cov,
                                  const CameraModel& cam,
                                  const Eigen::Matrix3f& cam_R_robot,
                                  const Eigen::Vector3f& cam_t_robot,
                                  float sigma_pitch, float sigma_height, float sigma_yaw)
    {
        PairObs o;
        o.vertex = vertex;
        o.assoc_prob = assoc_prob;

        const Eigen::Vector3f pc = cam_R_robot * p_robot + cam_t_robot;
        if (not (pc.norm() > 1e-3f)) return o;
        o.p_robot = p_robot;

        Eigen::Vector2d uv;
        if (not rc::img::project_with_model(cam, pc.cast<double>(), uv)) return o;
        Eigen::Matrix<double, 2, 3> P;
        if (not rc::img::project_jacobian_model(cam, pc.cast<double>(), P)) return o;

        o.uv_lidar = uv.cast<float>();
        o.uv_image = uv_image;
        o.r        = Eigen::Vector2f(
            static_cast<float>(rc::img::du_wrapped(
                static_cast<double>(o.uv_image.x()) - static_cast<double>(o.uv_lidar.x()), cam)),
            o.uv_image.y() - o.uv_lidar.y());
        o.range_m  = pc.norm();

        const Eigen::Matrix<float, 2, 3> Pf = P.cast<float>();
        o.P = Pf;
        // ★ THE CAMERA'S OWN AXES, in camera coordinates — literally (1,0,0) and (0,0,1), exactly as
        //   image_edge_source.cpp:277 defines them. This previously used cam_R_robot.col(0) and
        //   .col(2), which are the ROBOT's axes expressed in camera coordinates: a different thing.
        //   For this mount the up-axes coincide, so the yaw and height columns were unaffected — but
        //   col(0) is the robot's FORWARD axis, so what was labelled "pitch" was a rotation about the
        //   optical axis, i.e. a ROLL. Any pitch figure from the pair fit before this is void; the
        //   yaw result (+0.014 deg) stands, which is why it is worth saying which is which rather
        //   than quietly re-running everything.
        const Eigen::Vector3f x_cam(1.f, 0.f, 0.f);
        const Eigen::Vector3f z_cam(0.f, 0.f, 1.f);
        o.J.col(0) = sigma_pitch  * (Pf * x_cam.cross(pc));
        o.J.col(1) = sigma_height * (Pf * z_cam);
        o.J.col(2) = sigma_yaw    * (Pf * z_cam.cross(pc));
        // column 3 (dt) deliberately zero — see the note above.

        o.cov = cov;
        o.ok  = o.cov.allFinite() and o.J.allFinite() and o.r.allFinite();
        return o;
    }

    /// Builds the pose-free pair for one (triple point, corner match) with the same model_index.
    ///
    /// `J` mirrors image_edge_source.cpp's hcol columns EXACTLY — pitch = rotation about the camera x
    /// axis, height = translation along camera z, yaw = rotation about camera z — but keeps the full
    /// 2-vector per nuisance instead of contracting onto a contour normal. A triple point has no
    /// aperture problem, so contracting would throw away half the information for no reason.
    /// The dt column is left at zero: it needs the body twist and the image/LiDAR offset, and unlike
    /// the other three it is not a property of the mount at all.
    inline PairObs make_pair(const TriplePoint& tp,
                             const rc::CornerDetector::CornerMatch& cm,
                             const CameraModel& cam,
                             const Eigen::Matrix3f& cam_R_robot,
                             const Eigen::Vector3f& cam_t_robot,
                             float sigma_pitch, float sigma_height, float sigma_yaw)
    {
        // ★ z comes from the TRIPLE POINT, not from a constant: the same vertical wall-wall edge
        //   yields a corner at floor level and another at ceiling level, and this pairs with either.
        //   The LiDAR corner is a 2-D (x, y) at sensor height, but it is the SAME vertical edge, so
        //   its (x, y) is the triple point's (x, y) at whichever height that one sits. Taking z from
        //   the LiDAR would be wrong — it does not measure one.
        //   (This codebase's robot frame has its origin on the floor: the projection path forms
        //    e = p_room - pose with e.z() = p_room.z(), and P3Bot->body is identity.)
        const Eigen::Vector3f p_robot(cm.detected.x(), cm.detected.y(), tp.p_room.z());
        // The covariance needs this pair's own projection jacobian, so the geometry runs FIRST with
        // a placeholder covariance and the real one is installed below. Zero is finite, so the `ok`
        // flag computed there is exactly the geometry's own verdict.
        PairObs o = make_pair_from(tp.vertex, cm.assoc_prob, p_robot, tp.uv_meas,
                                   Eigen::Matrix2f::Zero(), cam, cam_R_robot, cam_t_robot,
                                   sigma_pitch, sigma_height, sigma_yaw);
        if (not o.ok) return o;
        const Eigen::Matrix<float, 2, 3> Pf = o.P;

        // ── Covariance: BOTH sensors' uncertainty, in pixels ─────────────────────────────────────
        // The image half comes from the triple point's own line-intersection covariance. The LiDAR
        // half is its detection precision pushed through the same projection. Using only one of them
        // would hand the residual to whichever sensor was quietly the noisier.
        Eigen::Matrix2f cov_img = tp.cov_uv;
        Eigen::Matrix2f cov_lid = Eigen::Matrix2f::Zero();
        // Λ_det can be RANK-1 when the two walls are near-parallel — a shallow corner leaves the
        // bisector direction unconstrained (corner_detector.h). Inverting it blindly gives infinities;
        // an eigen-floor keeps the unconstrained direction merely very uncertain, which is true.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(cm.information);
        if (es.info() == Eigen::Success)
        {
            Eigen::Vector2f ev = es.eigenvalues();
            for (int i = 0; i < 2; ++i) ev(i) = std::max(ev(i), 1e-4f);   // 1e-4 /m^2 -> sigma 100 m
            const Eigen::Matrix2f cov_xy =
                es.eigenvectors() * ev.cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
            const Eigen::Matrix<float, 2, 2> Pxy = (Pf * cam_R_robot).leftCols<2>();
            cov_lid = Pxy * cov_xy * Pxy.transpose();
        }
        o.cov_lidar = cov_lid;
        o.cov = cov_img + cov_lid + 0.01f * Eigen::Matrix2f::Identity();   // 0.1 px numerical floor
        o.ok  = o.cov.allFinite() and o.J.allFinite() and o.r.allFinite();
        return o;
    }

    /// ── THE CONSTANTS A RECORDED DRIVE NEEDS TO BE REPLAYABLE ───────────────────────────────────
    /// A pair CSV row is not self-sufficient. Rebuilding it under a PERTURBED extrinsic needs the
    /// camera model, the nominal mount, and the three prior sigmas that J's columns are scaled by —
    /// and, for the LiDAR leg of arm 7's attribution table, where the LiDAR sits in the robot frame,
    /// because a LiDAR yaw error rotates every corner about THAT point and not about the origin.
    /// These are constants of the run, so they are written once beside the CSV rather than on every
    /// row. ★ Without them a replay has to hard-code the mount, which is exactly how an offline tool
    /// and the agent come to disagree about the same drive while both look right.
    struct ReplayContext
    {
        std::string     robot, camera;
        CameraModel     cam;
        Eigen::Matrix3f cam_R_robot = Eigen::Matrix3f::Identity();
        Eigen::Vector3f cam_t_robot = Eigen::Vector3f::Zero();
        float sigma_pitch = 0.f, sigma_height = 0.f, sigma_yaw = 0.f;
        double offset_sigma_px = 0.0;   ///< what the agent's own solve was using, for comparison
        /// The correction the agent had already applied to this mount when the row was written,
        /// in prior-sigma units. A replay injects on top of the mount that ACTUALLY produced the
        /// rows, so it needs to know the mount was not the graph's.
        Eigen::Vector4d applied = Eigen::Vector4d::Zero();
        /// The LiDAR's origin in the robot frame. ⚠ `lidar_known == false` means it could not be
        /// resolved at write time; a replay must then REFUSE the LiDAR-injection leg rather than
        /// assume the origin — that leg's whole point is where the rotation centre is.
        Eigen::Vector3f lidar_t_robot = Eigen::Vector3f::Zero();
        bool            lidar_known = false;
    };

    /// One `key,value...` per line, C locale on both sides. CLAUDE.md: these machines run es_ES, so
    /// the writer imbues classic() and the reader uses from_chars — never `>>` or strtof.
    inline bool write_replay_context(const std::string& path, const ReplayContext& c)
    {
        std::ofstream f(path, std::ios::out | std::ios::trunc);
        if (not f.is_open()) return false;
        f.imbue(std::locale::classic());
        f << "# the run constants an offline replay needs (arm 7). Written when the pair CSV opens.\n"
          << "robot," << c.robot << "\ncamera," << c.camera << '\n'
          << "cam_kind," << static_cast<int>(c.cam.kind) << '\n'
          << "cam_intrinsics," << c.cam.fx << ',' << c.cam.fy << ',' << c.cam.cx << ',' << c.cam.cy
          << ',' << c.cam.width << ',' << c.cam.height << ',' << c.cam.azimuth_sign << ','
          << c.cam.azimuth_offset << '\n'
          << "cam_R_robot";
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) f << ',' << c.cam_R_robot(i, j);
        f << "\ncam_t_robot," << c.cam_t_robot.x() << ',' << c.cam_t_robot.y() << ','
          << c.cam_t_robot.z() << '\n'
          << "prior_sigmas," << c.sigma_pitch << ',' << c.sigma_height << ',' << c.sigma_yaw << '\n'
          << "offset_sigma_px," << c.offset_sigma_px << '\n';
        if (c.lidar_known)
            f << "lidar_t_robot," << c.lidar_t_robot.x() << ',' << c.lidar_t_robot.y() << ','
              << c.lidar_t_robot.z() << '\n';
        else
            f << "# lidar_t_robot UNRESOLVED at write time — the LiDAR injection leg must refuse\n";
        return f.good();
    }

    inline bool read_replay_context(const std::string& path, ReplayContext& c)
    {
        std::ifstream f(path);
        if (not f.is_open()) return false;
        std::string line;
        int seen = 0;
        while (std::getline(f, line))
        {
            if (line.empty() or line[0] == '#') continue;
            std::vector<std::string> tok;
            for (std::size_t a = 0, b; a <= line.size(); a = b + 1)
            {
                b = line.find(',', a);
                if (b == std::string::npos) b = line.size();
                tok.emplace_back(line.substr(a, b - a));
            }
            const auto num = [&](std::size_t i, auto& out) {
                if (i >= tok.size()) return false;
                const char* p = tok[i].data();
                const char* e = p + tok[i].size();
                while (p < e and *p == ' ') ++p;
                return std::from_chars(p, e, out).ec == std::errc{};
            };
            const std::string& k = tok[0];
            if      (k == "robot"  and tok.size() > 1) { c.robot = tok[1]; }
            else if (k == "camera" and tok.size() > 1) { c.camera = tok[1]; }
            else if (k == "cam_kind")
            {
                int kind = 0;
                if (num(1, kind)) { c.cam.kind = static_cast<CameraModel::Kind>(kind); ++seen; }
            }
            else if (k == "cam_intrinsics")
            {
                bool ok = num(1, c.cam.fx) and num(2, c.cam.fy) and num(3, c.cam.cx)
                          and num(4, c.cam.cy) and num(5, c.cam.width) and num(6, c.cam.height)
                          and num(7, c.cam.azimuth_sign) and num(8, c.cam.azimuth_offset);
                if (ok) { c.cam.valid = true; ++seen; }
            }
            else if (k == "cam_R_robot")
            {
                bool ok = true;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) ok = ok and num(static_cast<std::size_t>(1 + i * 3 + j), c.cam_R_robot(i, j));
                if (ok) ++seen;
            }
            else if (k == "cam_t_robot")
            {
                if (num(1, c.cam_t_robot.x()) and num(2, c.cam_t_robot.y()) and num(3, c.cam_t_robot.z())) ++seen;
            }
            else if (k == "prior_sigmas")
            {
                if (num(1, c.sigma_pitch) and num(2, c.sigma_height) and num(3, c.sigma_yaw)) ++seen;
            }
            else if (k == "offset_sigma_px") { num(1, c.offset_sigma_px); }
            else if (k == "lidar_t_robot")
            {
                if (num(1, c.lidar_t_robot.x()) and num(2, c.lidar_t_robot.y()) and num(3, c.lidar_t_robot.z()))
                    c.lidar_known = true;
            }
        }
        return seen == 5;   // kind, intrinsics, R, t, sigmas — everything a projection needs
    }

    /// ── THE PER-VERTEX OFFSET NUISANCE ──────────────────────────────────────────────────────────
    /// MEASURED 2026-09-02 over 395 171 ricoh pairs: the residual carries a systematic offset that
    /// belongs to the CORNER, not to the mount. Per-vertex means reach −17 px and spread sd 5.3 px
    /// across 23 vertices, while the mount signal being estimated is 1.1 px of yaw. The old model
    ///
    ///     r_i = −J_i θ + ε_i        ε independent
    ///
    /// has nowhere to put that, so it absorbs it into θ AND counts each of a vertex's thousands of
    /// sightings as independent confirmation. The reported yaw sigma came out 127x too small, against
    /// a design effect sqrt(n_rows/n_clusters) of 131 — the two agree, which names the mechanism.
    /// The fitted yaw was then smaller than its own honest error.
    ///
    /// The model gains the term it was missing, and the term is INTEGRATED OUT rather than estimated:
    ///
    ///     r_i = −J_i θ + δ_v(i) + ε_i,      δ_v ~ N(0, S),   S = offset_sigma_px² · I
    ///
    /// δ_v's value is not wanted, only its contamination removed. Marginalising is a Schur complement
    /// over the per-vertex partials below (derivation in VALIDATION_THREE_DEVICE_CORNERS §2 stage 1):
    ///
    ///     M_v = (S⁻¹ + D_v)⁻¹
    ///     H  += A_v − c_v M_v c_vᵀ,   b += b_v − c_v M_v e_v,   rTr += rTr_v − e_vᵀ M_v e_v
    ///
    /// ★ THIS IS A COVARIANCE TERM, NOT A GATE — the Woodbury common-mode marginalisation CLAUDE.md
    ///   already names for correlated mask points, applied one level up. Nothing is rejected and
    ///   nothing is clamped: `offset_sigma_px = 0` recovers the old estimator EXACTLY (that is the
    ///   default, so a running agent is unchanged until someone asks for the new term), and a large
    ///   sigma removes the level entirely.
    /// ★ THE LIMIT IS THE POINT. As n_v grows, M_v → D_v⁻¹ and the vertex contributes only
    ///   A_v − c_v D_v⁻¹ c_vᵀ — its WITHIN-vertex information. Seeing one corner a million times stops
    ///   buying anything. Yaw is a constant pixel shift with no range or bearing dependence, so within
    ///   a vertex it is EXACTLY degenerate with that vertex's own u-offset and loses nearly all of its
    ///   information; pitch and height are range-dependent (Δd = θ_p·d²/h) and survive. Marginalising
    ///   does not RECOVER yaw precision — it reveals we never had it.
    /// ⚠ A detector bias common to EVERY corner is yaw, and nothing here separates them.
    struct VertexBlock
    {
        Eigen::Matrix4d            A = Eigen::Matrix4d::Zero();              ///< Σ w Jᵀ W J
        Eigen::Matrix<double, 4, 2> c = Eigen::Matrix<double, 4, 2>::Zero(); ///< Σ w Jᵀ W
        Eigen::Matrix2d            D = Eigen::Matrix2d::Zero();              ///< Σ w W
        Eigen::Vector4d            b = Eigen::Vector4d::Zero();              ///< Σ w Jᵀ W r
        Eigen::Vector2d            e = Eigen::Vector2d::Zero();              ///< Σ w W r
        double                     rTr = 0.0;
        long                       n = 0;
        [[nodiscard]] bool finite() const
        { return A.allFinite() and c.allFinite() and D.allFinite() and b.allFinite() and e.allFinite(); }
    };

    /// The same 4-parameter normal-equation block stage 1 uses, so the two are directly comparable.
    /// Prior is the IDENTITY because J carries the prior sigma (see room_concept.h).
    struct Accum
    {
        Eigen::Matrix4d H = Eigen::Matrix4d::Zero();
        Eigen::Vector4d b = Eigen::Vector4d::Zero();
        double          rTr = 0.0;
        long            n = 0;
        /// Per-vertex partials, kept ALONGSIDE the aggregate above so the two models can be solved
        /// from one accumulation and compared without re-driving. ~23 entries in this apartment.
        std::map<int, VertexBlock> per_vertex;
        /// Prior sigma on a corner's own image offset, in PIXELS. 0 disables the nuisance and the
        /// solve is bit-for-bit the old one. ⚠ Measured from the same data it would be applied to
        /// (sd 5.3 px) this is an EMPIRICAL-BAYES hyperparameter, not a prior the data then confirms.
        /// ★ Pixels, not metres, on purpose: a 1/range fit does NOT distinguish a pixel-fixed
        ///   detector bias from a metric map offset on this data (both R² ≈ 0), so the pixel choice
        ///   assumes nothing that could not be measured. The metric version IS corner-as-landmark
        ///   refinement (DESIGN §7) and is the next step, not this one.
        double offset_sigma_px = 0.0;
        /// ── THE TOTAL CORRECTION ALREADY APPLIED TO THE MOUNT ───────────────────────────────────
        /// In prior-sigma units, in the sign of the SOLVE VECTOR `x` — which is `-Solution::p`, and
        /// therefore the sign every reporting site prints after negating `p` again. ⚠ Getting this
        /// backwards does not fail quietly: it drives the loop the wrong way and the correction runs
        /// away linearly (measured, before the sign was fixed: -53 degrees in 12 cycles on a 1 degree
        /// truth). The closed-loop selftest in tools/mount_replay.cpp exists to catch exactly that.
        /// It is not bookkeeping: it is where the PRIOR is centred.
        ///
        /// ★★★ WITHOUT IT A FEEDBACK LOOP RATCHETS. Applying the posterior mean and re-centring the
        ///   prior on the new mount removes the prior's pull, so each cycle applies a fresh shrunk
        ///   estimate of an error that is mostly prior, and a weakly-informed axis walks away from
        ///   the graph value one small honest step at a time. Anchoring the prior on the ORIGINAL
        ///   graph extrinsic makes the objective  ‖r + Jx‖²_W + ‖x − applied‖²  , whose fixed point
        ///   is the posterior mean of the TOTAL error relative to the graph — reached in one step
        ///   and then stationary.
        /// ★ The conservative behaviour falls out rather than being imposed: on an axis the data
        ///   does not inform (H → 0) the solve returns x → applied, the increment applied is −applied,
        ///   and the accumulated correction DECAYS BACK TO ZERO. No gate, no threshold — an
        ///   uninformed parameter simply stops being corrected, which is what its posterior says.
        Eigen::Vector4d applied = Eigen::Vector4d::Zero();
        /// Set when evidence was restored from a file written before the per-vertex partials existed.
        /// Such evidence cannot be marginalised — its rows carry no vertex — and must not be mixed
        /// with evidence that can, so the solve REFUSES the nuisance while it is set.
        bool legacy_unattributed = false;

        void add(const PairObs& o)
        {
            if (not o.ok) return;
            const Eigen::Matrix2d C = o.cov.cast<double>();
            const double det = C.determinant();
            if (not (det > 1e-12)) return;
            const Eigen::Matrix2d W = C.inverse();
            // ★ Weighted by assoc_prob: the detector's own posterior that this detection belongs to
            //   this model corner. A 0.6-probability association contributes 60% of a measurement,
            //   which is what it is — not a threshold, and not a full one either.
            const double w = std::clamp(static_cast<double>(o.assoc_prob), 0.0, 1.0);
            if (not (w > 1e-3)) return;
            // Columns 0-3 only: [4] is a per-contour map offset and a single paired corner carries
            // no information about it that is separable from the mount.
            const Eigen::Matrix<double, 2, 4> J = o.J.template leftCols<4>().template cast<double>();
            const Eigen::Vector2d r = o.r.cast<double>();
            const Eigen::Matrix<double, 4, 2> JtW = w * J.transpose() * W;
            H.noalias() += JtW * J;
            b.noalias() += JtW * r;
            rTr += w * r.dot(W * r);
            ++n;
            // A pair with no vertex cannot be attributed to a cluster. It must not be dropped (the
            // aggregate above still wants it) and must not be invented into vertex 0 either, so it
            // is recorded as unattributable and the nuisance refuses to run rather than guess.
            if (o.vertex < 0) { legacy_unattributed = true; return; }
            VertexBlock& v = per_vertex[o.vertex];
            v.A.noalias() += JtW * J;
            v.c.noalias() += JtW;
            v.D.noalias() += w * W;
            v.b.noalias() += JtW * r;
            v.e.noalias() += w * W * r;
            v.rTr += w * r.dot(W * r);
            ++v.n;
        }
        /// ── RE-REFERENCE THE EVIDENCE TO A MOUNT THAT HAS JUST BEEN CORRECTED ──────────────────
        /// Evidence is measured AGAINST a mount. The moment a correction `d` is applied to that
        /// mount, every residual already accumulated refers to the old one, and re-solving would
        /// hand back the same error a second time — the double-application that makes a feedback
        /// loop diverge instead of converge.
        ///
        /// Nothing has to be thrown away. The residual is linear in the mount, `r → r − J d`, and
        /// H, A, c and D do not depend on r at all, so only the right-hand sides move:
        ///
        ///     rTr −= 2 dᵀb − dᵀH d      b −= H d      (per vertex: rTr_v, b_v −= A_v d, e_v −= c_vᵀ d)
        ///
        /// ★ This is the SAME algebra as an injection (VALIDATION §2b), used in the opposite
        ///   direction: an injection asks what a wrong mount would do to the evidence, and a rebase
        ///   tells the evidence that the mount it was measured against has moved. Information is
        ///   preserved exactly — the posterior after `apply then rebase` is the posterior of a robot
        ///   that had the corrected mount all along.
        /// ⚠ `d` is in PRIOR-SIGMA UNITS, like everything else in this block, and carries the sign
        ///   of `x` (the solved vector) and NOT of the reported parameter `p = −x`.
        void rebase(const Eigen::Vector4d& d)
        {
            if (not d.allFinite() or d.isZero()) return;
            rTr -= 2.0 * d.dot(b) - d.dot(H * d);      // uses the OLD b, so it goes first
            b.noalias() -= H * d;
            for (auto& [vid, v] : per_vertex)
            {
                v.rTr -= 2.0 * d.dot(v.b) - d.dot(v.A * d);
                v.b.noalias() -= v.A * d;
                v.e.noalias() -= v.c.transpose() * d;
            }
        }

        /// Apply a physical correction `m` to the mount: re-reference the evidence to the mount that
        /// now exists, and move the prior's anchor with it so the total stays measured against the
        /// ORIGINAL graph extrinsic. The two must happen together — either alone is a bug — so there
        /// is one entry point.
        /// `m` is in prior-sigma units and in the sign of `x` (= `-Solution::p`); the increment that
        /// drives the remaining estimate to zero is exactly `m = x`, after which the loop is
        /// stationary rather than merely slower.
        void apply_correction(const Eigen::Vector4d& m)
        {
            if (not m.allFinite() or m.isZero()) return;
            rebase(m);
            applied += m;
        }

        void reset()
        { H.setZero(); b.setZero(); rTr = 0.0; n = 0; per_vertex.clear(); legacy_unattributed = false;
          applied.setZero(); }

        /// Returns {parameters in units of prior sigma, posterior sigma, chi2/dof, cond, ok}.
        /// SIGN: r = uv_image - uv_lidar and J = d(uv_pred)/d(nuisance). A mount error of x makes the
        /// PREDICTION wrong by J*x while the image measurement is right, so r = -J*x and the fit
        /// returns minus the parameter — identical to stage 1, deliberately, so the two can be
        /// compared without a sign convention standing between them.
        struct Solution
        {
            Eigen::Vector4d p = Eigen::Vector4d::Zero();
            Eigen::Vector4d sigma = Eigen::Vector4d::Ones();
            double chi2_dof = 0.0, cond = 0.0, rho = 0.0;
            int    rho_i = 0, rho_j = 1, informed = 0;
            bool   ok = false;
            // ── the nuisance, reported so a reader can tell WHICH model produced these numbers ──
            bool   marginalised = false;  ///< the per-vertex offset was integrated out
            int    clusters = 0;          ///< distinct vertices — the REAL sample size for the level
            double eff_params = 0.0;      ///< Σ tr(M_v D_v): how many of the 2·clusters offsets the
                                          ///< data actually paid for. → 2 per vertex as S grows.
        };
        /// `min_n` counts PAIRS. The nuisance does not change that: a solve with 30 pairs on one
        /// vertex is still one cluster, and reporting `clusters` is how that is made visible rather
        /// than defended against with a second minimum.
        [[nodiscard]] Solution solve(long min_n = 30) const
        {
            Solution s;
            if (n < min_n) return s;
            Eigen::Matrix4d Hm = H;
            Eigen::Vector4d bm = b;
            double          rm = rTr;
            // ⚠ REFUSED, not silently skipped: unattributed evidence cannot be marginalised, and
            //   mixing it with evidence that can would produce a number belonging to neither model.
            const bool want = offset_sigma_px > 0.0 and not legacy_unattributed and not per_vertex.empty();
            if (want)
            {
                const double s2 = offset_sigma_px * offset_sigma_px;
                const Eigen::Matrix2d Sinv = Eigen::Matrix2d::Identity() / s2;
                Hm.setZero(); bm.setZero(); rm = 0.0;
                for (const auto& [vtx, v] : per_vertex)
                {
                    if (v.n <= 0 or not v.finite()) continue;
                    const Eigen::Matrix2d M = (Sinv + v.D).inverse();
                    if (not M.allFinite()) continue;
                    Hm.noalias() += v.A - v.c * M * v.c.transpose();
                    bm.noalias() += v.b - v.c * (M * v.e);
                    rm += v.rTr - v.e.dot(M * v.e);
                    s.eff_params += (M * v.D).trace();
                    ++s.clusters;
                }
                if (s.clusters == 0) return s;
                s.marginalised = true;
            }
            else s.clusters = static_cast<int>(per_vertex.size());

            const Eigen::Matrix4d A = Hm + Eigen::Matrix4d::Identity();
            const Eigen::Matrix4d C = A.inverse();
            if (not C.allFinite()) return s;
            // The prior is centred on `applied`, not on zero — see the member's note. With
            // `applied` zero this is bit-for-bit the previous solve, which is what every result
            // recorded before 2026-09-03 rests on.
            const Eigen::Vector4d x = C * (bm - applied);
            // Exact for any prior centre, and reduces to `rm - x·bm` when it is zero:
            //   f(x) = rm − 2xᵀ(b+c) + xᵀ(H+I)x + cᵀc,  and at the optimum (H+I)x = b+c.
            const double chi2 = std::max(0.0, rm - x.dot(bm - applied) + applied.squaredNorm());
            // The offsets consume degrees of freedom too, and softly — tr(M_v D_v) is how much of
            // each vertex's 2 the data actually paid for. Ignoring it would inflate chi2/dof and
            // then inflate every sigma through `infl` below, hiding the improvement inside the fix.
            s.chi2_dof = chi2 / std::max(1.0, 2.0 * static_cast<double>(n) - 4.0 - s.eff_params);
            const double infl = std::sqrt(std::max(1.0, s.chi2_dof));
            s.p = -x;
            for (int i = 0; i < 4; ++i)
            {
                s.sigma(i) = std::sqrt(std::max(0.0, C(i, i))) * infl;
                if (s.sigma(i) < 0.9) s.informed |= (1 << i);
            }
            Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    R(i, j) = C(i, j) / std::sqrt(std::max(1e-300, C(i, i) * C(j, j)));
            const Eigen::Vector4d ev = Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(R).eigenvalues();
            s.cond = ev(3) / std::max(1e-12, ev(0));
            for (int i = 0; i < 4; ++i)
                for (int j = i + 1; j < 4; ++j)
                    if (std::abs(R(i, j)) > std::abs(s.rho)) { s.rho = R(i, j); s.rho_i = i; s.rho_j = j; }
            s.ok = true;
            return s;
        }
    };
}   // namespace rc::mount
