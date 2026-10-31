// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Image-edge channel: the shadow solve and triple-point logging.

#include "room_concept.h"
#include "../../common/status_stream/status_sink.h"
#include <pthread.h>   // pthread_setname_np: name the worker so a per-thread CPU sample attributes itself
#include "pointcloud_center_estimator.h"
#include <numeric>
#include <set>
#include "room_gn_solver.h"
#include "room_obs_weights.h"
#include "room_gn_solver.h"
#include "image_edge_accumulate.h"
#include "image_edge_ops.h"
#include <algorithm>
#include <ranges>
#include <fstream>
#include <filesystem>
#include <limits>
#include <locale>
#include <ctime>
#include <print>
#include <sstream>
#include <sys/stat.h>
#include <QDebug>

namespace rc
{


    // ── RGB edge shadow + systematic-residual monitor ────────────────────────────────────────────
    void RoomConcept::run_image_edge_shadow(const std::vector<Eigen::Vector3f>& poses_after,
                                            float boundary_weight, std::int64_t timestamp_ms,
                                            bool probe_pose)
    {
        // ── Health, so "no CSV" is diagnosable instead of ambiguous ───────────────────────────────
        // Three different failures land in the same silence: the camera never bound, it bound but
        // extracts nothing, or rows are being written and nobody looked. Count them apart and say so
        // once every 5 s. Without this the only evidence is an absent file, which is consistent with
        // all three.
        ++imgedge_calls_;
        if (window_mgr_.window.empty()) { ++imgedge_no_window_; return; }
        const auto& slot = window_mgr_.window.back();
        const auto& obs  = slot.image_edges;
        if (obs.empty())       { ++imgedge_no_obs_; imgedge_health(timestamp_ms); return; }
        if (not obs.cam.valid) { ++imgedge_no_cam_; imgedge_health(timestamp_ms); return; }

        const Eigen::Vector3f pose = poses_after.back();
        const float cth = std::cos(pose.z()), sth = std::sin(pose.z());
        Eigen::Matrix3f Rm; Rm << cth, sth, 0.f, -sth, cth, 0.f, 0.f, 0.f, 1.f;

        // ── Pass 1: the term's own numbers at the authority's pose ───────────────────────────────
        float trace_raw = 0.f, trace_eff = 0.f, sum_gamma = 0.f, chi2 = 0.f, loss_img = 0.f;
        int   n_used = 0;
        std::vector<float> abs_r;
        // Systematic-residual monitor: fit  r_k ~ b_const + b_invd * (fy / d_k).
        // Range SEPARATES the two mount errors, exactly as rotation-vs-time separates gyro scale
        // from gyro bias in calibration_estimator.h: a PITCH error is constant in pixels
        // (delta_psi = b_const / fy) while a HEIGHT error scales as fy/d (delta_h = b_invd). A
        // persistent, pose-independent, significant b is miscalibration; a fluctuating one is noise.
        double S11 = 0, S12 = 0, S22 = 0, Sy1 = 0, Sy2 = 0, Syy = 0, Sw = 0;

        for (const auto& seg : obs.segments)
        {
            const auto acc = rc::img::accumulate_segment(seg,
                [&](std::size_t k, Eigen::Matrix<float, 1, 3>& J) -> float
                {
                    const auto& smp = seg.samples[k];
                    const Eigen::Vector3f e(smp.p_room.x() - pose.x(), smp.p_room.y() - pose.y(), smp.p_room.z());
                    const Eigen::Vector3f p_robot = Rm * e;
                    const Eigen::Vector3f p_cam   = obs.cam_R_robot * p_robot + obs.cam_t_robot;
                    Eigen::Vector2d uv;
                    if (not rc::img::project_with_model(obs.cam, p_cam.cast<double>(), uv))
                        return std::numeric_limits<float>::quiet_NaN();
                    double du = uv.x() - static_cast<double>(smp.uv_meas.x());
                    if (obs.cam.kind != CameraModel::Kind::Pinhole)
                    {
                        while (du >  0.5 * obs.cam.width) du -= obs.cam.width;
                        while (du <= -0.5 * obs.cam.width) du += obs.cam.width;
                    }
                    const double dv = uv.y() - static_cast<double>(smp.uv_meas.y());
                    Eigen::Matrix<double, 2, 3> P;
                    if (not rc::img::project_jacobian_model(obs.cam, p_cam.cast<double>(), P))
                        return std::numeric_limits<float>::quiet_NaN();
                    Eigen::Matrix3f Jx;
                    Jx.col(0) = obs.cam_R_robot * (-Rm.col(0));
                    Jx.col(1) = obs.cam_R_robot * (-Rm.col(1));
                    Jx.col(2) = obs.cam_R_robot * Eigen::Vector3f(p_robot.y(), -p_robot.x(), 0.f);
                    J = smp.n_hat.transpose() * (P.cast<float>() * Jx);
                    const float r = smp.n_hat.x() * static_cast<float>(du)
                                  + smp.n_hat.y() * static_cast<float>(dv);
                    // ── The inlier weight, computed ONCE and shared by both monitors ──────────
                    // ★ SAME WEIGHT THE FACTOR USES: gamma/sigma^2, not 1/sigma^2. gamma is the
                    // mixture responsibility — this sample's posterior probability of being an
                    // inlier, an inlier Gaussian against a UNIFORM over the window actually
                    // searched. A bad match therefore removes itself by its own evidence, which is
                    // the only outlier handling here and needs no threshold. Ignoring it let
                    // mismatches dominate a 4-sample fit: r_rms ran 11.9 px against a stated sigma
                    // of 0.05.
                    // ★ THE INLIER VARIANCE MUST INCLUDE THE PREDICTION'S OWN SPREAD, not just the
                    // edge-localisation sigma. smp.h holds each nuisance's sensitivity ALREADY
                    // multiplied by that nuisance's prior sigma (px per unit-variance nuisance), so
                    // h.squaredNorm() IS the predicted px^2 variance contributed by an unknown
                    // mount pitch, height, boresight yaw and image/lidar offset. Widening by it is
                    // what makes "a residual consistent with a plausible mount error" an INLIER —
                    // exactly the sample that carries mount information. Weighting by sigma_px
                    // alone rejected everything: at r ~ 12 px against sigma_px ~ 0.05,
                    // exp(-0.5*(r/sigma)^2) underflows, gamma is 0 for every sample, and the pooled
                    // fit starved silently. An estimator that admits nothing and an estimator that
                    // has nothing to admit look identical from outside, which is why the sample
                    // counts are logged.
                    const float  s2  = smp.sigma_px * smp.sigma_px + smp.h.squaredNorm();
                    const float  gam = s2 > 0.f
                                     ? rc::img::responsibility(r, s2, smp.pi_vis, smp.search_L)
                                     : 0.f;
                    const double w   = s2 > 0.f ? static_cast<double>(gam) / static_cast<double>(s2)
                                                : 0.0;

                    // ── Monitor 1: FLOOR-JUNCTION samples only ───────────────────────────────
                    // The fy/d signature is what makes pitch and height separable, and a vertical
                    // corner does not carry it.
                    if (seg.class_id == ContourClass::FloorWall and smp.sigma_px > 0.f)
                    {
                        const float d = p_cam.norm();
                        if (d > 0.3f)
                        {
                            if (not (gam > 1e-6f)) return r;
                            const double x2 = obs.cam.fy / d;
                            S11 += w;        S12 += w * x2;   S22 += w * x2 * x2;
                            Sy1 += w * r;    Sy2 += w * x2 * r; Syy += w * r * r; Sw += w;
                        }
                    }
                    abs_r.push_back(std::fabs(r));
                    return r;
                },
                [](std::size_t) { return 0.0f; });
            trace_raw += acc.trace_raw;
            trace_eff += acc.trace_eff;
            sum_gamma += acc.sum_gamma;
            chi2      += acc.chi2;
            loss_img  += acc.loss;
            n_used    += acc.n_used;
        }
        if (n_used == 0) return;

        float r_rms = 0.f;
        for (float a : abs_r) r_rms += a * a;
        r_rms = std::sqrt(r_rms / static_cast<float>(std::max<std::size_t>(1, abs_r.size())));

        // Solve the 2x2 weighted normal equations for the monitor.
        float b_const = 0.f, b_invd = 0.f, se_const = 0.f, se_invd = 0.f;
        {
            const double det = S11 * S22 - S12 * S12;
            if (std::abs(det) > 1e-12 and Sw > 0)
            {
                b_const = static_cast<float>(( S22 * Sy1 - S12 * Sy2) / det);
                b_invd  = static_cast<float>((-S12 * Sy1 + S11 * Sy2) / det);
                se_const = static_cast<float>(std::sqrt(std::max(0.0,  S22 / det)));
                se_invd  = static_cast<float>(std::sqrt(std::max(0.0,  S11 / det)));
            }
        }

        // ── Pass 2: who won? Solve twice from the SAME start (the pattern of the old GN shadow).
        //    Shadow mode must not be able to change the published pose, so the window poses are
        //    restored afterwards by the caller (poses_after is what the authority produced).
        float dx = 0.f, dy = 0.f, dth = 0.f;
        if (probe_pose)
        {
            rc::gn::Input in;
            in.model = model_.get();
            in.params = &params;
            in.window = &window_mgr_.window;
            in.boundary_prior = &window_mgr_.boundary_prior;
            in.boundary_weight = boundary_weight;

            Params p_on  = params;  p_on.image_edge.enable = true;  p_on.image_edge.drive = true;
            rc::gn::Input in_on = in; in_on.params = &p_on;
            auto poses_on = poses_after;
            rc::gn::Options opts;
            const auto r_on = rc::gn::solve(in_on, poses_on, opts);
            if (r_on.ok)
            {
                const auto& a = poses_after.back();
                const auto& b = poses_on.back();
                dx = b.x() - a.x(); dy = b.y() - a.y();
                dth = std::atan2(std::sin(b.z() - a.z()), std::cos(b.z() - a.z()));
            }
        }

        // ── The DRIVING term's own consistency ──────────────────────────────────────────────────
        // Everything above describes the per-SAMPLE contour path. With useTriplePoints = true the
        // factor that actually moves the pose uses ONE residual per corner instead, weighted by that
        // corner's own cov_uv — so the sample path's chi2 diagnoses a different object from the one
        // being driven, and reporting it as if it were the same is how an arm gets graded on a
        // quantity it does not use.
        // chi2 here is exactly what ImageEdgeFactor weights by: r' cov_uv^-1 r, 2 DOF per corner.
        // Split MATCHED vs all, so occlusion contamination is separable from a genuine misfit rather
        // than averaged into it.
        int   tp_n = 0, tp_n_matched = 0, tp_occl = 0;
        float tp_ss = 0.f, tp_chi2 = 0.f, tp_ss_m = 0.f, tp_chi2_m = 0.f;
        for (const auto& t : obs.triple_points)
        {
            const Eigen::Matrix2f& C = t.cov_uv;
            const float det = C(0, 0) * C(1, 1) - C(0, 1) * C(1, 0);
            if (not std::isfinite(det) or not (det > 1e-12f)) continue;
            const Eigen::Vector2f r = t.resid_px;
            if (not r.allFinite()) continue;
            const float q = (C(1, 1) * r.x() * r.x() - 2.f * C(0, 1) * r.x() * r.y()
                             + C(0, 0) * r.y() * r.y()) / det;
            ++tp_n; tp_ss += r.squaredNorm(); tp_chi2 += q;
            if (t.pi_vis < 0.5f) { ++tp_occl; continue; }
            ++tp_n_matched; tp_ss_m += r.squaredNorm(); tp_chi2_m += q;
        }
        const float tp_rms   = tp_n ? std::sqrt(tp_ss / static_cast<float>(tp_n)) : 0.f;
        const float tp_c2    = tp_n ? tp_chi2 / (2.f * static_cast<float>(tp_n)) : 0.f;
        const float tp_rms_m = tp_n_matched ? std::sqrt(tp_ss_m / static_cast<float>(tp_n_matched)) : 0.f;
        const float tp_c2_m  = tp_n_matched ? tp_chi2_m / (2.f * static_cast<float>(tp_n_matched)) : 0.f;

        if (not image_edge_csv_.is_open())
        {
            image_edge_csv_.open(params.image_edge_csv, std::ios::out | std::ios::trunc);
            if (image_edge_csv_.is_open())
            {
                image_edge_csv_.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
                image_edge_csv_
                    // does it carry information?
                    << "ts_ms,frame_stamp,dt_img_lidar_ms,n_segments,n_used,sum_gamma,sigma_i,"
                       "r_rms_px,chi2_per_dof,trace_raw,trace_eff,info_ratio,loss_img,"
                    // under what MOTION? exposure smear scales with omega and is NOT modelled;
                    // ego-motion over the image-lidar offset IS (nuisance column [3]).
                       "omega,speed,"
                    // the DRIVING term (useTriplePoints), separately from the sample path above
                       "tp_n,tp_r_rms_px,tp_chi2_per_dof,tp_n_matched,tp_r_rms_matched,tp_chi2_matched,tp_occluded,"
                    // who won?
                       "dpose_valid,dpose_x,dpose_y,dpose_th,"
                    // is the mount calibrated?
                       "bias_const_px,se_const_px,bias_invd_m,se_invd_m,implied_dpitch_rad,implied_dheight_m\n";
            }
        }
        {   // Publish for the viewer BEFORE the CSV branch, so the plot does not depend on logging
            // being enabled — an instrument that only works when another instrument is on is how a
            // channel comes to look dead.
            std::scoped_lock lk(image_edge_stats_mutex_);
            image_edge_stats_.valid        = true;
            image_edge_stats_.chi2_per_dof = chi2 / std::max(1.f, sum_gamma);
            image_edge_stats_.r_rms_px     = r_rms;
            image_edge_stats_.loss_img     = loss_img;
            image_edge_stats_.n_used       = n_used;
            image_edge_stats_.n_segments   = static_cast<int>(obs.segments.size());
            image_edge_stats_.ts_ms        = timestamp_ms;
        }

        if (image_edge_csv_.is_open())
        {
            const float chi2_dof = chi2 / std::max(1.f, sum_gamma);
            image_edge_csv_
                << timestamp_ms << ',' << obs.frame_stamp << ',' << obs.dt_to_slot_ms
                << ',' << obs.segments.size() << ',' << n_used << ',' << sum_gamma
                << ',' << obs.sigma_i << ',' << r_rms << ',' << chi2_dof
                << ',' << trace_raw << ',' << trace_eff
                << ',' << (trace_raw / std::max(1e-9f, trace_eff))
                << ',' << loss_img
                << ',' << obs.body_twist.z()
                << ',' << std::hypot(obs.body_twist.x(), obs.body_twist.y())
                << ',' << tp_n << ',' << tp_rms << ',' << tp_c2
                << ',' << tp_n_matched << ',' << tp_rms_m << ',' << tp_c2_m << ',' << tp_occl
                << ',' << (probe_pose ? 1 : 0) << ',' << dx << ',' << dy << ',' << dth
                << ',' << b_const << ',' << se_const << ',' << b_invd << ',' << se_invd
                << ',' << (obs.cam.fy > 0.f ? b_const / obs.cam.fy : 0.f) << ',' << b_invd
                << '\n';
            image_edge_csv_.flush();
            ++imgedge_rows_;
        }
        log_triple_points(obs, timestamp_ms, pose);
        imgedge_health(timestamp_ms);
    }

    /// One row per detected triple point per frame. Diagnostic only — nothing consumes these yet.
    ///
    /// ★ THE CROSS-CHECK THAT MAKES THIS WORTH LOGGING SEPARATELY: du and dv here are the SAME
    ///   displacement the rigid-shift monitor reports as tx and ty, arrived at by a completely
    ///   different route — two per-segment line offsets intersected, rather than a two-parameter
    ///   regression over every sample in the frame. If the detector is right the medians agree. If
    ///   they disagree the detector is wrong, and that is worth knowing BEFORE anything is built on
    ///   top of it.
    void RoomConcept::log_triple_points(const ImageEdgeObs& obs, std::int64_t timestamp_ms,
                                        const Eigen::Vector3f& pose)
    {
        ++triple_frames_;
        if (obs.triple_points.empty()) return;
        if (not triple_csv_.is_open())
        {
            // Keyed by camera for the same reason the pair log is: one camera per run, and a fixed
            // filename means run 2 destroys run 1's record of a DIFFERENT device.
            triple_csv_.open("etc/image_edge_triple_" + (obs.camera.empty() ? "unknown" : obs.camera)
                                 + ".csv", std::ios::out | std::ios::trunc);
            if (triple_csv_.is_open())
            {
                triple_csv_.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
                triple_csv_ << "ts_ms,camera,vertex,at_ceiling,u_pred,v_pred,u_meas,v_meas,du,dv,"
                               "suu,svv,suv,cond,n_corner,n_floor,"
                               // depth_raw as published; pred_fwd and pred_range are what the MODEL
                               // says at this pose. depth_raw ~= pred_fwd means the value is the
                               // forward coordinate (assumed); depth_raw ~= pred_range, with the
                               // excess growing toward the image edge, means it is range along the
                               // ray and xyz_from_pixel_depth needs the other formula.
                               // ★ depth_raw / range_sigma / depth_dt_ms are −1/−1/0 for EVERY row
                               //   of a camera that advertises no depth stream (the ricoh panorama
                               //   is one). That is a correct report of an ABSENT channel, not a
                               //   failed measurement — `has_depth` says which of the two it is, so
                               //   an analysis is never left to guess from a column of sentinels.
                               "depth_raw,pred_fwd,pred_range,range_m,range_sigma,depth_dt_ms,"
                               "has_depth,pose_x,pose_y,pose_theta\n";
            }
        }
        if (not triple_csv_.is_open()) return;
        // What the MODEL says this corner's distance is, in both conventions, so the logged
        // depth_raw can be compared against each. Camera Y is forward (CameraAPI::ray_from_pixel).
        const float cth = std::cos(pose.z()), sth = std::sin(pose.z());
        Eigen::Matrix3f Rm; Rm << cth, sth, 0.f, -sth, cth, 0.f, 0.f, 0.f, 1.f;
        const auto to_cam = [&](const Eigen::Vector3f& p_room)
        {
            const Eigen::Vector3f e(p_room.x() - pose.x(), p_room.y() - pose.y(), p_room.z());
            return Eigen::Vector3f(obs.cam_R_robot * (Rm * e) + obs.cam_t_robot);
        };
        const auto pred_fwd   = [&](const TriplePoint& t) { return to_cam(t.p_room).y(); };
        const auto pred_range = [&](const TriplePoint& t) { return to_cam(t.p_room).norm(); };
        for (const auto& t : obs.triple_points)
        {
            triple_csv_ << timestamp_ms << ',' << obs.camera << ',' << t.vertex << ','
                        << (t.from == ContourClass::WallCeiling ? 1 : 0) << ','
                        << t.uv_pred.x() << ',' << t.uv_pred.y() << ','
                        << t.uv_meas.x() << ',' << t.uv_meas.y() << ','
                        << (t.uv_meas.x() - t.uv_pred.x()) << ','
                        << (t.uv_meas.y() - t.uv_pred.y()) << ','
                        << t.cov_uv(0, 0) << ',' << t.cov_uv(1, 1) << ',' << t.cov_uv(0, 1) << ','
                        << t.cond << ',' << t.n_corner << ',' << t.n_floor << ','
                        << t.depth_raw << ',' << pred_fwd(t) << ',' << pred_range(t) << ','
                        << t.range_m << ',' << t.range_sigma << ','
                        << (obs.depth_stamp_ms ? obs.depth_stamp_ms - timestamp_ms : 0) << ','
                        << (obs.depth_stamp_ms ? 1 : 0) << ','
                        << pose.x() << ',' << pose.y() << ',' << pose.z() << '\n';
            ++triple_rows_;
            if (t.from == ContourClass::WallCeiling) ++triple_ceil_; else ++triple_floor_;
        }
        triple_csv_.flush();

        // ── THE TRIPLE-POINT POSE FACTOR, IN SHADOW ──────────────────────────────────────────────
        // What it would be if it entered the loss: a 2-D projection residual per corner,
        //     r = uv_meas - project(pose, p_room),     W = cov_uv^-1
        // with the pose Jacobian the source already derives for its scalar samples, kept in full
        // rather than contracted onto a contour normal (a triple point has no aperture problem).
        //
        // ★ SELF-CONTAINED BY DESIGN. It forms its own 3x3 normal equations here instead of going
        //   through the GN solver, so there is no code path by which this can move the published
        //   pose — `drive` is not a flag that has to be respected, the influence does not exist.
        //   Building it this way first is the point: the gate for letting it drive is its own
        //   chi2/dof, and that has to be measured before the plumbing that could act on it exists.
        //
        // ★ WHAT THIS CANNOT SHOW. Corners sharing a floor segment are correlated (the detector
        //   pairs each vertex with whichever adjacent floor line carries more weight, so two corners
        //   of one wall can share it), and this treats them as independent. With a handful per frame
        //   and a covariance measured CONSERVATIVE by 36x in u / 2.3x in v against fixed-pose
        //   repeatability, the error is bounded and in the safe direction — but chi2 below 1 must be
        //   read as "conservative sigmas", not "better than the noise floor".
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d g = Eigen::Vector3d::Zero();
        double chi2 = 0.0; int n_used = 0;
        const float cth2 = std::cos(pose.z()), sth2 = std::sin(pose.z());
        Eigen::Matrix3f Rm2; Rm2 << cth2, sth2, 0.f, -sth2, cth2, 0.f, 0.f, 0.f, 1.f;
        for (const auto& t : obs.triple_points)
        {
            const Eigen::Vector3f e(t.p_room.x() - pose.x(), t.p_room.y() - pose.y(), t.p_room.z());
            const Eigen::Vector3f p_robot = Rm2 * e;
            const Eigen::Vector3f p_cam   = obs.cam_R_robot * p_robot + obs.cam_t_robot;
            Eigen::Vector2d uvp;
            if (not rc::img::project_with_model(obs.cam, p_cam.cast<double>(), uvp)) continue;
            Eigen::Matrix<double, 2, 3> P;
            if (not rc::img::project_jacobian_model(obs.cam, p_cam.cast<double>(), P)) continue;
            // d(p_cam)/d(pose) — identical to the source's Jx, kept as 2x3 instead of contracted.
            Eigen::Matrix3f Jx;
            Jx.col(0) = obs.cam_R_robot * (-Rm2.col(0));
            Jx.col(1) = obs.cam_R_robot * (-Rm2.col(1));
            Jx.col(2) = obs.cam_R_robot * Eigen::Vector3f(p_robot.y(), -p_robot.x(), 0.f);
            const Eigen::Matrix<double, 2, 3> J = P * Jx.cast<double>();
            const Eigen::Matrix2d C = t.cov_uv.cast<double>();
            if (not (C.determinant() > 1e-12)) continue;
            const Eigen::Matrix2d W = C.inverse();
            // Wrap-safe in u: on the Ricoh's equirectangular model the column axis is cyclic, and a
            // corner sitting on the seam would otherwise contribute a residual of nearly a full
            // image width. Identity on a pinhole.
            const Eigen::Vector2d r(
                rc::img::du_wrapped(static_cast<double>(t.uv_meas.x()) - uvp.x(), obs.cam),
                static_cast<double>(t.uv_meas.y()) - uvp.y());
            H.noalias() += J.transpose() * W * J;
            g.noalias() += J.transpose() * W * r;
            chi2 += r.dot(W * r);
            ++n_used;
        }
        if (n_used == 0) return;
        ++tps_frames_;
        const double dof = std::max(1.0, 2.0 * n_used - 3.0);
        // ★ TWO CHI-SQUARES, AND ONLY THE SECOND IS THE GATE.
        //   `chi2` is evaluated at the LOCALISER's pose, so it measures how much this factor
        //   disagrees with the incumbent — useful, but it conflates a wrong pose with wrong sigmas
        //   and cannot answer "is the covariance honest?". `chi2_post` is what is LEFT after the
        //   three pose DOF have absorbed everything they can: chi2 - g' H^-1 g. That is the part no
        //   pose could explain, so it is the one that must approach 1 before this may drive.
        //   Verified in simulation: with a deliberately wrong pose the pre-fit value runs into the
        //   thousands while the recovered dpose is correct to ~1.5 mm, which is exactly the
        //   confusion this split removes.
        const double c2d = chi2 / dof;
        // ★ A SINGLE CORNER IS NOT A WASTED FRAME. A factor CONTRIBUTES information; it never has to
        //   be invertible on its own, and in a multi-modal graph which landmarks are visible changes
        //   from frame to frame by design. One corner is a rank-2 constraint on three DOF, and the
        //   SDF and LiDAR terms supply the rest — that is what the graph is for.
        //   The n>=2 gate below belongs to this DIAGNOSTIC, which has to invert to report a dpose.
        //   Writing 0 for those frames (as this did) reports a real contribution as an absence, and
        //   dragged every dpose median to exactly 0.000 over 31.5% of frames. NaN now, never 0.
        const double NAd = std::numeric_limits<double>::quiet_NaN();
        Eigen::Vector3d dp = Eigen::Vector3d::Constant(NAd);
        double cond = NAd, chi2_post = NAd, c2d_post = NAd;
        // The information this frame CONTRIBUTES, which is defined whatever the rank: reported so a
        // one-corner frame can be seen doing its job instead of looking like a hole in the data.
        const double info_trace = H.trace();
        if (n_used >= 2)
        {
            // ★ CORRELATION-NORMALISED, and the raw form is a trap this file already warns about
            //   for the mount fits: H mixes METRES and RADIANS, so its raw condition number is
            //   unit-dependent and says nothing about geometry. Measured on a real two-corner view
            //   the raw number reads 418 while the unit-free one reads 97 — and the posterior is
            //   5.6 mm / 2.4 mm / 0.083 deg, i.e. excellent. The raw number made a good factor look
            //   degenerate. What the normalised form shows is STRUCTURE, not weakness: corr(x,theta)
            //   ~ 0.98, the ordinary bearing-only ambiguity when the visible corners sit ahead and
            //   clustered, with y determined independently.
            const Eigen::Matrix3d Cv = H.inverse();
            if (Cv.allFinite())
            {
                Eigen::Matrix3d Rc = Eigen::Matrix3d::Identity();
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        Rc(i, j) = Cv(i, j) / std::sqrt(std::max(1e-300, Cv(i, i) * Cv(j, j)));
                const Eigen::Vector3d ev =
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(Rc).eigenvalues();
                cond = ev(2) / std::max(1e-12, ev(0));
            }
            // A tiny ridge only so a single visible corner cannot produce a NaN. It is NOT a prior:
            // if the geometry cannot determine a pose component, `cond` is what says so.
            dp = (H + 1e-9 * Eigen::Matrix3d::Identity()).ldlt().solve(g);
            chi2_post = std::max(0.0, chi2 - g.dot(dp));
            c2d_post  = chi2_post / dof;
        }
        if (std::isfinite(c2d_post)) { tps_chi2_sum_ += c2d_post; ++tps_chi2_n_; }
        if (not triple_pose_csv_.is_open())
        {
            triple_pose_csv_.open("etc/image_edge_triple_pose.csv", std::ios::out | std::ios::trunc);
            if (triple_pose_csv_.is_open())
            {
                triple_pose_csv_.imbue(std::locale::classic());
                triple_pose_csv_ << "ts_ms,n_corners,chi2,dof,chi2_per_dof,"
                                    "chi2_post,chi2_post_per_dof,info_trace,"
                                    "dpose_x,dpose_y,dpose_th,cond_H,pose_x,pose_y,pose_theta\n";
            }
        }
        if (triple_pose_csv_.is_open())
        {
            triple_pose_csv_ << timestamp_ms << ',' << n_used << ',' << chi2 << ',' << dof << ','
                             << c2d << ',' << chi2_post << ',' << c2d_post << ','
                             << info_trace << ','
                             << dp(0) << ',' << dp(1) << ',' << dp(2) << ','
                             << cond << ',' << pose.x() << ',' << pose.y() << ',' << pose.z() << '\n';
            triple_pose_csv_.flush();
        }
    }

} // namespace rc
