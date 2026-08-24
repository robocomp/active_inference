/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#include "image_edge_source.h"

#include <algorithm>
#include <cmath>

#include <occlusion/occlusion.h>

#include "image_edge_ops.h"

namespace rc
{
namespace
{
    /// One 3-D segment of a structural contour, in the ROOM frame.
    struct Contour3D
    {
        Eigen::Vector3f a, b;
        ContourClass    cls;
    };

    /// room -> robot, then robot -> camera. `pose` = [x, y, theta] of room<-robot.
    inline Eigen::Vector3f to_camera(const Eigen::Vector3f& p_room, const Eigen::Vector3f& pose,
                                     const Eigen::Matrix3f& cam_R_robot, const Eigen::Vector3f& cam_t_robot)
    {
        const float c = std::cos(pose.z()), s = std::sin(pose.z());
        const Eigen::Vector3f e(p_room.x() - pose.x(), p_room.y() - pose.y(), p_room.z());
        // R(-theta) * e
        const Eigen::Vector3f p_robot( c * e.x() + s * e.y(),
                                      -s * e.x() + c * e.y(),
                                       e.z());
        return cam_R_robot * p_robot + cam_t_robot;
    }

    inline float median_of(std::vector<float>& v)
    {
        if (v.empty()) return 0.f;
        const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
        std::nth_element(v.begin(), mid, v.end());
        return *mid;
    }
}  // namespace

ImageEdgeObs ImageEdgeSource::extract(const GrayFrame& frame,
                                      const CameraModel& model,
                                      const Eigen::Matrix3f& cam_R_robot,
                                      const Eigen::Vector3f& cam_t_robot,
                                      const Eigen::Vector3f& pose,
                                      const Eigen::Matrix3f& pose_cov,
                                      const Eigen::Vector3f& body_twist,
                                      std::int64_t dt_ms,
                                      Stats* stats) const
{
    ImageEdgeObs obs;
    obs.frame_stamp   = frame.stamp;
    obs.dt_to_slot_ms = dt_ms;
    obs.sigma_i       = frame.sigma_i;
    obs.cam_R_robot   = cam_R_robot;
    obs.cam_t_robot   = cam_t_robot;
    obs.cam           = model;

    Stats st;
    st.sigma_i = frame.sigma_i;
    if (not frame.ok() or not model.valid or polygon_.size() < 3)
    {
        if (stats) *stats = st;
        return obs;
    }

    const int   W = frame.width, H = frame.height;
    const auto* g = frame.gray.data();
    const std::size_t np = polygon_.size();

    // ── 1. Enumerate structural contours in the ROOM frame ───────────────────────────────────────
    // Vertical wall-wall corners first: their image normal is HORIZONTAL, so the mount pitch and
    // height nuisances barely project onto them (h ~ 0 in those columns) and they carry bearing,
    // which is the DOF the image genuinely adds. The floor junction carries range but is exposed to
    // delta_d = theta_pitch * d^2 / h — 1 degree is 14 cm at 3 m — which is why it is separately gated.
    std::vector<Contour3D> contours;
    if (cfg_.use_wall_corners)
        for (std::size_t i = 0; i < np; ++i)
            contours.push_back({{polygon_[i].x(), polygon_[i].y(), 0.f},
                                {polygon_[i].x(), polygon_[i].y(), cfg_.room_height},
                                ContourClass::WallCorner});
    if (cfg_.use_floor_junction)
        for (std::size_t i = 0; i < np; ++i)
        {
            const auto& a = polygon_[i];
            const auto& b = polygon_[(i + 1) % np];
            contours.push_back({{a.x(), a.y(), 0.f}, {b.x(), b.y(), 0.f}, ContourClass::FloorWall});
        }
    if (cfg_.use_wall_ceiling)
        for (std::size_t i = 0; i < np; ++i)
        {
            const auto& a = polygon_[i];
            const auto& b = polygon_[(i + 1) % np];
            contours.push_back({{a.x(), a.y(), cfg_.room_height},
                                {b.x(), b.y(), cfg_.room_height}, ContourClass::WallCeiling});
        }
    st.n_contours = static_cast<int>(contours.size());

    // Camera position in the room frame, for the occlusion sightline.
    const Eigen::Vector2f cam_xy(pose.x(), pose.y());

    std::vector<float> sig_list, len_list;

    for (const auto& c : contours)
    {
        const float seg_len = (c.b - c.a).norm();
        if (seg_len < 1e-3f) continue;
        // ★ Arc-length-uniform in 3-D, NOT uniform in pixels. Uniform-in-pixels concentrates samples
        //   at the near end, which is exactly where the height nuisance is weakest — it would
        //   silently reweight the estimator toward the least informative geometry.
        const int K = std::max(2, static_cast<int>(std::ceil(seg_len / std::max(1e-3f, cfg_.sample_spacing_m))));

        ImageEdgeSegment seg;
        seg.class_id = c.cls;
        seg.samples.reserve(static_cast<std::size_t>(K));

        for (int k = 0; k < K; ++k)
        {
            const float t = (static_cast<float>(k) + 0.5f) / static_cast<float>(K);
            const Eigen::Vector3f p_room = c.a + t * (c.b - c.a);

            // ── project the sample and its two neighbours (for the projected tangent) ────────────
            const float dt_par = 0.5f / static_cast<float>(K);
            const Eigen::Vector3f p_prev = c.a + std::max(0.f, t - dt_par) * (c.b - c.a);
            const Eigen::Vector3f p_next = c.a + std::min(1.f, t + dt_par) * (c.b - c.a);

            Eigen::Vector2d uv, uv_p, uv_n;
            const Eigen::Vector3f pc  = to_camera(p_room, pose, cam_R_robot, cam_t_robot);
            const Eigen::Vector3f pcp = to_camera(p_prev, pose, cam_R_robot, cam_t_robot);
            const Eigen::Vector3f pcn = to_camera(p_next, pose, cam_R_robot, cam_t_robot);
            if (not rc::img::project_with_model(model, pc.cast<double>(),  uv))   continue;
            if (not rc::img::project_with_model(model, pcp.cast<double>(), uv_p)) continue;
            if (not rc::img::project_with_model(model, pcn.cast<double>(), uv_n)) continue;
            if (uv.x() < 1.0 or uv.y() < 1.0 or uv.x() >= W - 2 or uv.y() >= H - 2) continue;
            st.n_projected++;

            // Projected tangent -> image-space normal. Computed from the PROJECTION, so it curves
            // correctly on a panorama; a 3-D tangent projected as a direction would not.
            double du = uv_n.x() - uv_p.x();
            if (model.kind != CameraModel::Kind::Pinhole)
            {   // the panorama wraps: fold before differencing (ricoh_projection_overlay.cpp:99)
                while (du >  0.5 * model.width) du -= model.width;
                while (du <= -0.5 * model.width) du += model.width;
            }
            Eigen::Vector2f tang(static_cast<float>(du), static_cast<float>(uv_n.y() - uv_p.y()));
            if (tang.norm() < 1e-6f) continue;
            tang.normalize();
            const Eigen::Vector2f n_hat(-tang.y(), tang.x());

            // ── occlusion PRIOR (not a cull) ────────────────────────────────────────────────────
            // A wall between the camera and this sample hides it. Non-convex rooms make this real:
            // one wall routinely occludes another's corner. Kept as a continuous prior so the term
            // degrades at a box edge instead of flickering.
            float pi_vis = 1.f;
            {
                const Eigen::Vector2f tgt(p_room.x(), p_room.y());
                // own_wall_skip: every structural contour LIES ON the polygon, so the segment it
                // belongs to must not count as its own occluder.
                if (rc::occlusion::walls_block(cam_xy, tgt, polygon_, 0.08f))
                    pi_vis = 0.02f;      // strongly disbelieved, never exactly 0 (keeps the mixture finite)
            }
            if (pi_vis < 0.5f) st.n_occluded++;
            st.n_visible++;

            // ── search half-length, DERIVED from the posterior, not a pixel constant ─────────────
            // sigma_pred^2 = J * Sigma_x * J^T with the SAME J the factor uses, plus the shared
            // nuisance spread. Right after a relocalisation Sigma_x is large -> a wide window -> more
            // competing peaks -> the responsibility drops -> the term self-mutes exactly when it
            // would be most dangerous. As the pose tightens the window narrows and the term sharpens.
            Eigen::Matrix<float, IMAGE_EDGE_NUISANCES, 1> hcol;
            hcol.setZero();
            float sigma_pred = 2.f;
            {
                // d(u,v)/dp_cam — the SAME helper the factor uses, so the search window and the
                // Jacobian that consumes it can never be derived from different maths.
                Eigen::Matrix<double, 2, 3> Pd;
                if (not rc::img::project_jacobian_model(model, pc.cast<double>(), Pd)) continue;
                const Eigen::Matrix<float, 2, 3> P = Pd.cast<float>();

                // dp_cam/dx for x = [x, y, theta] of room<-robot.
                const float cth = std::cos(pose.z()), sth = std::sin(pose.z());
                Eigen::Matrix3f dRm;                       // d(R(-theta)e)/d(x,y) = -R(-theta)
                dRm <<  cth,  sth, 0.f,
                       -sth,  cth, 0.f,
                        0.f,  0.f, 1.f;
                const Eigen::Vector3f e(p_room.x() - pose.x(), p_room.y() - pose.y(), p_room.z());
                const Eigen::Vector3f p_robot = dRm * e;
                Eigen::Matrix3f Jx;
                Jx.col(0) = cam_R_robot * (-dRm.col(0));
                Jx.col(1) = cam_R_robot * (-dRm.col(1));
                // d/dtheta of R(-theta)e is (+p_robot.y, -p_robot.x, 0), rotated into the camera.
                // ★ Verified symbolically; the opposite sign is the natural mistake here and it
                //   flips the yaw column, which converges just as prettily on the wrong answer.
                Jx.col(2) = cam_R_robot * Eigen::Vector3f(p_robot.y(), -p_robot.x(), 0.f);

                const Eigen::Matrix<float, 1, 3> Jrow = n_hat.transpose() * P * Jx;
                const float v_pred = (Jrow * pose_cov * Jrow.transpose())(0, 0);
                sigma_pred = std::sqrt(std::max(0.f, v_pred));

                // ── nuisance sensitivities, all of the form n^T P (.) ────────────────────────────
                // Shared WITHIN a contour segment, which is why the Woodbury correction is applied
                // per segment. These are what stop N correlated samples claiming sqrt(N) precision.
                const Eigen::Vector3f x_cam(1.f, 0.f, 0.f), z_cam(0.f, 0.f, 1.f);
                const Eigen::Matrix<float, 1, 2> nP_pre = n_hat.transpose();
                const auto contract = [&](const Eigen::Vector3f& d) -> float
                { return (nP_pre * (P * d))(0, 0); };
                hcol(0) = cfg_.mount_pitch_sigma  * contract(x_cam.cross(pc));   // boresight pitch
                hcol(1) = cfg_.mount_height_sigma * contract(z_cam);             // mount height
                hcol(2) = cfg_.mount_yaw_sigma    * contract(z_cam.cross(pc));   // boresight yaw
                // image/lidar dt: ego-motion during the offset. A VARIANCE, not a correction — a
                // dead-reckoned fix would reintroduce the graph pose (CLAUDE.md: ego-motion
                // downweights via the interaction-matrix variance added to R, not a motion gate).
                const float dts = 1e-3f * static_cast<float>(dt_ms);
                const Eigen::Vector3f vel_robot(body_twist.x(), body_twist.y(), 0.f);
                const Eigen::Vector3f dp = cam_R_robot * (-vel_robot * dts)
                                         + cam_R_robot * Eigen::Vector3f(p_robot.y(), -p_robot.x(), 0.f)
                                           * (-body_twist.z() * dts);
                hcol(3) = contract(dp);
            }

            float L = cfg_.search_sigmas * std::sqrt(sigma_pred * sigma_pred + hcol.squaredNorm() + 4.f);
            if (L > static_cast<float>(cfg_.max_search_px)) { L = static_cast<float>(cfg_.max_search_px); st.n_clamped++; }
            L = std::max(2.f, L);

            // ── normal search: peak of |dI/dn| along +-n_hat ─────────────────────────────────────
            const int steps = static_cast<int>(std::lround(L));
            float best_mag = 0.f, best_s = 0.f, sum_g2 = 0.f;
            int   best_i = 0;
            std::vector<float> prof;
            prof.reserve(static_cast<std::size_t>(2 * steps + 1));
            for (int i = -steps; i <= steps; ++i)
            {
                const float u = static_cast<float>(uv.x()) + static_cast<float>(i) * n_hat.x();
                const float v = static_cast<float>(uv.y()) + static_cast<float>(i) * n_hat.y();
                float gd = 0.f;
                if (not rc::img::dir_derivative(g, W, H, u, v, n_hat, gd)) { prof.push_back(0.f); continue; }
                const float m = std::fabs(gd);
                prof.push_back(m);
                sum_g2 += gd * gd;
                if (m > best_mag) { best_mag = m; best_s = static_cast<float>(i); best_i = static_cast<int>(prof.size()) - 1; }
            }
            if (best_mag <= 0.f or prof.size() < 3) continue;

            // Sub-pixel: parabola through the three samples around the peak.
            if (best_i > 0 and best_i + 1 < static_cast<int>(prof.size()))
                best_s += rc::img::parabolic_vertex(prof[best_i - 1], prof[best_i], prof[best_i + 1]);

            // ── precision from the Cramer-Rao bound. A flat wall gives sum_g2 -> 0 -> sigma -> inf
            //    -> weight 0, so no visibility threshold is needed anywhere. ────────────────────────
            const float sigma_px = rc::img::crb_sigma_px(sum_g2, frame.sigma_i);
            if (not std::isfinite(sigma_px)) continue;
            st.n_searched++;

            ImageEdgeSample smp;
            smp.p_room   = p_room;
            smp.n_hat    = n_hat;
            smp.uv_meas  = Eigen::Vector2f(static_cast<float>(uv.x()) + best_s * n_hat.x(),
                                           static_cast<float>(uv.y()) + best_s * n_hat.y());
            smp.sigma_px = sigma_px;
            smp.pi_vis   = pi_vis;
            smp.search_L = L;
            smp.h        = hcol;
            seg.samples.push_back(smp);

            sig_list.push_back(sigma_px);
            len_list.push_back(L);
        }

        if (not seg.samples.empty())
            obs.segments.push_back(std::move(seg));
    }

    st.med_sigma_px  = median_of(sig_list);
    st.med_search_px = median_of(len_list);
    if (stats) *stats = st;
    return obs;
}

}  // namespace rc
