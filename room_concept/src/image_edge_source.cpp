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
#include "image_edge_accumulate.h"   // responsibility(): the SAME weight the factor and monitors use

namespace rc
{
namespace
{
    /// One 3-D segment of a structural contour, in the ROOM frame.
    struct Contour3D
    {
        Eigen::Vector3f a, b;
        ContourClass    cls;
        int             vertex = -1;   ///< polygon vertex (WallCorner) or edge's first vertex
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
    // Column axis is CYCLIC on a 360 model and not on a pinhole. Everything that samples the image
    // below takes this, so the seam is closed in exactly one place. See image_edge_ops.h::bilinear.
    const int wrap_u = (model.kind == CameraModel::Kind::Pinhole) ? 0 : W;

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
                                ContourClass::WallCorner, static_cast<int>(i)});
    if (cfg_.use_floor_junction)
        for (std::size_t i = 0; i < np; ++i)
        {
            const auto& a = polygon_[i];
            const auto& b = polygon_[(i + 1) % np];
            contours.push_back({{a.x(), a.y(), 0.f}, {b.x(), b.y(), 0.f}, ContourClass::FloorWall,
                                static_cast<int>(i)});
        }
    if (cfg_.use_wall_ceiling)
        for (std::size_t i = 0; i < np; ++i)
        {
            const auto& a = polygon_[i];
            const auto& b = polygon_[(i + 1) % np];
            contours.push_back({{a.x(), a.y(), cfg_.room_height},
                                {b.x(), b.y(), cfg_.room_height}, ContourClass::WallCeiling,
                                static_cast<int>(i)});
        }
    st.n_contours = static_cast<int>(contours.size());

    // Camera position in the room frame, for the occlusion sightline.
    const Eigen::Vector2f cam_xy(pose.x(), pose.y());

    std::vector<float> sig_list, len_list;

    // ── Per-segment line offset, for the triple-point detector ───────────────────────────────────
    // Each contour is modelled as its PREDICTED line displaced perpendicular by one scalar. The
    // direction is taken from the model and only the offset is fitted: the direction is far better
    // known than the offset (it comes from the polygon and the pose, both well constrained), and
    // letting it float would trade a well-posed 1-parameter fit for an ill-posed 2-parameter one on
    // samples that already span only a few pixels of lateral range.
    struct SegFit
    {
        int             vertex = -1;
        ContourClass    cls    = ContourClass::WallCorner;
        double          w = 0.0, ws = 0.0;      ///< weight, and weight * along-normal offset
        double          wh4 = 0.0;              ///< weight * this segment's map-offset sensitivity
        double          wv  = 0.0;              ///< weight * pi_vis, for the corner's carried visibility
        Eigen::Vector2d wn = Eigen::Vector2d::Zero();
    };
    std::vector<SegFit> fits;

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
        seg.vertex   = c.vertex;
        SegFit fit; fit.vertex = c.vertex; fit.cls = c.cls;
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
            // Vertical bounds are real on every model — the top and bottom rows are the ends of the
            // sensor. The HORIZONTAL bound is not, on a panorama: project_with_model already folded u
            // into [0, W), so rejecting the columns beside the seam would blank a strip of azimuth
            // for no reason other than where the manufacturer chose to cut the sphere.
            if (uv.y() < 1.0 or uv.y() >= H - 2) continue;
            if (wrap_u == 0 and (uv.x() < 1.0 or uv.x() >= W - 2)) continue;
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

                // ── AND BY THE OBJECTS IN THE ROOM ───────────────────────────────────────────────
                // A table or a chair between the camera and a wall junction does NOT make the edge
                // search fail — it makes it SUCCEED on the object's edge, returning a confidently
                // wrong sub-pixel position with a small Cramer-Rao sigma. The mixture would then have
                // to reject it on residual size alone, and an object edge lying near the predicted
                // junction would not be rejected at all. Explaining those samples away with the
                // models that predict them is the point of this block.
                // ★ SOFT, like the wall case and for the same reason: an object at a contour's edge
                // would otherwise flicker the term in and out. pi_vis MULTIPLIES, so several partial
                // occluders compound rather than one winning.
                // ★ SOFTENED BY THE OBJECT'S OWN POSITION UNCERTAINTY. An anchor known to +/-30 cm
                // cannot say which specific ray it blocks, so its claim is widened by map_pos_sigma
                // and weakened with it — a vague object explains away vaguely.
                for (const auto& a : anchors_)
                {
                    if (not (a.radius_m > 0.f)) continue;      // no stated footprint: no claim
                    const Eigen::Vector2f c(a.pose_world.x(), a.pose_world.y());
                    const Eigen::Vector2f d = tgt - cam_xy;
                    const float len2 = d.squaredNorm();
                    if (len2 < 1e-9f) continue;
                    // Closest approach of the camera->sample ray to the object centre, clamped to the
                    // segment: an object BEHIND the wall is not an occluder.
                    const float t = std::clamp((c - cam_xy).dot(d) / len2, 0.f, 1.f);
                    const float miss = ((cam_xy + t * d) - c).norm();
                    const float reach = a.radius_m + std::max(0.f, a.map_pos_sigma);
                    if (miss >= reach) continue;
                    // How much of the object's width the ray passes through, in [0,1]: a graze
                    // explains away a little, a central hit almost everything. No threshold — the
                    // geometry supplies the number.
                    const float depth = 1.f - miss / std::max(reach, 1e-6f);
                    const float claim = std::clamp(depth * a.p_exists, 0.f, 1.f);
                    pi_vis *= std::max(0.02f, 1.f - claim);
                }
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

                // ── [4] THIS CONTOUR'S OWN POSITION IN THE MAP ───────────────────────────────────────
                // The four columns above are GLOBAL to the frame, so a wall simply in the wrong place had
                // nowhere to go but the residual, and every sample along it counted as an independent
                // measurement of a position they all share. Measured: per-vertex median residuals spread
                // 1.74 px (u) and 0.81 px (v) against a fixed-pose repeatability of 0.019 and 0.093 px.
                //
                // ★ THE SENSITIVE DIRECTION IS NOT ASSUMED, IT FALLS OUT. A displacement of this contour
                //   in the room PLANE is two unknowns, but only one combination moves the image along
                //   this sample's normal, and the norm over the plane picks it automatically:
                //       h4 = sigma_wall * || [ contract(e_x_cam), contract(e_y_cam) ] ||
                //   For a floor junction, sliding the wall ALONG itself moves the line not at all, so
                //   that direction contributes ~0 and the norm selects the perpendicular by itself — no
                //   per-class branch, and it stays correct for a vertical corner where both directions
                //   matter.
                // ★ It is the WORST-CASE direction, so the marginalisation is mildly conservative: a
                //   real displacement along a specific direction removes less information than this
                //   allows. Conservative is the correct side here — the failure being fixed is a term
                //   claiming more independent evidence than it has.
                {
                    const Eigen::Matrix3f R_cam_room = cam_R_robot * dRm;
                    const float cxr = contract(R_cam_room.col(0));
                    const float cyr = contract(R_cam_room.col(1));
                    hcol(4) = cfg_.wall_position_sigma * std::hypot(cxr, cyr);
                }
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
                if (not rc::img::dir_derivative(g, W, H, u, v, n_hat, gd, wrap_u)) { prof.push_back(0.f); continue; }
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

            // ★ SAME WEIGHT THE FACTOR AND THE MONITORS USE: gamma / (sigma_px^2 + |h|^2). The
            //   mixture responsibility is the only outlier handling anywhere in this term, and a
            //   line fit that ignored it would let one mismatched sample drag the whole offset —
            //   which is exactly how the pooled mount fit once ran r_rms 11.9 px against a stated
            //   sigma of 0.05. `best_s` IS the measurement: uv_meas = uv_pred + best_s * n_hat.
            const float s2f = sigma_px * sigma_px + hcol.squaredNorm();
            if (s2f > 0.f)
            {
                const float gam = rc::img::responsibility(-best_s, s2f, pi_vis, L);
                if (gam > 1e-6f)
                {
                    const double wf = static_cast<double>(gam) / static_cast<double>(s2f);
                    fit.w   += wf;
                    fit.ws  += wf * static_cast<double>(best_s);
                    fit.wh4 += wf * static_cast<double>(hcol(4));
                    fit.wv  += wf * static_cast<double>(pi_vis);
                    fit.wn  += wf * Eigen::Vector2d(n_hat.x(), n_hat.y());
                }
            }

            sig_list.push_back(sigma_px);
            len_list.push_back(L);
        }

        if (not seg.samples.empty())
            obs.segments.push_back(std::move(seg));
        if (fit.w > 0.0) fits.push_back(fit);
    }

    // ── 3. Triple points: floor + wall + wall, i.e. a polygon vertex at z = 0 ─────────────────────
    // See rc::img::TriplePoint for why this feature is worth having and why it costs no new image
    // processing. Construction: the vertical corner line and the floor junction line each translate
    // along their own normal by one measured offset, so their intersection moves by the Delta that
    // satisfies BOTH, which is a 2x2 solve:
    //     n_c . Delta = delta_c        n_f . Delta = delta_f
    // ★ uv_meas is a MEASUREMENT and is stored as one, not as a residual. The line DIRECTIONS come
    //   from the model (hence from the pose) but the crossing point is where the image says the
    //   corner is; a factor may later predict it from any pose. The direction's pose dependence is
    //   second order and is what lets this be extracted once per frame.
    {
        const auto find_fit = [&fits](ContourClass cls, int vertex) -> const SegFit*
        {
            for (const auto& f : fits)
                if (f.cls == cls and f.vertex == vertex) return &f;
            return nullptr;
        };
        // ★ BOTH ENDS OF EACH VERTICAL CORNER. The wall-wall edge runs floor to ceiling, so it can
        //   be intersected with the floor junction OR the wall-ceiling junction, giving two distinct
        //   0-D features per vertex at known heights. The CEILING one is the less occluded of the
        //   two by a wide margin — furniture, people and clutter all sit on the floor, which is
        //   exactly where a floor corner is blocked — and on a panoramic camera it is the better
        //   feature outright, since the floor corner's compensating advantage (readable ZED depth)
        //   does not exist for a 360 model.
        const ContourClass horiz[2] = {ContourClass::FloorWall, ContourClass::WallCeiling};
        for (std::size_t i = 0; i < np; ++i)
        {
            const SegFit* fc = find_fit(ContourClass::WallCorner, static_cast<int>(i));
            if (fc == nullptr) continue;
        for (const ContourClass hcls : horiz)
        {
            // Either horizontal edge meets this vertex: edge i leaves it, edge i-1 arrives at it.
            // Take whichever carries more weight rather than averaging two DIFFERENT lines into one.
            const SegFit* f1 = find_fit(hcls, static_cast<int>(i));
            const SegFit* f2 = find_fit(hcls, static_cast<int>((i + np - 1) % np));
            const SegFit* ff = (f1 and f2) ? (f1->w >= f2->w ? f1 : f2) : (f1 ? f1 : f2);
            if (ff == nullptr) continue;

            // ★ The normals must be CONSISTENT within a segment or a single-offset line is the wrong
            //   model — a mean normal well short of unit length means they disagreed, and the fitted
            //   offset would be an average of displacements along different directions.
            const Eigen::Vector2d nc = fc->wn / fc->w, nf = ff->wn / ff->w;
            if (nc.norm() < 0.9 or nf.norm() < 0.9) continue;
            const Eigen::Vector2d nch = nc.normalized(), nfh = nf.normalized();

            Eigen::Matrix2d A; A.row(0) = nch.transpose(); A.row(1) = nfh.transpose();
            const double det = A.determinant();
            // |det| = |sin(angle between the normals)|. Near zero means the two lines are parallel
            // in the image and their crossing is undefined — which happens when the vertex is seen
            // edge-on. Reported through `cond`, and skipped only when the solve is meaningless.
            if (not std::isfinite(det) or std::abs(det) < 1e-3) continue;

            const Eigen::Vector3f p_room(polygon_[i].x(), polygon_[i].y(),
                                         hcls == ContourClass::WallCeiling ? cfg_.room_height : 0.f);
            const Eigen::Vector3f p_cam = to_camera(p_room, pose, cam_R_robot, cam_t_robot);
            Eigen::Vector2d uvp;
            if (not rc::img::project_with_model(model, p_cam.cast<double>(), uvp)) continue;

            const Eigen::Vector2d d(fc->ws / fc->w, ff->ws / ff->w);
            const Eigen::Matrix2d Ai = A.inverse();
            const Eigen::Vector2d delta = Ai * d;
            // ── Line-offset variance: TWO parts, and the second does not average down ────────────
            // 1/w is the standard error of the weighted MEAN, which shrinks as 1/N. That is the
            // right answer for independent per-sample noise and the WRONG one for the thing that
            // actually dominates: nuisance column [4], this contour's own position in the map, is
            // COMMON MODE across the segment. Sampling a misplaced wall more finely does not locate
            // it any better, so that variance must be added, not averaged.
            //
            // ★ MEASURED, and it is why this was found: with only the 1/w term the factor's post-fit
            //   chi2/dof ran 6.99 — residuals ~2.6x the stated sigma AFTER the pose had absorbed
            //   everything it could, which is precisely the per-corner systematic (1.74 px in u,
            //   0.81 px in v across six vertices) that column [4] exists to represent. The column
            //   was reaching the per-sample weights and stopping there.
            // ★ The magnitude is a prediction, not a fit: sigma_wall 0.015 m at ~4 m and fy 448 is
            //   1.68 px, against the 1.74 px measured. Nothing was tuned to make those agree.
            const double h4c = fc->wh4 / fc->w;      // weighted mean sensitivity over each segment
            const double h4f = ff->wh4 / ff->w;
            Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
            S(0, 0) = 1.0 / fc->w + h4c * h4c;
            S(1, 1) = 1.0 / ff->w + h4f * h4f;

            TriplePoint tp;
            tp.vertex   = static_cast<int>(i);
            tp.from     = hcls;
            tp.p_room   = p_room;
            tp.uv_pred  = uvp.cast<float>();
            // Fold back onto the cyclic column axis, so a corner detected just past the seam is
            // reported at a column that exists. Identity on a pinhole (wrap_u is 0 there).
            Eigen::Vector2d uvm = uvp + delta;
            if (wrap_u > 0)
            {
                uvm.x() = std::fmod(uvm.x(), static_cast<double>(wrap_u));
                if (uvm.x() < 0.0) uvm.x() += static_cast<double>(wrap_u);
            }
            tp.uv_meas  = uvm.cast<float>();
            tp.cov_uv   = (Ai * S * Ai.transpose()).cast<float>();
            // Weighted over BOTH segments: a crossing is only as visible as the pair that
            // formed it, and the two can disagree (a ceiling join in the clear, its floor
            // counterpart behind a wall).
            tp.pi_vis   = static_cast<float>((fc->wv + ff->wv) / std::max(1e-12, fc->w + ff->w));
            tp.n_corner = static_cast<int>(fc->w);
            tp.n_floor  = static_cast<int>(ff->w);
            const double c = std::min(1.0 - 1e-12, std::abs(nch.dot(nfh)));
            tp.cond     = static_cast<float>((1.0 + c) / (1.0 - c));
            st.n_triple++;
            if (tp.pi_vis < 0.5f) st.n_triple_occl++;
            obs.triple_points.push_back(tp);
        }
        }
    }

    st.med_sigma_px  = median_of(sig_list);
    st.med_search_px = median_of(len_list);
    if (stats) *stats = st;
    return obs;
}

}  // namespace rc
