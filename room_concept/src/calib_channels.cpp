#include <genericworker.h>   // FIRST: DSR's signal emitter is not self-contained (see pose_publisher.cpp)

#include "calib_channels.h"

#include "mount_calibrator.h"
#include "room_concept.h"
#include "room_viewer.h"

#include <dsr/api/dsr_api.h>

#include <QDateTime>
#include <QDebug>
#include <QString>

#include <algorithm>
#include <cmath>
#include <locale>

namespace rc
{
void CalibChannels::place_triple_points_in_room(rc::ImageEdgeObs &obs,
                                                 const rc::CameraIngestor &ing,
                                                 const Eigen::Vector3f &pose)
{
    const Eigen::Matrix3f Rcr = obs.cam_R_robot;          // robot -> camera
    const Eigen::Matrix3f Rrc = Rcr.transpose();          // camera -> robot
    const Eigen::Vector3f o_rob = -Rrc * obs.cam_t_robot; // camera centre in robot coords
    const float c = std::cos(pose.z()), sn = std::sin(pose.z());
    const auto to_room = [&](const Eigen::Vector3f &v, bool is_point)
    {
        const Eigen::Vector3f r(c * v.x() - sn * v.y(), sn * v.x() + c * v.y(), v.z());
        return is_point ? Eigen::Vector3f(r.x() + pose.x(), r.y() + pose.y(), r.z()) : r;
    };
    const Eigen::Vector3f o_room = to_room(o_rob, true);

    // ONE map, pixel -> room plane, used both for the point and for its uncertainty. Writing it once
    // is the point: a covariance propagated through a SECOND, hand-written copy of this geometry would
    // be a model of the map rather than the map, and the two would drift apart silently.
    // The plane a corner lies in is its own height: 0 for a floor corner, room_height for a ceiling
    // one. A ray nearly parallel to that plane meets it nowhere useful — the intersection runs away to
    // infinity — so it is skipped rather than clamped.
    const auto intersect = [&](const rc::TriplePoint &tp, double u, double v,
                               Eigen::Vector3f &out) -> bool
    {
        const Eigen::Vector3f d_cam = ing.ray_from_pixel(u, v);
        if (not (d_cam.norm() > 1e-6f)) return false;
        const Eigen::Vector3f d_room = to_room(Rrc * d_cam, false);
        if (std::abs(d_room.z()) < 1e-3f) return false;
        const float tt = (tp.p_room.z() - o_room.z()) / d_room.z();
        if (not (tt > 0.f) or not std::isfinite(tt)) return false;   // behind the camera
        out = o_room + tt * d_room;
        return out.allFinite();
    };

    for (auto &t : obs.triple_points)
    {
        Eigen::Vector3f p;
        if (not intersect(t, t.uv_meas.x(), t.uv_meas.y(), p)) continue;
        t.p_room_meas = p;
        // Range along the ray, so a consumer has one whether or not a depth stream exists. NOT the
        // same quantity as depth_raw, which stays what the sensor said.
        t.range_m = (p - o_room).norm();

        // ── cov_uv (px^2) carried into the room plane (m^2) ─────────────────────────────────────
        // A CENTRED difference of the intersection itself, one pixel apart, gives d(x, y)/d(u, v)
        // directly in metres per pixel. No angular rule is assumed and no focal length is quoted:
        // whatever the camera model does — pinhole, equirectangular, cylindrical — and whatever the
        // grazing geometry does to the v axis (the d^2/h amplification a floor corner suffers and a
        // bearing does not) is already inside the map being differenced. Half a pixel each side stays
        // within the frame for any uv the extractor could have produced, and the map is smooth there.
        // If either probe fails the corner simply keeps cov_room zero, and the consumer must say so
        // rather than draw the width a zero covariance would imply.
        Eigen::Vector3f p_up, p_um, p_vp, p_vm;
        if (intersect(t, t.uv_meas.x() + 0.5, t.uv_meas.y(), p_up) and
            intersect(t, t.uv_meas.x() - 0.5, t.uv_meas.y(), p_um) and
            intersect(t, t.uv_meas.x(), t.uv_meas.y() + 0.5, p_vp) and
            intersect(t, t.uv_meas.x(), t.uv_meas.y() - 0.5, p_vm))
        {
            Eigen::Matrix2f J;                       // d(x, y) / d(u, v), metres per pixel
            J.col(0) = (p_up - p_um).head<2>();
            J.col(1) = (p_vp - p_vm).head<2>();
            const Eigen::Matrix2f C = J * t.cov_uv * J.transpose();
            if (C.allFinite()) t.cov_room = C;
        }
    }
}

void CalibChannels::pump_calib_channels()
{
    const auto res = room_concept_.get_last_result();
    if (not res.has_value() or room_polygon_.size() < 3) return;

    for (auto &chp : calib_channels_)
    {
        auto &ch = *chp;
        if (not ch.bound)
        {
            if (not ch.ingestor->bind_camera(params.LIDAR_ROBOT_FRAME)) continue;
            ch.bound = true;
            ch.source->set_room_polygon(room_polygon_);
            auto ic = ch.source->config();
            ic.room_height = params.room_height;
            ch.source->set_config(ic);
            ch.ingestor->start();
            qInfo() << "[camcal] channel" << QString::fromStdString(ch.name) << "bound";
        }
        if (not ch.loaded)
        {
            ch.loaded = true;
            ch.calib.set_camera(ch.name, params.LIDAR_ROBOT_FRAME);
            // Same model on every camera, or the Calib window would show two mounts judged by two
            // different notions of uncertainty side by side.
            ch.calib.set_vertex_offset_sigma_px(params.IMAGE_EDGE_MOUNT_VERTEX_OFFSET_SIGMA_PX);
            const std::string path = ch.calib.path();
            const std::size_t k_aux = ch.calib.load(path);
            mount_.reconcile_mount_nominal(ch.calib, *ch.ingestor, ch.name);
            mount_.push_mount_correction(*ch.ingestor, ch.calib.applied(), ch.name, "resumed");
            if (k_aux > 0)
                qInfo().nospace() << "[camcal] " << QString::fromStdString(ch.name)
                                  << " resumed from " << QString::fromStdString(path)
                                  << " (" << k_aux << " pairs)";
        }

        // ── This channel's own column in the Calib window ────────────────────────────────────────
        // ★ THE SECOND ESTIMATOR HAD NO WAY OUT. Only the driving camera's pool ever reached the
        //   viewer (see mount_pair_update), so this channel's evidence accumulated in its own file
        //   with nothing on screen able to show it, while the window's single camera block was
        //   captioned with whichever camera reported last. One column per camera now, each fed by
        //   its own estimator.
        // ★ BEFORE take_latest(), not after the pairing loop. Placed after it, a channel that is
        //   bound but currently sees no corner — including one that has just RESUMED a full evidence
        //   file from disk — would never push, and its column would read "no evidence" while the
        //   evidence sat on disk. The column must describe the estimator, not this tick's luck.
        //   Wall clock, 5 s, matching the driving camera's window so both columns are the same age.
        if (const auto now_ms = QDateTime::currentMSecsSinceEpoch();
            viewer() and now_ms - ch.viz_ms >= 5000)
        {
            ch.viz_ms = now_ms;
            if (const auto sol = ch.calib.solve(); sol.ok)
            {
                Eigen::Matrix<float, rc::camcal::P_COUNT, 1> pv, sv;
                // Same unit conversion as the driving camera: the estimator works in units of the
                // PRIOR SIGMA, so value AND sigma are both scaled back to physical here — scaling
                // only the value would put a dimensionless uncertainty beside a physical number.
                const double psig[4] = {params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                                        params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                                        params.IMAGE_EDGE_MOUNT_YAW_SIGMA, 1.0};
                for (int i = 0; i < rc::camcal::P_COUNT; ++i)
                {
                    pv(i) = static_cast<float>(sol.p(i)     * psig[i]);
                    sv(i) = static_cast<float>(sol.sigma(i) * psig[i]);
                }
                viewer()->set_camera_calibration(pv, sv, sol.informed,
                                                static_cast<float>(sol.cond),
                                                ch.calib.pairs(), ch.name);
            }
        }

        // ── The ceiling the contour is projected at, REFRESHED ──────────────────────────────────────
    // ★ It used to be set only inside the bind block above, so a ceiling measured one second after
    //   binding never reached the projection — and with the wire broken it never arrived at all. The
    //   room polygon beside it was already refreshed every tick for exactly this reason.
    // ⚠ A change here moves every wall-ceiling corner, which is what the mount's HEIGHT parameter
    //   reads, so it is logged when it moves rather than changing the geometry quietly.
    if (image_edge_bound_)
    {
        const float mz = room_concept_.measured_ceiling();
        const float want = mz > 1.5f ? mz : params.room_height;
        if (auto ic = image_edge_source_->config(); std::abs(ic.room_height - want) > 1e-4f)
        {
            qInfo().noquote() << QString::asprintf(
                "[imgedge] wall-ceiling contour re-projected: room_height %.4f -> %.4f m (%s;"
                " the scenario states %.4f)", ic.room_height, want,
                mz > 1.5f ? "MEASURED by the LiDAR" : "stated in the scenario", params.room_height);
            if (mz > 1.5f)
                qInfo().noquote() << QString::asprintf(
                    "         its uncertainty is +/- %.3f m — the ceiling corners' z now carries that,"
                    " and the mount's height can no longer silently absorb it",
                    room_concept_.measured_ceiling_sigma());
            ic.room_height = want;
            image_edge_source_->set_config(ic);
        }
    }

    rc::GrayFrame frame;
        if (not ch.ingestor->take_latest(frame)) continue;

        const Eigen::Affine2f &rp = res->robot_pose;
        const Eigen::Vector3f pose(rp.translation().x(), rp.translation().y(),
                                   std::atan2(rp.linear()(1, 0), rp.linear()(0, 0)));
        const std::int64_t dt_ms = static_cast<std::int64_t>(frame.stamp) - res->timestamp_ms;
        auto obs = ch.source->extract(frame, ch.ingestor->model(), ch.ingestor->cam_R_robot(),
                                      ch.ingestor->cam_t_robot(), pose, res->covariance,
                                      Eigen::Vector3f::Zero(), dt_ms, nullptr);
        place_triple_points_in_room(obs, *ch.ingestor, pose);
        // Each camera's corners go to ITS OWN window: the ZED popup shows the ZED's corners even
        // though the Ricoh is the one driving, which is the comparison worth being able to see.
        if (viewer()) viewer()->set_triple_points(obs.triple_points, ch.name);
        if (obs.triple_points.empty() or res->corner_matches.empty()) continue;

        for (const auto &tp : obs.triple_points)
        {
            const auto it = std::ranges::find_if(res->corner_matches,
                [&](const auto &m) { return m.model_index == tp.vertex; });
            if (it == res->corner_matches.end() or it->suppressed) continue;
            const auto pr = rc::mount::make_pair(tp, *it, ch.ingestor->model(),
                                                 ch.ingestor->cam_R_robot(),
                                                 ch.ingestor->cam_t_robot(),
                                                 params.IMAGE_EDGE_MOUNT_PITCH_SIGMA,
                                                 params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA,
                                                 params.IMAGE_EDGE_MOUNT_YAW_SIGMA);
            if (not pr.ok) continue;
            ch.calib.add(pr);
            ++ch.pairs;
            // The SAME row the driving camera writes. Without it this channel produced evidence
            // nobody could re-derive: arm 7 needs both cameras' mounts re-solved under one
            // injection, and the closure recomputed from the two, all from a single drive.
            if (not ch.csv.is_open()) mount_.open_pair_log(ch.csv, ch.name, *ch.ingestor);
            rc::MountCalibrator::write_pair_row(ch.csv, ch.name, static_cast<std::int64_t>(frame.stamp), pr,
                           tp.from == rc::ContourClass::WallCeiling,
                           it->angle_deg, it->assoc_chi2_val, it->n_rivals, it->runnerup_chi2,
                           ch.ingestor->mount_correction());
            // Per ROW, not per frame: see the note on the driving camera's call — the local
            // pixel-to-angle scale varies across a pinhole's field, and this channel IS the pinhole.
            const Eigen::Vector2f ppr = rc::img::px_per_rad_at(obs.cam, pr.uv_image);
            if (ppr.x() > 0.f and ppr.y() > 0.f)
                mount_.loop_closure_observe(ch.name, tp.vertex,
                                     tp.from == rc::ContourClass::WallCeiling,
                                     static_cast<double>(pr.r.x() / ppr.x()),
                                     static_cast<double>(pr.r.y() / ppr.y()),
                                     static_cast<std::int64_t>(frame.stamp));
        }
        if (ch.pairs % 2000 < 12 and ch.pairs > 0)
        {
            if (const auto sol_apply = ch.calib.solve(); sol_apply.ok)
                mount_.apply_mount_solve(ch.calib, *ch.ingestor, sol_apply, ch.name);
            ch.calib.save(ch.calib.path());
            if (ch.csv.is_open()) ch.csv.flush();   // same cadence as the evidence beside it
            if (const auto sol = ch.calib.solve(); sol.ok)
                qInfo().nospace().noquote()
                    << "[camcal] " << QString::fromStdString(ch.name) << " " << ch.pairs
                    << " pairs | yaw "
                    << QString::number(-sol.p(2) * params.IMAGE_EDGE_MOUNT_YAW_SIGMA * 180.0 / M_PI,
                                       'f', 4)
                    << " deg | cond " << QString::number(sol.cond, 'f', 1);
        }
    }
}

void CalibChannels::pump_image_edges()
{
    if (not camera_ingestor_ or not image_edge_source_) return;

    // Bind lazily and keep retrying: the camera node, its media descriptor and the RT chain all come
    // up asynchronously, and a miss here is normal for the first few seconds.
    // ── IN ESTIMATE MODE THE ROOM POLYGON ARRIVES LATE, AND FROM THE ESTIMATOR ───────────────────
    // room_polygon_ is loaded from the SVG in Given mode and CLEARED by configure_room_estimate(), so
    // in Estimate mode it stayed empty for the life of the run — the bind below returns on
    // `size() < 3` for ever and the extractor is never given anything to project. That is the whole
    // reason no ricoh corners appear, and it reads as "bound, nothing extracted" in the [imgedge]
    // line because the camera binds fine; it is the geometry that is missing.
    // LOCALIZING is when a layout exists, and RoomConcept::polygon_vertices() is the frozen one it
    // handed to the corner detector at the same moment. Taken once: after this the layout does not
    // change, by definition of the state.
    if (room_polygon_.size() < 3 and room_concept_.localizing()
        and room_concept_.polygon_vertices().size() >= 3)
    {
        room_polygon_ = room_concept_.polygon_vertices();
        room_polygon_offset_ = Eigen::Vector2f::Zero();   // the frozen layout IS the room frame
        image_edge_source_->set_room_polygon(room_polygon_);
        qInfo().noquote() << QString("[imgedge] room polygon taken from the frozen layout: %1 vertices — "
                                     "RGB corner extraction can start").arg(room_polygon_.size());
    }

    if (not image_edge_bound_)
    {
        // The room polygon and the robot-frame name are both resolved asynchronously during
        // start-up, so this retries until all of it is available rather than binding once and
        // failing permanently.
        if (room_polygon_.size() < 3) return;
        if (not camera_ingestor_->bind_camera(params.LIDAR_ROBOT_FRAME)) return;
        image_edge_bound_ = true;
        image_edge_source_->set_room_polygon(room_polygon_);
        // The objects the room currently believes in, so the extraction can explain away wall
        // samples they stand in front of. Refreshed each tick below, not only at bind: furniture
        // appears, moves and is forgotten while the agent runs.
        image_edge_source_->set_object_anchors(room_concept_.object_anchors());
        // room_height is read from the graph after construction, so refresh it here too — and
        // PREFER THE MEASURED CEILING when the LiDAR has one. The startup geometry check already
        // locates the ceiling plane with a likelihood test (annulus vs wall-top) and found 3.01 m
        // against a stated 3.00; until 2026-09-03 that number only capped the wall band and was then
        // discarded, while the wall-ceiling contour this module projects used the hand-typed
        // constant. A stated ceiling that is a few cm wrong is a pure SCALE error on every range the
        // contour implies, and a Manhattan estimator cannot see it.
        auto ic = image_edge_source_->config();
        ic.room_height = params.room_height;
        const float measured_ceiling = room_concept_.measured_ceiling();
        if (measured_ceiling > 1.5f)
        {
            ic.room_height = measured_ceiling;
            if (std::abs(measured_ceiling - params.room_height) > 0.05f)
                qWarning() << "[imgedge] the LiDAR measures the ceiling at" << measured_ceiling
                           << "m but the scenario states" << params.room_height
                           << "m; using the measurement. A stated ceiling that is wrong scales every"
                           << "range the wall-ceiling contour implies.";
        }
        image_edge_source_->set_config(ic);
        qInfo() << "[imgedge] bound to" << QString::fromStdString(params.IMAGE_EDGE_CAMERA)
                << "in frame" << QString::fromStdString(params.LIDAR_ROBOT_FRAME)
                << "| polygon" << room_polygon_.size() << "pts, room_height" << ic.room_height
                << (measured_ceiling > 1.5f ? "(measured by the LiDAR)" : "(stated in the scenario)");
    }

    rc::GrayFrame frame;
    image_edge_source_->set_object_anchors(room_concept_.object_anchors());
    if (not camera_ingestor_->take_latest(frame)) return;    // no fresh image this tick

    const auto res = room_concept_.get_last_result();
    if (not res.has_value()) return;

    // The pose at which to PREDICT and SEARCH. This is the localizer's own current estimate, which is
    // correct and is NOT the circularity: the search only decides WHICH image edge each model contour
    // is matched to. The residual the factor then minimises is a function of the state VARIABLE, and
    // uv_meas is held fixed while it does — which is also what LM's accept/reject test requires.
    const Eigen::Affine2f& rp = res->robot_pose;
    const Eigen::Vector3f pose(rp.translation().x(), rp.translation().y(),
                               std::atan2(rp.linear()(1, 0), rp.linear()(0, 0)));

    const std::int64_t dt_ms = static_cast<std::int64_t>(frame.stamp) - res->timestamp_ms;

    // Body twist, for the image/LiDAR dt nuisance column. A VARIANCE, never a correction: dead-
    // reckoning the pose forward here would reintroduce exactly the graph-pose dependency we removed.
    // Measured by differencing two published localizer results, so it needs no graph read and no
    // command channel (the commanded velocity is not what the robot did).
    Eigen::Vector3f twist = Eigen::Vector3f::Zero();
    if (image_edge_prev_ts_ > 0 and res->timestamp_ms > image_edge_prev_ts_)
    {
        const float dt_s = 1e-3f * static_cast<float>(res->timestamp_ms - image_edge_prev_ts_);
        if (dt_s > 1e-3f and dt_s < 1.0f)
        {
            const Eigen::Vector3f d = pose - image_edge_prev_pose_;
            const float dth = std::atan2(std::sin(d.z()), std::cos(d.z()));
            // Rotate the room-frame displacement into the robot frame: the nuisance is expressed
            // there, because that is where the lever arm to the camera lives.
            const float c = std::cos(pose.z()), sn = std::sin(pose.z());
            twist = Eigen::Vector3f(( c * d.x() + sn * d.y()) / dt_s,
                                    (-sn * d.x() +  c * d.y()) / dt_s,
                                    dth / dt_s);
        }
    }
    image_edge_prev_pose_ = pose;
    image_edge_prev_ts_   = res->timestamp_ms;

    rc::ImageEdgeSource::Stats st;
    auto obs = image_edge_source_->extract(frame, camera_ingestor_->model(),
                                           camera_ingestor_->cam_R_robot(),
                                           camera_ingestor_->cam_t_robot(),
                                           pose, res->covariance, twist, dt_ms, &st);
    // The correction that was inside the extrinsic used above. Recorded on the observation so a
    // shadow solve can remove it and re-create the nominal mount for THESE measurements — factor B.
    obs.mount_correction = camera_ingestor_->mount_correction();
    // Provenance travels WITH the evidence from here on: every downstream consumer (the pair log,
    // the triple log, the viewer overlay) then reports the camera this observation actually came
    // from, not the one the config names at the moment it is asked.
    obs.camera = params.IMAGE_EDGE_CAMERA;

    // ── Range for the triple points, from the ZED depth plane ────────────────────────────────────
    // Zero-copy: the pixel list is known now (the corners were detected from the RGB frame above),
    // so probe_depth reads exactly these pixels inside the loaned SHM view and copies no frame.
    // The depth frame is whatever is newest at this instant; its stamp is recorded beside the RGB
    // stamp rather than assumed equal, because they are different streams from different threads.
    if (not obs.triple_points.empty())
    {
        std::vector<Eigen::Vector2f> uv;
        uv.reserve(obs.triple_points.size());
        for (const auto& t : obs.triple_points) uv.push_back(t.uv_meas);
        std::vector<float> dm;
        std::int64_t dstamp = 0;
        if (camera_ingestor_->probe_depth(uv, 2, dm, dstamp) > 0)
        {
            obs.depth_stamp_ms = dstamp;
            for (std::size_t k = 0; k < obs.triple_points.size(); ++k)
            {
                auto& t = obs.triple_points[k];
                t.depth_raw = dm[k];
                Eigen::Vector3d xyz;
                if (not rc::img::xyz_from_pixel_depth(camera_ingestor_->model(),
                                                      t.uv_meas.x(), t.uv_meas.y(), dm[k], xyz))
                    continue;
                t.p_cam_meas = xyz.cast<float>();
                t.range_m    = static_cast<float>(xyz.norm());
                // camera -> robot -> room. cam_R_robot maps robot into camera, so its transpose
                // brings the point back; then the pose rotation, which is R(+theta) — the inverse of
                // the R(-theta) the projection path applies.
                const Eigen::Vector3f p_rb = camera_ingestor_->cam_R_robot().transpose()
                                           * (t.p_cam_meas - camera_ingestor_->cam_t_robot());
                const float cs = std::cos(pose.z()), sn = std::sin(pose.z());
                t.p_room_meas = Eigen::Vector3f(pose.x() + cs * p_rb.x() - sn * p_rb.y(),
                                                pose.y() + sn * p_rb.x() + cs * p_rb.y(),
                                                p_rb.z());
                // ★ The sigma is a PLACEHOLDER and is marked as one. A depth sigma is a property of
                //   the sensor at that range and this camera's has not been measured here; the
                //   LiDAR-anchored depth-correction work in retina is where that number should come
                //   from. Writing a plausible constant and treating it as measured is how a term
                //   acquires unearned authority, so nothing may consume this until it is real.
                t.range_sigma = -1.f;
            }
        }
    }
    // Place the corners in the room BEFORE anything consumes them: the 2-D canvas needs it, and it
    // works for a camera with no depth at all, which the panorama is.
    place_triple_points_in_room(obs, *camera_ingestor_, pose);
    mount_.mount_pair_update(obs, res->corner_matches, static_cast<std::int64_t>(frame.stamp));
    // The overlay draws these on whichever window IS this camera; the name travels with them so a
    // ricoh corner can never be painted onto a zed frame at a plausible-looking wrong position.
    if (viewer()) viewer()->set_triple_points(obs.triple_points, params.IMAGE_EDGE_CAMERA);
    room_concept_.set_image_edges(std::move(obs));

    if (const auto [polls, hits] = camera_ingestor_->depth_stats();
        polls > 0 and polls % 100 == 0)
        qInfo().nospace().noquote()
            << "[depth] " << hits << "/" << polls << " polls delivered a frame ("
            << QString::number(100.0 * hits / polls, 'f', 1) << "%)"
            << (hits == 0 ? "  <- subscriber exists and NOTHING arrives: check the producer is "
                            "publishing depth and that the frame is under MAX_IMAGE_BYTES (3.69 MB), "
                            "which the plane drops SILENTLY" : "");

    // ~1/s liveness line. n_searched == 0 with n_projected > 0 means the contours project but carry
    // no gradient — a real answer (blank walls), not a plumbing failure, and the two are worth being
    // able to tell apart from the log alone.
    const auto now = QDateTime::currentMSecsSinceEpoch();
    if (now - last_image_edge_log_ms_ > 1000)
    {
        last_image_edge_log_ms_ = now;
        qInfo() << "[imgedge] contours" << st.n_contours << "projected" << st.n_projected
                << "visible" << st.n_visible << "searched" << st.n_searched
                << "occluded" << st.n_occluded
                << "| corners" << st.n_triple << "(" << st.n_triple_occl << "behind a wall)"
                << "| med sigma" << st.med_sigma_px << "px"
                << "med L" << st.med_search_px << "px sigma_i" << st.sigma_i
                << "| dt_img_lidar" << dt_ms << "ms";
    }
}

    void CalibChannels::configure()
    {
    // RGB edge alignment. Constructed ONLY when enabled: with ImageEdge.enable = false there is no
    // subscriber, no thread and no extraction, so the feature is exactly free when off.
    if (params.IMAGE_EDGE_ENABLE)
    {
        camera_ingestor_ = std::make_unique<rc::CameraIngestor>(G, params.IMAGE_EDGE_CAMERA);
        // Set BEFORE the first bind_camera(): the correction is applied where the extrinsic is read.
        camera_ingestor_->set_mount_yaw_correction(params.IMAGE_EDGE_MOUNT_YAW_CORR);
        // Throttle the grey CONVERSION, never the drain — see CameraIngestor and the config note.
        camera_ingestor_->set_min_convert_interval_ms(params.IMAGE_EDGE_MIN_CONVERT_MS);
        mount_.set_driving_ingestor(camera_ingestor_.get());

        image_edge_source_ = std::make_unique<rc::ImageEdgeSource>();
        rc::ImageEdgeSource::Config ic;
        ic.use_wall_corners    = params.IMAGE_EDGE_USE_WALL_CORNERS;
        ic.use_floor_junction  = params.IMAGE_EDGE_USE_FLOOR_JUNCTION;
        ic.use_wall_ceiling    = params.IMAGE_EDGE_USE_WALL_CEILING;
        ic.sample_spacing_m    = params.IMAGE_EDGE_SAMPLE_SPACING_M;
        ic.search_sigmas       = params.IMAGE_EDGE_SEARCH_SIGMAS;
        ic.max_search_px       = params.IMAGE_EDGE_MAX_SEARCH_PX;
        ic.mount_pitch_sigma   = params.IMAGE_EDGE_MOUNT_PITCH_SIGMA;
        ic.mount_height_sigma  = params.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA;
        ic.mount_yaw_sigma     = params.IMAGE_EDGE_MOUNT_YAW_SIGMA;
        ic.wall_position_sigma = params.IMAGE_EDGE_WALL_POS_SIGMA;
        ic.room_height         = params.room_height;
        image_edge_source_->set_config(ic);

        // ── Extra CALIBRATION channels ─────────────────────────────────────────────────────────
        // ★ AFTER image_edge_source_ is built and configured. This block sat BEFORE it and
        //   called image_edge_source_->config() on a null unique_ptr — a segfault on the
        //   very first start, before the graph was even up. Second time today that adding
        //   setup beside related code put it ahead of the thing it depends on; the
        //   dependency is on CONSTRUCTION ORDER and nothing in the surrounding code shows it.──
        // The driving camera is skipped: it already has an ingestor above, and running it twice
        // would feed its own evidence file from two extractions of the same frames — the exact
        // double-count the triple-point/segment choice exists to avoid.
        for (const auto &cam : params.CALIB_CAMERAS)
        {
            if (cam == params.IMAGE_EDGE_CAMERA) continue;
            auto ch = std::make_unique<CalibChannel>();
            ch->name = cam;
            ch->ingestor = std::make_unique<rc::CameraIngestor>(G, cam);
            // ★ The CALIB interval, not the driving one. A calibration channel's frames never
            //   reach the loss — they feed an estimator that accumulates H and b over minutes — so
            //   skipping frames here costs coverage, not evidence. Sharing one number with the
            //   driving camera would pay for this saving with the pose factor's rate.
            ch->ingestor->set_min_convert_interval_ms(params.IMAGE_EDGE_CALIB_MIN_CONVERT_MS);
            // ★ The yaw correction is per CAMERA and is NOT shared. It was measured for the zed; a
            //   second camera has its own mount and applying one camera's correction to another
            //   would be a fabricated extrinsic.
            ch->source = std::make_unique<rc::ImageEdgeSource>();
            ch->source->set_config(ic);   // the same config, from the local `ic` — see below
            calib_channels_.push_back(std::move(ch));
            qInfo() << "[camcal] calibration channel for" << QString::fromStdString(cam)
                    << "(does not drive the pose)";
        }
    }
    }

    void CalibChannels::start()
    {
        // ── ONLY THE DRIVING CAMERA STARTS HERE, AS IT ALWAYS DID ───────────────────────────────
        // The extraction briefly started every calibration channel from this point too. That was the
        // single run-time difference the review found, and it is reverted: each auxiliary channel is
        // started lazily inside pump_calib_channels() once its own bind_camera() succeeds. The
        // difference is not academic — a camera that is CONFIGURED BUT NEVER BINDABLE (wrong node
        // name, a sensor this robot does not have) would otherwise cost a live ingest thread doing
        // 1 Hz descriptor discovery and frame drain for the entire run, and the fleet's own note on
        // that failure is that a media plane for a sensor the robot lacks bridges for ever.
        // Starting a reader before there is anything to read is not free and buys nothing.
        if (camera_ingestor_) camera_ingestor_->start();
    }

    void CalibChannels::stop()
    {
        // ⚠ REVIEW 2026-09-13, TWO NOTES.
        //   (a) the baseline never destroyed the auxiliary ingestors at shutdown; this does. Benign,
        //       but it is not verbatim either.
        //   (b) MountCalibrator::driving_ is a RAW pointer into camera_ingestor_, handed over in
        //       configure(), and is NOT cleared here — so after this it dangles, and the guard in
        //       mount_pair_update() (`not driving_`) can no longer detect it, where at baseline that
        //       same guard read the owning unique_ptr and was a true guard. Unreachable today: the
        //       only caller is pump_image_edges(), which returns on its own null camera_ingestor_,
        //       and request_shutdown() runs on the GUI thread and then _Exit()s, so no compute tick
        //       can interleave — safe by a property of SHUTDOWN, not by construction. Closed below.
        // The calibrator holds a RAW pointer to the driving ingestor (handed over in configure()), so
        // it is told BEFORE the object goes. Its own `not driving_` guard is then a true guard again,
        // as it was at the baseline where it read the owning unique_ptr.
        mount_.set_driving_ingestor(nullptr);
        camera_ingestor_.reset();          // drop the RGB readers before the graph goes
        for (auto& chp : calib_channels_) chp->ingestor.reset();
    }

    void CalibChannels::set_room_polygon(std::vector<Eigen::Vector2f> poly, const Eigen::Vector2f& offset)
    {
        // ⚠ REVIEW 2026-09-13: this pushes the polygon into image_edge_source_ below, which the
        //   baseline did only later, on bind, inside pump_image_edges(). Same VALUE (the bind path
        //   still re-sets it); only the timing is earlier. Recorded because it is behaviour, not
        //   mechanics.
        room_polygon_ = std::move(poly);
        room_polygon_offset_ = offset;
        if (image_edge_source_ and room_polygon_.size() >= 3)
            image_edge_source_->set_room_polygon(room_polygon_);
    }

    std::string CalibChannels::convert_stats_line() const
    {
        std::string out;
        const auto add = [&out](const std::string& name, std::pair<long, long> st)
        {
            out += " " + name + " " + std::to_string(st.second) + "/" + std::to_string(st.first);
        };
        if (camera_ingestor_) add(params.IMAGE_EDGE_CAMERA, camera_ingestor_->convert_stats());
        for (const auto& chp : calib_channels_)
            if (chp->ingestor) add(chp->name, chp->ingestor->convert_stats());
        return out;
    }

}   // namespace rc
