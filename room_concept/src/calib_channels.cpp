#include <genericworker.h>   // FIRST: DSR's signal emitter is not self-contained (see pose_publisher.cpp)

#include "calib_channels.h"
#include "image_edge_ops.h"   // rc::img::xyz_from_pixel_depth
#include "status_reporter.h"

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
    // ── THROTTLE ONCE LOCALIZING (2026-10-10) ────────────────────────────────────────────────────────────────
    // perf, parked after promotion: the channel cost ~15-20% of a core at the camera's full rate, while the localiser
    // early-exits on ~100% of frames — so the corners feed almost only the MOUNT estimate, which moves slowly. Fewer
    // frames are converted (the ingest thread's own throttle) and so fewer are extracted here.
    // ★COMPUTE BUDGET, not a model constant: ImageEdge.localizingConvertIntervalMs (0 = never throttle).
    if (not localizing_throttled_ and room_concept_.localizing() and params.IMAGE_EDGE_LOCALIZING_CONVERT_MS > 0)
    {
        camera_ingestor_->set_min_convert_interval_ms(params.IMAGE_EDGE_LOCALIZING_CONVERT_MS);
        localizing_throttled_ = true;
        qInfo().noquote() << QString("[imgedge] LOCALIZING: camera conversion throttled to one frame per %1 ms")
                                 .arg(params.IMAGE_EDGE_LOCALIZING_CONVERT_MS);
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

    // ── The extrinsic and the ceiling the contour is projected at, REFRESHED every tick ──────────────────────────────────────
    // ★ It used to be set only inside the bind block above, so a ceiling measured one second after
    //   binding never reached the projection — and with the wire broken it never arrived at all. The
    //   room polygon beside it was already refreshed every tick for exactly this reason.
    // ⚠ A change here moves every wall-ceiling corner, which is what the mount's HEIGHT parameter
    //   reads, so it is logged when it moves rather than changing the geometry quietly.
    // (Until 2026-10-29 this block sat inside the auxiliary calibration-channel loop and so ran only while a
    //  zed calibration channel existed; it belongs to the driving camera.)
    if (image_edge_bound_)
    {
        camera_ingestor_->refresh_graph_extrinsic();   // a committed body->camera edge reaches the image-edge term
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
        // Throttle the grey CONVERSION, never the drain — see CameraIngestor and the config note.
        camera_ingestor_->set_min_convert_interval_ms(params.IMAGE_EDGE_MIN_CONVERT_MS);

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
    }
    }

    void CalibChannels::start()
    {
        if (camera_ingestor_) camera_ingestor_->start();
    }

    void CalibChannels::stop()
    {
        camera_ingestor_.reset();          // drop the RGB readers before the graph goes
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

    std::vector<CalibChannels::StreamStat> CalibChannels::stream_stats() const
    {
        std::vector<StreamStat> out;
        const auto add = [&out](const std::string& name, const rc::CameraIngestor& ing)
        {
            out.push_back({name, static_cast<long long>(ing.frames_ingested()),
                           static_cast<long long>(ing.ms_since_last_frame())});
        };
        if (camera_ingestor_) add(params.IMAGE_EDGE_CAMERA, *camera_ingestor_);
        return out;
    }

    std::string CalibChannels::convert_stats_line() const
    {
        std::string out;
        const auto add = [&out](const std::string& name, std::pair<long, long> st)
        {
            out += " " + name + " " + std::to_string(st.second) + "/" + std::to_string(st.first);
        };
        if (camera_ingestor_) add(params.IMAGE_EDGE_CAMERA, camera_ingestor_->convert_stats());
        return out;
    }

}   // namespace rc
