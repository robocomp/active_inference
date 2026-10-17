// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Wall-SLAM channel, observe side, plus the open-door point weighting it shares with the solver.

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

    std::vector<Eigen::Vector3f> RoomConcept::filter_through_door(
        const std::vector<Eigen::Vector3f>& points_robot, int* removed) const
    {
        if (removed != nullptr) *removed = 0;
        if (doors_robot_.empty() or points_robot.empty()) return points_robot;
        std::vector<Eigen::Vector3f> kept;
        kept.reserve(points_robot.size());
        const Eigen::Vector2f origin(0.f, 0.f);
        for (const auto& p : points_robot)
        {
            // Any crossing at all drops the point from the DRAWING. The SDF keeps the continuous
            // weight — a half-open door discounts by half there — but a canvas cannot draw half a
            // point, and "is it removing them?" is the question this view exists to answer.
            // ½ = the mixture's own even split: drawn when the return is more likely the room's than the next one's.
            if (DoorApertures::weight(doors_robot_, origin, p.head<2>(), params.rfe_obs_sigma) >= 0.5f) kept.push_back(p);
            else if (removed != nullptr) ++(*removed);
        }
        return kept;
    }

    Eigen::VectorXf RoomConcept::door_point_weights(const ScanPoints& points_robot) const
    {
        if (doors_robot_.empty() or points_robot.empty())
            return {};
        Eigen::VectorXf w(static_cast<Eigen::Index>(points_robot.size()));
        const Eigen::Vector2f origin(0.f, 0.f);
        bool any = false;
        for (std::size_t i = 0; i < points_robot.size(); ++i)
        {
            const auto& p = points_robot[i];
            w(static_cast<Eigen::Index>(i)) = DoorApertures::weight(doors_robot_, origin, Eigen::Vector2f(p.x(), p.y()),
                                                                    params.rfe_obs_sigma);
            if (w(static_cast<Eigen::Index>(i)) < 1.f) any = true;
        }
        return any ? w : Eigen::VectorXf{};   // nothing crossed an open door: report the plain median
    }

    Eigen::Matrix3f RoomConcept::slot_pose_precision(std::size_t i) const
    {
        const auto& W = window_mgr_.window;
        if (i == 0)
        {
            const auto& bp = window_mgr_.boundary_prior;
            if (bp.valid and W.size() > 1 and bp.precision.allFinite())
                return bp.precision;
            Eigen::Matrix3f g = Eigen::Matrix3f::Zero();
            g(0, 0) = g(1, 1) = 1.f / (params.wall_gauge_sigma_xy * params.wall_gauge_sigma_xy);
            g(2, 2) = 1.f / (params.wall_gauge_sigma_theta * params.wall_gauge_sigma_theta);
            return g;
        }
        if (i < W.size())
        {
            const Eigen::Matrix3f& C = W[i].pose_cov;
            if (C.allFinite() and C.determinant() > 1e-18f)
                return C.inverse();
        }
        return Eigen::Matrix3f::Zero();
    }

    std::unordered_map<std::uint64_t, Eigen::Matrix2f> RoomConcept::window_wall_information() const
    {
        std::unordered_map<std::uint64_t, Eigen::Matrix2f> out;
        if (not estimating()) return out;
        gn::Input in; in.params = &params;
        in.walls = const_cast<wallmap::WallMap*>(&wall_map_);   // read-only: find() and fields
        const auto& W = window_mgr_.window;
        for (std::size_t i = 0; i < W.size(); ++i)
        {
            const auto& slot = W[i];
            if (slot.wall_assoc.empty()) continue;
            const Eigen::Matrix3f prec = slot_pose_precision(i);
            for (const auto& [id, info] : gn::wall_marginal_information_of_slot(in, slot, slot.pose, &prec))
            {
                auto [it, fresh] = out.try_emplace(id, Eigen::Matrix2f::Zero());
                it->second += info;
            }
        }
        return out;
    }

    void RoomConcept::record_wall_input(const std::vector<Eigen::Vector3f>& points, const std::vector<Eigen::Vector2f>& pts,
                                        const Eigen::VectorXf& weights, const Eigen::Vector3f& pose, std::int64_t timestamp_ms)
    {
        if (params.record_wall_input)
        {
            if (not wall_input_rec_.is_open())
            {
                std::filesystem::create_directories("tmp");
                const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count();
                wall_input_rec_.open("tmp/wall_input_" + std::to_string(ms) + ".bin", std::ios::binary | std::ios::trunc);
                if (wall_input_rec_.is_open())
                {
                    // v2 (2026-10-05): odom is THIS frame's delta, not the stride accumulator (stride_span.h)
                    const std::uint32_t version = 2;
                    wall_input_rec_.write("WIN1", 4);
                    wall_input_rec_.write(reinterpret_cast<const char*>(&version), sizeof(version));
                    qInfo() << "[room][wall-slam] RecordWallInput: recording wall-step input to tmp/wall_input_" << ms << ".bin";
                }
            }
            if (wall_input_rec_.is_open())
            {
                const auto put = [&](const auto& v) { wall_input_rec_.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
                put(static_cast<std::int64_t>(timestamp_ms));
                put(static_cast<std::uint8_t>(rec_stride_replace_ ? 1 : 0));
                for (int i = 0; i < 3; ++i) put(rec_odom_delta_[i]);
                for (int i = 0; i < 3; ++i) put(pose[i]);
                for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) put(current_covariance(r, c));
                put(static_cast<std::uint32_t>(pts.size()));
                for (size_t i = 0; i < pts.size(); ++i)
                {
                    put(pts[i].x()); put(pts[i].y()); put(points[i].z());
                    put(weights.size() > static_cast<long>(i) ? weights(static_cast<long>(i)) : 1.f);
                }
            }
        }
    }

    void RoomConcept::wall_slam_observe(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& pose,
                                        std::int64_t timestamp_ms)
    {
        std::vector<Eigen::Vector2f> pts;
        pts.reserve(points.size());
        for (const auto& p : points) pts.emplace_back(p.x(), p.y());
        // ── AFTER PROMOTION THE WALL MAP HAS NO CONSUMER (2026-10-10) ────────────────────────────────────
        // The adopted map layout is the given room (derived_polygon_ = map_layout_), the refiner edits it from the
        // shadow SLAM map, and the SDF localiser solves against it; the wall map's polygon is not published and the
        // model polygon is not overwritten from it. Its segmentation + association cost 1.8-3 ms/frame (measured,
        // t_wall_ms) for nothing. Only the input RECORDING continues (offline replays), with unit weights.
        if (map_layout_adopted_)
        {
            record_wall_input(points, pts, Eigen::VectorXf::Ones(static_cast<long>(pts.size())), pose, timestamp_ms);
            return;
        }
        const auto seg = wallseg::segment(pts, params.wall_seg, wall_rng_);

        // The sidecar sees the robot-frame scan, the map-frame pose, that pose's covariance, and —
        // for the Manhattan gauge — THE AGENT'S OWN FITTED WALL SEGMENTS. Reusing the segmenter
        // here is not convenience: an estimator that reads consecutive point differences assumes
        // an azimuthally ordered scan, which is true of a synthesised one and not of this input,
        // and that assumption cost a confident, stable, 56-degree-wrong gauge.
        if (box_channel_.enabled())
        {
            std::vector<float> sphi, slen;
            sphi.reserve(seg.segments.size()); slen.reserve(seg.segments.size());
            for (const auto& sg : seg.segments)
            { sphi.push_back(sg.phi); slen.push_back(std::abs(sg.s_max - sg.s_min)); }
            box_channel_.observe(points, pose, box_pose_cov_, sphi, slen);
            // The SAME scan, below the wall band: furniture, for navigation only. It never enters
            // the layout, the registration or the gauge — see Params::obs_z_min.
            box_channel_.observe_obstacles(points, pose);
        }

        // Observation weights (range / incidence) — the SAME as the SDF term's, from each segment's own
        // normal, in ONE call over every claimed point so the normalisation is per slot, not per wall.
        Eigen::VectorXf weights = Eigen::VectorXf::Ones(static_cast<long>(pts.size()));
        if (params.far_points_weight or params.incidence_angle_weight)
        {
            std::vector<int> idx;
            std::vector<Eigen::Vector2f> nrm;
            for (const auto& sg : seg.segments)
                for (int i : sg.inliers) { idx.push_back(i); nrm.push_back(sg.normal); }
            if (not idx.empty())
            {
                ScanPoints P(idx.size(), Eigen::Vector3f::Zero());
                Eigen::Matrix<float, -1, 2> N(static_cast<Eigen::Index>(idx.size()), 2);
                const float c = std::cos(pose.z()), s = std::sin(pose.z());
                for (size_t k = 0; k < idx.size(); ++k)
                {
                    P[k].x() = pts[static_cast<size_t>(idx[k])].x();
                    P[k].y() = pts[static_cast<size_t>(idx[k])].y();
                    // weights_from_normals wants ROOM-frame normals (it rotates the rays by θ itself)
                    N(static_cast<Eigen::Index>(k), 0) = c * nrm[k].x() - s * nrm[k].y();
                    N(static_cast<Eigen::Index>(k), 1) = s * nrm[k].x() + c * nrm[k].y();
                }
                const Eigen::VectorXf wa = weights_from_normals(params, P, pose.z(), N, &doors_robot_);
                for (size_t k = 0; k < idx.size(); ++k) weights(idx[k]) = wa(static_cast<Eigen::Index>(k));
            }
        }
        // Z attribution BEFORE anything is logged: per-segment mean z, and the frame's z profile.
        last_seg_z_.assign(seg.segments.size(), 0.f);
        for (size_t si = 0; si < seg.segments.size(); ++si)
        {
            double zs = 0.0;
            for (int i : seg.segments[si].inliers) zs += points[static_cast<size_t>(i)].z();
            last_seg_z_[si] = seg.segments[si].inliers.empty() ? 0.f
                              : static_cast<float>(zs / static_cast<double>(seg.segments[si].inliers.size()));
        }
        {
            std::vector<float> zs;
            zs.reserve(points.size());
            for (const auto& p : points) zs.push_back(p.z());
            std::sort(zs.begin(), zs.end());
            wall_z_mean_ = zs.empty() ? 0.f : std::accumulate(zs.begin(), zs.end(), 0.f) / static_cast<float>(zs.size());
            wall_z_p95_  = zs.empty() ? 0.f : zs[static_cast<size_t>(0.95f * static_cast<float>(zs.size() - 1))];
            wall_z_max_  = zs.empty() ? 0.f : zs.back();
        }
        wall_frame_ts_ = timestamp_ms;

        // ── DIAGNOSTIC RECORDER (RoomShape.RecordWallInput, default OFF) ─────────────────────────────
        // Exactly what the wall step receives this frame, so a live run can be replayed through the offline
        // harness core and the live INPUT separated from the agent's INTEGRATION loop. Read-only.
        // Binary, native little-endian: "WIN1" + u32 version, then per frame: i64 ts, u8 stride_replace,
        // f32 odom[3], f32 pose[3], f32 cov[9] (row-major), u32 n, n x (f32 x, f32 y, f32 z, f32 weight).
        record_wall_input(points, pts, weights, pose, timestamp_ms);

        // FROZEN: observe() still runs, because association is what feeds the residual log and the
        // canvas, and a frozen map that cannot say how far the scan falls from it is worth nothing.
        // But its structural and geometric state is put back exactly as it was: no births, no deaths,
        // no splices, no extent creep, no candidate promoted while we are not looking. Restoring is
        // used rather than a flag threaded through observe() because it cannot be partially right —
        // whatever new mutation is added inside observe() next month is frozen too, by construction.
        if (wall_frozen_)
        {
            // θ₀ FREEZES WITH THE WALLS. Restoring only walls+order left update_theta0() free to keep
            // moving the room's reference direction from every segment — and manhattan_polygon(), the
            // polygon that is actually PUBLISHED, projects the walls onto θ₀'s classes. So a frozen
            // wall set still produced a changing published layout: measured live, the agent flip-
            // flopped "republished delimiting_polygon with 6 vertices (was 10)" and back, every few
            // frames, each one a structure change that restales the wall_i nodes for every consumer.
            // Freezing the geometry means freezing everything the published geometry is a function of.
            const auto saved_walls = wall_map_.walls;
            const auto saved_order = wall_map_.order;
            const float saved_theta0 = wall_map_.theta0;
            const float saved_theta0_info = wall_map_.theta0_information;
            const bool  saved_theta0_born = wall_map_.theta0_born;
            last_wall_frame_ = wall_map_.observe(seg, pts, weights, pose, current_covariance, timestamp_ms);
            wall_map_.walls = saved_walls;
            wall_map_.order = saved_order;
            wall_map_.theta0 = saved_theta0;
            wall_map_.theta0_information = saved_theta0_info;
            wall_map_.theta0_born = saved_theta0_born;
            wall_map_.candidates.clear();
        }
        else
            last_wall_frame_ = wall_map_.observe(seg, pts, weights, pose, current_covariance, timestamp_ms);
        last_wall_segments_ = seg.segments;
        window_mgr_.newest().wall_assoc = last_wall_frame_.assoc;

        if (not wall_events_csv_.is_open())
        {
            wall_events_csv_.open("etc/wall_slam_events.csv", std::ios::trunc);
            wall_events_csv_.imbue(std::locale::classic());
            wall_events_csv_ << "ts_ms,event,id,phi,d,npts,frames,dF_or_lodds,nearest_wall,nearest_chi2,seg_z_mean\n";
        }
        for (const auto& d : last_wall_frame_.deaths_info)
            wall_events_csv_ << timestamp_ms << ",death," << d.id << ",,," << d.points_seen << ','
                             << d.frames_seen << ',' << d.lodds << ",,,\n";
        for (const auto& b : last_wall_frame_.births_info)
            wall_events_csv_ << timestamp_ms << ",birth," << b.id << ',' << b.phi << ',' << b.d << ','
                             << b.npts << ',' << b.frames << ',' << b.dF << ',' << b.nearest_wall << ','
                             << b.nearest_chi2 << ','
                             << ((b.seg >= 0 and b.seg < static_cast<int>(last_seg_z_.size())) ? last_seg_z_[static_cast<size_t>(b.seg)] : -1.f)
                             << "\n";
        wall_events_csv_.flush();
        // Every birth is loggable: after the duplicate fix a birth is a rare, meaningful event, and
        // nearest_chi2 is the number that says whether the gate refused a twin (small-ish, tens) or
        // this line truly matches nothing (huge/inf).
        for (const auto& d : last_wall_frame_.deaths_info)
            qInfo().noquote() << QString("[room][wall-slam] DEATH wall %1: existence log-odds %2 after %3 frames / %4 points — the beams walked through it")
                                     .arg(d.id).arg(d.lodds, 0, 'f', 1).arg(d.frames_seen).arg(d.points_seen);
        for (const auto& b : last_wall_frame_.births_info)
            qInfo().noquote() << QString("[room][wall-slam] BIRTH wall %1: phi=%2 d=%3 npts=%4 frames=%5 dF=%6 | nearest wall %7 chi2=%8")
                                     .arg(b.id).arg(b.phi, 0, 'f', 3).arg(b.d, 0, 'f', 3).arg(b.npts).arg(b.frames)
                                     .arg(b.dF, 0, 'f', 1).arg(b.nearest_wall).arg(b.nearest_chi2, 0, 'f', 1);
        // Rate-limited health line (~every 200 frames): the numbers that say whether association is
        // doing its job. assoc/segs near 1 and twins ≈ births ≈ 0 is a healthy converged map.
        wall_stat_frames_++;
        wall_stat_assoc_  += last_wall_frame_.segments_associated;
        wall_stat_segs_   += static_cast<int>(seg.segments.size());
        wall_stat_twins_  += last_wall_frame_.twins_fused;
        wall_stat_births_ += last_wall_frame_.births;
        wall_stat_deaths_ += last_wall_frame_.deaths;
        wall_stat_contained_ += last_wall_frame_.splice_rejected;
        ++wall_frames_since_rederive_;
        wall_rejected_since_rederive_ += last_wall_frame_.splice_rejected;
        if (wall_stat_frames_ >= 200)
        {
            // splice_rej counts qualified candidates that found no valid place on the polygon —
            // a refusal is never silent (the residual agent's "refused vs never asked" lesson).
            qInfo().noquote() << QString("[room][wall-slam] health over %1 frames: assoc %2/%3 segments, twins fused %4, births %5, deaths %6, %7, walls %8, candidates %9")
                                     .arg(wall_stat_frames_).arg(wall_stat_assoc_).arg(wall_stat_segs_)
                                     .arg(wall_stat_twins_).arg(wall_stat_births_).arg(wall_stat_deaths_)
                                     .arg(QString("splice_rej %1").arg(wall_stat_contained_))
                                     .arg(wall_map_.walls.size()).arg(wall_map_.candidates.size());
            wall_stat_frames_ = wall_stat_assoc_ = wall_stat_segs_ = wall_stat_twins_ = wall_stat_births_ = 0;
            wall_stat_deaths_ = 0; wall_stat_contained_ = 0;
        }
        // ── DO THE RETURNS FALL SHORT OF THE WALL WE FITTED TO THEM? ────────────────────────
        // One row per associated segment: the median signed perpendicular residual of its returns
        // about the wall, in the map frame. Positive = the returns sit on the room side, i.e. short
        // of the fitted wall. This is the systematic component the paper's corner floor represents,
        // and it was only ever measured offline (the harness prints it per wall; the live agent had
        // nothing) or inferred indirectly from recovered room dimensions.
        //
        // ⚠ IT LIVES HERE, in the per-frame observe stage, and NOT in wall_slam_after_solve, because
        // that runs downstream of the pose solve and the prediction early-exit skips it. Measured on
        // the first frozen-layout run: with the map fixed and right, the early exit took 2612 of 2622
        // frames — 99.6% — so every writer inside after_solve fell to one row every few seconds and
        // this log went dark at exactly the moment it became worth reading. It also carries its own
        // tick: sharing layout_trace_tick_ meant its flush condition stopped advancing too, so rows
        // were written and never reached the disk.
        if (not last_wall_frame_.assoc.empty())
        {
            if (not resid_csv_.is_open())
            {
                std::filesystem::create_directories("tmp");
                resid_csv_.open("tmp/wall_residual.csv", std::ios::out | std::ios::trunc);
                if (resid_csv_.is_open())
                {
                    resid_csv_.imbue(std::locale::classic());
                    resid_csv_ << "frame,ts_ms,wall_id,npts,pda,resid_med_m\n";
                }
            }
            if (resid_csv_.is_open())
            {
                for (const auto& a : last_wall_frame_.assoc)
                    if (a.resid_n > 0)
                        resid_csv_ << resid_tick_ << ',' << wall_frame_ts_ << ',' << a.wall_id
                                   << ',' << a.resid_n << ',' << a.pda << ',' << a.resid_med << '\n';
                if ((++resid_tick_ % 20u) == 0) resid_csv_.flush();
            }
        }

        // ── EVERY WALL'S GEOMETRY, EVERY FRAME ───────────────────────────────────────────────
        // A published room 6.03 x 4.00 m became 0.51 x 4.00 m within 25 frames of first passing the
        // publish bar, with the corner sigma UNCHANGED at 0.035 through the collapse — so no existing
        // log could say which wall moved, when, or whether its normal flipped. 6.03 = 3.27 + 2.76 and
        // 3.27 - 2.76 = 0.51, i.e. one of the two long walls ended up on the WRONG SIDE of the origin,
        // which is a rigid-transform question and not an estimation one. phi and d together answer it:
        // a sign flip of d with phi unchanged is a wall that jumped across the room, while phi moving
        // by pi with d negated is the same line re-expressed and harmless. Four walls at 20 Hz, so the
        // file is small; it is what turns "the room collapsed" into "wall 7 did this on frame 1381".
        if (not wallgeom_csv_.is_open())
        {
            std::filesystem::create_directories("tmp");
            wallgeom_csv_.open("tmp/wall_geometry.csv", std::ios::out | std::ios::trunc);
            if (wallgeom_csv_.is_open())
            {
                wallgeom_csv_.imbue(std::locale::classic());
                wallgeom_csv_ << "frame,ts_ms,wall_id,k,phi_rad,d_m,s_min,s_max,frames_seen,points_seen,"
                                 "lodds,sigma_d,theta0,frozen,reanchored,class_nats\n";
            }
        }
        if (wallgeom_csv_.is_open())
        {
            for (const auto& w : wall_map_.walls)
            {
                const float lam = w.information(1, 1);
                wallgeom_csv_ << resid_tick_ << ',' << wall_frame_ts_ << ',' << w.id << ',' << w.k << ','
                              << w.phi << ',' << w.d << ',' << (w.has_extent ? w.s_min : 0.f) << ','
                              << (w.has_extent ? w.s_max : 0.f) << ',' << w.frames_seen << ','
                              << w.points_seen << ',' << w.exist_lodds << ','
                              << ((lam > 1e-9f) ? 1.f / std::sqrt(lam) : -1.f) << ','
                              << wall_map_.theta0 << ',' << (wall_frozen_ ? 1 : 0) << ','
                              << (wall_reanchored_ ? 1 : 0) << ','
                              // How far this wall disagrees with its own Manhattan class, in nats.
                              // Above -log(manhattan_off_prior) (2.3 at the default 0.1) the wall is
                              // better explained by no class at all, and the layout must not be
                              // pinned yet. This is the column that would have shown w988 at 16 nats
                              // while every corner sigma looked healthy.
                              << wall_map_.classify(w.phi).cost << '\n';
            }
            if ((resid_tick_ % 20u) == 0) wallgeom_csv_.flush();
        }

        if (wall_map_.walls.size() > 64 and wall_map_.walls.size() % 32 == 0)
            qWarning() << "[room][wall-slam]" << wall_map_.walls.size()
                       << "walls — far more than a room has. If they are twins of a few real ones,"
                       << "raise RoomShape.WallMapSigmaD (the corner detector's map_sigma lesson).";
    }

    void RoomConcept::record_slam_shadow(UpdateResult& r)   // non-const: hands the shadow pose to the refiner (2026-10-10)
    {
        // ── SHADOW (2026-10-10): plain SLAM (own map, own pose) vs the pose the SDF localiser published this frame.
        // Called on BOTH paths — the early exit ends most localizing frames, and placing this only after the full
        // solve recorded almost nothing (the first live run's log was empty). Flushed about once a second.
        if (not map_layout_adopted_) return;
        const Eigen::Vector3f sp = plain_slam_.pose();
        r.slam_pose = sp;   // the refiner accumulates its evidence at THIS pose (layout_refiner.h)
        r.slam_pose_valid = true;
        const Eigen::Vector2f t = r.robot_pose.translation();
        const float th = std::atan2(r.robot_pose.linear()(1, 0), r.robot_pose.linear()(0, 0));
        const float dpos = std::hypot(sp.x() - t.x(), sp.y() - t.y());
        const float dth  = std::remainder(sp.z() - th, 2.f * static_cast<float>(M_PI));
        slam_disagree_.store(dpos, std::memory_order_relaxed);
        if (not shadow_csv_.is_open())
        {
            shadow_csv_.open("tmp/slam_shadow.csv", std::ios::out | std::ios::trunc);
            shadow_csv_.imbue(std::locale::classic());
            shadow_csv_ << "ts_ms,sdf_x,sdf_y,sdf_th,slam_x,slam_y,slam_th,d_pos_m,d_th_rad,early_exit\n";
        }
        shadow_csv_ << r.timestamp_ms << ',' << t.x() << ',' << t.y() << ',' << th << ',' << sp.x() << ',' << sp.y()
                    << ',' << sp.z() << ',' << dpos << ',' << dth << ',' << (r.iterations_used <= 0 ? 1 : 0) << '\n';
        if (r.timestamp_ms - shadow_flush_ms_ >= 1000) { shadow_csv_.flush(); shadow_flush_ms_ = r.timestamp_ms; }
    }

    void RoomConcept::adopt_map_layout_as_given()
    {
        std::vector<Eigen::Vector2f> L;
        {
            std::scoped_lock lk(wall_map_mutex_);
            L = map_layout_;
            if (L.size() >= 3) derived_polygon_ = L;
        }
        if (L.size() < 3) return;
        map_layout_adopted_ = true;
        map_ready_ = true;
        wall_frozen_ = true;                              // layout_state() == Localizing from the next frame
        init_polygon_vertices_ = L;                       // the corner channel's model (a GIVEN layout)
        corner_detector_.set_model_corners(init_polygon_vertices_);
        if (model_ != nullptr and model_->has_state())
            model_->update_polygon_vertices(L);           // the SDF the localiser solves against
        qInfo().noquote() << QString("[room][startup-proto] ESTIMATING -> LOCALIZING: the map layout (%1 vertices) is now "
                                     "the GIVEN room; the pose is solved against its SDF from here on (plain SLAM stops).")
                                 .arg(L.size());
    }

    void RoomConcept::refine_given_layout(const std::vector<Eigen::Vector2f>& L)
    {
        // ── LAYOUT REFINEMENT AFTER PROMOTION (2026-10-10, layout_refiner.h) ──────────────────────────────────
        // The same writes as adopt_map_layout_as_given, minus the state flags: map_layout_ is what
        // wall_slam_after_solve copies into derived_polygon_ every frame (-> nominal_room_polygon -> the published
        // polygon), the corner channel's model, and the SDF. No re-anchor: an edit is a local change to the vertex
        // list in the frame the room already has, so the published frame (T_pub_int_) is untouched; the POSE moves
        // only as far as the moved wall moves the SDF solution — which is the point (toward the shadow SLAM pose).
        if (not map_layout_adopted_ or L.size() < 4) return;
        {
            std::scoped_lock lk(wall_map_mutex_);
            map_layout_ = L;
            derived_polygon_ = L;
        }
        init_polygon_vertices_ = L;
        corner_detector_.set_model_corners(init_polygon_vertices_);
        if (model_ != nullptr and model_->has_state())
            model_->update_polygon_vertices(L);
        qInfo().noquote() << QString("[room][refine] given room REFINED: %1 vertices; the SDF is rebuilt, frame unchanged").arg(L.size());
    }

} // namespace rc
