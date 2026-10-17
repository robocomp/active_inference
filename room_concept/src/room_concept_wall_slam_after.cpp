// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Wall-SLAM channel, after the solve: map update, re-anchoring, polygon room replacement.

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

    void RoomConcept::wall_slam_after_solve(UpdateResult& res)
    {
        // Both of the map's global repair paths are off while frozen: a merge changes the wall set
        // and a re-derivation replaces the whole cycle, which is precisely what "fixed" excludes.
        if (wall_frozen_) { }
        else
        {
        wall_map_.merge_indistinguishable();
        // GLOBAL re-derivation on the bench's cadence (WallMap::Params::rederive_*): trace the
        // observed free space around the solved pose and adopt its cycle iff it explains more.
        // Until 2026-09-02 only the bench ran this — the agent's polygon could only move by the
        // local splice jumps, and the bench IoU graded an algorithm the agent never executed.
        if (wall_map_.params.enable_rederive
            and (wall_frames_since_rederive_ >= wall_map_.params.rederive_every_frames
                 or wall_rejected_since_rederive_ >= wall_map_.params.rederive_after_rejections))
        {
            const int frames = wall_frames_since_rederive_, rejected = wall_rejected_since_rederive_;
            wall_frames_since_rederive_ = 0;
            wall_rejected_since_rederive_ = 0;
            const Eigen::Vector2f robot_xy = window_mgr_.newest().pose.head<2>();
            if (wall_map_.re_derive(robot_xy))
            {
                ++wall_rederives_;
                qInfo().noquote() << QString("[room][wall-slam] GLOBAL re-derivation adopted (#%1): %2 walls in the cycle, after %3 frames / %4 rejected splices")
                                         .arg(wall_rederives_).arg(wall_map_.order.size()).arg(frames).arg(rejected);
                if (wall_events_csv_.is_open())
                    wall_events_csv_ << wall_frame_ts_ << ",rederive," << wall_map_.order.size() << ",,,"
                                     << rejected << ',' << frames << ",,,,\n" << std::flush;
            }
        }
        }
        // ★The publish test reads carried + WINDOW information: carried holds only dropped slots, so a parked
        // robot (window never slides) never published a fully visible room (live Room2, 09-17).
        auto poly = wall_map_.build_polygon_with(window_wall_information());
        // The PUBLISHED layout: exactly Manhattan by construction — the output-stage projection
        // on a COPY (WallMap::manhattan_polygon). The raw polygon keeps every in-loop role
        // (re-anchor trigger and bbox, model update, status): six in-loop hard-Manhattan
        // variants were measured to degrade estimation; only what leaves the agent is projected.
        // LOCALIZING republishes the polygon it froze, rather than projecting and decorating afresh.
        // See frozen_pub_ for why: decorate() re-decides level-2 features from the grid every frame.
        auto pub = frozen_pub_.has_value() ? *frozen_pub_ : wall_map_.manhattan_polygon_with(window_wall_information());
        if (poly.closed and poly.verts.size() >= 3)
        {
            // ── AND THE WALLS MUST AGREE WITH THEIR OWN CLASSES BEFORE THE FRAME IS PINNED ───────
            // publishable is a corner-sigma test, and corner sigma cannot see a wall that is still
            // annealing onto its Manhattan class: all four corners can be individually sharp while
            // one wall sits 11.5 deg out (measured, wall w988 — it put the long dimension 0.14 m
            // long and the median |SDF| at 0.116 m). Both the re-anchor and the freeze are ONE-SHOT,
            // so pinning the frame on such a layout locks the tilt in permanently — theta0 can no
            // longer rotate it and no re-derivation will run.
            // The test adds no constant: classify() already prices a wall's disagreement with its
            // class in nats against the class-less component's own prior.
            float mh_worst = 0.f; std::uint64_t mh_wall = 0;
            const bool mh_ok = wall_map_.cycle_manhattan_converged(poly, &mh_worst, &mh_wall);
            if (not mh_ok and not wall_reanchored_ and (wall_mh_log_tick_++ % 100) == 0)
                qInfo().noquote() << QString("[room][wall-slam] layout publishable but NOT pinned yet: wall %1 "
                                             "disagrees with its Manhattan class by %2 nats (class-less costs %3). "
                                             "Still annealing — the re-anchor and the freeze are one-shot, so they wait.")
                                         .arg(mh_wall).arg(mh_worst, 0, 'f', 1)
                                         .arg(-std::log(std::clamp(wall_map_.params.manhattan_off_prior, 1e-6f, 1.f - 1e-6f)), 0, 'f', 1);
            if (poly.publishable and mh_ok and not wall_reanchored_)
            {
                // One-shot re-anchoring to the frame the fleet expects: origin on the Manhattan-aligned
                // bbox centre of the polygon, θ₀ = 0. Everything in the map frame moves together.
                const float rot = wall_map_.theta0_born ? wall_map_.theta0 : 0.f;
                const Eigen::Rotation2Df Rm(-rot);
                Eigen::Vector2f lo = Rm * poly.verts.front(), hi = lo;
                for (const auto& v : poly.verts) { const Eigen::Vector2f q = Rm * v; lo = lo.cwiseMin(q); hi = hi.cwiseMax(q); }
                const Eigen::Vector2f c = Eigen::Rotation2Df(rot) * ((lo + hi) * 0.5f);
                // ★ NOT while a start-up proto-room owns the published frame (2026-10-09). That frame is frozen at
                //   the robot's initial pose, so re-centring the internal one buys nothing; and this early in a run
                //   the re-anchor rotated the MAP while the robot pose was not re-expressed with it — the localiser
                //   then snapped the robot to the rotated map (heading -0.53 -> 2.88 rad while driving) and the
                //   controller steered by the jump. Internal frame == published frame until promotion.
                if (not freeze_internal_frame_.load(std::memory_order_relaxed))
                    reanchor_map_frame(c, rot);
                wall_reanchored_ = true;
                poly = wall_map_.build_polygon_with(window_wall_information());
                pub  = wall_map_.manhattan_polygon_with(window_wall_information());
                map_ready_ = true;
                wall_frozen_ = wall_map_.params.freeze_when_publishable;
                if (wall_frozen_)
                {
                    frozen_pub_ = pub;   // this shape, from here on
                    // ── AND THE LEARNT LAYOUT BECOMES THE CORNER DETECTOR'S MODEL ────────────────
                    // The corner channel is gated on init_polygon_vertices_ being non-empty — that is
                    // the GIVEN layout, filled only when a polygon is loaded from file, and
                    // configure_room_estimate() clears it. So in Estimate mode corner tracking never
                    // ran at all, before or after the freeze: there was no model to match detections
                    // against, which is why no LiDAR corners appear once the room is stable.
                    // Entering LOCALIZING is exactly when that model exists, so this is where it is
                    // handed over. The published polygon is used, not the raw one: it is the
                    // axis-aligned shape every other consumer sees.
                    // ⚠ NOT via set_polygon_room(), which REPLACES model_ with a fresh Model and
                    // would reset the window, the covariance and the pose in the middle of a run. The
                    // model's geometry is already current — update_polygon_vertices() refreshes it
                    // every frame — so only these two things are missing.
                    if (pub.verts.size() >= 3)
                    {
                        init_polygon_vertices_ = pub.verts;
                        corner_detector_.set_model_corners(init_polygon_vertices_);
                        qInfo().noquote() << QString("[room][wall-slam] corner tracking armed on the frozen layout: "
                                                     "%1 model corners").arg(init_polygon_vertices_.size());
                    }
                }
                res.covariance = current_covariance;
                qInfo() << "[room][wall-slam] polygon CLOSED and publishable:" << poly.verts.size()
                        << "vertices, worst corner sigma" << poly.worst_corner_sigma << "m. Map frame re-anchored:"
                        << "origin moved by (" << c.x() << "," << c.y() << ") m, rotated" << rot * 180.f / static_cast<float>(M_PI) << "deg.";
                if (wall_frozen_)
                {
                    // Loud on purpose: from here the layout is an input, so every later number —
                    // pose, residual, corner sigma — is measured against a room that no longer moves.
                    // A silent freeze would look exactly like an estimator that simply stopped.
                    float w_lo = 1e9f, w_hi = -1e9f, h_lo = 1e9f, h_hi = -1e9f;
                    for (const auto& v : pub.verts)
                    { w_lo = std::min(w_lo, v.x()); w_hi = std::max(w_hi, v.x());
                      h_lo = std::min(h_lo, v.y()); h_hi = std::max(h_hi, v.y()); }
                    qWarning().noquote() << QString("[room][wall-slam] LAYOUT FROZEN (RoomShape.FreezeLayoutWhenPublishable): "
                                                   "%1 vertices, %2 x %3 m, worst corner sigma %4 m. No further births, deaths, "
                                                   "merges, re-derivations or wall updates — the map is now GIVEN.")
                                                .arg(pub.verts.size()).arg(w_hi - w_lo, 0, 'f', 3).arg(h_hi - h_lo, 0, 'f', 3)
                                                .arg(poly.worst_corner_sigma, 0, 'f', 4);
                    // Do not re-init the model here (init_from_state() resets the prediction state; under
                    // torch it also aborted on a second parameter registration). It does not need re-seeding:
                    // update_polygon_vertices() already refreshes its geometry and half_extents from
                    // the wall polygon every frame, by design, without touching the pose tensors.
                    // The magenta box was never a stale MODEL — it was the viewer drawing that bbox
                    // centred on the scene origin (see Viewer2D::update_estimated_room_rect), which is
                    // only where the room is after the re-anchor. It is now dropped once the wall map
                    // has a polygon of its own.
                    //
                    // ── IS THE FROZEN ROOM ACTUALLY AXIS-ALIGNED? ───────────────────────────────
                    // The re-anchor above rotates the whole map by theta0, so every wall should land
                    // on a multiple of 90 deg and the room should sit square to the axes. That is an
                    // ASSERTION about theta0, not a fact: it is estimated from the segments, and if
                    // it is off by d degrees when this one-shot fires, the room is left tilted by d
                    // for ever — the re-anchor never runs again, and under the freeze theta0 cannot
                    // correct it either. So the residual tilt is printed per wall and as a worst
                    // case, because a viewer looking at a tilted cloud in an axis-aligned frame can
                    // see that something is wrong but not which of the two is responsible.
                    {
                        float worst_tilt = 0.f;
                        QStringList phis;
                        for (const auto& w : wall_map_.walls)
                        {
                            const float deg = w.phi * 180.f / static_cast<float>(M_PI);
                            float off = std::fmod(std::fabs(deg), 90.f);
                            if (off > 45.f) off = 90.f - off;
                            worst_tilt = std::max(worst_tilt, off);
                            phis << QString("w%1 %2deg(off %3)").arg(w.id).arg(deg, 0, 'f', 2).arg(off, 0, 'f', 2);
                        }
                        qWarning().noquote()
                            << QString("[room][wall-slam] frozen frame check: theta0 %1 deg (born %2), "
                                       "worst wall tilt off axis %3 deg | %4")
                                   .arg(wall_map_.theta0 * 180.f / static_cast<float>(M_PI), 0, 'f', 2)
                                   .arg(wall_map_.theta0_born ? "yes" : "NO")
                                   .arg(worst_tilt, 0, 'f', 2)
                                   .arg(phis.join(", "));
                    }
                }
            }
            if (model_ != nullptr and model_->has_state() and not map_layout_adopted_)
                model_->update_polygon_vertices(poly.verts);   // in-loop consumer: RAW on purpose (an ADOPTED map layout stays)
            std::scoped_lock lk(wall_map_mutex_);
            // Publish the projected polygon only. Falling back to the raw one would alternate
            // vertex counts and a 1-2° tilt frame to frame, which the scene graph reports as a
            // structure change and door_concept re-keys on; a projection that fails to close keeps
            // the last good published layout instead (logged once per stretch).
            if (pub.closed and pub.verts.size() >= 3)
            {
                derived_polygon_ = pub.verts;
                projection_failed_logged_ = false;
            }
            else if (not projection_failed_logged_)
            {
                qWarning().noquote() << "[room][wall-slam] projected polygon did not close"
                                     << (pub.status.empty() ? "" : QString::fromStdString("— " + pub.status))
                                     << "; keeping the last published layout";
                projection_failed_logged_ = true;
            }
        }
        // The per-status line is gone from the terminal: every one of its numbers — walls, candidates,
        // closed, publishable, worst corner sigma — is a column of etc/wall_slam.csv, written every
        // frame, and the status string changes often enough during estimation to bury everything else.
        last_wall_status_ = poly.status;
        res.wall_view.segments    = last_wall_segments_;
        // The canvas shows the WORKING SET, not the graveyard: walls in the polygon plus walls a
        // segment touched in the last two seconds. Everything else still exists in the map (and in
        // the CSV) but stopped earning screen space — 350 grey immortals were the whole picture.
        {
            std::set<std::uint64_t> in_poly(poly.wall_of_edge.begin(), poly.wall_of_edge.end());
            res.wall_view.walls.clear();
            // ★WITH THE WINDOW'S INFORMATION (09-17). `information` alone is what has been ABSORBED, i.e. only
            // dropped slots: parked it is still the 0.50 m rectangle prior, so the canvas printed sd=0.50 and
            // "not publishable" about a room that had been published at 2 cm. Display only; nothing is stored.
            const auto win_info = window_wall_information();
            for (const auto& w : wall_map_.walls)
                if (in_poly.contains(w.id) or wall_frame_ts_ - w.last_seen_ms <= 2000)
                {
                    auto wv = w;
                    if (const auto it = win_info.find(w.id); it != win_info.end() and it->second.allFinite())
                        wv.information += it->second;
                    res.wall_view.walls.push_back(wv);
                }
        }
        res.wall_view.polygon     = (pub.closed and pub.verts.size() >= 3) ? pub : poly;

        // ── THE BOX SIDECAR PUBLISHES, IF IT IS ON ───────────────────────────────────────────
        // Everything above ran exactly as it ships. This replaces only the polygon that leaves the
        // agent, so an A/B on the robot is one config key and the wall map remains the thing to
        // fall back to. The box polygon cannot have fewer than 4 vertices — the smallest
        // representable region is one box — so the 3-vertex room that started this redesign is not
        // rejected here, it is unrepresentable.
        box_pose_cov_ = res.covariance;                     // for the NEXT observe()
        if (box_channel_.enabled())
        {
            const bool changed = box_channel_.step();
            const auto bverts = box_channel_.polygon();
            if (bverts.size() >= 4)
            {
                wallmap::Polygon bp;
                bp.verts = bverts;
                bp.closed = true;
                // A box layout is publishable as soon as it exists: it is closed, Manhattan and
                // simply connected BY CONSTRUCTION, so there is no shape test left to pass. What
                // remains uncertain is its SIZE, and that is carried in the corner sigmas below
                // rather than in a boolean.
                bp.publishable = true;
                const auto bcov = box_channel_.layout().polygon_cov(box_channel_.params().sigma_flat);
                bp.corners.resize(bverts.size());
                float worst = 0.f;
                for (size_t i = 0; i < bverts.size() and i < bcov.size(); ++i)
                {
                    // A vertex IS two offsets, so its covariance is a sub-block lookup — never an
                    // intersection of two lines, which is what diverged to a corner at
                    // (-11752, -14336) m with sigma 441051 in the representation this replaces.
                    bp.corners[i].p = bverts[i];
                    const float sg = std::sqrt(std::max(0.f, bcov[i].trace() * 0.5f));
                    bp.corners[i].sigma = sg;
                    worst = std::max(worst, sg);
                }
                bp.worst_corner_sigma = worst;
                bp.status = "box layout";
                res.wall_view.polygon = bp;
            }
            if (changed)
                qInfo().noquote()
                    << QString("[room][boxes] %1 boxes, %2 verts | rms %3 m | proposed %4 admitted %5 "
                               "removed %6 | last dL %7 nats | %8 cells | sigma_pose %9 m")
                           .arg(box_channel_.boxes()).arg(bverts.size())
                           .arg(box_channel_.rms(), 0, 'f', 3)
                           .arg(box_channel_.proposed()).arg(box_channel_.admitted())
                           .arg(box_channel_.removed())
                           .arg(box_channel_.last_dL(), 0, 'f', 0)
                           .arg(box_channel_.cells())
                           .arg(box_channel_.last_sigma(), 0, 'f', 3);
        }
        res.wall_view.seg_to_wall = last_wall_frame_.seg_to_wall;
        res.wall_view.theta0_born = wall_map_.theta0_born;
        res.wall_view.theta0      = wall_map_.theta0;
        res.wall_view.candidates  = static_cast<int>(wall_map_.candidates.size());
        res.wall_view.births      = last_wall_frame_.births;
        res.wall_view.map_ready   = map_ready_.load();

        // One analysis row per frame. Small, append-only, truncated per run.
        if (not wall_csv_.is_open())
        {
            wall_csv_.open("etc/wall_slam.csv", std::ios::trunc);
            wall_csv_.imbue(std::locale::classic());
            wall_csv_ << "ts_ms,x,y,theta,npts,z_mean,z_p95,z_max,segs,assoc,twins,births,deaths,"
                         "walls,candidates,closed,publishable,worst_corner_sigma,map_ready\n";
        }
        {
            const auto st = (model_ != nullptr and model_->has_state()) ? model_->get_state()
                                                                        : Eigen::Matrix<float, 5, 1>::Zero().eval();
            wall_csv_ << wall_frame_ts_ << ',' << st[2] << ',' << st[3] << ',' << st[4] << ','
                      << last_wall_segments_.size() << ',' << wall_z_mean_ << ',' << wall_z_p95_ << ','
                      << wall_z_max_ << ',' << last_wall_frame_.seg_to_wall.size() << ','
                      << last_wall_frame_.segments_associated << ',' << last_wall_frame_.twins_fused << ','
                      << last_wall_frame_.births << ',' << last_wall_frame_.deaths << ','
                      << wall_map_.walls.size() << ',' << wall_map_.candidates.size() << ','
                      << (poly.closed ? 1 : 0) << ',' << (poly.publishable ? 1 : 0) << ','
                      << poly.worst_corner_sigma << ',' << (map_ready_.load() ? 1 : 0) << "\n";
            wall_csv_.flush();
        }

        // ── THE PUBLISHED LAYOUT, WITH THE UNCERTAINTY ATTACHED TO IT ────────────────────────
        // etc/wall_slam.csv above records only the WORST corner sigma, which cannot be checked
        // against anything: grading a layout's uncertainty needs each vertex and the sigma claimed
        // for it, so the two can be matched to ground truth one-to-one. The offline harness has
        // written this trace for a while (RunConfig::poly_csv) and the live agent never did, which
        // is why every calibration number so far comes from simulated observations. Same columns and
        // same semantics as the harness, so one analysis script reads both.
        // Semicolon-separated because the vertex list keeps its commas; classic locale because this
        // machine runs es_ES and a comma decimal separator would make it unreadable (CLAUDE.md).
        if (not layout_csv_.is_open())
        {
            std::filesystem::create_directories("tmp");
            layout_csv_.open("tmp/layout_trace.csv", std::ios::out | std::ios::trunc);
            if (layout_csv_.is_open())
            {
                layout_csv_.imbue(std::locale::classic());
                layout_csv_ << "# frame;ts_ms;model_room_x,model_room_y,model_room_phi;verts;corner_sigma;"
                               "edge_sigma_d;closed,publishable\n";
            }
        }
        if (layout_csv_.is_open() and not poly.verts.empty())
        {
            // ⚠ get_state() is [width, length, x, y, phi] of the ROOM MODEL — NOT the robot pose.
            // Mislabelling this column as the robot's cost an hour: in estimate mode the model room
            // is not optimised at all (no_sdf), it exists for the viewer and the SDF paths, so it
            // drifts away from the wall landmarks that ARE the estimate. Measured in Webots room 2:
            // model room phi 0.5-1.1 deg against a wall-map polygon tilted 3.8 deg, i.e. ~3 deg
            // apart, which is ~40 cm at a far corner of a 6 m room — and the viewer draws both.
            // All-zero here means has_state() == false, not a room at the origin.
            const auto st = (model_ != nullptr and model_->has_state()) ? model_->get_state()
                                                                       : Eigen::Matrix<float, 5, 1>::Zero().eval();
            layout_csv_ << layout_trace_tick_++ << ';' << wall_frame_ts_ << ';'
                        << st[2] << ',' << st[3] << ',' << st[4] << ';';
            for (std::size_t i = 0; i < poly.verts.size(); ++i)
                layout_csv_ << (i ? " " : "") << poly.verts[i].x() << ',' << poly.verts[i].y();
            layout_csv_ << ';';
            for (std::size_t i = 0; i < poly.corners.size(); ++i)
                layout_csv_ << (i ? " " : "") << poly.corners[i].sigma;
            layout_csv_ << ';';
            // Per-edge sigma_d comes from the wall the edge belongs to: 1/sqrt(Lambda_dd).
            for (std::size_t i = 0; i < poly.wall_of_edge.size(); ++i)
            {
                const auto* w = wall_map_.find(poly.wall_of_edge[i]);
                const float lam = (w != nullptr) ? w->information(1, 1) : 0.f;
                layout_csv_ << (i ? " " : "") << ((lam > 1e-9f) ? 1.f / std::sqrt(lam) : -1.f);
            }
            layout_csv_ << ';' << (poly.closed ? 1 : 0) << ',' << (poly.publishable ? 1 : 0) << '\n';
            if ((layout_trace_tick_ % 20u) == 0) layout_csv_.flush();
        }
        // ── THE MAP LAYOUT REPLACES THE WALL MAP'S POLYGON WHILE PLAIN SLAM DRIVES (2026-10-09) ────────────────
        // Live, the wall estimator froze at its first-frame seed or merged rooms on this apartment, while the layout
        // read off the SLAM map scored IoU 0.95 vs ground truth. derived_polygon_ feeds nominal_room_polygon() ->
        // the scene graph's published polygon, promotion and the wall nodes; wall_view.polygon feeds the viewer.
        if (plain_slam_driving() or map_layout_adopted_)
        {
            std::scoped_lock lk(wall_map_mutex_);
            if (map_layout_.size() >= 3)
            {
                derived_polygon_ = map_layout_;
                wallmap::Polygon P;
                P.verts = map_layout_;
                P.closed = true;
                P.publishable = true;
                res.wall_view.polygon = std::move(P);
                res.wall_view.map_ready = true;   // the viewer draws the clean room outline only when this is set
            }
        }
    }

    void RoomConcept::reanchor_map_frame(const Eigen::Vector2f& c, float rot)
    {
        // The box sidecar lives in the map frame too, and its voxel map spans the whole session.
        // Skipping it here leaves every accumulated return stale by this exact transform.
        box_channel_.reanchor(c, rot);
        const Eigen::Rotation2Df Rm(-rot);
        const auto wrap = [](float a) { return std::atan2(std::sin(a), std::cos(a)); };
        const auto xf = [&](const Eigen::Vector3f& p)
        {
            const Eigen::Vector2f q = Rm * (p.head<2>() - c);
            return Eigen::Vector3f(q.x(), q.y(), wrap(p.z() - rot));
        };
        Eigen::Matrix3f A = Eigen::Matrix3f::Identity();
        A.topLeftCorner<2, 2>() = Rm.toRotationMatrix();

        auto poses = read_window_poses();
        for (auto& p : poses) p = xf(p);
        write_window_poses(poses);
        auto& bp = window_mgr_.boundary_prior;
        if (bp.valid)
        {
            bp.mu = xf(bp.mu);
            bp.precision = A * bp.precision * A.transpose();   // Δ' = A·Δ ⇒ Λ' = A Λ Aᵀ
        }
        // The model's own pose tensors — NOT set_robot_pose(), which clears the window.
        if (model_ != nullptr and model_->has_state())
        {
            const auto st = model_->get_state();
            const Eigen::Vector3f p = xf(Eigen::Vector3f(st[2], st[3], st[4]));
            model_->set_pose(p);
            model_->has_prediction = false;
            model_->robot_prev_pose = std::nullopt;
        }
        current_covariance = A * current_covariance * A.transpose();
        if (has_smoothed_pose_) smoothed_pose_ = xf(smoothed_pose_);
        if (stride_has_admitted_) stride_last_admitted_ = xf(stride_last_admitted_);
        if (last_good_pose_valid_) last_good_pose_ = xf(last_good_pose_);
        {
            std::scoped_lock lk(object_anchors_mutex_);
            const Eigen::Matrix2f R2 = Rm.toRotationMatrix();
            for (auto& [id, e] : landmark_estimates_)
            {
                e.p = R2 * (e.p - c);
                e.information = R2 * e.information * R2.transpose();
            }
        }
        wall_map_.reanchor(c, rot);

        // ★ THE PUBLISHED FRAME MUST NOT MOVE WITH THE GAUGE. Everything above rewrote the INTERNAL
        // frame, which is only a gauge and is nobody else's business. But a frame published as a fixed
        // transform of it — the proto-room — would move with it, and that is a teleport to every
        // consumer holding it: one re-anchor was measured at 92 deg and 0.74 m in a single frame.
        // So hand the publisher the same move to absorb. p_new = Rm*(p - c) ⇒ T_new_old below.
        Eigen::Affine2f T_new_old = Eigen::Affine2f::Identity();
        T_new_old.linear()        = Rm.toRotationMatrix();
        T_new_old.translation()   = -(Rm * c);
        pending_reanchor_         = T_new_old * pending_reanchor_;   // compose, never overwrite
        has_pending_reanchor_     = true;
    }

    void RoomConcept::set_polygon_room(const std::vector<Eigen::Vector2f>& polygon_vertices)
    {
        if (polygon_vertices.size() < 3)
        {
            std::cerr << "set_polygon_room: Need at least 3 vertices" << std::endl;
            return;
        }

        // Vertices are in room frame (where user clicked on the viewer)
        // Keep current robot pose if we have one, otherwise start at origin
        float init_x = 0.0f;
        float init_y = 0.0f;
        float init_phi = 0.0f;

        if (model_ != nullptr and model_->has_state())
        {
            // Preserve current robot pose. has_state(), not just non-null: the previous model may
            // have been allocated and never filled, and get_state() would throw here.
            const auto state = model_->get_state();
            init_x = state[2];
            init_y = state[3];
            init_phi = state[4];
        }

        model_ = std::make_shared<Model>();
        model_->init_from_polygon(polygon_vertices, init_x, init_y, init_phi, params.wall_height);

        // Reset state but keep covariance reasonable
        last_lidar_timestamp = 0;
        last_update_result = UpdateResult{};
        current_covariance = Eigen::Matrix3f::Identity() * 0.1f;
        needs_orientation_search_ = true;  // Will search for best orientation on first update
        has_smoothed_pose_ = false;  // Reset smoothing
        tracking_step_count_ = 0;  // Reset early exit tracking
        prediction_early_exits_ = 0;
        current_velocity_weights_ = Eigen::Vector3f::Ones();  // Reset velocity weights
        prev_sdf_mse_ = 0.f;  // Reset boundary quality gate
        u_b_init_ = false;    // reseed hierarchical boundary log-precision to g(v) after a reset
        window_mgr_.clear(); reset_stride_state();
        rerun_room_polygon_sent_ = false;

        // Initialize corner detector with model polygon + graded-covariance tuning from config.
        {
            auto& cp = corner_detector_.params();
            cp.wall_band      = params.corner_wall_band;
            cp.base_sigma     = params.corner_base_sigma;
            cp.orient_tau_deg = params.corner_orient_tau_deg;
            cp.merge_chi2         = params.corner_merge_chi2;
            cp.merge_prior_sigma  = params.corner_merge_prior_sigma;
            cp.min_wall_map_sigmas  = params.corner_min_wall_map_sigmas;
            cp.min_yield_map_sigmas = params.corner_min_yield_map_sigmas;
            cp.yield_leak           = params.corner_yield_leak;
            cp.yield_warmup         = params.corner_yield_warmup;
            cp.yield_release_factor = params.corner_yield_release_factor;
        }
        corner_detector_.set_model_corners(polygon_vertices);

        // Report the landmark exclusion once, with the geometry that justified it — a silent drop of
        // model corners is exactly the kind of thing that later reads as "the detector is broken".
        if (const auto& dropped = corner_detector_.short_wall_dropped_indices(); !dropped.empty())
        {
            const int N = static_cast<int>(polygon_vertices.size());
            QStringList detail;
            for (const int i : dropped)
            {
                const float w_in  = (polygon_vertices[i] - polygon_vertices[(i + N - 1) % N]).norm();
                const float w_out = (polygon_vertices[(i + 1) % N] - polygon_vertices[i]).norm();
                detail << QString("v%1(%2/%3m)").arg(i).arg(w_in, 0, 'f', 3).arg(w_out, 0, 'f', 3);
            }
            qInfo().noquote()
                << "[corners]" << dropped.size() << "of" << N
                << "polygon vertices refused LANDMARK status: adjacent walls shorter than"
                << params.corner_min_wall_map_sigmas * corner_detector_.params().map_sigma
                << "m, which the traced layout cannot assert. They remain in the polygon and the SDF."
                << "\n           " << detail.join(' ');
        }

        // A fresh polygon invalidates every per-vertex tally (indices may not even mean the same thing).
        corner_vertex_stats_.clear();
        corner_stats_frames_ = 0;
    }

} // namespace rc
