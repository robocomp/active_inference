// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
// NEW-ROOM BENCH — how well does room_concept's CURRENT room-building pipeline estimate a room it has never seen?
//
// Included by wall_slam_selftest.cpp (just before main) so it shares, UNCHANGED, the random-room generator
// (random_room / apartamento_room), the simulated LiDAR (scan), the base + encoder models, the bumper, the
// controller chain (TrajectoryController + RouteSpline) and the scoring helpers (polygon_iou, point_to_poly,
// segments_cross). Only the ESTIMATOR / EXPLORER / PROMOTION of run_boxes are replaced, by the agent's own units:
//
//   pose        rc::slam::ScanSlam (scan_slam.h)                         room_concept_update.cpp:341-375
//   raster      rc::freespace::FreeSpacePolygon (free_space_polygon.h)   room_scene_graph.cpp step_startup_proto
//   outline     FreeSpacePolygon::manhattan_layout(start, 0.18f)         room_scene_graph.cpp:2012
//   exploration rc::layout::plan_coverage + FeasibleGrid + boxed_in_handoff   room_scene_graph.cpp:2377-2516
//   promotion   outline pool (last 60) -> LayoutRefiner::description_lengths -> shortest   :2025-2036, :2125-2158
//   refinement  LayoutRefiner::propose every 5 s after promotion; contested-cell explorer :2478-2490, :4528-4600
//
// NOT used: the box-cover estimator (room_boxes*.h, BoxLayout) and the WallMap. Each block below cites the live
// lines it mirrors; the glue that lives in RoomSceneGraph tied to DSR (publish throttle, pool, claim/refuse
// protocol) is reproduced line by line with the DSR/Qt calls removed and wall-clock replaced by simulated time.
// DEVIATIONS from the live agent are listed in the report AND marked "DEVIATION" below.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#pragma once

#include <deque>
#include <numeric>

#include "free_space_polygon.h"
#include "layout_explorer.h"
#include "layout_refiner.h"
#include "scan_slam.h"

struct NewRoomRun
{
    float iou = 0.f;            ///< published outline vs truth, map frame -> world with the TRUE start pose (no gauge freedom)
    float iou_rot4 = 0.f;       ///< the box bench's score: best of the four quarter-turns about the start pose
    int   est_verts = 0, truth_verts = 0;
    bool  exact = false;
    double pose_med = -1.0;     ///< median over frames of |slam xy - true xy| in the map frame (m)
    double rot_err_deg = -1.0;  ///< median over frames of |slam heading - true heading| (deg)
    float seen_frac = 0.f;
    int   promo_frame = -1;     ///< frame at which promotion fired (-1 = never)
    bool  promoted = false;
    int   edits_adopted = 0;    ///< layout edits the refiner adopted after promotion
    int   frames = 0;           ///< frames actually run
    int   frames_out = 0;       ///< executed pose outside the TRUE polygon
    int   frames_blocked = 0;   ///< translation refused by the bumper (body within body_radius of a wall)
    bool  left_room = false;
    bool  explorer_finished = false;   ///< layout_finished_ latched (the promotion trigger)
    std::string stop = "budget";
    int   pool_max = 0, publishes = 0, replans = 0, visits = 0, refused = 0;
    float dl_best = 0.f, dl_latest = 0.f;   ///< promotion's description lengths (nats) of the chosen / latest outline
    Poly  est_poly;             ///< world frame
    std::vector<Eigen::Vector2f> traj;
    double wall_s = 0.0;
    float iou_promo = -1.f; int verts_promo = 0;   ///< the OUTLINE STAGE alone: the adopted outline at the moment of promotion, before any refiner edit
    double start_clear = 0.0;   ///< true clearance of the start pose to the nearest wall (m); the bumper refuses motion below body_radius 0.23
};

inline NewRoomRun run_newroom(const Poly& room, unsigned seed, const std::vector<Eigen::Vector2f>& wp,
                              int max_frames, int start_rot, float start_frac)
{
    NewRoomRun R;
    R.truth_verts = static_cast<int>(room.size());
    const auto t_wall0 = std::chrono::steady_clock::now();

    // ── the start (same mechanics as run_boxes: WS_BOXES_START_ROT / WS_BOXES_START_FRAC) ───────────────
    std::vector<Eigen::Vector2f> wp_use = wp;
    if (not wp_use.empty() and start_rot != 0)
    {
        const int n = static_cast<int>(wp_use.size());
        std::rotate(wp_use.begin(), wp_use.begin() + (((start_rot % n) + n) % n), wp_use.end());
    }
    auto truth = circuit(wp_use, 0.025f, 1);   // only its START POSE is used: the explorer drives from there
    if (start_frac > 0.f and not truth.empty())
    {
        const float fr = start_frac - std::floor(start_frac);
        const auto off = static_cast<long>(fr * static_cast<float>(truth.size()));
        if (off > 0 and off < static_cast<long>(truth.size())) std::rotate(truth.begin(), truth.begin() + off, truth.end());
    }
    std::mt19937 rng(seed);
    RunConfig cfg; cfg.exec_motion = true;
    BaseModel base;
    EncoderModel enc; enc.sigma_xy = cfg.odom_sigma_xy; enc.sigma_th = cfg.odom_sigma_th;
    Eigen::Vector3f exec_pose = truth[0];
    const Eigen::Vector3f origin = truth[0];
    R.start_clear = point_to_poly(origin.head<2>(), room);
    const rc::boxch::Params bparams;               // only for the bumper's body_radius, shared with the box bench
    const float dt_ms = 1000.f * cfg.dt;

    const auto to_map = [&](const Eigen::Vector3f& p)
    {
        const float c = std::cos(-origin.z()), s = std::sin(-origin.z());
        const Eigen::Vector2f d = p.head<2>() - origin.head<2>();
        return Eigen::Vector3f(c * d.x() - s * d.y(), s * d.x() + c * d.y(), wrap(p.z() - origin.z()));
    };
    const auto wp_to_world = [&](const Eigen::Vector2f& m)
    {
        const float c = std::cos(origin.z()), s = std::sin(origin.z());
        return Eigen::Vector2f(c * m.x() - s * m.y() + origin.x(), s * m.x() + c * m.y() + origin.y());
    };

    // ── the agent's units ────────────────────────────────────────────────────────────────────────────────
    rc::slam::ScanSlam slam;                         // default Params, as RoomConcept::plain_slam_ (room_concept.h:1156)
    slam.set_pose(Eigen::Vector3f::Zero());          // seeded at the initial pose; step() adds each frame's delta
    rc::freespace::Params fp;                        // config.toml [StartupProto]: Cell .05, HalfSpan 20, ErodeM 0, MaxVerts 64
    fp.cell = 0.05f; fp.half_span = 20.f; fp.erode_m = 0.f; fp.max_verts = 64;
    rc::freespace::FreeSpacePolygon free_space(fp);
    rc::refine::LayoutRefiner refiner;               // default Params, as RoomSceneGraph::refiner_
    constexpr float kRobotW = 0.460f, kRobotL = 0.480f;                       // room_config.h ROBOT_WIDTH / ROBOT_LENGTH
    const float body_circ = 0.5f * std::hypot(kRobotW, kRobotL);
    const float footprint_area = static_cast<float>(M_PI) * 0.25f * (kRobotW * kRobotW + kRobotL * kRobotL);   // room_scene_graph_explored.cpp:56
    const Eigen::Vector2f lidar_origin(0.f, 0.f);    // DEVIATION: the sim LiDAR sits at the body centre (live: 0.155 m offset)

    // ── RoomSceneGraph state (room_scene_graph.h) ────────────────────────────────────────────────────────
    std::vector<Eigen::Vector2f> fs_published, fs_free_ring, refine_contested;
    std::deque<std::vector<Eigen::Vector2f>> layout_pool;
    std::optional<std::vector<Eigen::Vector2f>> map_layout;      // room_concept's set_map_layout(): the layout the outline gave
    bool startup_promoted = false, layout_finished = false;
    std::int64_t fs_last_ms = 0, refine_last_ms = 0, layout_plan_ms = -1000000, layout_stall_ms = 0;
    std::optional<Eigen::Vector2f> layout_target;
    std::vector<std::pair<Eigen::Vector2f, std::int64_t>> layout_refused;
    Eigen::Vector2f layout_stall_anchor = Eigen::Vector2f::Zero();
    std::int64_t armed_at_ms = 0;
    int layout_visits = 0;
    rc::layout::Plan last_plan;
    bool refine_adopted_last = true;             // the last refiner evaluation adopted an edit (or none ran yet)
    bool refine_ran = false;

    // ── the agent's controller chain, as run_boxes (WS_TC_PLAIN arm: the agent's route pipeline) ─────────
    rc::TrajectoryController tcontrol;
    tcontrol.set_footprint(rc::RobotFootprint::shadow());
    const rc::RobotFootprint body_fp = rc::RobotFootprint::shadow();
    rc::RouteSpline route_spline;
    tcontrol.params.pd_cross_track_gain = 1.4f;
    bool have_route = false, route_done = false, blocked_prev = false;
    std::vector<Eigen::Vector3f> cloud_room_prev;

    std::vector<Eigen::Vector2f> traj_seen;
    std::vector<double> perr, rerr;
    bool left_state = false;

    // The belief the route is optimised against: the polygon the controller is handed (confirmed free space,
    // else the published outline), signed distance, inside positive. NEVER the truth.
    const auto belief_dist = [&](const Eigen::Vector2f& wpt)
    {
        const Eigen::Vector2f m = to_map(Eigen::Vector3f(wpt.x(), wpt.y(), 0.f)).head<2>();
        const auto& ring = fs_free_ring.size() >= 3 ? fs_free_ring : fs_published;
        if (ring.size() < 3) return 0.f;
        const float d = point_to_poly(m, ring);
        return rc::freespace::point_in_polygon(m, ring) ? d : -d;
    };

    for (int f = 0; f < max_frames; ++f)
    {
        const std::int64_t now = static_cast<std::int64_t>(f) * static_cast<std::int64_t>(dt_ms);   // simulated "steady_ms"
        Eigen::Vector3f cmd = Eigen::Vector3f::Zero();
        if (f > 0 and have_route)
        {
            Eigen::Affine2f rp = Eigen::Affine2f::Identity();
            rp.translation() = exec_pose.head<2>();
            rp.linear() = Eigen::Rotation2Df(exec_pose.z() - static_cast<float>(M_PI_2)).toRotationMatrix();   // agent theta: forward at +90 deg
            const rc::ControlOutput out = tcontrol.compute(rp, cloud_room_prev);
            if (out.goal_reached) route_done = true;
            cmd = Eigen::Vector3f(out.adv, out.side, -out.rot);                                                // FRAME conversion (2/2)
        }
        Eigen::Vector3f bodyd = (f == 0) ? Eigen::Vector3f::Zero() : base.execute(cmd, cfg.dt);

        // ── the bumper, verbatim from run_boxes (physics: a translation that would penetrate is refused) ──
        bool blocked = false, probe_in = true;
        if (f > 0)
        {
            const Eigen::Vector3f probe = compose(exec_pose, bodyd);
            probe_in = rc::corner_visibility::point_in_polygon(probe.head<2>(), room);
            const float clr = point_to_poly(probe.head<2>(), room);
            if ((not probe_in) or clr < bparams.body_radius)
            {
                blocked = true;
                bodyd.x() = 0.f; bodyd.y() = 0.f;
                base.vel.x() = 0.f; base.vel.y() = 0.f;
            }
        }
        exec_pose = compose(exec_pose, bodyd);
        blocked_prev = blocked;
        if (blocked) ++R.frames_blocked;
        // VALIDITY: the executed pose, tested against the TRUE polygon (centre-outside; the bumper makes this a rare event)
        if (not rc::corner_visibility::point_in_polygon(exec_pose.head<2>(), room)) { ++R.frames_out; left_state = true; }
        const Eigen::Vector3f meas = (f == 0) ? Eigen::Vector3f::Zero() : enc.measure(bodyd, rng);

        const auto pts = scan(room, exec_pose, cfg.n_rays, cfg.scan_sigma, rng);
        R.frames = f + 1;
        if (pts.size() < 20) continue;
        cloud_room_prev.clear(); cloud_room_prev.reserve(pts.size());
        {
            const float cw = std::cos(exec_pose.z()), sw = std::sin(exec_pose.z());
            for (const auto& q : pts)
                cloud_room_prev.emplace_back(cw * q.x() - sw * q.y() + exec_pose.x(), sw * q.x() + cw * q.y() + exec_pose.y(), 0.9f);
        }

        // ── POSE: plain scan-to-map SLAM (room_concept_update.cpp:358-364). The odometry delta is the encoder
        //    reading rotated into the map frame by the SLAM heading, as run_boxes integrates `est`.
        const float cs = std::cos(slam.pose().z()), sn = std::sin(slam.pose().z());
        const Eigen::Vector3f odom_delta(cs * meas.x() - sn * meas.y(), sn * meas.x() + cs * meas.y(), meas.z());
        const Eigen::Vector3f spose = slam.step(odom_delta, pts);
        Eigen::Affine2f pose_pub = Eigen::Affine2f::Identity();     // T_pub_int_ = identity: the published frame IS the initial pose
        pose_pub.translation() = spose.head<2>();
        pose_pub.linear() = Eigen::Rotation2Df(spose.z()).toRotationMatrix();
        const Eigen::Vector2f robot = spose.head<2>();

        const Eigen::Vector3f tm = to_map(exec_pose);
        perr.push_back((spose.head<2>() - tm.head<2>()).norm());
        static const bool trace = std::getenv("WS_NR_TRACE") != nullptr;   // diagnostics only
        if (trace and f % 100 == 0)
            std::fprintf(stderr, "[nr] f=%d true=(%.2f,%.2f,%.2f) slam=(%.2f,%.2f) target=%s route=%d blk=%d promoted=%d fin=%d why=%s pub=%zu\n",
                         f, exec_pose.x(), exec_pose.y(), exec_pose.z(), spose.x(), spose.y(),
                         layout_target ? (std::to_string(layout_target->x()) + "," + std::to_string(layout_target->y())).c_str() : "none",
                         have_route ? 1 : 0, R.frames_blocked, startup_promoted ? 1 : 0, layout_finished ? 1 : 0,
                         last_plan.why.c_str(), fs_published.size());
        rerr.push_back(std::abs(wrap(spose.z() - tm.z())) * 180.0 / M_PI);
        traj_seen.push_back(exec_pose.head<2>());
        if (f % 12 == 0) R.traj.push_back(exec_pose.head<2>());

        std::vector<Eigen::Vector2f> ends; ends.reserve(pts.size());
        for (const auto& p : pts) if (p.allFinite()) ends.push_back(pose_pub * p);

        // ═════ step_startup_proto (room_scene_graph.cpp:1945) / step_layout_refiner (:4528) ══════════════
        if (not startup_promoted)
        {
            // :1964 — promotion = the layout is publishable AND the explorer has nothing left worth visiting
            if (map_layout.has_value() and layout_finished)
            {
                // adopt_most_probable_layout() (:2125-2158)
                if (fs_published.size() >= 4 and layout_pool.size() >= 2)
                {
                    std::vector<Eigen::Vector2f> surf, fr; std::vector<float> cs_, cf_;
                    free_space.evidence_cells(surf, fr, &cs_, &cf_);
                    const std::vector<std::vector<Eigen::Vector2f>> cands(layout_pool.begin(), layout_pool.end());
                    const auto& latest = *map_layout;      // published_polygon() = room_concept's map layout = the pool's back
                    const auto dl = refiner.description_lengths(cands, latest, surf, fr, cs_, cf_);
                    const auto best = static_cast<std::size_t>(std::ranges::distance(dl.begin(), std::ranges::min_element(dl)));
                    R.dl_best = dl[best]; R.dl_latest = dl.back();
                    if (best + 1 != cands.size() and std::isfinite(dl[best])) map_layout = cands[best];
                }
                startup_promoted = true;
                R.promoted = true; R.promo_frame = f;
                R.iou_promo = polygon_iou(to_world(*map_layout, origin), room); R.verts_promo = static_cast<int>(map_layout->size());
                fs_published = *map_layout;                // promote_startup_proto: the learnt polygon is the published one
                refine_last_ms = now;                      // :2196
                layout_target.reset(); layout_refused.clear(); armed_at_ms = 0;
                have_route = false; tcontrol.stop();
                layout_plan_ms = -1000000;
            }
            else
            {
                // accumulate, every update (:1976-1990)
                free_space.add_scan(pose_pub * lidar_origin, ends);
                free_space.add_footprint(pose_pub.translation(), body_circ + free_space.params().erode_m);
                // publish, throttled (:1996-2000)
                if (now - fs_last_ms >= 1000)             // StartupProto.PublishPeriodMs = 1000
                {
                    fs_last_ms = now;
                    if (auto fr = free_space.polygon(pose_pub.translation()); not fr.empty()) fs_free_ring = std::move(fr);
                    std::vector<Eigen::Vector2f> ring;
                    // :2012 — the Manhattan outline of the SLAM map IS the layout
                    if (auto lay = free_space.manhattan_layout(pose_pub.translation(), 0.18f);
                        lay.size() >= 4 and rc::freespace::point_in_polygon(pose_pub.translation(), lay))
                    {
                        map_layout = lay;                  // room_concept_->set_map_layout (identity frame)
                        constexpr std::size_t kLayoutPoolSize = 60;
                        std::erase(layout_pool, lay);
                        layout_pool.push_back(lay);
                        if (layout_pool.size() > kLayoutPoolSize) layout_pool.pop_front();
                        R.pool_max = std::max<int>(R.pool_max, static_cast<int>(layout_pool.size()));
                        ring = std::move(lay);
                    }
                    if (ring.empty()) ring = fs_free_ring;
                    if (not ring.empty() and ring != fs_published)
                    {
                        fs_published = std::move(ring);
                        ++R.publishes;
                        if (layout_target.has_value() and not rc::freespace::point_in_polygon(*layout_target, fs_published))
                            layout_target.reset();         // :2113 a held target the new ring no longer contains is dropped
                    }
                }
            }
        }
        else
        {
            // step_layout_refiner (:4528): accumulate at the shadow SLAM pose, evaluate every 5 s
            free_space.add_scan(pose_pub * lidar_origin, ends);
            if (now - refine_last_ms >= 5000 and fs_published.size() >= 4)
            {
                refine_last_ms = now;
                std::vector<Eigen::Vector2f> surf, fr; std::vector<float> cs_, cf_;
                free_space.evidence_cells(surf, fr, &cs_, &cf_);
                const auto Rr = refiner.propose(fs_published, surf, fr, cs_, cf_);
                refine_contested = Rr.contested;
                refine_ran = true;
                refine_adopted_last = not Rr.edits.empty();
                if (not Rr.edits.empty())
                {
                    R.edits_adopted += static_cast<int>(Rr.edits.size());
                    fs_published = Rr.polygon;             // refine_given_layout + the published polygon
                    map_layout = Rr.polygon;
                }
            }
        }

        // ═════ dsr_update_layout_affordance (room_scene_graph.cpp:2254) ═════════════════════════════════
        // DEVIATION (claim protocol): live, the controller CLAIMS the offered standpoint, executes it and the
        // manager latches a completion. Here the claim is immediate, and the same three outcomes are derived from
        // the sim: goal_reached = completion (:2287-2300 / :2336-2345), the bumper refusing the translation =
        // the controller's refusal / OutsideRoom (refuse_held), no 0.1 m progress in EXEC_STALL_TIMEOUT_S = the
        // stall watchdog (:2318-2329), 45 s = the lease (:2333).
        if (fs_published.size() < 3) continue;
        constexpr std::int64_t kExecutionLeaseMs = 45000, kRefusedMs = 30000;
        constexpr float kStallTimeoutS = 25.f, kStallProgressM = 0.10f;
        std::erase_if(layout_refused, [now](const auto& r) { return r.second <= now; });
        bool replan = not layout_target.has_value();
        const auto refuse_held = [&]()
        {
            if (layout_target.has_value()) { layout_refused.emplace_back(*layout_target, now + kRefusedMs); ++R.refused; }
            layout_target.reset(); layout_stall_ms = 0; have_route = false; tcontrol.stop(); replan = true;
        };
        if (layout_target.has_value())
        {
            if (route_done and have_route)           // the afford_room COMPLETED level (:2336)
            {
                ++layout_visits; ++R.visits;
                layout_refused.emplace_back(*layout_target, now + kRefusedMs);   // a visited standpoint is set aside like a refused one (:2342)
                layout_target.reset(); layout_stall_ms = 0; have_route = false; replan = true;
            }
            else if (blocked_prev) refuse_held();
            else
            {
                if (layout_stall_ms == 0 or (robot - layout_stall_anchor).norm() > kStallProgressM)
                { layout_stall_anchor = robot; layout_stall_ms = now; }
                else if (now - layout_stall_ms > static_cast<std::int64_t>(1000.f * kStallTimeoutS)) refuse_held();
                else if (now - armed_at_ms > kExecutionLeaseMs) refuse_held();
            }
        }
        // DEVIATION: live re-plans every update while it holds no target; here at most once a second (CPU only —
        // the plan is a pure function of the maps, which move far slower than that).
        if (replan and not layout_target.has_value() and now - layout_plan_ms < 1000) replan = false;
        if (replan and startup_promoted and refine_contested.empty())
        {   // :2381 — nothing contested, nothing to test: the explorer has nothing left to do after promotion
            layout_plan_ms = now;
            last_plan = {}; last_plan.finished = true; last_plan.why = "nothing_contested";
            if (refine_ran and not refine_adopted_last) { R.stop = "refined"; break; }
            continue;
        }
        if (replan)
        {
            rc::layout::ExplorerParams ep;
            ep.clearance = 0.625f;                         // config.toml StartupProto.ExplorerClearance
            const auto& feasible = fs_free_ring.size() >= 3 ? fs_free_ring : fs_published;   // :2400
            const float path_clear = body_circ + 0.05f + 0.025f;
            rc::layout::FeasibleGrid grid(feasible, ep.cell, std::min(path_clear, ep.clearance));
            // (no block_if: this bench has no low-obstacle layer, so matter_within is false everywhere)
            grid.free_if([&](const Eigen::Vector2f& c) { return (c - robot).norm() <= path_clear; });
            constexpr float kRefusedRadiusM = 0.75f;
            const auto refused_now = [&](const Eigen::Vector2f& v)
            { return std::ranges::any_of(layout_refused, [&](const auto& r) { return (r.first - v).norm() < kRefusedRadiusM; }); };
            const auto standable = [&](const Eigen::Vector2f& v)
            {
                if ((v - robot).norm() < 0.25f + body_circ) return false;
                if (not rc::freespace::point_in_polygon(v, fs_published)) return false;
                float dmin = std::numeric_limits<float>::infinity();
                for (std::size_t i = 0; i < fs_published.size(); ++i)
                {
                    const Eigen::Vector2f a = fs_published[i], b = fs_published[(i + 1) % fs_published.size()];
                    const Eigen::Vector2f ab = b - a;
                    const float t = std::clamp((v - a).dot(ab) / std::max(ab.squaredNorm(), 1e-9f), 0.f, 1.f);
                    dmin = std::min(dmin, (v - (a + t * ab)).norm());
                }
                return dmin >= ep.clearance;
            };
            const auto deferred = [&](const Eigen::Vector2f& v) { return refused_now(v); };
            if (startup_promoted and not refine_contested.empty())
            {   // :2478-2490 — the testing behaviour: gain = contested area a scan from the standpoint would see
                const auto mask = free_space.cells_mask(refine_contested);
                const auto& Rl = refiner.last();
                const float cell2 = free_space.params().cell * free_space.params().cell;
                const float min_gain = (Rl.kappa + std::log(float(std::max<std::size_t>(4, fs_published.size())))) / std::max(Rl.w, 1e-3f) * cell2;
                last_plan = rc::layout::plan_coverage(grid, robot, ep, standable,
                    [&](const Eigen::Vector2f& v) { return free_space.visible_area(v, 8.f, 180, mask); }, min_gain, deferred);
            }
            else
            {   // :2456-2475 — the survey: unseen cells inside/bordering the outline or the free ring
                auto countable = free_space.countable_mask(fs_published, 0.40f);
                if (fs_free_ring.size() >= 3)
                {
                    const auto ring_mask = free_space.countable_mask(fs_free_ring, 0.40f, 8.f);
                    for (std::size_t k = 0; k < countable.size() and k < ring_mask.size(); ++k)
                        countable[k] = countable[k] or ring_mask[k];
                }
                last_plan = rc::layout::plan_coverage(grid, robot, ep, standable,
                    [&](const Eigen::Vector2f& v) { return free_space.visible_unknown_area(v, 8.f, 180, &countable); },
                    footprint_area, deferred);
            }
            rc::layout::boxed_in_handoff(last_plan, layout_visits);
            if (last_plan.finished
                or (layout_visits > 0 and map_layout.has_value() and (last_plan.why == "no_gain" or last_plan.why == "unreachable")))
                if (not startup_promoted) { layout_finished = true; R.explorer_finished = true; }
            layout_plan_ms = now;
            ++R.replans;
            if (last_plan.ok)
            {
                layout_target = last_plan.target;
                layout_stall_ms = 0; armed_at_ms = now;
                // hand the target to the controller: the explorer's A* cells, from the robot, as run_boxes hands plan_path()
                std::vector<Eigen::Vector2f> room_path;
                room_path.push_back(exec_pose.head<2>());
                for (const auto& w : last_plan.path) room_path.push_back(wp_to_world(w));
                if (last_plan.path.empty() or (last_plan.path.back() - last_plan.target).norm() > 1e-3f)
                    room_path.push_back(wp_to_world(last_plan.target));
                if (room_path.size() >= 2)
                {
                    rc::RouteOptimizerConfig opt;
                    opt.enabled = true;
                    opt.distance = belief_dist;
                    opt.distance_gradient = [&](const Eigen::Vector2f& w)
                    {
                        const float e = 0.01f;
                        return Eigen::Vector2f((belief_dist(w + Eigen::Vector2f(e, 0.f)) - belief_dist(w - Eigen::Vector2f(e, 0.f))) / (2.f * e),
                                               (belief_dist(w + Eigen::Vector2f(0.f, e)) - belief_dist(w - Eigen::Vector2f(0.f, e))) / (2.f * e));
                    };
                    opt.d_target = body_fp.circumscribed_radius() + 0.6f;
                    opt.rho = 0.7f * 0.7f / 1.0f;
                    opt.sigma_a = 0.30f;
                    opt.support_radius = [bf = body_fp](float heading_yaw, const Eigen::Vector2f& dir) { return bf.support_radius_yaw(heading_yaw, dir); };
                    opt.clearance_floor = body_fp.inscribed_radius();
                    opt.iterations = 30;
                    opt.safety_bias = 0.75f;
                    opt.w_jerk = 0.5f;
                    const auto is_free = [&](const Eigen::Vector2f& w, float heading)
                    {
                        const float d = belief_dist(w);
                        const Eigen::Vector2f g = opt.distance_gradient(w);
                        const Eigen::Vector2f toward = g.norm() > 1e-6f ? Eigen::Vector2f(-g.normalized()) : Eigen::Vector2f(1.f, 0.f);
                        return d >= body_fp.support_radius_yaw(heading, toward);
                    };
                    bool curved = false;
                    if (route_spline.build(room_path, 0.05f, is_free, 0.40f, &opt))
                    {
                        tcontrol.set_path_presmoothed(route_spline.samples());
                        tcontrol.set_route(&route_spline, /*force_reset=*/true);
                        curved = true;
                    }
                    if (not curved) { tcontrol.set_path(room_path); tcontrol.set_route(nullptr, true); }
                    have_route = true; route_done = false;
                }
            }
            else
            {
                layout_target.reset();
                if (have_route) { have_route = false; tcontrol.stop(); }
                // a promoted room whose contested cells are unreachable/finished: the refiner has nothing more to test
                if (startup_promoted and refine_ran and not refine_adopted_last and (last_plan.finished or last_plan.why == "no_gain"
                                                                                      or last_plan.why == "unreachable"))
                { R.stop = "refined"; break; }
            }
        }
    }
    R.left_room = left_state;
    R.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wall0).count();

    // ── score: the PUBLISHED outline, whatever its state (a never-promoted run is scored as such and flagged) ──
    R.est_verts = static_cast<int>(fs_published.size());
    R.exact = R.est_verts == R.truth_verts;
    auto med = [](std::vector<double> x) { if (x.empty()) return -1.0; std::sort(x.begin(), x.end()); return x[x.size() / 2]; };
    R.pose_med = med(perr); R.rot_err_deg = med(rerr);
    if (fs_published.size() >= 3)
    {
        R.est_poly = to_world(fs_published, origin);
        R.iou = polygon_iou(R.est_poly, room);
        for (int k = 0; k < 4; ++k)
        {
            const float a4 = static_cast<float>(k) * kPi * 0.5f;
            const float ca = std::cos(a4), sa = std::sin(a4);
            Poly rot;
            for (const auto& v : fs_published) rot.push_back({ca * v.x() - sa * v.y(), sa * v.x() + ca * v.y()});
            R.iou_rot4 = std::max(R.iou_rot4, polygon_iou(to_world(rot, origin), room));
        }
    }
    // ── boundary actually observed: verbatim copy of run_boxes' block (wall_slam_selftest.cpp "boundary actually observed") ──
    {
        const auto inside = [&](const Eigen::Vector2f& p)
        {
            bool in = false;
            for (size_t i = 0, j = room.size() - 1; i < room.size(); j = i++)
                if ((room[i].y() > p.y()) != (room[j].y() > p.y())
                    and p.x() < (room[j].x() - room[i].x()) * (p.y() - room[i].y()) / (room[j].y() - room[i].y()) + room[i].x())
                    in = not in;
            return in;
        };
        const auto blocked_seg = [&](const Eigen::Vector2f& a, const Eigen::Vector2f& b)
        {
            for (size_t e = 0; e < room.size(); ++e)
                if (segments_cross(a, b, room[e], room[(e + 1) % room.size()])) return true;
            return false;
        };
        double tot = 0.0, seen = 0.0;
        for (size_t e = 0; e < room.size(); ++e)
        {
            const Eigen::Vector2f a = room[e], b = room[(e + 1) % room.size()];
            const float len = (b - a).norm();
            if (len < 1e-4f) continue;
            const int ns = std::max(2, static_cast<int>(len / 0.10f));
            Eigen::Vector2f nrm(-(b - a).y() / len, (b - a).x() / len);
            const Eigen::Vector2f mid = 0.5f * (a + b);
            if (not inside(mid + 1e-3f * nrm)) nrm = -nrm;
            int ok = 0;
            for (int k = 0; k < ns; ++k)
            {
                const float t = (static_cast<float>(k) + 0.5f) / static_cast<float>(ns);
                const Eigen::Vector2f q = a + t * (b - a) + 1e-3f * nrm;
                for (const auto& o : traj_seen)
                    if ((q - o).norm() <= 15.f and not blocked_seg(o, q)) { ++ok; break; }
            }
            tot += len; seen += len * static_cast<double>(ok) / ns;
        }
        R.seen_frac = tot > 0.0 ? static_cast<float>(seen / tot) : 0.f;
    }
    return R;
}

/// CSV column header + row (decimal POINT: the stream is imbued with the classic locale by the caller).
inline const char* newroom_csv_header()
{
    return "room_id,order,true_verts,iou,iou_rot4,est_verts,exact_vertices,pose_err_med_m,rot_err_med_deg,seen_frac,"
           "frames_to_promotion,promoted,refiner_edits,left_room,frames_outside,frames_blocked,frames_run,stop,"
           "explorer_finished,pool_max,publishes,replans,visits,refused,dl_best,dl_latest,wall_s,seed,style,start_rot,start_frac,start_clearance_m,iou_promo,verts_promo";
}

inline void newroom_csv_row(std::ostream& o, int id, int order, const NewRoomRun& r, unsigned seed, const char* style,
                            int start_rot, float start_frac)
{
    o << id << ',' << order << ',' << r.truth_verts << ',' << r.iou << ',' << r.iou_rot4 << ',' << r.est_verts << ','
      << (r.exact ? 1 : 0) << ',' << r.pose_med << ',' << r.rot_err_deg << ',' << r.seen_frac << ','
      << r.promo_frame << ',' << (r.promoted ? 1 : 0) << ',' << r.edits_adopted << ',' << (r.left_room ? 1 : 0) << ','
      << r.frames_out << ',' << r.frames_blocked << ',' << r.frames << ',' << r.stop << ','
      << (r.explorer_finished ? 1 : 0) << ',' << r.pool_max << ',' << r.publishes << ',' << r.replans << ','
      << r.visits << ',' << r.refused << ',' << r.dl_best << ',' << r.dl_latest << ',' << r.wall_s << ','
      << seed << ',' << style << ',' << start_rot << ',' << start_frac << ',' << r.start_clear << ',' << r.iou_promo << ',' << r.verts_promo << '\n';
}
