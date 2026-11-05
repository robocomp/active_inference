// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Relocalisation: the mixture search, episodes, the lost belief and rival modes.

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

    void RoomConcept::write_search_episode(const SearchEpisode& e)
    {
        if (not recovery_log_.is_open())
            return;
        recovery_log_ << e.ts_ms << ',' << e.trigger << ',' << e.n_lidar
                      << ',' << e.incumbent_x << ',' << e.incumbent_y << ',' << e.incumbent_theta
                      << ',' << e.incumbent_loss << ',' << e.good_thr
                      << ',' << e.stage
                      << ',' << e.best_x << ',' << e.best_y << ',' << e.best_theta << ',' << e.best_loss
                      << ',' << e.topk_best << ',' << e.topk_worst << ',' << e.ess
                      << ',' << e.cov_xx << ',' << e.cov_yy << ',' << e.cov_tt
                      << ',' << e.n_evals << ',' << e.jump_m << ',' << e.jump_rad
                      << ',' << (e.success ? 1 : 0) << ',' << e.duration_ms
                      << ',' << e.beta << ',' << e.n_points
                      << ',' << e.n_modes << ',' << e.mode_entropy << ',' << e.top_weight << ',' << e.second_weight
                      << ',' << e.n_yaws << ',' << e.n_segments << ',' << e.p_lost << '\n';
        recovery_log_.flush();
    }

    // =========================================================================
    //  Relocalisation: mixture search, lost belief, rival modes (reloc_search.h)
    // =========================================================================
    rc::reloc::Params RoomConcept::reloc_params() const
    {
        rc::reloc::Params rp;
        rp.sigma_sdf         = params.sigma_sdf;
        rp.outlier_frac      = params.reloc_outlier_frac;
        rp.wall_offset_sigma = params.reloc_wall_offset_sigma;
        rp.body_clearance    = params.reloc_body_clearance;
        rp.max_points        = params.reloc_max_points;
        rp.max_modes         = params.reloc_max_modes;
        // The room's open doorways, seated on its boundary — the tracker's per-beam doorway mixture inside ℓ, evaluated
        // per hypothesis (reloc_search.h, Aperture; Task R2). Never while the room is being estimated: there is no room
        // polygon of THIS room to seat them on, and relocalisation is disarmed then anyway.
        // ★ "NOT WHILE A ROOM IS LEARNT" IS searching(), NOT estimating() (2026-11-01, pingpong39 leg 6). estimating() stays
        //   true for the rest of the run after the first crossing (runtime_estimate_ is never cleared), so every search in
        //   a RECALLED room — frozen, map ready, localising — ran with NO doorways: the returns through the door the body
        //   had just come through were scored as room or clutter, a 4.35 m / 179° twin explained them better, and the
        //   search moved there one frame after the recall (tools/reloc_handover_replay: without the aperture it picks that
        //   exact pose; with door_7's, at any openness >= 0.5 and P(shut) up to 0.95, it keeps the pose by the door).
        if (not searching())
            rp.apertures = rc::reloc::reloc_apertures(
                rc::reloc::seated_doors(door_apertures(), rt_.passage, current_room_polygon(), false), params.reloc_hazard);
        rp.beyond_extent     = DoorApertures::kBeyondExtentM;
        // ★ WHILE THE HAND-OVER'S DOOR ANCHOR LIVES, SEARCH THE ENTRY AREA (Params::local_seeds). The robot has just come
        //   through a known door: its pose is the anchor carried by odometry, forward or reversed. Each component seeds the
        //   polish at its mean and at ±1σ, ±2σ of its own heading spread (the yaw is what a wrong placement gets wrong; the
        //   position is the door's, within the polish's basin). The anchor retires by information (step_rivals), and the
        //   room-wide search is back for a kidnap after that. A component of weight 0 (impossible) seeds nothing.
        if (not rt_.door_anchor.empty())
            for (const auto& an : rt_.door_anchor)
            {
                if (not (an.w > 0.f) or not an.pose.allFinite()) continue;
                const float sth = std::sqrt(std::max(0.f, an.cov(2, 2)));
                for (const int k : {-2, -1, 0, 1, 2})
                    rp.local_seeds.emplace_back(an.pose.x(), an.pose.y(),
                                                std::remainder(an.pose.z() + float(k) * sth, 2.f * static_cast<float>(M_PI)));
            }
        return rp;
    }

    std::vector<Eigen::Vector2f> RoomConcept::current_room_polygon() const
    {
        std::vector<Eigen::Vector2f> poly;
        if (model_ == nullptr)
            return poly;
        if (model_->use_polygon and not model_->polygon_vertices.empty())
            poly = model_->polygon_vertices;
        else if (model_->has_state())
        {
            const float hw = model_->half_extents.x(), hh = model_->half_extents.y();
            poly = {{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}};
        }
        return poly;
    }

    bool RoomConcept::grid_search_initial_pose(const std::vector<Eigen::Vector3f>& lidar_points,
                                               float /*grid_resolution*/,
                                               float /*angle_resolution*/)
    {
        // The START-UP search only (bootstrap_initialization_from_lidar, validate_seed_pose): the seeded pose is polished
        // and scored alongside every structural hypothesis, so it stands exactly when it is the top mode's basin. Once the
        // agent runs, the pose is moved by evidence alone (step_rivals); nothing re-enters here.
        if (model_ == nullptr or not model_->has_state() or lidar_points.empty())
            return false;

        SearchEpisode ep;
        ep.ts_ms   = last_update_result.timestamp_ms;
        ep.n_lidar = static_cast<int>(lidar_points.size());
        ep.trigger = search_trigger_;
        ep.stage   = 10;
        struct EpisodeWriter
        {
            RoomConcept* self; SearchEpisode* e;
            ~EpisodeWriter() { self->write_search_episode(*e); }
        } ep_writer{this, &ep};
        struct SearchFlag
        {
            std::atomic<bool>& active;
            std::atomic<std::int64_t>& end_ms;
            SearchFlag(std::atomic<bool>& a, std::atomic<std::int64_t>& e) : active(a), end_ms(e)
            { active.store(true, std::memory_order_relaxed); }
            ~SearchFlag()
            {
                end_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count(), std::memory_order_relaxed);
                active.store(false, std::memory_order_relaxed);
            }
        } search_flag(grid_search_active_, grid_search_end_ms_);

        const auto poly = current_room_polygon();
        if (poly.size() < 3)
        {
            // Same failure the legacy lattice hit as a segfault on 2026-09-12: a degenerate model mid-republish.
            qWarning() << "[reloc] no usable room polygon (" << poly.size() << "vertices) — keeping the pose.";
            return false;
        }
        const auto rp = reloc_params();
        if (poly != reloc_raster_poly_ or not reloc_raster_.valid())
        {
            reloc_raster_.build(poly, rp);
            reloc_raster_poly_ = poly;
        }
        if (not reloc_raster_.valid())
            return false;

        std::vector<Eigen::Vector2f> pts2;
        pts2.reserve(lidar_points.size());
        for (const auto& pt : lidar_points)
            pts2.emplace_back(pt.x(), pt.y());

        const auto st = model_->get_state();
        const Eigen::Vector3f incumbent(st[2], st[3], st[4]);
        float inc_med = std::numeric_limits<float>::quiet_NaN();
        rc::reloc::log_likelihood(poly, pts2, incumbent, rp, reloc_raster_.extent(), &inc_med);
        ep.incumbent_x = incumbent.x(); ep.incumbent_y = incumbent.y(); ep.incumbent_theta = incumbent.z();
        ep.incumbent_loss = inc_med;

        auto res = rc::reloc::search(reloc_raster_, poly, pts2, {incumbent}, rp,
                                     static_cast<std::uint32_t>(ep.ts_ms & 0xffffffff));
        auto chi2_of = [](Eigen::Vector3f d, const Eigen::Matrix3f& S)
        {
            d.z() = std::remainder(d.z(), 2.f * static_cast<float>(M_PI));
            const float c = d.dot(S.ldlt().solve(d));
            return std::isfinite(c) ? c : std::numeric_limits<float>::infinity();
        };
        // A basin of ℓ is as wide as the likelihood's noise scale: σ_sdf in translation, and in rotation σ_sdf / (median
        // range of the scan) — the turn that moves the scan's typical point by σ_sdf. No new constant.
        Eigen::Matrix3f basin = Eigen::Matrix3f::Zero();
        {
            std::vector<float> ranges;
            ranges.reserve(pts2.size());
            for (const auto& q : pts2) ranges.push_back(q.norm());
            float r_med = 1.f;
            if (not ranges.empty())
            {
                const auto mid = ranges.begin() + static_cast<std::ptrdiff_t>(ranges.size() / 2);
                std::ranges::nth_element(ranges, mid);
                r_med = std::max(*mid, rp.sigma_sdf);
            }
            basin(0, 0) = basin(1, 1) = rp.sigma_sdf * rp.sigma_sdf;
            basin(2, 2) = (rp.sigma_sdf / r_med) * (rp.sigma_sdf / r_med);
        }
        ep.n_evals = res.n_evals; ep.duration_ms = res.duration_ms;
        ep.n_modes = static_cast<int>(res.modes.size()); ep.mode_entropy = res.entropy; ep.ess = res.ess;
        ep.n_yaws = res.n_yaws; ep.n_segments = res.n_segments;
        ep.n_points = std::min(rp.max_points, static_cast<int>(pts2.size()));
        if (res.modes.empty())
        {
            qWarning() << "[reloc] the search returned no mode — keeping the pose.";
            return false;
        }
        const auto& top = res.modes.front();   // modes come sorted by weight
        ep.top_weight = top.weight;
        {
            float second = 0.f;
            for (std::size_t k = 1; k < res.modes.size(); ++k) second = std::max(second, res.modes[k].weight);
            ep.second_weight = second;
        }
        ep.best_x = top.pose.x(); ep.best_y = top.pose.y(); ep.best_theta = top.pose.z();
        ep.best_loss = top.median_abs;
        ep.jump_m   = (top.pose.head<2>() - incumbent.head<2>()).norm();
        ep.jump_rad = std::abs(std::remainder(top.pose.z() - incumbent.z(), 2.f * static_cast<float>(M_PI)));
        ep.success = true;

        // Did the search CONFIRM the pose it was handed? It did when the incumbent belongs to the top mode's BASIN:
        //   (1) polishing the incumbent reaches the top mode — same local optimum, by the χ² merge the search uses
        //       for two seeds, under the pair's combined covariance; and
        //   (2) that polish never left the incumbent's own basin (`basin` above).
        // Both halves are needed (measured in reloc_selftest when either was missing): without (1) a same-place re-find a
        // few cm off read as a MOVE; without (2) a stale pose METRES away polished all the way into the top mode.
        const auto inc_pol = rc::reloc::polish(poly, pts2, incumbent, rp, reloc_raster_.extent());
        const bool same_optimum = inc_pol.pose.allFinite()
            and chi2_of(top.pose - inc_pol.pose, top.cov + inc_pol.cov) < rp.chi2_merge;
        Eigen::Matrix3f S_travel = basin + inc_pol.cov;
        if (current_covariance.allFinite()) S_travel += current_covariance;
        const bool stayed_in_basin = inc_pol.pose.allFinite()
            and chi2_of(inc_pol.pose - incumbent, S_travel) < rp.chi2_merge;
        const bool confirmed = same_optimum and stayed_in_basin;

        // Commit the top mode with the SECOND MOMENT of the whole mixture about it: Σ_k w_k (Σ_k + d_k d_kᵀ).
        // One decisive mode ⇒ its own Laplace Σ; a symmetric twin 6 m away ⇒ metres of variance, the honest statement.
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (const auto& m : res.modes)
        {
            Eigen::Vector3f d = m.pose - top.pose;
            d.z() = std::remainder(d.z(), 2.f * static_cast<float>(M_PI));
            cov += m.weight * (m.cov + d * d.transpose());
        }
        ep.cov_xx = cov(0, 0); ep.cov_yy = cov(1, 1); ep.cov_tt = cov(2, 2);

        // A search that CONFIRMED the pose commits nothing: the estimate of that same basin is better than one scan's polish.
        if (not confirmed)
        {
            commit_relocalised_pose(top.pose, cov);
            qWarning().noquote() << QString("[reloc] %1: moved %2 m / %3° to (%4, %5, %6°), w=%7, %8 ms")
                .arg(ep.trigger).arg(ep.jump_m, 0, 'f', 2).arg(ep.jump_rad * 57.2958f, 0, 'f', 0)
                .arg(top.pose.x(), 0, 'f', 2).arg(top.pose.y(), 0, 'f', 2).arg(top.pose.z() * 57.2958f, 0, 'f', 0)
                .arg(top.weight, 0, 'f', 2).arg(res.duration_ms, 0, 'f', 1);
        }
        return true;
    }

    void RoomConcept::commit_relocalised_pose(const Eigen::Vector3f& pose, const Eigen::Matrix3f& cov)
    {
        // A pose jump that is not the late fit's own (a rival swap, a search) makes the ring's stored poses and an in-flight
        // job meaningless: the ring is cleared, the job is dropped at apply by its stale reloc_epoch_ (bumped below).
        if (not committing_own_fit_) { late_ring_.clear(); late_ref_valid_ = false; }
        model_->set_pose(pose);
        model_->has_prediction  = false;
        model_->robot_prev_pose = std::nullopt;
        // ★ THE PREDICTION BASE. build_motion_prior_selection() predicts the next frame from
        //   last_update_result.robot_pose, NOT from the model tensors. Writing only the tensors — what this commit
        //   did until 2026-09-17 — let the very next frame re-predict from the OLD pose and silently undo the
        //   relocalisation; live, 2277 consecutive "successful" searches each committed and were each discarded.
        if (last_update_result.ok)
        {
            last_update_result.robot_pose.translation() = pose.head<2>();
            last_update_result.robot_pose.linear() = Eigen::Rotation2Df(pose.z()).toRotationMatrix();
            last_update_result.state[2] = pose.x();
            last_update_result.state[3] = pose.y();
            last_update_result.state[4] = pose.z();
            last_update_result.covariance = cov;
        }
        smoothed_pose_            = pose;
        has_smoothed_pose_        = true;
        last_good_pose_           = pose;
        last_good_pose_valid_     = true;
        needs_orientation_search_ = false;
        tracking_step_count_      = 0;
        current_covariance        = cov;
        // The window's slots and boundary prior belong to the abandoned trajectory; keeping them would pull the
        // solve straight back to it.
        window_mgr_.clear(); reset_stride_state();
        reloc_epoch_.fetch_add(1, std::memory_order_relaxed);   // release the publisher's clamp for this jump
    }

    // One tape frame per relocalisation_step call: the pose update() just committed, the odometry increment + covariance the
    // PREDICT step used (not differences of estimated poses), and the scan as the relocaliser receives it. The room polygon +
    // doorway apertures go in as a record whenever they change, ahead of the first frame that uses them.
    void RoomConcept::record_scan_tape(const UpdateResult& res, const std::vector<Eigen::Vector3f>& lidar_points)
    {
        if (scan_tape_failed_ or not res.ok) return;
        if (not scan_tape_)
        {
            std::string path = params.scan_tape_path;
            if (const auto k = path.find("{ts}"); k != std::string::npos)
                path.replace(k, 4, std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                                                      std::chrono::system_clock::now().time_since_epoch()).count()));
            scan_tape_ = std::make_unique<rc::tape::Writer>(path);
            if (not scan_tape_->is_open())
            {
                qWarning() << "[ScanTape] cannot open" << QString::fromStdString(path) << "- tape disabled";
                scan_tape_.reset(); scan_tape_failed_ = true; return;
            }
            qInfo() << "[ScanTape] recording to" << QString::fromStdString(path);
        }
        rc::tape::RoomRecord now;
        now.room = room_name();
        now.polygon = current_room_polygon();
        for (const auto& d : door_apertures())
            now.apertures.push_back({d.a, d.b, d.n_out, d.p_open, d.prior_closed, d.half_depth, d.recess, d.sigma_along, d.name});
        if (not tape_has_room_ or rc::tape::room_state_changed(tape_state_, now))
        {
            scan_tape_->write_room(now);
            tape_state_ = std::move(now);
            tape_has_room_ = true;
        }
        rc::tape::Frame fr;
        fr.t_ms = res.timestamp_ms;
        fr.pose = {res.robot_pose.translation().x(), res.robot_pose.translation().y(),
                   std::atan2(res.robot_pose.linear()(1, 0), res.robot_pose.linear()(0, 0))};
        fr.cov = res.covariance;
        fr.odo_delta = get_predictor_delta();
        fr.odo_cov = get_predictor_cov();
        fr.room = tape_state_.room;
        fr.pts_robot.reserve(lidar_points.size());
        for (const auto& q : lidar_points) fr.pts_robot.emplace_back(q.x(), q.y());
        scan_tape_->write(fr);
    }

    void RoomConcept::relocalisation_step(const UpdateResult& res, const std::vector<Eigen::Vector3f>& lidar_points)
    {
        // ONE mechanism: the rival-hypothesis belief (room_concept_reloc_rivals.cpp; docs/superpowers/specs/
        // 2026-11-01-relocaliser-redesign.md). It replaced three that fought (the RecoveryManager, the map-trust relocaliser
        // and the periodic symmetry check) and then the lost-belief HMM + mixture search that succeeded them.
        //
        // ARMED ONLY AGAINST A FIXED LAYOUT: map ready AND not SEARCHING (a given layout, or an estimated one
        // that has been frozen). While SEARCHING, the walls are estimated FROM this pose's own trajectory, so:
        //   • there is no independent map to be lost against — a misfit is as much the map's error as the pose's;
        //   • the solver fixes the gauge by pinning the first window slot to the origin until a boundary prior
        //     exists, so a relocalisation's window clear is not a jump but a reset of the map frame.
        // Measured live 2026-09-17 (Webots Room2, estimate mode, FreezeLayoutWhenPublishable=false): the first
        // search moved the pose 3.25 m to the symmetric twin, the gauge snapped it to exactly (0,0,0) in a map
        // built in the old frame, the search re-fired every frame for 4 s, and the wall map collapsed 4 → 2.
        //
        // map_ready also keeps the older contract: entering SEARCHING for a proto-room clears it, which is how the
        // agent LEAVES relocalisation, and after the room is retired nothing here may write the robot→room pose
        // again (mark_room_rt_not_current's ring must stay the edge's last write).
        if (params.scan_tape_enabled) record_scan_tape(res, lidar_points);
        if (belief_reset_requested_.exchange(false, std::memory_order_relaxed))
        {
            // ★ A HAND-OVER THROUGH A DOOR (note_passage_handover; passage_prior.h). What the crossing knows enters the
            //   model as GEOMETRY, frame by frame: the doorway term tempers the misfit while the body is in the passage
            //   and expires as it leaves; the door anchor is H0's entry prior (step_rivals, spec sec 5).
            {
                std::scoped_lock lk(passage_mutex_);
                rt_.passage = std::exchange(rt_.pending_passage, std::nullopt);
            }
            rt_.door_anchor.clear();
            rt_.door_anchor_frames = 0;
            if (rt_.passage.has_value())
            {
                rt_.door_anchor = rc::passage::anchor_at_handover(*rt_.passage);
                handover_ts_ms_ = res.timestamp_ms;
            }
            clear_rival_belief();   // a new room: the rivals, their epoch, any in-flight proposal (spec sec 5)
        }
        // Disarmed while the room transition is open (crossing a door of this room, or just entered through a face that
        // still lives — set_transition_open): the misfit is the other room seen through the door, not a lost pose.
        if (params.reloc_enabled and map_guided_checks_allowed() and not searching() and not transition_open())
        {
            if (late_pending_.load(std::memory_order_acquire) and not lidar_points.empty())
                if (not late_forced_done_) run_late_recall_fit(lidar_points);   // multi-scan fit (room_concept_reloc_late_fit.cpp)
            step_door_anchor(res);
            step_rivals(res, lidar_points);
        }
        else
            clear_rival_belief();
        // The prediction of the NEXT frame is built on the pose as it stands now (after any commit above): the door
        // anchor's odometry increment is measured against it (step_door_anchor).
        if (model_ != nullptr and model_->has_state())
        {
            const auto st = model_->get_state();
            rt_.anchor_prev_est = Eigen::Vector3f(st[2], st[3], st[4]);
            rt_.anchor_prev_valid = true;
        }
    }

    // ★ THE ADOPTION TEST STARTS FROM THE RECOGNITION EVIDENCE. The question the verdict answers ("is this room K?") is the
    //   one ltsm's recogniser already answered from its own evidence, so its posterior log-odds ln P(K)/P(new) is the PRIOR
    //   of the sequential test, not 0. Starting at 0 discarded that evidence and made the test hostage to the moment of
    //   adoption, which is exactly when transients happen (the pose jump settling, the scan that triggered the swap): three
    //   bounded outlier scans (3 ln ε = −3.6 < −ln 19) rejected a correct K (gn_selftest late_recall (g)).
    //   The COST, stated: a wrong K that was strongly recognised needs proportionally more contrary scans to be undone
    //   (each worth at most ln ε ≈ −1.2 nats, so ≈ (prior + ln 19)/1.2 of them), and one unremarkable scan can Accept a K
    //   whose prior already sits at the bound. The decision is still taken only on a scan (accumulate_late_evidence): the
    //   prior alone never decides. ⚠ the recogniser and this test both read the LiDAR of the same room; they are taken as
    //   independent (different features: layout shape vs the SDF misfit at the relocalised pose), which they are not fully.
    void RoomConcept::seed_reloc_prior(const std::vector<rc::reloc::PriorMode>& modes, double prior_log_odds)
    {
        { std::scoped_lock lk(late_prior_mutex_); late_prior_ = modes; }
        late_log_odds_ = std::isfinite(prior_log_odds) ? prior_log_odds : 0.0;
        late_forced_done_ = false;
        if (late_fit_.valid()) { late_fit_.wait(); late_fit_ = {}; }   // a stale job belongs to the previous proposal
        late_ring_.clear(); late_ref_valid_ = false;   // the commit below moves the pose: relative poses across it are meaningless
        late_verdict_.store(static_cast<int>(modes.empty() ? LateVerdict::None : LateVerdict::Pending), std::memory_order_release);
        late_pending_.store(not modes.empty(), std::memory_order_release);
        belief_reset_requested_.store(true, std::memory_order_relaxed);
        if (modes.empty()) return;
        // The pose moves WITH the frame: the prior's heaviest mean, now (the fit refines it on the following scans).
        if (model_ != nullptr and model_->has_state())
        {
            const auto best = std::ranges::max_element(modes, {}, &rc::reloc::PriorMode::w);
            commit_relocalised_pose(best->mu, best->S);
        }
    }

    void RoomConcept::cancel_late_recall()
    {
        { std::scoped_lock lk(late_prior_mutex_); late_prior_.clear(); }
        late_pending_.store(false, std::memory_order_release);
        late_verdict_.store(static_cast<int>(LateVerdict::None), std::memory_order_release);
    }

    // ── A REMEMBERED LAYOUT AS A HYPOTHESIS (room-transition protocol C2.7) ──────────────────────────────────────
    // The fit is the late adoption's, unchanged: K becomes the GIVEN layout and seed_reloc_prior arms the forced search +
    // the sequential verdict. What is new is that it may LOSE without cost: everything it overwrites in the survey is
    // snapshotted here and put back by end_hypothesis_fit, and the pose the agent PUBLISHES meanwhile is plain SLAM's
    // shadow (it keeps running on its own scan map while a layout is adopted, room_concept_update.cpp), which is exactly
    // the pose the survey was publishing — so a wrong prior (try32: 3.77 m / 89°) never reaches a consumer.
    // The WallMap needs no snapshot: while map_layout_adopted_ it is not observed (wall_slam_observe) and nothing here
    // re-configures it (begin_room_estimation would, which is why a rejection does NOT go through it).
    bool RoomConcept::begin_hypothesis_fit(std::vector<Eigen::Vector2f> K_int, const std::vector<rc::reloc::PriorMode>& modes,
                                           double prior_log_odds)
    {
        if (hypothesis_.has_value() or K_int.size() < 3 or modes.empty() or not plain_slam_driving()
            or model_ == nullptr or not model_->has_state())
            return false;
        HypothesisSnapshot h;
        {
            std::scoped_lock lk(wall_map_mutex_);
            h.map_layout = map_layout_;
            h.derived_polygon = derived_polygon_;
        }
        h.init_polygon = init_polygon_vertices_;
        h.model_polygon = model_->polygon_vertices;
        h.map_layout_adopted = map_layout_adopted_;
        h.map_ready = map_ready_.load();
        h.wall_frozen = wall_frozen_;
        h.wall_reanchored = wall_reanchored_;
        h.init_use_polygon = init_use_polygon_;
        h.belief_reset_requested = belief_reset_requested_.load(std::memory_order_relaxed);
        h.covariance = current_covariance;
        hypothesis_ = std::move(h);
        // The survey's rivals live in P's frame and would switch the pose away from the proposal while it is judged:
        // seed_reloc_prior's belief reset (below) drops them, and the survey's relocaliser starts afresh on a rejection.
        set_map_layout(std::move(K_int));
        adopt_map_layout_as_given();
        seed_reloc_prior(modes, prior_log_odds);
        return true;
    }

    std::optional<rc::publish::SurveyPose> RoomConcept::survey_pose_during_fit() const
    {
        if (not hypothesis_.has_value()) return std::nullopt;
        return rc::publish::SurveyPose{plain_slam_.pose(), plain_slam_pred_, hypothesis_->covariance};
    }

    void RoomConcept::end_hypothesis_fit()
    {
        if (not hypothesis_.has_value()) return;
        cancel_late_recall();
        const auto h = std::move(*hypothesis_);
        hypothesis_.reset();
        {
            std::scoped_lock lk(wall_map_mutex_);
            map_layout_ = h.map_layout;
            derived_polygon_ = h.derived_polygon;
        }
        init_polygon_vertices_ = h.init_polygon;
        corner_detector_.set_model_corners(init_polygon_vertices_);
        if (h.model_polygon.size() >= 3 and model_ != nullptr and model_->has_state())
            model_->update_polygon_vertices(h.model_polygon);
        map_layout_adopted_ = h.map_layout_adopted;
        map_ready_ = h.map_ready;
        wall_frozen_ = h.wall_frozen;
        wall_reanchored_ = h.wall_reanchored;
        init_use_polygon_ = h.init_use_polygon;
        // The pose goes back to the survey's own: plain SLAM's shadow, which never followed K. (A commit clears the window
        // and the prediction base, exactly as the fit's own seeding did; plain SLAM rewrites the newest slot every frame.)
        commit_relocalised_pose(plain_slam_.pose(), h.covariance);
        belief_reset_requested_.store(h.belief_reset_requested, std::memory_order_relaxed);
        qInfo().noquote() << "[room][hypothesis] remembered layout released: the survey is restored (layout) and the pose is plain SLAM's again";
    }

    void RoomConcept::keep_hypothesis_fit() { hypothesis_.reset(); }

    // ★ THE ADOPTION VERDICT ACCUMULATES. One scan during the jump is noisy, so the verdict is the log-odds of "K explains
    //   the scans" summed over scans (the construction of stability_belief.h): per scan the LLR of the tracker's median |SDF|
    //   under the tracking emission vs the broad one — the two densities of the verdict below. NO forgetting: the
    //   window is bounded by model events (the fence / the proposal or room vanishing, RoomSceneGraph::step_proposal).
    //   FLAGGED: ±ln 19 is a SEQUENTIAL test boundary (Wald's SPRT, α = β = 0.05: ln((1−β)/α) = ln 19), not a 0-1-loss
    //   decision: another scan is cheap while a false Undo (or a missed one) is expensive. Weak evidence stays Pending,
    //   which ltsm reads as no objection.
    //   Each scan's likelihood is HEAVY-TAILED: the tracking emission is a one-sided Gaussian in ln s, so a single transient
    //   scan (a person passing, the post-jump settling) would read tens of nats against it. The numerator is therefore the
    //   mixture (1−ε)·tracking + ε·broad, ε = the relocaliser's own clutter fraction (reloc_outlier_frac, the constant ℓ
    //   already uses for returns that fit no wall): one scan's LLR is bounded below by ln ε, with no clamp. NOTE: ε is
    //   reused with a PER-SCAN meaning here (the fraction of whole scans that are transients), where ℓ means a per-RETURN
    //   fraction; the same number stands for both until a per-scan rate is measured.
    //   The sum starts from the recogniser's log-odds of K (seed_reloc_prior), not from 0. And each LLR is
    //   weighted by the view's novelty (step_rivals's view-change weight): repeated identical views are one
    //   observation.
    void RoomConcept::accumulate_late_evidence(float s, float novelty)
    {
        if (not late_pending_.load(std::memory_order_acquire) or not std::isfinite(s)) return;
        // The two densities of the verdict, stateless (no learning: the tracking emission is the prior one). Tracking: a
        // ONE-SIDED Gaussian in ln s (a fit BETTER than tracking's typical one is not evidence against it) centred on
        // ln(RelocTrackSMedian), with the prior spread of ln s plus the same again for a freshly fitted single pose (which
        // carries pose error too). Broad: log-uniform in ln s over [RelocLostSMin, RelocLostSMax].
        const double smin = std::max(params.reloc_lost_s_min, 1e-4f);
        const double smax = std::max(static_cast<double>(params.reloc_lost_s_max), smin * 1.001);
        const double sig2 = static_cast<double>(params.reloc_track_log_sigma) * params.reloc_track_log_sigma;
        const double v    = 2.0 * std::max(sig2, 1e-6);
        const double dev  = std::max(0.0, std::log(std::clamp(static_cast<double>(s), smin, smax))
                                          - std::log(std::max(static_cast<double>(params.reloc_track_s_median), 1e-4)));
        const double lt = -0.5 * std::log(2.0 * M_PI * v) - 0.5 * dev * dev / v;
        const double lb = -std::log(std::log(smax / smin));
        const double eps = std::clamp(static_cast<double>(params.reloc_outlier_frac), 1e-9, 1.0 - 1e-9);   // numerical guard
        const double hi = std::max(lt + std::log1p(-eps), lb + std::log(eps));
        const double log_mix = hi + std::log(std::exp(lt + std::log1p(-eps) - hi) + std::exp(lb + std::log(eps) - hi));
        late_log_odds_ += std::clamp(static_cast<double>(novelty), 0.0, 1.0) * (log_mix - lb);   // novelty is a [0,1] weight
        const double bound = std::log(19.0);   // FLAGGED above
        if (std::abs(late_log_odds_) < bound)
            return;   // undecided: keep reading the next scan
        const bool accepted = late_log_odds_ > 0.0;
        { std::scoped_lock lk(late_prior_mutex_); late_prior_.clear(); }
        late_pending_.store(false, std::memory_order_release);
        late_verdict_.store(static_cast<int>(accepted ? LateVerdict::Accepted : LateVerdict::Rejected), std::memory_order_release);
        qInfo().noquote() << QString("[reloc] late recall: %1 (log-odds %2)")
            .arg(accepted ? "ACCEPTED — K's layout explains the scans" : "REJECTED — no pose in K's layout explains the scans")
            .arg(late_log_odds_, 0, 'f', 1);
    }

} // namespace rc
