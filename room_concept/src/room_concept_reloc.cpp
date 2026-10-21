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
                                               float grid_resolution,
                                               float /*angle_resolution*/)
    {
        last_search_confirmed_incumbent_ = false;
        last_search_moved_ = false;
        if (model_ == nullptr or not model_->has_state() or lidar_points.empty())
            return false;

        SearchEpisode ep;
        ep.ts_ms   = last_update_result.timestamp_ms;
        ep.n_lidar = static_cast<int>(lidar_points.size());
        ep.trigger = search_trigger_;
        ep.stage   = 10;
        ep.p_lost  = pending_episode_p_lost_;
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
        const bool with_prior = search_trigger_ == "lost" and not res.modes.empty() and std::isfinite(pending_episode_p_lost_);

        std::vector<double> log_post;
        int top_idx = res.modes.empty() ? -1 : 0, keep_idx = -1;
        bool keep_travelled = false;   // the held hypothesis' own optimum is off the incumbent by more than ℓ's basin
        if (with_prior)
        {
            auto mp = mode_posterior(res, poly, pts2, incumbent, rp, basin);   // the ★ notes are there
            log_post = std::move(mp.log_post);
            top_idx = mp.top_idx; keep_idx = mp.keep_idx; keep_travelled = mp.keep_travelled;
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
        const auto top = res.modes[static_cast<std::size_t>(top_idx)];
        ep.top_weight = top.weight;
        last_search_top_weight_ = top.weight;
        {
            float second = 0.f;
            for (std::size_t k = 0; k < res.modes.size(); ++k)
                if (static_cast<int>(k) != top_idx) second = std::max(second, res.modes[k].weight);
            ep.second_weight = second;
        }
        ep.best_x = top.pose.x(); ep.best_y = top.pose.y(); ep.best_theta = top.pose.z();
        ep.best_loss = top.median_abs;
        ep.jump_m   = (top.pose.head<2>() - incumbent.head<2>()).norm();
        ep.jump_rad = std::abs(std::remainder(top.pose.z() - incumbent.z(), 2.f * static_cast<float>(M_PI)));
        ep.success = true;

        rc::reloc::Decision decision = rc::reloc::Decision::Move;
        if (with_prior)
        {
            decision = rc::reloc::msprt_decide(log_post, top_idx, keep_idx, static_cast<double>(params.reloc_hazard));
            if (decision == rc::reloc::Decision::Keep and keep_travelled)
                decision = rc::reloc::Decision::Move;   // the held hypothesis IS the top mode; take its optimum
            last_search_confirmed_incumbent_ = decision == rc::reloc::Decision::Keep;
            if (decision == rc::reloc::Decision::Keep)
                last_search_top_weight_ = top.weight;   // the incumbent's own posterior weight — relocalise_from_belief folds it
        }
        else
        {
            // Did the search CONFIRM the pose it was handed? It did when the incumbent belongs to the top mode's BASIN:
            //   (1) polishing the incumbent reaches the top mode — same local optimum, by the χ² merge the search uses
            //       for two seeds, under the pair's combined covariance; and
            //   (2) that polish never left the incumbent's own basin (`basin` above).
            // Both halves are needed, each measured in reloc_selftest when it was missing:
            //   • without (1), comparing the raw incumbent against the top mode's single-scan Σ (1–2 cm) read a same-
            //     place re-find a few cm off the window's estimate as a MOVE — 4 no-op relocalisations in one run of H,
            //     each clearing the window and releasing the publisher's clamp;
            //   • without (2), a stale pose METRES away polished all the way into the top mode and "confirmed" it —
            //     F kidnaps left uncorrected (rect 15 → 13 of 15).
            const auto inc_pol = rc::reloc::polish(poly, pts2, incumbent, rp, reloc_raster_.extent());
            const bool same_optimum = inc_pol.pose.allFinite()
                and chi2_of(top.pose - inc_pol.pose, top.cov + inc_pol.cov) < rp.chi2_merge;
            Eigen::Matrix3f S_travel = basin + inc_pol.cov;
            if (current_covariance.allFinite()) S_travel += current_covariance;
            const bool stayed_in_basin = inc_pol.pose.allFinite()
                and chi2_of(inc_pol.pose - incumbent, S_travel) < rp.chi2_merge;
            last_search_confirmed_incumbent_ = same_optimum and stayed_in_basin;
        }

        // MOVE ONLY IF THE NEW MODE EXPLAINS THE SCAN THE WAY TRACKING DOES. A search always finds SOME pose:
        // with a door open onto the next room, over half the returns lie beyond the walls, and the lattice
        // happily warps them onto other walls — measured in reloc_selftest, 267 of 346 searches under a 200°
        // through-door view moved the pose, median 1.9 m / 88°. After a real kidnap the right pose drops back
        // to a tracking-sized residual; in a changed view NO pose does. The comparison uses the learnt tracking
        // emission against the broad one — the same two densities the lost belief runs on, no new constant.
        last_search_moved_ = false;
        rt_.last_search_held  = false;
        if (not last_search_confirmed_incumbent_ and not explains_like_tracking(top.median_abs))
        {
            ep.success = false;
            rivals_.clear();
            rival_ref_valid_ = false;
            publish_rival_snapshot();
            qWarning().noquote() << QString("[reloc] %1: best mode (%2 m / %3° away) leaves median |SDF| %4 m — "
                                            "no pose explains this view; keeping the pose")
                .arg(ep.trigger).arg(ep.jump_m, 0, 'f', 2).arg(ep.jump_rad * 57.2958f, 0, 'f', 0)
                .arg(top.median_abs, 0, 'f', 3);
            return true;
        }

        // HOLD: no hypothesis is decisive and the held pose is still alive. Nothing is committed; every other mode is
        // carried as a rival with its posterior log-odds against the HELD pose, so the following scans decide (the
        // rival SPRT switches as soon as one of them passes the bound) — and an exact twin, whose odds never move,
        // never takes the pose.
        if (decision == rc::reloc::Decision::Hold)
        {
            ep.success = false;
            rt_.last_search_held = true;
            const auto& held = res.modes[static_cast<std::size_t>(keep_idx)];
            // The held hypothesis carries the whole mixture's spread about it — an undecided twin metres away is metres
            // of variance, which is what the pose honestly is until the rivals decide. Published in BOTH branches
            // (reviewer I-3: the untravelled hold kept a ~0.3 m σ with a twin 3–6 m away at ~50%).
            Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
            for (const auto& m : res.modes)
            {
                Eigen::Vector3f d = m.pose - held.pose;
                d.z() = std::remainder(d.z(), 2.f * static_cast<float>(M_PI));
                cov += m.weight * (m.cov + d * d.transpose());
            }
            ep.cov_xx = cov(0, 0); ep.cov_yy = cov(1, 1); ep.cov_tt = cov(2, 2);
            // Taken at its own optimum when it travelled; otherwise the pose stays and the spread is PUBLISHED on every
            // frame the episode stays open (relocalisation_step → rt_.held_spread → run()), not written into the
            // tracker's covariance once (reviewer: later held frames went back to cm σ while the twin pended).
            if (keep_travelled)
                commit_relocalised_pose(held.pose, cov);
            rivals_.clear();
            const double l_keep = log_post[static_cast<std::size_t>(keep_idx)];
            for (std::size_t k = 0; k < res.modes.size(); ++k)
                if (static_cast<int>(k) != keep_idx)
                    rivals_.push_back({res.modes[k].pose, static_cast<float>(log_post[k] - l_keep)});
            rival_ref_pose_  = keep_travelled ? held.pose : incumbent;
            rival_ref_valid_ = true;
            publish_rival_snapshot();
            qWarning().noquote() << QString("[reloc] %1: HELD — top mode %2 m / %3° away has w=%4, the held pose w=%5: "
                                            "no hypothesis decisive (bound %6 nats); %7 rival(s) carry the episode")
                .arg(ep.trigger).arg(ep.jump_m, 0, 'f', 2).arg(ep.jump_rad * 57.2958f, 0, 'f', 0)
                .arg(top.weight, 0, 'f', 2).arg(res.modes[static_cast<std::size_t>(keep_idx)].weight, 0, 'f', 2)
                .arg(std::log((1.0 - params.reloc_hazard) / params.reloc_hazard), 0, 'f', 1).arg(rivals_.size());
            return true;
        }
        last_search_moved_ = not last_search_confirmed_incumbent_;

        // Commit the top mode with the SECOND MOMENT of the whole mixture about it: Σ_k w_k (Σ_k + d_k d_kᵀ).
        // One decisive mode ⇒ its own Laplace Σ; a symmetric twin 6 m away ⇒ metres of variance, which is the
        // honest statement — the old commit published a fixed Identity()*0.1 on exactly these paths.
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (const auto& m : res.modes)
        {
            Eigen::Vector3f d = m.pose - top.pose;
            d.z() = std::remainder(d.z(), 2.f * static_cast<float>(M_PI));
            cov += m.weight * (m.cov + d * d.transpose());
        }
        ep.cov_xx = cov(0, 0); ep.cov_yy = cov(1, 1); ep.cov_tt = cov(2, 2);

        // A search that CONFIRMED the pose commits nothing: the window's estimate of that same basin, built over
        // many scans, is better than one scan's polish, and writing the polish over it would be a needless jump.
        if (last_search_moved_)
            commit_relocalised_pose(top.pose, cov);

        // The other modes are CARRIED, not discarded: step_rival_modes scores them on every following scan.
        rivals_.clear();
        for (std::size_t k = 0; k < res.modes.size(); ++k)
            if (static_cast<int>(k) != top_idx)
                rivals_.push_back({res.modes[k].pose, res.modes[k].log_weight - top.log_weight});
        rival_ref_pose_  = top.pose;
        rival_ref_valid_ = true;
        publish_rival_snapshot();

        if (not last_search_confirmed_incumbent_)
            qWarning().noquote() << QString("[reloc] %1: moved %2 m / %3° to (%4, %5, %6°), w=%7, %8 rival mode(s), %9 ms")
                .arg(ep.trigger).arg(ep.jump_m, 0, 'f', 2).arg(ep.jump_rad * 57.2958f, 0, 'f', 0)
                .arg(top.pose.x(), 0, 'f', 2).arg(top.pose.y(), 0, 'f', 2).arg(top.pose.z() * 57.2958f, 0, 'f', 0)
                .arg(top.weight, 0, 'f', 2).arg(rivals_.size()).arg(res.duration_ms, 0, 'f', 1);
        return true;
    }

    void RoomConcept::commit_relocalised_pose(const Eigen::Vector3f& pose, const Eigen::Matrix3f& cov)
    {
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

    void RoomConcept::relocalisation_step(const UpdateResult& res, const std::vector<Eigen::Vector3f>& lidar_points)
    {
        // ONE mechanism (see Params, "Relocalisation"; reloc_search.h). It replaces three that fought:
        // the RecoveryManager (10 bad frames + 60-frame cooldown, reset by every symmetry flip), the
        // map-trust relocaliser (never enabled) and the periodic symmetry check, which teleported a LOST
        // robot to a mirror image on 74 of 80 checks (09-12) and on a door OPENING with the robot 1.6 m
        // inside (09-16) — both judged a median, which cannot tell "wrong pose" from "unmodelled view".
        //
        // ARMED ONLY AGAINST A FIXED LAYOUT: map ready AND not SEARCHING (a given layout, or an estimated one
        // that has been frozen). While SEARCHING, the walls are estimated FROM this pose's own trajectory, so:
        //   • there is no independent map to be lost against — a misfit is as much the map's error as the pose's;
        //   • the solver fixes the gauge by pinning the first window slot to the origin until a boundary prior
        //     exists, so a relocalisation's window clear is not a jump but a reset of the map frame.
        // Measured live 2026-09-17 (Webots Room2, estimate mode, FreezeLayoutWhenPublishable=false): the first
        // search moved the pose 3.25 m to the symmetric twin, the gauge snapped it to exactly (0,0,0) in a map
        // built in the old frame, the search re-fired every frame for 4 s, and the wall map collapsed 4 → 2.
        // With the freeze ON, Searching and map-ready never overlap, which is why those runs were clean.
        //
        // map_ready also keeps the older contract: entering SEARCHING for a proto-room clears it, which is how the
        // agent LEAVES relocalisation, and after the room is retired nothing here may write the robot→room pose
        // again (mark_room_rt_not_current's ring must stay the edge's last write).
        if (belief_reset_requested_.exchange(false, std::memory_order_relaxed))
        {
            reset_lost_belief();
            // ★ A HAND-OVER THROUGH A DOOR (note_passage_handover; passage_prior.h). What the crossing knows enters the
            //   model as GEOMETRY, frame by frame: the doorway term tempers the misfit while the body is in the passage
            //   and expires as it leaves; the door anchor is a component of the search's prior. Measured on the plain
            //   reset (rtD1e/f/g2): s 0.28–2.75 m in the first frames (the 360° scan sees the room just left through the
            //   door; door weights / prediction statistic lag the room swap), P(lost) 0.500–0.503 and a search 60–220 ms
            //   after every recall; in rtD1e it moved a pose 0.04 m from the truth 4.40 m / 179° to the twin. (8ce48e5e
            //   seeded MISMATCH with p_cross instead; reviewer I-2: that hid a WRONG recall for a hazard-set ~18 s.)
            {
                std::scoped_lock lk(passage_mutex_);
                rt_.passage = std::exchange(rt_.pending_passage, std::nullopt);
            }
            rt_.door_anchor.clear();
            rt_.door_anchor_frames = 0;
            if (rt_.passage.has_value())
                rt_.door_anchor = rc::passage::anchor_at_handover(*rt_.passage);
            rt_.hold_open = false;
            belief_prev_pose_valid_ = false;
            if (not rivals_.empty()) { rivals_.clear(); publish_rival_snapshot(); }
            rival_ref_valid_ = false;
        }
        if (params.reloc_enabled and map_guided_checks_allowed() and not searching())
        {
            step_door_anchor(res);
            // Rivals first: they are scored on THIS scan against the pose update() just committed.
            step_rival_modes(lidar_points);

            // s is the worse of the post-fit residual and the prediction error — kept from the old trigger for the
            // reason recorded there: Adam re-solving a chronically wrong prediction every frame makes the post-fit
            // number alone look healthy. Both are MEDIAN |SDF| in metres.
            const float s = std::max(res.sdf_mse, res.pred_sdf_median);
            // How new this frame's view is: the pose step since the last frame, measured against what tracking
            // resolves — the learnt typical misfit s_track = exp(E[ln s]) in translation, s_track / (mean range) in
            // rotation. Driving, a frame's step is ~1 of those (ν ≈ 0.4–1); parked, the pose jitters by 0.1 mm
            // (ν ≈ 0). σ_sdf (0.15 m) would be the wrong scale: a driving robot would then read as parked.
            float view_change = 1.f;
            {
                const Eigen::Vector3f pose(res.robot_pose.translation().x(), res.robot_pose.translation().y(),
                                           std::atan2(res.robot_pose.linear()(1, 0), res.robot_pose.linear()(0, 0)));
                if (belief_prev_pose_valid_ and not lidar_points.empty())
                {
                    double r_sum = 0.0;
                    for (const auto& q : lidar_points) r_sum += std::hypot(q.x(), q.y());
                    const double s_track = std::isfinite(track_mu_ln_s_) ? std::exp(static_cast<double>(track_mu_ln_s_))
                                                                         : static_cast<double>(params.reloc_track_s_median);
                    const double sig = std::max(1e-3, s_track);
                    const double r_mean = std::max(sig, r_sum / static_cast<double>(lidar_points.size()));
                    const double dp2 = (pose.head<2>() - belief_prev_pose_.head<2>()).squaredNorm() / (sig * sig);
                    const double dth = std::remainder(pose.z() - belief_prev_pose_.z(), 2.0 * M_PI);
                    const double dt2 = dth * dth / ((sig / r_mean) * (sig / r_mean));
                    view_change = static_cast<float>(1.0 - std::exp(-0.5 * (dp2 + dt2)));
                }
                belief_prev_pose_ = pose;
                belief_prev_pose_valid_ = true;
            }
            // The passage just used is a doorway of THIS room too, whatever the graph's parenting says yet (live: the
            // crossed door hangs from the room left behind for a while, so the scene graph's doorway term read exactly
            // 0 right after every recall). Two independent reasons the room does not owe the view combine as a union.
            float outside = outside_prob();
            rt_.passage_term = 0.f;
            if (rt_.passage.has_value())
            {
                rt_.passage_term = rc::passage::doorway_term(*rt_.passage, res.robot_pose.translation(),
                                                          res.covariance.topLeftCorner<2, 2>());
                outside = 1.f - (1.f - outside) * (1.f - rt_.passage_term);
            }
            const float p_lost = update_lost_belief(s, outside, view_change);
            // A HELD episode (msprt_decide → Hold) is carried by its rivals: step_rival_modes above scores them on
            // every scan and switches the moment one is decisive. It ends when they resolve or empty, or when the
            // belief itself says the episode is over (P(lost) back under ½ — the same decision as the trigger).
            if (rivals_.empty() or p_lost < 0.5f)
                rt_.hold_open = false;
            if (p_lost >= 0.5f and not rt_.hold_open)   // ⚠ the equal-cost Bayes decision, flagged
                relocalise_from_belief(lidar_points, s, p_lost);
            // RETIRE THE DOOR ANCHOR once its window is over: the body has left the passage (doorway term past ½) AND
            // tracking has recovered (P(track) past ½) — the same equal-cost decision as the trigger, both flagged.
            // Its information is about the hand-over; carried for minutes by odometry that over-reports rotation ~41%
            // on Shadow (memory: command-prior-over-rotates-shadow), it would become a biased prior, not a helpful one.
            if (not rt_.door_anchor.empty() and rt_.passage_term < 0.5f and belief_[kTrack] >= 0.5)
                rt_.door_anchor.clear();
            // A HELD episode publishes its mixture spread on EVERY frame it stays open (added to the next published
            // covariance in run()): Σ_k w_k d_k d_kᵀ over the held pose (d = 0) and its rivals, w from the carried odds.
            rt_.held_spread.setZero();
            if (rt_.hold_open and not rivals_.empty() and model_ != nullptr and model_->has_state())
            {
                const auto st = model_->get_state();
                const Eigen::Vector3f cur(st[2], st[3], st[4]);
                double hi = 0.0;
                for (const auto& rv : rivals_) hi = std::max(hi, static_cast<double>(rv.llr));
                double z = std::exp(-hi);
                for (const auto& rv : rivals_) z += std::exp(static_cast<double>(rv.llr) - hi);
                for (const auto& rv : rivals_)
                {
                    Eigen::Vector3f d = rv.pose - cur;
                    d.z() = std::remainder(d.z(), 2.f * static_cast<float>(M_PI));
                    rt_.held_spread += static_cast<float>(std::exp(static_cast<double>(rv.llr) - hi) / z) * (d * d.transpose());
                }
            }
            pose_undecided_.store(belief_[kLost] >= 0.5 or rt_.hold_open, std::memory_order_relaxed);
        }
        else
        {
            if (not rivals_.empty()) { rivals_.clear(); publish_rival_snapshot(); }
            rival_ref_valid_ = false;
            rt_.hold_open = false;
            rt_.held_spread.setZero();
            reset_lost_belief();
            pose_undecided_.store(false, std::memory_order_relaxed);
        }
        // The prediction of the NEXT frame is built on the pose as it stands now (after any commit above): the door
        // anchor's odometry increment is measured against it (step_door_anchor).
        if (model_ != nullptr and model_->has_state())
        {
            const auto st = model_->get_state();
            rt_.anchor_prev_est = Eigen::Vector3f(st[2], st[3], st[4]);
            rt_.anchor_prev_valid = true;
        }
    }

    void RoomConcept::reset_lost_belief()
    {
        const double h = std::clamp(static_cast<double>(params.reloc_hazard), 1e-9, 0.25);
        belief_ = {1.0 - 2.0 * h, h, h};
        belief_init_ = true;
        lost_logodds_ = static_cast<float>(std::log(h / (1.0 - h)));
        p_track_display_.store(static_cast<float>(belief_[kTrack]), std::memory_order_relaxed);
    }

    void RoomConcept::set_belief_mismatch()
    {
        const double h = std::clamp(static_cast<double>(params.reloc_hazard), 1e-9, 0.25);
        belief_ = {h, h, 1.0 - 2.0 * h};
        belief_init_ = true;
        lost_logodds_ = static_cast<float>(std::log(h / (1.0 - h)));
        p_track_display_.store(static_cast<float>(belief_[kTrack]), std::memory_order_relaxed);
    }

    double RoomConcept::log_tracking_emission(float s, double extra_var)
    {
        if (not std::isfinite(track_mu_ln_s_) or not std::isfinite(track_var_ln_s_))
        {
            track_mu_ln_s_  = std::log(std::max(params.reloc_track_s_median, 1e-4f));
            track_var_ln_s_ = params.reloc_track_log_sigma * params.reloc_track_log_sigma;
        }
        const double smin = std::max(params.reloc_lost_s_min, 1e-4f);
        const double smax = std::max(static_cast<double>(params.reloc_lost_s_max), smin * 1.001);
        const double ls = std::log(std::clamp(static_cast<double>(s), smin, smax));
        const double v  = std::max(static_cast<double>(track_var_ln_s_), 1e-6) + std::max(0.0, extra_var);
        // ONE-SIDED. s is a misfit: a fit BETTER than tracking's typical one is not evidence against tracking.
        // A two-sided Gaussian made it so, and once the learnt spread tightened it rejected the true pose after a
        // kidnap for fitting too WELL — measured in reloc_selftest F: 13 of 45 relocalisations refused with the
        // true pose at s = 0.016–0.021 m against a learnt median of 0.023. Below the mode the density is held at
        // its peak (an unnormalised tail of at most a factor 2, which cancels in every ratio this is used for).
        const double dev = std::max(0.0, ls - static_cast<double>(track_mu_ln_s_));
        return -0.5 * std::log(2.0 * M_PI * v) - 0.5 * dev * dev / v;
    }

    double RoomConcept::log_broad_emission(float /*s*/) const
    {
        const double smin = std::max(params.reloc_lost_s_min, 1e-4f);
        const double smax = std::max(static_cast<double>(params.reloc_lost_s_max), smin * 1.001);
        return -std::log(std::log(smax / smin));   // log-uniform density in ln s
    }

    float RoomConcept::update_lost_belief(float s, float outside, float view_change)
    {
        if (not belief_init_)
            reset_lost_belief();
        if (not std::isfinite(s) or s <= 0.f)
            return static_cast<float>(belief_[kLost]);

        // PREDICT. TRACKING leaves for LOST or MISMATCH at the hazard; LOST persists (only a search or the
        // optimiser's own re-convergence ends it, the latter at the hazard); MISMATCH ends as the view returns,
        // on the tracking memory's timescale, and can hide a kidnap at the hazard. P(LOST) never falls below h,
        // so a kidnap after an hour of good tracking needs the same evidence as one after a minute.
        const double h = std::clamp(static_cast<double>(params.reloc_hazard), 1e-9, 0.25);
        // ★ THE VIEW RETURNS ONLY IF IT CAN CHANGE (2026-10-18). MISMATCH → TRACKING is "the view came back", and a
        //   parked robot's view cannot come back — it is the same view. At a constant 1/memory per frame that leak
        //   was refuted by the very next scan (the emission cannot tell LOST from MISMATCH), so the LOST:MISMATCH
        //   odds grew by (1 + leave) every frame and crossed ½ after memory·ln(1 + 1/(memory·h)) = 358 frames —
        //   18.6 s, measured, with the robot still to 0.1 mm: a search every 18.6 s, for 20 minutes, each one
        //   confirming the pose it was handed. view_change ∈ [0,1] is how far this frame's pose moved relative to
        //   the likelihood's own basin (relocalisation_step); parked it is ~0, driving it is ~1. The kidnap hazard
        //   h is NOT scaled: a robot can be picked up while parked, and that must stay detectable.
        const double nu = std::clamp(static_cast<double>(view_change), 0.0, 1.0);
        const double leave = nu / std::max(1, params.reloc_track_memory_frames);
        const auto& b = belief_;
        std::array<double, 3> pr{
            b[kTrack] * (1.0 - 2.0 * h) + b[kLost] * h + b[kMismatch] * leave,
            b[kTrack] * h + b[kLost] * (1.0 - h) + b[kMismatch] * h,
            b[kTrack] * h + b[kMismatch] * (1.0 - leave - h)};

        // UPDATE. Past an open doorway the room no longer owes the scan an explanation, so the misfit stops
        // being evidence: the broad-vs-tracking log-ratio is tempered by P(still inside) — the old semantics.
        const double w = 1.0 - std::clamp(static_cast<double>(outside), 0.0, 1.0);
        const double lt = log_tracking_emission(s);
        const double lb = log_tracking_emission(s) + w * (log_broad_emission(s) - lt);
        const double top = std::max(lt, lb);
        std::array<double, 3> post{pr[kTrack] * std::exp(lt - top), pr[kLost] * std::exp(lb - top),
                                   pr[kMismatch] * std::exp(lb - top)};
        const double z = post[0] + post[1] + post[2];
        if (z > 0.0 and std::isfinite(z))
            for (auto& x : post) x /= z;
        else
            post = pr;
        belief_ = post;
        lost_logodds_ = static_cast<float>(std::log(std::max(belief_[kLost], 1e-300) / std::max(1.0 - belief_[kLost], 1e-300)));
        p_track_display_.store(static_cast<float>(belief_[kTrack]), std::memory_order_relaxed);

        // LEARN the tracking emission from the frames it explains, weighted by P(TRACKING), forgetting over
        // reloc_track_memory_frames. A lost robot or a changed view stops teaching it.
        // ★ AND A REPEATED VIEW TEACHES IT ONCE (2026-10-18): parked, every frame is the same look, and its misfit
        //   is common-mode, not fresh noise. Counted as independent, minutes of identical (clamped) samples drove
        //   track_var_ln_s_ onto its 1e-6 floor, after which one real step in s — a door node deleted under a
        //   parked robot — took P(LOST) from −14.7 log-odds to a search in 1–2 frames (9 searches in 10 s, live).
        //   So the learning rate is weighted by how new the view is, the same view_change as the transition.
        const double smin = std::max(params.reloc_lost_s_min, 1e-4f);
        const double smax = std::max(static_cast<double>(params.reloc_lost_s_max), smin * 1.001);
        const double ls = std::log(std::clamp(static_cast<double>(s), smin, smax));
        const double a = nu * belief_[kTrack] / std::max(1, params.reloc_track_memory_frames);
        const double dev = ls - track_mu_ln_s_;
        track_mu_ln_s_  = static_cast<float>(track_mu_ln_s_ + a * dev);
        track_var_ln_s_ = static_cast<float>(std::max(1e-6, track_var_ln_s_ + a * (dev * dev - track_var_ln_s_)));
        return static_cast<float>(belief_[kLost]);
    }

    void RoomConcept::relocalise_from_belief(const std::vector<Eigen::Vector3f>& lidar_points, float s, float p_lost)
    {
        qWarning().noquote() << QString("[reloc] P(lost)=%1 at s=%2 m — mixture search").arg(p_lost, 0, 'f', 3).arg(s, 0, 'f', 3);
        search_trigger_ = "lost";
        ++rt_.searches;
        pending_episode_p_lost_ = p_lost;
        const bool ran = grid_search_initial_pose(lidar_points, 0.5f, static_cast<float>(M_PI_4));
        pending_episode_p_lost_ = std::numeric_limits<float>::quiet_NaN();
        if (ran and last_search_moved_)
            reset_lost_belief();
        else if (ran and rt_.last_search_held)
        {
            // HELD: the search found hypotheses that explain the view, none decisive, the held pose still alive. The
            // belief keeps P(lost) as it is (the search neither confirmed nor refuted the pose); the open episode is
            // now the rivals' (relocalisation_step does not re-search while it is open).
            rt_.hold_open = true;
        }
        else if (ran and last_search_confirmed_incumbent_)
        {
            // ★ THE SEARCH IS EVIDENCE, NOT A RESET (2026-10-18). It returned the incumbent's basin with posterior
            //   weight w: P(confirm | not lost) = w, P(confirm | lost) = 1 − w. LOST is updated by that ratio and
            //   the rest is split between TRACKING and MISMATCH by this frame's own emission — instead of throwing
            //   the outcome away and restarting from the prior, which is what made the next ramp identical.
            const double w  = std::clamp(static_cast<double>(last_search_top_weight_), 1e-9, 1.0 - 1e-9);
            const double pl = std::clamp(static_cast<double>(p_lost), 1e-12, 1.0 - 1e-12);
            const double h  = std::clamp(static_cast<double>(params.reloc_hazard), 1e-9, 0.25);
            const double l  = std::max(h, pl * (1.0 - w) / (pl * (1.0 - w) + (1.0 - pl) * w));
            const double lt = log_tracking_emission(s), lb = log_broad_emission(s), m = std::max(lt, lb);
            const double qt = std::exp(lt - m) / (std::exp(lt - m) + std::exp(lb - m));
            belief_ = {(1.0 - l) * qt, l, (1.0 - l) * (1.0 - qt)};
            belief_init_ = true;
            p_track_display_.store(static_cast<float>(belief_[kTrack]), std::memory_order_relaxed);
            lost_logodds_ = static_cast<float>(std::log(l / (1.0 - l)));
        }
        else
            set_belief_mismatch();        // nothing explains this view better: not lost, the view changed
    }

} // namespace rc
