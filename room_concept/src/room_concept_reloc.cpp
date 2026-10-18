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
        // ★ THE MODES ARE A POSTERIOR, NOT A LIKELIHOOD (2026-10-18). The search weighs each mode by how well it
        //   explains the scan alone. In a symmetric room (live: room_2 promoted as a 4-vertex rectangle) the true
        //   pose and its 180° twin explain it EQUALLY (w = 0.50 each), so the top was a coin toss and the pose
        //   flipped 34 times in 200 s, dragging both doors to the mirrored side. A lost search still has a prior:
        //   P(lost) of being anywhere in the room (uniform over area × 2π) and 1 − P(lost) of being where tracking
        //   left it, under its covariance + the mode's own + the likelihood's basin. A twin metres away now has
        //   to WIN on the scan; a tie stays put, and a real kidnap (P(lost) → 1, a decisive misfit) still moves.
        if (search_trigger_ == "lost" and not res.modes.empty() and std::isfinite(pending_episode_p_lost_))
        {
            const double pl = std::clamp(static_cast<double>(pending_episode_p_lost_), 1e-9, 1.0 - 1e-9);
            double area2 = 0.0;
            for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
                area2 += static_cast<double>(poly[j].x()) * poly[i].y() - static_cast<double>(poly[i].x()) * poly[j].y();
            const double log_uniform = -std::log(std::max(0.5 * std::abs(area2), 1e-6) * 2.0 * M_PI);
            std::vector<float> ranges;   // the basin's rotational width: σ_sdf over the scan's median range
            ranges.reserve(pts2.size());
            for (const auto& q : pts2) ranges.push_back(q.norm());
            float r_scale = 1.f;
            if (not ranges.empty())
            {
                const auto mid = ranges.begin() + static_cast<std::ptrdiff_t>(ranges.size() / 2);
                std::ranges::nth_element(ranges, mid);
                r_scale = std::max(*mid, rp.sigma_sdf);
            }
            Eigen::Matrix3f basin = Eigen::Matrix3f::Zero();
            basin(0, 0) = basin(1, 1) = rp.sigma_sdf * rp.sigma_sdf;
            basin(2, 2) = (rp.sigma_sdf / r_scale) * (rp.sigma_sdf / r_scale);
            const Eigen::Matrix3f S_inc = current_covariance.allFinite() ? Eigen::Matrix3f(current_covariance + basin) : basin;
            float lw_max = -std::numeric_limits<float>::infinity();
            for (auto& m : res.modes)
            {
                Eigen::Vector3f d = m.pose - incumbent;
                d.z() = std::remainder(d.z(), 2.f * static_cast<float>(M_PI));
                const Eigen::Matrix3f S = S_inc + m.cov;
                const Eigen::LDLT<Eigen::Matrix3f> ldlt(S);
                const double maha = d.dot(ldlt.solve(d));
                const double logdet = ldlt.vectorD().array().abs().max(1e-30f).log().sum();
                const double log_near = std::isfinite(maha) ? -0.5 * maha - 0.5 * (3.0 * std::log(2.0 * M_PI) + logdet)
                                                            : -std::numeric_limits<double>::infinity();
                const double a = std::log(1.0 - pl) + log_near, b = std::log(pl) + log_uniform;
                const double hi = std::max(a, b);
                m.log_weight += static_cast<float>(hi + std::log(std::exp(a - hi) + std::exp(b - hi)));
                lw_max = std::max(lw_max, m.log_weight);
            }
            double z = 0.0;
            for (const auto& m : res.modes) z += std::exp(static_cast<double>(m.log_weight - lw_max));
            res.entropy = 0.f;
            float sw2 = 0.f;
            for (auto& m : res.modes)
            {
                m.log_weight = static_cast<float>(m.log_weight - lw_max - std::log(z));
                m.weight = std::exp(m.log_weight);
                if (m.weight > 0.f) res.entropy -= m.weight * m.log_weight;
                sw2 += m.weight * m.weight;
            }
            res.ess = sw2 > 0.f ? 1.f / sw2 : 0.f;
            std::ranges::sort(res.modes, std::ranges::greater{}, &rc::reloc::Mode::weight);
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
        const auto& top = res.modes.front();
        ep.top_weight = top.weight;
        ep.second_weight = res.modes.size() > 1 ? res.modes[1].weight : 0.f;
        ep.best_x = top.pose.x(); ep.best_y = top.pose.y(); ep.best_theta = top.pose.z();
        ep.best_loss = top.median_abs;
        ep.jump_m   = (top.pose.head<2>() - incumbent.head<2>()).norm();
        ep.jump_rad = std::abs(std::remainder(top.pose.z() - incumbent.z(), 2.f * static_cast<float>(M_PI)));
        ep.success = true;

        // Did the search CONFIRM the pose it was handed? It did when the incumbent belongs to the top mode's BASIN:
        //   (1) polishing the incumbent reaches the top mode — same local optimum, by the χ² merge the search uses
        //       for two seeds, under the pair's combined covariance; and
        //   (2) that polish never left the incumbent's own basin. A basin of ℓ is as wide as the likelihood's noise
        //       scale: σ_sdf in translation, and in rotation σ_sdf / (median range of the scan) — the turn that
        //       moves the scan's typical point by σ_sdf. No new constant: both come from the model and the data.
        // Both halves are needed, each measured in reloc_selftest when it was missing:
        //   • without (1), comparing the raw incumbent against the top mode's single-scan Σ (1–2 cm) read a same-
        //     place re-find a few cm off the window's estimate as a MOVE — 4 no-op relocalisations in one run of H,
        //     each clearing the window and releasing the publisher's clamp;
        //   • without (2), a stale pose METRES away polished all the way into the top mode and "confirmed" it —
        //     F kidnaps left uncorrected (rect 15 → 13 of 15).
        {
            const auto inc_pol = rc::reloc::polish(poly, pts2, incumbent, rp, reloc_raster_.extent());
            auto chi2_of = [](Eigen::Vector3f d, const Eigen::Matrix3f& S)
            {
                d.z() = std::remainder(d.z(), 2.f * static_cast<float>(M_PI));
                const float c = d.dot(S.ldlt().solve(d));
                return std::isfinite(c) ? c : std::numeric_limits<float>::infinity();
            };
            const bool same_optimum = inc_pol.pose.allFinite()
                and chi2_of(top.pose - inc_pol.pose, top.cov + inc_pol.cov) < rp.chi2_merge;

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
            Eigen::Matrix3f basin = Eigen::Matrix3f::Zero();
            basin(0, 0) = basin(1, 1) = rp.sigma_sdf * rp.sigma_sdf;
            basin(2, 2) = (rp.sigma_sdf / r_med) * (rp.sigma_sdf / r_med);
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
        for (std::size_t k = 1; k < res.modes.size(); ++k)
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
        if (params.reloc_enabled and map_guided_checks_allowed() and not searching())
        {
            // Rivals first: they are scored on THIS scan against the pose update() just committed.
            step_rival_modes(lidar_points);

            // s is the worse of the post-fit residual and the prediction error — kept from the old trigger for the
            // reason recorded there: Adam re-solving a chronically wrong prediction every frame makes the post-fit
            // number alone look healthy. Both are MEDIAN |SDF| in metres.
            const float s = std::max(res.sdf_mse, res.pred_sdf_median);
            const float p_lost = update_lost_belief(s, outside_prob());
            if (p_lost >= 0.5f)   // ⚠ the equal-cost Bayes decision, flagged
                relocalise_from_belief(lidar_points, s, p_lost);
        }
        else
        {
            if (not rivals_.empty()) { rivals_.clear(); publish_rival_snapshot(); }
            rival_ref_valid_ = false;
            reset_lost_belief();
        }
    }

    void RoomConcept::publish_rival_snapshot()
    {
        std::vector<rc::reloc::PoseHypothesis> snap;
        snap.reserve(rivals_.size());
        for (const auto& rv : rivals_) snap.push_back({rv.pose, rv.llr});
        std::scoped_lock lk(rival_snapshot_mutex_);
        rival_snapshot_ = std::move(snap);
    }

    void RoomConcept::reset_lost_belief()
    {
        const double h = std::clamp(static_cast<double>(params.reloc_hazard), 1e-9, 0.25);
        belief_ = {1.0 - 2.0 * h, h, h};
        belief_init_ = true;
        lost_logodds_ = static_cast<float>(std::log(h / (1.0 - h)));
    }

    void RoomConcept::set_belief_mismatch()
    {
        const double h = std::clamp(static_cast<double>(params.reloc_hazard), 1e-9, 0.25);
        belief_ = {h, h, 1.0 - 2.0 * h};
        belief_init_ = true;
        lost_logodds_ = static_cast<float>(std::log(h / (1.0 - h)));
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

    float RoomConcept::update_lost_belief(float s, float outside)
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
        const double leave = 1.0 / std::max(1, params.reloc_track_memory_frames);
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

        // LEARN the tracking emission from the frames it explains, weighted by P(TRACKING), forgetting over
        // reloc_track_memory_frames. A lost robot or a changed view stops teaching it.
        const double smin = std::max(params.reloc_lost_s_min, 1e-4f);
        const double smax = std::max(static_cast<double>(params.reloc_lost_s_max), smin * 1.001);
        const double ls = std::log(std::clamp(static_cast<double>(s), smin, smax));
        const double a = belief_[kTrack] / std::max(1, params.reloc_track_memory_frames);
        const double dev = ls - track_mu_ln_s_;
        track_mu_ln_s_  = static_cast<float>(track_mu_ln_s_ + a * dev);
        track_var_ln_s_ = static_cast<float>(std::max(1e-6, track_var_ln_s_ + a * (dev * dev - track_var_ln_s_)));
        return static_cast<float>(belief_[kLost]);
    }

    void RoomConcept::relocalise_from_belief(const std::vector<Eigen::Vector3f>& lidar_points, float s, float p_lost)
    {
        qWarning().noquote() << QString("[reloc] P(lost)=%1 at s=%2 m — mixture search").arg(p_lost, 0, 'f', 3).arg(s, 0, 'f', 3);
        search_trigger_ = "lost";
        pending_episode_p_lost_ = p_lost;
        const bool ran = grid_search_initial_pose(lidar_points, 0.5f, static_cast<float>(M_PI_4));
        pending_episode_p_lost_ = std::numeric_limits<float>::quiet_NaN();
        if (ran and last_search_moved_)
            reset_lost_belief();
        else if (ran and last_search_confirmed_incumbent_ and log_tracking_emission(s) > log_broad_emission(s))
            reset_lost_belief();          // the pose was fine and so is its fit: a false alarm
        else
            set_belief_mismatch();        // nothing explains this view better: not lost, the view changed
    }

    void RoomConcept::step_rival_modes(const std::vector<Eigen::Vector3f>& lidar_points)
    {
        if (rivals_.empty() or model_ == nullptr or not model_->has_state() or lidar_points.empty())
            return;
        const auto st = model_->get_state();
        const Eigen::Vector3f cur(st[2], st[3], st[4]);
        if (not rival_ref_valid_)
        {
            rival_ref_pose_ = cur;
            rival_ref_valid_ = true;
        }
        // Robot-frame motion since the last step is hypothesis-independent: every mode moved the same way
        // relative to itself. Δ = ref⁻¹ ∘ cur, then rival ← rival ∘ Δ.
        const float cr = std::cos(rival_ref_pose_.z()), sr = std::sin(rival_ref_pose_.z());
        const Eigen::Vector2f dw = cur.head<2>() - rival_ref_pose_.head<2>();
        const Eigen::Vector2f d_robot(cr * dw.x() + sr * dw.y(), -sr * dw.x() + cr * dw.y());
        const float dth = std::remainder(cur.z() - rival_ref_pose_.z(), 2.f * static_cast<float>(M_PI));
        for (auto& rv : rivals_)
        {
            const float c = std::cos(rv.pose.z()), s = std::sin(rv.pose.z());
            rv.pose.x() += c * d_robot.x() - s * d_robot.y();
            rv.pose.y() += s * d_robot.x() + c * d_robot.y();
            rv.pose.z()  = std::remainder(rv.pose.z() + dth, 2.f * static_cast<float>(M_PI));
        }
        rival_ref_pose_ = cur;

        const auto poly = current_room_polygon();
        if (poly.size() < 3)
            return;
        const auto rp = reloc_params();
        if (poly != reloc_raster_poly_ or not reloc_raster_.valid())
        {
            reloc_raster_.build(poly, rp);
            reloc_raster_poly_ = poly;
        }
        std::vector<Eigen::Vector2f> pts2;
        const int stride = std::max(1, static_cast<int>(lidar_points.size()) / std::max(1, rp.max_points));
        pts2.reserve(lidar_points.size() / static_cast<std::size_t>(stride) + 1);
        for (std::size_t i = 0; i < lidar_points.size(); i += static_cast<std::size_t>(stride))
            pts2.emplace_back(lidar_points[i].x(), lidar_points[i].y());

        const float extent = reloc_raster_.extent();
        const float ll_cur = rc::reloc::log_likelihood(poly, pts2, cur, rp, extent);
        for (auto& rv : rivals_)
            rv.llr += rc::reloc::log_likelihood(poly, pts2, rv.pose, rp, extent) - ll_cur;
        // A mode carried out of the room cannot be the robot.
        std::erase_if(rivals_, [&](const RivalMode& rv)
            { return not std::isfinite(rv.llr) or not rc::reloc::inside_polygon(poly, rv.pose.head<2>()); });
        publish_rival_snapshot();
        if (rivals_.empty())
            return;

        // Wald's SPRT: switch when the accumulated ratio passes log((1−α)/α), α = the hazard. A symmetric twin
        // (ratio ≡ 0) never passes; after a switch the old pose needs the same margin to come back, so noise
        // around a near-tie cannot flip-flop the published pose.
        const double alpha = std::clamp(static_cast<double>(params.reloc_hazard), 1e-9, 0.5);
        const float bound = static_cast<float>(std::log((1.0 - alpha) / alpha));
        const auto best = std::ranges::max_element(rivals_, {}, &RivalMode::llr);
        if (best->llr <= bound)
            return;
        // Same explain-away as the search: a rival that out-scores the pose but still leaves a misfit tracking
        // would not produce is a warp of an unexplained view, not the robot's pose.
        float rival_med = std::numeric_limits<float>::quiet_NaN();
        rc::reloc::log_likelihood(poly, pts2, best->pose, rp, extent, &rival_med);
        if (not explains_like_tracking(rival_med))
            return;

        SearchEpisode ep;
        ep.ts_ms = last_update_result.timestamp_ms;
        ep.trigger = "mode_switch";
        ep.stage = 10;
        ep.n_lidar = static_cast<int>(lidar_points.size());
        ep.incumbent_x = cur.x(); ep.incumbent_y = cur.y(); ep.incumbent_theta = cur.z();
        ep.best_x = best->pose.x(); ep.best_y = best->pose.y(); ep.best_theta = best->pose.z();
        ep.jump_m = (best->pose.head<2>() - cur.head<2>()).norm();
        ep.jump_rad = std::abs(std::remainder(best->pose.z() - cur.z(), 2.f * static_cast<float>(M_PI)));
        ep.top_weight = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(best->llr))));
        ep.n_modes = static_cast<int>(rivals_.size()) + 1;
        ep.success = true;
        write_search_episode(ep);
        qWarning().noquote() << QString("[reloc] mode switch: rival out-explained the pose by %1 nats (bound %2) — "
                                        "moving %3 m / %4°")
            .arg(best->llr, 0, 'f', 1).arg(bound, 0, 'f', 1).arg(ep.jump_m, 0, 'f', 2).arg(ep.jump_rad * 57.2958f, 0, 'f', 0);

        const Eigen::Vector3f new_pose = best->pose;
        const float llr = best->llr;
        rivals_.erase(best);
        for (auto& rv : rivals_) rv.llr -= llr;          // ratios are now against the new pose
        rivals_.push_back({cur, -llr});                  // the old pose becomes a rival, needing the same margin back

        commit_relocalised_pose(new_pose, current_covariance);
        rival_ref_pose_ = new_pose;
        reset_lost_belief();
        publish_rival_snapshot();
    }

} // namespace rc
