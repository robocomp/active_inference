// RoomConcept, relocalisation part 2 (split out of room_concept_reloc.cpp at Task R1b to keep both under the
// file-length lint): the mode posterior of a lost search, the door anchor's odometry, and the rival modes.

#include "room_concept.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <QDebug>

namespace rc
{
    void RoomConcept::note_passage_handover(const rc::passage::Passage& p)
    {
        { std::scoped_lock lk(passage_mutex_); rt_.pending_passage = p; }
        belief_reset_requested_.store(true, std::memory_order_relaxed);
    }

    // ── THE MODE POSTERIOR OF A LOST SEARCH (split out of grid_search_initial_pose, Task R1b) ──
    // ★ THE MODES ARE A POSTERIOR, NOT A LIKELIHOOD (2026-10-18). The search weighs each mode by how well it
    //   explains the scan alone. In a symmetric room (live: room_2 promoted as a 4-vertex rectangle) the true
    //   pose and its 180° twin explain it EQUALLY (w = 0.50 each), so the top was a coin toss and the pose
    //   flipped 34 times in 200 s, dragging both doors to the mirrored side. A lost search still has a prior:
    //   P(lost) of being anywhere in the room (uniform over area × 2π) and 1 − P(lost) of being where tracking
    //   left it, under its covariance + the hypothesis' own + the likelihood's basin.
    // ★ AND THE HELD POSE IS ONE OF THE HYPOTHESES (2026-10-21, Task R1). The prior alone did not stop the coin
    //   flip: it only says which mode is NEAR the pose, and when no mode is, every mode gets the same uniform
    //   floor and the twins tie again. Live rtD1g2: at the three moves the incumbent's own σ was 2–7 cm but the
    //   incumbent itself was already 3.7–6 m off (a missed hand-over, R1b), so no mode was near it (w 0.503/0.497,
    //   0.506/0.494, 0.500/0.500 — all three moved). The same tie follows from a covariance inflated to metres
    //   (reloc_selftest I). So KEEP is scored like any mode — the incumbent polished on the search's
    //   own point set, Laplace evidence × the same prior — and the pose moves only when msprt_decide says the
    //   posterior moves it (rc::reloc::msprt_decide: the top is decisive, or the held pose is rejected).
    RoomConcept::ModePosterior RoomConcept::mode_posterior(rc::reloc::Result& res, const std::vector<Eigen::Vector2f>& poly,
                                                           const std::vector<Eigen::Vector2f>& pts2,
                                                           const Eigen::Vector3f& incumbent, const rc::reloc::Params& rp,
                                                           const Eigen::Matrix3f& basin)
    {
        auto chi2_of = [](Eigen::Vector3f d, const Eigen::Matrix3f& S)
        {
            d.z() = std::remainder(d.z(), 2.f * static_cast<float>(M_PI));
            const float c = d.dot(S.ldlt().solve(d));
            return std::isfinite(c) ? c : std::numeric_limits<float>::infinity();
        };
        std::vector<double> log_post;
        int top_idx = 0, keep_idx = -1;
        bool keep_travelled = false;
        const double pl = std::clamp(static_cast<double>(pending_episode_p_lost_), 1e-9, 1.0 - 1e-9);
        double area2 = 0.0;
        for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
            area2 += static_cast<double>(poly[j].x()) * poly[i].y() - static_cast<double>(poly[i].x()) * poly[j].y();
        const double log_uniform = -std::log(std::max(0.5 * std::abs(area2), 1e-6) * 2.0 * M_PI);
        const Eigen::Matrix3f S_inc = current_covariance.allFinite() ? Eigen::Matrix3f(current_covariance + basin) : basin;
        const auto log_prior = [&](const Eigen::Vector3f& pose, const Eigen::Matrix3f& cov)
        {
            Eigen::Vector3f d = pose - incumbent;
            d.z() = std::remainder(d.z(), 2.f * static_cast<float>(M_PI));
            const Eigen::Matrix3f S = S_inc + cov;
            const Eigen::LDLT<Eigen::Matrix3f> ldlt(S);
            const double maha = d.dot(ldlt.solve(d));
            const double logdet = ldlt.vectorD().array().abs().max(1e-30f).log().sum();
            const double log_near = std::isfinite(maha) ? -0.5 * maha - 0.5 * (3.0 * std::log(2.0 * M_PI) + logdet)
                                                        : -std::numeric_limits<double>::infinity();
            // LOST is not KIDNAPPED. With the tracker lost, odometry still carries the door anchor (passage_prior.h),
            // unless the robot was carried off since the hand-over: the hazard h per frame, over the frames since.
            double b = std::log(pl) + log_uniform;
            if (rt_.door_anchor.has_value())
            {
                const double h = std::clamp(static_cast<double>(params.reloc_hazard), 1e-9, 0.25);
                const double p_kid = 1.0 - std::pow(1.0 - h, static_cast<double>(rt_.door_anchor_frames + 1));
                Eigen::Vector3f da = pose - rt_.door_anchor->pose;
                da.z() = std::remainder(da.z(), 2.f * static_cast<float>(M_PI));
                const Eigen::LDLT<Eigen::Matrix3f> la(Eigen::Matrix3f(rt_.door_anchor->cov + basin + cov));
                const double ma = da.dot(la.solve(da));
                const double lda = la.vectorD().array().abs().max(1e-30f).log().sum();
                const double log_door = std::isfinite(ma) ? -0.5 * ma - 0.5 * (3.0 * std::log(2.0 * M_PI) + lda)
                                                          : -std::numeric_limits<double>::infinity();
                const double x1 = std::log1p(-p_kid) + log_door, x2 = std::log(p_kid) + log_uniform;
                const double hx = std::max(x1, x2);
                b = std::log(pl) + hx + std::log(std::exp(x1 - hx) + std::exp(x2 - hx));
            }
            const double a = std::log(1.0 - pl) + log_near;
            const double hi = std::max(a, b);
            return hi + std::log(std::exp(a - hi) + std::exp(b - hi));
        };
        for (const auto& m : res.modes)
            log_post.push_back(static_cast<double>(rc::reloc::log_evidence(m)) + log_prior(m.pose, m.cov));

        // KEEP = the incumbent's own basin: its polish on the SAME points the modes were scored on, provided the
        // polish stayed inside the incumbent's basin. A polish that travelled out of it means the scan itself
        // rejects the held pose (a stale pose metres away polishes into whatever mode is downhill — reloc_selftest F).
        const auto keep = rc::reloc::polish(poly, rc::reloc::polish_points(pts2, rp), incumbent, rp, reloc_raster_.extent());
        Eigen::Matrix3f S_travel = basin + keep.cov;
        if (current_covariance.allFinite()) S_travel += current_covariance;
        const bool keep_alive = keep.pose.allFinite() and std::isfinite(keep.log_lik)
                                and chi2_of(keep.pose - incumbent, S_travel) < rp.chi2_merge;
        // Alive but TRAVELLED: the incumbent's covariance admits its basin's optimum, yet that optimum is further
        // from the point estimate than ℓ's own basin — with an inflated covariance a hypothesis metres wide. Then
        // keeping the hypothesis means taking its optimum, not the stale point (reloc_selftest J).
        keep_travelled = keep_alive and chi2_of(keep.pose - incumbent, basin + keep.cov) >= rp.chi2_merge;
        if (keep_alive)
        {
            for (std::size_t k = 0; k < res.modes.size() and keep_idx < 0; ++k)
                if (chi2_of(res.modes[k].pose - keep.pose, res.modes[k].cov + keep.cov) < rp.chi2_merge)
                    keep_idx = static_cast<int>(k);   // the held pose IS this mode
            if (keep_idx < 0)
            {
                res.modes.push_back(keep);
                log_post.push_back(static_cast<double>(rc::reloc::log_evidence(keep)) + log_prior(keep.pose, keep.cov));
                keep_idx = static_cast<int>(res.modes.size()) - 1;
            }
        }
        const double hi = *std::ranges::max_element(log_post);
        double z = 0.0;
        for (const double l : log_post) z += std::exp(l - hi);
        res.entropy = 0.f;
        float sw2 = 0.f;
        for (std::size_t k = 0; k < res.modes.size(); ++k)
        {
            auto& m = res.modes[k];
            m.log_weight = static_cast<float>(log_post[k] - hi - std::log(z));
            m.weight = std::exp(m.log_weight);
            if (m.weight > 0.f) res.entropy -= m.weight * m.log_weight;
            sw2 += m.weight * m.weight;
        }
        res.ess = sw2 > 0.f ? 1.f / sw2 : 0.f;
        top_idx = static_cast<int>(std::ranges::max_element(log_post) - log_post.begin());
        return {std::move(log_post), top_idx, keep_idx, keep_travelled};
    }

    void RoomConcept::step_door_anchor(const UpdateResult& res)
    {
        // Carry the door anchor by THIS frame's odometry increment: the motion prior's prediction minus the pose it was
        // built on (pred[k] − est[k−1], the publisher's measured-motion identity), in the robot frame of est[k−1], with
        // the selected motion prior's covariance rotated into that frame. Never the tracker's correction: the anchor
        // exists for the frames in which the tracker cannot be trusted.
        if (not rt_.door_anchor.has_value() or not rt_.anchor_prev_valid)
            return;
        const Eigen::Vector3f& b = rt_.anchor_prev_est;
        const float c = std::cos(b.z()), sn = std::sin(b.z());
        const Eigen::Vector2f dw(res.pred_x - b.x(), res.pred_y - b.y());
        const Eigen::Vector3f u(c * dw.x() + sn * dw.y(), -sn * dw.x() + c * dw.y(),
                                std::remainder(res.pred_theta - b.z(), 2.f * static_cast<float>(M_PI)));
        Eigen::Matrix3f Su = Eigen::Matrix3f::Zero();
        if (last_selected_prior_.valid and last_selected_prior_.covariance_eigen.allFinite())
        {
            Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
            R(0, 0) = c; R(0, 1) = sn; R(1, 0) = -sn; R(1, 1) = c;   // world → robot frame of est[k−1]
            Su = R * last_selected_prior_.covariance_eigen * R.transpose();
        }
        if (u.allFinite())
        {
            rc::passage::propagate(*rt_.door_anchor, u, Su);
            ++rt_.door_anchor_frames;
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
        // Wald's SPRT: switch when the accumulated ratio passes log((1−α)/α), α = the hazard. A symmetric twin
        // (ratio ≡ 0) never passes; after a switch the old pose needs the same margin to come back, so noise
        // around a near-tie cannot flip-flop the published pose.
        const double alpha = std::clamp(static_cast<double>(params.reloc_hazard), 1e-9, 0.5);
        const float bound = static_cast<float>(std::log((1.0 - alpha) / alpha));
        // A mode carried out of the room cannot be the robot — and a mode whose ratio has fallen past the LOWER
        // bound −log((1−α)/α) is REJECTED by the same test (Wald's other half; reviewer I-4: a held episode whose
        // rivals were all wrong stayed open for ever, since only leaving the room ever removed one).
        std::erase_if(rivals_, [&](const RivalMode& rv)
            { return not std::isfinite(rv.llr) or rv.llr <= -bound or not rc::reloc::inside_polygon(poly, rv.pose.head<2>()); });
        publish_rival_snapshot();
        if (rivals_.empty())
            return;

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
