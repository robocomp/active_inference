// RoomConcept, relocalisation: the rival-hypothesis belief (docs/superpowers/specs/2026-11-01-relocaliser-redesign.md sec 3-5, 7).
//
// The incumbent H0 is the tracker's own pose and covariance. A rival is a competing pose with ONE number, lambda = ln posterior
// odds against H0 (reloc_rivals.h). A background proposer (reloc_proposer.h) only ever ADDS rivals, priced by their Bayes
// factor (birth_lambda); each novel scan moves lambda by the log likelihood ratio of the SAME robust likelihood the search
// uses (reloc_rival_score.h); a rival that reaches +ln 19 becomes the incumbent. There is no lost belief, no "should I
// search?" state and no explains-like-tracking gate: effort is scheduling, the jump is evidence.

#include "room_concept.h"
#include "reloc_rival_score.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <QDebug>

namespace rc
{
    namespace
    {
        Eigen::Vector3f se2_compose(const Eigen::Vector3f& a, const Eigen::Vector3f& d)   // a o d, d in a's frame
        {
            const float c = std::cos(a.z()), s = std::sin(a.z());
            return {a.x() + c * d.x() - s * d.y(), a.y() + s * d.x() + c * d.y(), rc::rivals::wrap_pi(a.z() + d.z())};
        }

        double polygon_area(const std::vector<Eigen::Vector2f>& poly)
        {
            double a2 = 0.0;
            for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
                a2 += static_cast<double>(poly[j].x()) * poly[i].y() - static_cast<double>(poly[i].x()) * poly[j].y();
            return 0.5 * std::abs(a2);
        }
    }

    void RoomConcept::clear_rival_belief()
    {
        rivals_.clear();
        ++room_epoch_;            // a proposal still in flight belongs to the room (or pose) we left: poll() results are dropped
        rival_job_.reset();
        rv_ref_valid_ = false;
        rv_h0s_valid_ = false;
        rv_prev_cov_valid_ = false;
        view_prev_pose_valid_ = false;
        rv_novelty_acc_ = 0.f;
        rt_.held_spread.setZero();
        pose_undecided_.store(false, std::memory_order_relaxed);
        p_rival_pub_.store(0.f, std::memory_order_relaxed);
        publish_rival_snapshot();
    }

    // The rivals as the planner sees them (pose_hypotheses()): pose + ln odds against H0. Empty <=> the pose is unambiguous.
    void RoomConcept::publish_rival_snapshot()
    {
        std::vector<rc::reloc::PoseHypothesis> snap;
        snap.reserve(rivals_.rivals().size());
        for (const auto& rv : rivals_.rivals()) snap.push_back({rv.pose, rv.lambda});
        std::scoped_lock lk(rival_snapshot_mutex_);
        rival_snapshot_ = std::move(snap);
    }

    void RoomConcept::step_rivals(const UpdateResult& res, const std::vector<Eigen::Vector3f>& lidar_points)
    {
        const auto t_start = std::chrono::steady_clock::now();
        if (not rivals_ready_)
        {
            rc::rivals::Params bp;
            bp.pi_kid = std::clamp(params.reloc_hazard, 1e-9f, 0.25f);        // RelocHazard: the kidnap rate, per frame
            bp.k_max = std::max(1, params.reloc_max_modes);
            bp.ln_eps = std::log(std::clamp(params.reloc_outlier_frac, 1e-6f, 1.f));   // ln(RelocOutlierFrac)
            rivals_ = rc::rivals::Belief(bp);
            rivals_ready_ = true;
        }
        const Eigen::Vector3f pose0(res.robot_pose.translation().x(), res.robot_pose.translation().y(),
                                    std::atan2(res.robot_pose.linear()(1, 0), res.robot_pose.linear()(0, 0)));
        // H0's own Laplace covariance: run() adds last frame's published spread to res.covariance before this call.
        Eigen::Matrix3f cov0 = tracker_cov_valid_ ? tracker_cov_ : res.covariance;
        tracker_cov_valid_ = false;
        if (not cov0.allFinite()) cov0 = Eigen::Matrix3f::Identity();
        if (not pose0.allFinite() or lidar_points.empty() or model_ == nullptr or not model_->has_state())
            return;
        const auto poly = current_room_polygon();
        if (poly.size() < 3)
            return;

        // ── how new this view is: the pose step since the last frame, in the likelihood's own basin; w in [0,1] ──
        float w = 1.f;
        if (view_prev_pose_valid_)
        {
            double r_sum = 0.0;
            for (const auto& q : lidar_points) r_sum += std::hypot(q.x(), q.y());
            const double sig = std::max(1e-3, static_cast<double>(params.reloc_track_s_median));   // what tracking resolves
            const double r_mean = std::max(sig, r_sum / static_cast<double>(lidar_points.size()));
            const double dp2 = (pose0.head<2>() - view_prev_pose_.head<2>()).squaredNorm() / (sig * sig);
            const double dth = std::remainder(pose0.z() - view_prev_pose_.z(), 2.0 * M_PI);
            const double dt2 = dth * dth / ((sig / r_mean) * (sig / r_mean));
            w = static_cast<float>(1.0 - std::exp(-0.5 * (dp2 + dt2)));
        }
        view_prev_pose_ = pose0;
        view_prev_pose_valid_ = true;
        rv_novelty_acc_ += w;
        // ★ THE LATE RECALL'S VERDICT READS NORMAL TRACKING AFTER ITS FIRST FIT (restored 2026-11-06; lost in af46a885 with the
        //   legacy lost-belief code that held the only per-scan call). Without it the verdict had ONE observation — the
        //   multi-scan fit — and a recognition whose fit alone did not cross ±ln 19 stayed Pending for ever (explore_tour9:
        //   the re-entered start room judged for minutes, the exploration drive holding on it). Each scan's misfit, weighted by
        //   how new this view is (w above): a parked robot's repeated scans are one observation, not many.
        if (late_pending_.load(std::memory_order_acquire) and late_forced_done_ and not std::exchange(late_fit_committed_this_frame_, false))
            accumulate_late_evidence(std::max(res.sdf_mse, res.pred_sdf_median), w);

        // ── carry: every rival (and the pending proposal) moves with the ROBOT'S ODOMETRY increment since the last step ──
        // The motion prior's increment (global frame, what update()'s predict step added), rotated into the body frame of the
        // previous pose. NOT H0's pose difference: a wrong H0 jumps (the tracker's own symmetry flip is 4 m / 180 deg), and a
        // rival dragged by that jump is never at the truth. No valid prior (no odometry yet) = no motion.
        Eigen::Vector3f d = Eigen::Vector3f::Zero();
        if (rv_ref_valid_ and last_selected_prior_.valid and last_selected_prior_.delta_pose.allFinite())
        {
            const float c = std::cos(rv_ref_.z()), s = std::sin(rv_ref_.z());
            const Eigen::Vector3f& g = last_selected_prior_.delta_pose;
            d = {c * g.x() + s * g.y(), -s * g.x() + c * g.y(), rc::rivals::wrap_pi(g.z())};
        }
        rv_ref_ = pose0;
        rv_ref_valid_ = true;
        rivals_.carry(d);
        if (rv_h0s_valid_) rv_h0s_ = se2_compose(rv_h0s_, d);
        if (rival_job_) rival_job_->odo = se2_compose(rival_job_->odo, d);

        // ── shared, lazily built per frame ──
        if (poly != reloc_raster_poly_ or not reloc_raster_.valid())
        {
            reloc_raster_.build(poly, reloc_params());
            reloc_raster_poly_ = poly;
        }
        std::optional<rc::reloc::Params> rp_cache;
        auto rp = [&]() -> const rc::reloc::Params& { if (not rp_cache) rp_cache = reloc_params(); return *rp_cache; };
        std::vector<Eigen::Vector2f> pts2;
        auto points = [&]() -> const std::vector<Eigen::Vector2f>&
        {
            if (pts2.empty())
            {
                const int stride = std::max(1, static_cast<int>(lidar_points.size()) / std::max(1, params.reloc_max_points));
                pts2.reserve(lidar_points.size() / static_cast<std::size_t>(stride) + 1);
                for (std::size_t i = 0; i < lidar_points.size(); i += static_cast<std::size_t>(stride))
                    pts2.emplace_back(lidar_points[i].x(), lidar_points[i].y());
            }
            return pts2;
        };

        // ── the door anchor retires BY INFORMATION (one currency: nats of pose information over the room x
        // circle): the tracker holds at least as much as the anchor carried by odometry since the hand-over, and the body has
        // left the doorway. The anchor is H0's entry prior (sec 5): once the tracker's own covariance is tighter it IS the prior.
        if (not rt_.door_anchor.empty())
        {
            rt_.passage_term = rt_.passage.has_value()
                                   ? rc::passage::doorway_term(*rt_.passage, res.robot_pose.translation(), cov0.topLeftCorner<2, 2>())
                                   : 0.f;
            const double h = std::clamp(static_cast<double>(params.reloc_hazard), 1e-9, 0.25);
            const double p_kid = 1.0 - std::pow(1.0 - h, static_cast<double>(rt_.door_anchor_frames + 1));
            const double a_area = std::max(polygon_area(poly), 1e-6);
            const double anchor_kl = (1.0 - p_kid) * rc::reloc::anchor_information(rt_.door_anchor, a_area);
            const double track_kl = rc::reloc::gaussian_information(cov0, a_area);
            if (rt_.passage_term < 0.5f and track_kl >= anchor_kl)
            {
                qInfo().noquote() << QString("[reloc/rivals] door anchor retired %1 ms after the hand-over (%2 frames): tracker %3 nats >= anchor %4 nats")
                                         .arg(res.timestamp_ms - handover_ts_ms_).arg(rt_.door_anchor_frames)
                                         .arg(track_kl, 0, 'f', 1).arg(anchor_kl, 0, 'f', 1);
                rt_.door_anchor.clear();
            }
        }

        // ── H0's prior this frame (spec sec 3, 5): its odometry prediction, or the door anchor while that lives ──
        const double area = std::max(polygon_area(poly), 1e-6);
        auto prior_now = [&]() -> RivalPrior
        {
            RivalPrior pr;
            pr.volume = static_cast<float>(area * 2.0 * M_PI);
            pr.pred = {res.pred_x, res.pred_y, res.pred_theta};
            // P_pred = the previous posterior's covariance grown by the odometry's own (what the predict step does): the
            // odometry covariance ALONE is ~0 for a parked robot and would price every tracker correction at 1e6 nats.
            pr.P = cov0;   // the tracker's own covariance (Hessian + odometry growth, spec sec 3); NOT last frame's: it collapses when parked
            if (last_selected_prior_.valid and last_selected_prior_.covariance_eigen.allFinite())
                pr.P += last_selected_prior_.covariance_eigen;
            // sec 5: after a hand-over H0's prior is N(mu_entry, Sigma_entry) -- the door anchor (forward / reversed),
            // carried by odometry so its covariance already carries the growth. The component that best explains H0's pose
            // is the prior (max-term of the mixture); births draw from the entry volume, not the whole room.
            const rc::passage::Anchor* best = nullptr;
            double best_lp = -std::numeric_limits<double>::infinity();
            for (const auto& an : rt_.door_anchor)
            {
                if (not (an.w > 0.f) or not an.pose.allFinite() or not an.cov.allFinite()) continue;
                Eigen::Vector3f da = pose0 - an.pose;
                da.z() = rc::rivals::wrap_pi(da.z());
                const Eigen::Matrix3f S = an.cov + 1e-8f * Eigen::Matrix3f::Identity();   // numerical: a zero-variance door
                const double lp = std::log(static_cast<double>(an.w)) - 0.5 * da.dot(S.inverse() * da)
                                  - 0.5 * std::log(std::max(static_cast<double>((2.f * std::numbers::pi_v<float> * S).determinant()), 1e-30));
                if (lp > best_lp) { best_lp = lp; best = &an; }
            }
            if (best != nullptr)
            {
                pr.pred = best->pose;
                pr.P = best->cov;
                pr.ln_w = std::log(static_cast<double>(best->w));
                // V_entry: the +-3 sigma box of the anchor (216 sqrt|S|), never larger than the room.
                const float v_entry = 216.f * std::sqrt(std::max(best->cov.determinant(), 1e-12f));
                pr.volume = std::min(pr.volume, v_entry);
            }
            pr.P += 1e-8f * Eigen::Matrix3f::Identity();   // numerical: a parked robot's odometry covariance can be exactly 0
            pr.volume = std::max(pr.volume, 1e-3f);
            return pr;
        };
        // ONE prior over the pose for EVERY hypothesis (spec sec 3): p(x) = (1-pi) N(x; pred, P_pred) + pi / V -- consistent with
        // the odometry prediction (or the entry area), or kidnapped (uniform over V). Then, in Laplace form,
        //   lambda_r = [l_r - l_0] + [ln p(x_r) + 1/2 ln|2 pi S_r|] - [ln p(x_0) + 1/2 ln|2 pi S_0|].
        // For x_r far from the prediction and x_0 near it this IS birth_lambda (sec 4.2: ln pi/(1-pi) + dl + ln(sqrt|2 pi S_r|/V)
        // - ln N(x_0) - 1/2 ln|2 pi S_0|). It differs only where sec 4.2 is silent: a rival AT the prediction while the tracker
        // jumped off it (M: tracker diverged from the door anchor; the anchor-consistent pose is then favoured, a twin is not),
        // and a tracker that jumped by itself (H): there the rival is not "kidnapped", and an overconfident P_pred cannot price
        // the incumbent below ln(V/pi) (the unbounded -ln N(x_0) of sec 4.2 was 2e4 nats on a 5 cm tracker correction).
        auto ln_prior = [&](const Eigen::Vector3f& x, const RivalPrior& pr)
        {
            Eigen::Vector3f dx = x - pr.pred;
            dx.z() = rc::rivals::wrap_pi(dx.z());
            const double pi_k = rivals_.params().pi_kid;
            const double ln_n = -static_cast<double>(rc::rivals::half_log_det_2pi(pr.P)) - 0.5 * dx.dot(pr.P.inverse() * dx);
            const double a1 = std::log1p(-pi_k) + ln_n + pr.ln_w, b1 = std::log(pi_k) - std::log(static_cast<double>(pr.volume));
            const double hi = std::max(a1, b1);
            return hi + std::log(std::exp(a1 - hi) + std::exp(b1 - hi));
        };
        auto price = [&](float dl, const Eigen::Vector3f& x_r, const Eigen::Matrix3f& cov_r, const Eigen::Vector3f& inc,
                         const Eigen::Matrix3f& inc_cov, const RivalPrior& pr)
        {
            return static_cast<float>(dl + ln_prior(x_r, pr) + rc::rivals::half_log_det_2pi(cov_r)
                                      - ln_prior(inc, pr) - rc::rivals::half_log_det_2pi(inc_cov));
        };

        // A basin of ell is as wide as the likelihood's noise scale : sigma_sdf in translation,
        // sigma_sdf / (median range of the scan) in rotation. No new constant.
        auto basin_of = [](const std::vector<Eigen::Vector2f>& pts, float sigma)
        {
            std::vector<float> ranges;
            ranges.reserve(pts.size());
            for (const auto& q : pts) ranges.push_back(q.norm());
            float r_med = 1.f;
            if (not ranges.empty())
            {
                const auto mid = ranges.begin() + static_cast<std::ptrdiff_t>(ranges.size() / 2);
                std::ranges::nth_element(ranges, mid);
                r_med = std::max(*mid, sigma);
            }
            Eigen::Matrix3f b = Eigen::Matrix3f::Zero();
            b(0, 0) = b(1, 1) = sigma * sigma;
            b(2, 2) = (sigma / r_med) * (sigma / r_med);
            return b;
        };
        // H0 evaluated at ITS OWN MODE OF ell (sec 4.1: Laplace on both sides). The tracker's pose is the optimum of the
        // tracker's SDF objective, not of the relocaliser's ell; in a doorway view the two differ by ~14 cm = ~60 nats
        // (selftest U), which would be charged to H0 on every scan against rivals that sit at ell's own optimum. The polish
        // from the tracker's pose, accepted only while it stays inside the same basin (else the tracker's pose stands).
        auto h0_star = [&](const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& pose, const Eigen::Matrix3f& cov,
                           const rc::reloc::Params& rp_, const Eigen::Matrix3f& basin) -> Eigen::Vector3f
        {
            const auto pp = rc::reloc::polish_points(pts_robot, rp_);
            const auto bl = rc::reloc::beyond_lines(pp, rp_);
            const auto k = rc::reloc::polish(poly, pp, pose, rp_, reloc_raster_.extent(), &bl);
            if (not k.pose.allFinite() or not k.cov.allFinite()) return pose;
            Eigen::Vector3f dk = k.pose - pose;
            dk.z() = rc::rivals::wrap_pi(dk.z());
            return dk.dot((basin + k.cov + cov).inverse() * dk) < rp_.chi2_merge ? k.pose : pose;
        };

        // ── 0. the pose the start-up search left: a rival, priced like any other (its birth price is its floor) ──
        if (startup_rival_.has_value())
        {
            const auto sr = *startup_rival_;
            startup_rival_.reset();
            if (sr.pose.allFinite() and sr.cov.allFinite() and rc::reloc::inside_polygon(poly, sr.pose.head<2>()))
            {
                const Eigen::Matrix3f basin0 = basin_of(points(), rp().sigma_sdf);
                Eigen::Vector3f dh = sr.pose - pose0;
                dh.z() = rc::rivals::wrap_pi(dh.z());
                if (dh.dot((cov0 + sr.cov + basin0).inverse() * dh) > 9.f)   // the identity test of births: not H0's own basin
                {
                    rc::rivals::Rival r;
                    r.pose = sr.pose; r.cov = sr.cov;
                    r.lambda_birth = price(0.f, sr.pose, sr.cov, pose0, cov0, prior_now());
                    r.lambda = r.lambda_birth;   // no evidence yet: step 1 below scores it on this very scan
                    r.id = rv_next_id_++;
                    if (std::isfinite(r.lambda)) rivals_.add(std::move(r));
                }
            }
        }

        // ── 1. score the rivals that existed before this frame on THIS scan, update, test for a winner ──
        // Every hypothesis, H0 included, is evaluated at ITS OWN local mode of ell (a few GN steps from where it stands): the
        // rival is a tracked pose like the incumbent. Compared at carried/tracked poses instead, a twin carried by exact motion
        // beat a tracker pose 1 cm off by ~1.5 nats a scan -- a drift that took it from -28 to the band in ~20 s (selftest V1).
        if (not rivals_.rivals().empty())
        {
            const auto& pp_all = points();
            // The tracking evaluation uses every OTHER point of the search's polish subset: half the cost (hot path < 1 ms), the
            // same ell on both sides of every ratio, half the evidence per scan (conservative). The proposer keeps the full set.
            auto pp = rc::reloc::polish_points(pp_all, rp());
            {
                std::vector<Eigen::Vector2f> half;
                half.reserve(pp.size() / 2 + 1);
                for (std::size_t i = 0; i < pp.size(); i += 2) half.push_back(pp[i]);
                pp = std::move(half);
            }
            const auto bl = rc::reloc::beyond_lines(pp, rp());
            const Eigen::Matrix3f basin = basin_of(pp_all, rp().sigma_sdf);
            // A few GN steps per scan: each hypothesis is re-seated on its own mode every frame, so it is never far from it.
            rc::reloc::Params rp_track = rp();
            rp_track.polish_iters = 3;   // ⚠ a compute budget (hot path < 1 ms), not a model quantity; the full 40 is the proposer's
            struct Star { float ll; Eigen::Vector3f pose; };
            auto ell_star = [&](const Eigen::Vector3f& pose, const Eigen::Matrix3f& cov) -> Star
            {
                const float l_here = rc::reloc::log_likelihood(poly, pp, pose, rp_track, reloc_raster_.extent(), nullptr, &bl);
                const auto k = rc::reloc::polish(poly, pp, pose, rp_track, reloc_raster_.extent(), &bl);
                if (not k.pose.allFinite() or not k.cov.allFinite() or not std::isfinite(k.log_lik)) return {l_here, pose};
                Eigen::Vector3f dk = k.pose - pose;
                dk.z() = rc::rivals::wrap_pi(dk.z());
                // k.log_lik is the polish's own (EM-annealed sigma) figure: NOT comparable with the fixed-sigma ell used everywhere
                // else here. Re-score the polished POSE with the one ell. NOT max(start, polished): that selection favours the side
                // that started further from the optimum (a tied twin drifted +0.4 nats a scan into a swap, selftest V1).
                if (dk.dot((basin + k.cov + cov).inverse() * dk) >= rp_track.chi2_merge) return {l_here, pose};
                return {rc::reloc::log_likelihood(poly, pp, k.pose, rp_track, reloc_raster_.extent(), nullptr, &bl), k.pose};
            };
            // H0 is polished from where ITS mode was last frame (carried), not from the tracker's pose afresh: three steps from a
            // tracker pose a centimetre off do not converge, from last frame's mode they do, which is what the rivals get.
            // Back to the tracker's pose when that mode left the tracker's basin (or after a swap / a room change).
            Eigen::Vector3f h0_start = pose0;
            if (rv_h0s_valid_)
            {
                Eigen::Vector3f dh = rv_h0s_ - pose0;
                dh.z() = rc::rivals::wrap_pi(dh.z());
                if (dh.dot((basin + cov0).inverse() * dh) < rp_track.chi2_merge) h0_start = rv_h0s_;
            }
            const auto h0_st = ell_star(h0_start, cov0);
            rv_h0s_ = h0_st.pose; rv_h0s_valid_ = true;
            const float l0 = h0_st.ll;
            std::vector<float> dl_v;
            dl_v.reserve(rivals_.rivals().size());
            for (std::size_t i = 0; i < rivals_.rivals().size(); ++i)
            {
                const auto st = ell_star(rivals_.rivals()[i].pose, rivals_.rivals()[i].cov);
                dl_v.push_back(st.ll - l0);
                rivals_.set_pose(i, st.pose);   // tracked: re-seated on its own mode
            }
            rivals_.update(dl_v, w);
            // Tracked rivals can slide into one another's basin or into H0's: two hypotheses in one basin of ell are ONE (the
            // identity test of births, basin included). A rival that has become H0 is H0's own mass; of two that have merged the
            // better odds survive. (Rebuilt only when something merged: the common frame allocates nothing.)
            {
                const auto& cur = rivals_.rivals();
                bool any = false;
                std::vector<char> drop(cur.size(), 0);
                for (std::size_t i = 0; i < cur.size(); ++i)
                {
                    Eigen::Vector3f dh = cur[i].pose - h0_st.pose;
                    dh.z() = rc::rivals::wrap_pi(dh.z());
                    if (dh.dot((cov0 + cur[i].cov + basin).inverse() * dh) <= 9.f) { drop[i] = 1; any = true; continue; }
                    for (std::size_t j = 0; j < i; ++j)
                    {
                        if (drop[j]) continue;
                        Eigen::Vector3f dj = cur[i].pose - cur[j].pose;
                        dj.z() = rc::rivals::wrap_pi(dj.z());
                        if (dj.dot((cur[i].cov + cur[j].cov + basin).inverse() * dj) <= 9.f)
                        { (cur[i].lambda > cur[j].lambda ? drop[j] : drop[i]) = 1; any = true; break; }
                    }
                }
                if (any)
                {
                    std::vector<rc::rivals::Rival> keep;
                    for (std::size_t i = 0; i < cur.size(); ++i) if (not drop[i]) keep.push_back(cur[i]);
                    rivals_.clear();
                    for (auto& r : keep) rivals_.add(std::move(r));
                }
            }
        }

        // ── 2. a finished proposal: births, priced in the frame the scan was taken ──
        if (auto done = proposer_.poll())
        {
            rv_last_search_ms_ = done->second.duration_ms;
            if (done->first == room_epoch_ and rival_job_)   // a proposal from another room / pose is dropped
            {
                const RivalJob& job = *rival_job_;
                // H0 on the same footing as every mode: the search was seeded at H0 (`around`), so the mode inside H0's basin IS
                // H0 polished by the very routine that polished the rivals -- ell(H0*) is the better of that and the tracker's pose.
                // (Only when no mode landed in H0's basin is H0 polished here.)
                auto in_h0_basin = [&](const rc::reloc::Mode& m)
                {
                    Eigen::Vector3f dm = m.pose - job.pose0;
                    dm.z() = rc::rivals::wrap_pi(dm.z());
                    return dm.dot((job.cov0 + m.cov + job.basin).inverse() * dm) <= 9.f;
                };
                float ll0 = rc::rivals::score(reloc_raster_, poly, job.pts, job.pose0, job.rp);
                bool h0_mode = false;
                for (const auto& m : done->second.modes)
                    if (m.pose.allFinite() and m.cov.allFinite() and std::isfinite(m.log_lik) and in_h0_basin(m))
                    { ll0 = std::max(ll0, rc::rivals::score(reloc_raster_, poly, job.pts, m.pose, job.rp)); h0_mode = true; }
                if (not h0_mode)
                    ll0 = std::max(ll0, rc::rivals::score(reloc_raster_, poly, job.pts, h0_star(job.pts, job.pose0, job.cov0, job.rp, job.basin), job.rp));
                for (const auto& m : done->second.modes)
                {
                    if (not m.pose.allFinite() or not m.cov.allFinite() or not std::isfinite(m.log_lik)) continue;
                    // THRESHOLD IN DISGUISE: "this mode IS H0" = Mahalanobis <= 3 under the summed covariances PLUS ell's own
                    // basin (sigma_sdf in translation, sigma_sdf / median range in rotation: `basin`, no new constant), the
                    // same identity test Belief::add uses between rivals. A better optimum inside H0's basin is the tracker's to
                    // refine (its SDF solve), not a rival: that would be 14 cm of model mismatch priced as a kidnap (selftest U).
                    if (in_h0_basin(m)) continue;
                    rc::rivals::Rival r;
                    r.pose = se2_compose(m.pose, job.odo);   // carried by H0's motion since the scan
                    r.cov = m.cov;
                    r.lambda = price(rc::rivals::score(reloc_raster_, poly, job.pts, m.pose, job.rp) - ll0, m.pose, m.cov, job.pose0, job.cov0, job.prior);
                    r.lambda_birth = price(0.f, m.pose, m.cov, job.pose0, job.cov0, job.prior);   // the price without the evidence: the floor
                    r.id = rv_next_id_++;
                    if (std::isfinite(r.lambda) and std::isfinite(r.lambda_birth)
                        and rc::reloc::inside_polygon(poly, r.pose.head<2>()))
                    {
                        // Two proposals inside one basin of ell are ONE rival (the same identity test as H0's, basin included):
                        // Belief::add merges only at 3 sigma of the Laplace covariances, a few mm, and a re-proposed twin 8 cm away
                        // became a second twin accumulating its own odds (selftest V1). The merged rival keeps the better odds.
                        auto all = rivals_.rivals();
                        bool merged = false;
                        for (auto& e : all)
                        {
                            Eigen::Vector3f de = e.pose - r.pose;
                            de.z() = rc::rivals::wrap_pi(de.z());
                            if (de.dot((e.cov + r.cov + job.basin).inverse() * de) <= 9.f)
                            {
                                if (r.lambda > e.lambda) { e.lambda = r.lambda; e.pose = r.pose; e.cov = r.cov; }
                                merged = true;
                                break;
                            }
                        }
                        if (merged)
                        {
                            rivals_.clear();
                            for (auto& e : all) rivals_.add(std::move(e));
                        }
                        else
                            rivals_.add(std::move(r));
                    }
                }
            }
            rival_job_.reset();
        }
        // ── 3. a winner becomes the incumbent: the tracker is reset to its pose, the old incumbent becomes a rival ──
        Eigen::Vector3f pose_pub = pose0;
        Eigen::Matrix3f cov_pub = cov0;
        // ★ NO SWAP WHILE A RECOGNITION IS JUDGED (2026-11-12, clean56). The hypothesis under test is "P is K AT ltsm's
        //   T(P<-K)": it fixes the robot's pose in K up to the chain's covariance (cm). A rival metres away is a DIFFERENT
        //   hypothesis ("P is K, the chain is wrong by metres"), whose prior ltsm itself puts at ~e^-10, not this one's
        //   support. Letting it win made the test "does P fit ANYWHERE in K": clean56 relocalised the bath 1.4 -> 5.3 ->
        //   7.2 -> 9.9 m across the big room until a corner explained it, the verdict then scored those fits as "K explains
        //   the scans" and ACCEPTED with a 7.9 m / 88 deg correction. Held, the verdict scores the scans at the pose the
        //   recognition predicted (the multi-scan fit already searches within the prior's covariance), and a room that does
        //   not fit THERE is rejected by its own misfit. The rivals keep their odds; they compete again once the fit ends.
        const bool judging = hypothesis_fit_active();
        if (const auto wi = rivals_.winner(); wi and judging and not std::exchange(rv_swap_held_logged_, true))
            qInfo().noquote() << QString("[reloc/rivals] a rival out-explains the incumbent (lambda %1) while a recognition is "
                                         "judged: held — the hypothesis fixes the pose at ltsm's prior")
                                     .arg(rivals_.rivals().at(*wi).lambda, 0, 'f', 1);
        if (not judging) rv_swap_held_logged_ = false;
        if (const auto wi = judging ? std::optional<std::size_t>{} : rivals_.winner())
        {
            RivalPrior pr = prior_now();
            const rc::rivals::Rival won0 = rivals_.rivals().at(*wi);
            // The new incumbent's own prior is centred on it (its Laplace covariance): the price of every OTHER pose against
            // it is then the bare Occam price, not a chi-square against the OLD incumbent's odometry prediction.
            pr.pred = won0.pose; pr.P = won0.cov + 1e-8f * Eigen::Matrix3f::Identity(); pr.ln_w = 0.f;
            rivals_.swap(*wi, pose0, cov0, price(0.f, pose0, cov0, won0.pose, won0.cov, pr));
            std::vector<rc::rivals::Rival> all = rivals_.rivals();
            rivals_.clear();
            for (std::size_t i = 0; i < all.size(); ++i)
            {
                auto r = all[i];
                if (i != *wi)   // ratios are now against the new incumbent: its odds and its price move with it
                {
                    r.lambda -= won0.lambda;
                    r.lambda_birth = price(0.f, r.pose, r.cov, won0.pose, won0.cov, pr);
                    r.lambda = std::max(r.lambda, r.lambda_birth);
                }
                rivals_.add(std::move(r));
            }
            const float jump_m = (won0.pose.head<2>() - pose0.head<2>()).norm();
            const float jump_rad = std::abs(rc::rivals::wrap_pi(won0.pose.z() - pose0.z()));
            SearchEpisode ep;
            ep.ts_ms = res.timestamp_ms;
            ep.trigger = "rival_swap";
            ep.stage = 10;
            ep.n_lidar = static_cast<int>(lidar_points.size());
            ep.incumbent_x = pose0.x(); ep.incumbent_y = pose0.y(); ep.incumbent_theta = pose0.z();
            ep.best_x = won0.pose.x(); ep.best_y = won0.pose.y(); ep.best_theta = won0.pose.z();
            ep.jump_m = jump_m; ep.jump_rad = jump_rad;
            ep.top_weight = 1.f / (1.f + std::exp(-won0.lambda));
            ep.n_modes = static_cast<int>(all.size()) + 1;
            ep.success = true;
            write_search_episode(ep);
            qWarning().noquote() << QString("[reloc/rivals] swap: rival out-explained the incumbent, lambda %1 (band %2) -- moving %3 m / %4 deg")
                .arg(won0.lambda, 0, 'f', 1).arg(rivals_.params().log_band, 0, 'f', 1).arg(jump_m, 0, 'f', 2).arg(jump_rad * 57.2958f, 0, 'f', 0);
            ++rv_swaps_;
            commit_relocalised_pose(won0.pose, won0.cov);
            pose_pub = won0.pose;
            cov_pub = won0.cov;
            rv_ref_ = won0.pose;
            rv_h0s_valid_ = false;
            view_prev_pose_ = won0.pose;
            ++room_epoch_;        // the in-flight proposal was priced against the old incumbent: drop it
            rival_job_.reset();
        }

        // ── 4. scheduling: the proposer is never on the 50 ms path; one job in flight ──
        {
            const bool gaining = std::ranges::any_of(rivals_.rivals(), [](const rc::rivals::Rival& r) { return r.lambda > r.lambda_birth; });
            const bool entering = not rt_.door_anchor.empty();
            // Base 2 Hz; every novel scan (one unit of accumulated view_change) while a rival is gaining or the hand-over is
            // fresh. A duty cap of 25% of one core (gap >= 4 x the last search's duration) is a scheduling constant, flagged:
            // it can delay a proposal, it cannot move the pose.
            const std::int64_t gap = res.timestamp_ms - rv_last_submit_ms_;
            const std::int64_t duty = static_cast<std::int64_t>(4.f * rv_last_search_ms_);
            const bool due = (gap >= 500) or ((gaining or entering) and rv_novelty_acc_ >= 1.f);
            if (due and gap >= duty and not rival_job_ and not proposer_.busy())
            {
                RivalJob job;
                job.pose0 = pose_pub; job.cov0 = cov_pub; job.prior = prior_now();
                job.pts = points();
                job.rp = rp();
                job.basin = basin_of(job.pts, job.rp.sigma_sdf);
                if (proposer_.submit(room_epoch_, reloc_raster_, poly, job.pts, job.rp, pose_pub, cov_pub))
                {
                    ++rv_submits_;
                    rival_job_ = std::move(job);
                    rv_last_submit_ms_ = res.timestamp_ms;
                    rv_novelty_acc_ = 0.f;
                }
            }
        }

        rv_prev_cov_ = cov_pub; rv_prev_cov_valid_ = true;

        // ── 5. publish: H0 (or the winner), its covariance widened by the mixture spread while rivals carry mass ──
        rt_.held_spread = rivals_.spread(pose_pub);
        const float pr_mass = rivals_.p_rival();
        p_rival_pub_.store(pr_mass, std::memory_order_relaxed);
        pose_undecided_.store(pr_mass >= 0.5f, std::memory_order_relaxed);   // the majority of the posterior mass is off H0
        publish_rival_snapshot();
        if (rv_timing_on_)
            rv_timing_us_.push_back(static_cast<float>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_start).count()) * 1e-3f);
    }
}   // namespace rc
