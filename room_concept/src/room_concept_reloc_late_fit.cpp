// ADOPTION AFTER A RECOGNITION SWAP (spec 2026-11-01-relocaliser-redesign sec 6).
// A JOINT fit of the last M keyframes against the
// remembered room K, around ltsm's T(K<-P) prior with its covariance (rc::rivals::multi_scan_fit, which calls the
// relocaliser's own per-point likelihood). The fit runs on a one-off worker with deep-copied inputs (no DSR, no Qt); the
// compute thread only pushes keyframes, launches the job and applies a finished result.
// Keyframes: the tracker's own pose estimates (already in the current frame) with a subsampled scan. Their RELATIVE poses
// are used, so slow drift cancels; the ring is cleared by seed_reloc_prior (the pose jump of the adoption itself).
#include "room_concept.h"

#include <chrono>
#include <cmath>
#include <future>
#include <QDebug>

namespace rc
{
    namespace
    {
        Eigen::Vector3f compose(const Eigen::Vector3f& a, const Eigen::Vector3f& d)   // a o d, d in a's frame
        {
            const float c = std::cos(a.z()), s = std::sin(a.z());
            return {a.x() + c * d.x() - s * d.y(), a.y() + s * d.x() + c * d.y(), std::remainder(a.z() + d.z(), 2.f * static_cast<float>(M_PI))};
        }
        Eigen::Vector3f between(const Eigen::Vector3f& a, const Eigen::Vector3f& b)   // b in a's frame
        {
            const float c = std::cos(a.z()), s = std::sin(a.z());
            const Eigen::Vector2f d = b.head<2>() - a.head<2>();
            return {c * d.x() + s * d.y(), -s * d.x() + c * d.y(), std::remainder(b.z() - a.z(), 2.f * static_cast<float>(M_PI))};
        }
        // FLAGGED scheduling constant: keyframes per fit. The brief asks 10-20; ~1.5 s of scans at the lidar rate, enough path
        // for a scan's blind direction to be seen by another keyframe. Not a gate on the result.
        constexpr std::size_t kLateKeyframes = 15;
    }

    void RoomConcept::run_late_recall_fit(const std::vector<Eigen::Vector3f>& lidar_points)
    {
        std::vector<rc::reloc::PriorMode> prior;
        { std::scoped_lock lk(late_prior_mutex_); prior = late_prior_; }
        if (prior.empty() or model_ == nullptr or not model_->has_state() or lidar_points.empty())
        {
            if (prior.empty()) { late_ring_.clear(); late_ref_valid_ = false; if (late_fit_.valid()) late_fit_.wait(), late_fit_ = {}; }
            return;
        }
        const auto st = model_->get_state();
        const Eigen::Vector3f now(st[2], st[3], st[4]);
        const auto rp = reloc_params();

        // 1. a finished job: carry its pose to NOW by the motion since its latest keyframe, commit, start the verdict.
        if (late_fit_.valid() and late_fit_.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            const auto fit = late_fit_.get();
            late_fit_ = {};
            if (fit.epoch != reloc_epoch_.load(std::memory_order_relaxed) or fit.poly != current_room_polygon())
            {   // the pose was moved by someone else, or the room changed, while the job ran: its frame is gone
                qInfo() << "[reloc] late recall: multi-scan fit dropped (pose or room changed while it ran)";
                late_ring_.clear(); late_ref_valid_ = false;
                return;
            }
            if (std::isfinite(fit.mode.log_lik) and fit.mode.pose.allFinite() and fit.mode.cov.allFinite())
            {
                const Eigen::Vector3f at_now = compose(fit.mode.pose, between(fit.submit_pose, now));
                committing_own_fit_ = true;
                commit_relocalised_pose(at_now, fit.mode.cov);
                committing_own_fit_ = false;
                late_forced_done_ = true;
                clear_rival_belief();   // the pose moved: the rivals / in-flight proposals were priced against the old one
                ++rt_.searches;
                qInfo().noquote() << QString("[reloc] late recall: multi-scan fit of %1 keyframes -> (%2, %3, %4 deg), sd (%5 m, %6 deg), median |SDF| %7 m")
                    .arg(kLateKeyframes).arg(at_now.x(), 0, 'f', 2).arg(at_now.y(), 0, 'f', 2).arg(at_now.z() * 57.2958f, 0, 'f', 1)
                    .arg(std::sqrt(fit.mode.cov(0, 0)), 0, 'f', 3).arg(std::sqrt(fit.mode.cov(2, 2)) * 57.2958f, 0, 'f', 2)
                    .arg(fit.mode.median_abs, 0, 'f', 3);
                accumulate_late_evidence(fit.mode.median_abs, 1.f);   // the first views are entirely new
                return;
            }
            qWarning() << "[reloc] late recall: the multi-scan fit produced no pose — fitting again";
        }

        // 2. this frame's keyframe.
        std::vector<Eigen::Vector2f> pts2;
        pts2.reserve(lidar_points.size());
        for (const auto& q : lidar_points) pts2.emplace_back(q.x(), q.y());
        if (not late_ref_valid_) { late_ref_pose_ = now; late_ref_valid_ = true; }
        late_ring_.push_back({now, rc::reloc::polish_points(pts2, rp)});
        while (late_ring_.size() > kLateKeyframes) late_ring_.pop_front();

        // 3. a full ring and no job in flight: launch.
        if (late_fit_.valid() or late_ring_.size() < kLateKeyframes) return;
        const auto poly = current_room_polygon();
        if (poly.size() < 3) return;
        if (poly != reloc_raster_poly_ or not reloc_raster_.valid()) { reloc_raster_.build(poly, rp); reloc_raster_poly_ = poly; }
        if (not reloc_raster_.valid()) return;
        const auto best = std::ranges::max_element(prior, {}, &rc::reloc::PriorMode::w);
        // The prior is for the pose at the ring's first frame (the pose the adoption committed); carry it to the latest keyframe.
        const Eigen::Vector3f prior_now = compose(best->mu, between(late_ref_pose_, now));
        std::vector<std::pair<Eigen::Vector3f, std::vector<Eigen::Vector2f>>> scans;
        scans.reserve(late_ring_.size());
        for (const auto& kf : late_ring_) scans.emplace_back(between(now, kf.pose), kf.pts);
        late_fit_ = std::async(std::launch::async,
            [raster = reloc_raster_, poly, epoch = reloc_epoch_.load(std::memory_order_relaxed), scans = std::move(scans), prior_now, S = best->S, rp, now]() mutable -> LateFit
            { return {rc::rivals::multi_scan_fit(raster, poly, scans, prior_now, S, rp), now, epoch, poly}; });
    }
}
