#include "trajectory_harvester.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace sc::calib
{

Pair make_pair_from_motion(double dth, double d_fwd, double d_lat, double path, double rotated, const HarvesterConfig& cfg)
{
    Pair p;
    p.dth = dth; p.d_fwd = d_fwd; p.d_lat = d_lat; p.path = path; p.rotated = rotated;
    const double s = 2.0 * std::sin(0.5 * dth);
    p.J << s, 0.0, 0.0,
           0.0, -s, d_fwd;
    const double v = cfg.reg_sigma_m * cfg.reg_sigma_m + std::pow(cfg.odo_sigma_per_m * path, 2)
                   + std::pow(cfg.odo_sigma_per_rad * rotated, 2);
    p.var = Eigen::Vector2d(v, v);
    p.info = p.J.transpose() * p.var.cwiseInverse().asDiagonal() * p.J;
    return p;
}

// ───────────────────────────── MountEvidence ─────────────────────────────

MountEvidence::MountEvidence(const HarvesterConfig& cfg)
{
    Hp_ = Eigen::Vector3d(1.0 / (cfg.prior_sigma_xy * cfg.prior_sigma_xy), 1.0 / (cfg.prior_sigma_xy * cfg.prior_sigma_xy),
                          1.0 / (cfg.prior_sigma_yaw * cfg.prior_sigma_yaw)).asDiagonal();
}
void MountEvidence::add(const Pair& p, double y_fwd, double y_lat)
{
    if (not std::isfinite(y_fwd) or not std::isfinite(y_lat)) return;
    const Eigen::Vector2d y(y_fwd, y_lat);
    H_ += p.info;
    b_ += p.J.transpose() * p.var.cwiseInverse().asDiagonal() * y;
    ++n_;
}
Eigen::Vector3d MountEvidence::mean() const { return (H_ + Hp_).ldlt().solve(b_); }
Eigen::Vector3d MountEvidence::sigma() const { return (H_ + Hp_).inverse().diagonal().cwiseMax(0.0).cwiseSqrt(); }

// ───────────────────────────── TrajectoryHarvester ─────────────────────────────

TrajectoryHarvester::TrajectoryHarvester(HarvesterConfig cfg) : cfg_(cfg)
{
    H_prior_ = Eigen::Vector3d(1.0 / (cfg.prior_sigma_xy * cfg.prior_sigma_xy), 1.0 / (cfg.prior_sigma_xy * cfg.prior_sigma_xy),
                               1.0 / (cfg.prior_sigma_yaw * cfg.prior_sigma_yaw)).asDiagonal();
}

TrajectoryHarvester::Node TrajectoryHarvester::advance(const Node& n, double dt)
{
    Node r = n;
    const double thm = n.th + 0.5 * n.rot * dt;
    r.x += (n.adv * std::cos(thm) - n.side * std::sin(thm)) * dt;
    r.y += (n.adv * std::sin(thm) + n.side * std::cos(thm)) * dt;
    r.th += n.rot * dt;
    r.path += std::hypot(n.adv, n.side) * dt;
    r.rot_acc += std::abs(n.rot) * dt;
    return r;
}

void TrajectoryHarvester::add_odometry(std::int64_t ts, double adv, double side, double rot)
{
    if (not std::isfinite(adv) or not std::isfinite(side) or not std::isfinite(rot)) return;
    Node n;
    if (not hist_.empty())
    {
        if (ts <= hist_.back().ts) return;
        n = advance(hist_.back(), double(ts - hist_.back().ts) * 1e-3);
    }
    n.ts = ts; n.adv = adv; n.side = side; n.rot = rot;
    hist_.push_back(n);
    const std::int64_t keep = std::int64_t(cfg_.history_s * 1000.0);
    while (hist_.size() > 2 and hist_[1].ts < ts - keep) hist_.pop_front();
}

bool TrajectoryHarvester::over_budget() const
{
    if (ring_.size() < 2) return false;
    return ring_.size() > cfg_.max_keyframes
        or double(ring_.back().stamp_ms - ring_.front().stamp_ms) * 1e-3 > cfg_.ring_span_s;
}

bool TrajectoryHarvester::add_keyframe(std::int64_t stamp, bool stalled)
{
    if (stalled)
    {   // HOLD: nothing of a stalled stream is ingested, and no pair may span the gap
        ++ep_.dropped_stalled;
        finish();
        have_kf_ = false;
        return false;
    }
    if (have_kf_ and double(stamp - last_kf_ms_) * 1e-3 < cfg_.keyframe_spacing_s) return false;   // budget decimator
    if (hist_.empty() or stamp < hist_.front().ts) return false;
    // the node in force at `stamp`, advanced over the remainder of its hold interval
    auto it = std::upper_bound(hist_.begin(), hist_.end(), stamp, [](std::int64_t t, const Node& n) { return t < n.ts; });
    const Node n = advance(*(it - 1), double(stamp - (it - 1)->ts) * 1e-3);
    Keyframe k{next_id_++, stamp, n.x, n.y, n.th, n.path, n.rot_acc};
    ring_.push_back(k);
    last_kf_ms_ = stamp; have_kf_ = true;
    if (ep_.keyframes == 0) ep_.t_begin_ms = stamp;
    ep_.t_end_ms = stamp;
    ++ep_.keyframes;
    while (over_budget()) commit(false);
    return true;
}

void TrajectoryHarvester::finish() { while (ring_.size() >= 2) commit(true); ring_.clear(); }

Pair TrajectoryHarvester::pair_of(const Keyframe& a, const Keyframe& b) const
{
    const double dth = b.th - a.th, thm = a.th + 0.5 * dth;
    const double cx = b.x - a.x, cy = b.y - a.y;
    Pair p = make_pair_from_motion(dth, cx * std::cos(thm) + cy * std::sin(thm), -cx * std::sin(thm) + cy * std::cos(thm),
                                   b.path - a.path, b.rot - a.rot, cfg_);
    p.id_a = a.id; p.id_b = b.id; p.stamp_a_ms = a.stamp_ms; p.stamp_b_ms = b.stamp_ms;
    return p;
}

void TrajectoryHarvester::commit(bool final_flush)
{
    const int K = int(ring_.size());
    if (K < 2) { ring_.clear(); return; }
    const Eigen::Matrix3d Sigma = (H_acc_ + H_prior_).inverse();

    // v(a,b) = tr(Sigma I_ab): first-order log det gain; additive over disjoint pairs => segmentation by DP.
    std::vector<std::vector<Pair>> P(K, std::vector<Pair>(K));
    std::vector<std::vector<double>> v(K, std::vector<double>(K, 0.0));
    for (int a = 0; a < K; ++a)
        for (int b = a + 1; b < K; ++b)
        {
            P[a][b] = pair_of(ring_[a], ring_[b]);
            v[a][b] = (Sigma * P[a][b].info).trace();
        }
    // best[i] = best value over keyframes [0, i); from[i] = start of the segment ending at i-1 (or -1: none)
    std::vector<double> best(K + 1, 0.0);
    std::vector<int> from(K + 1, -1);
    for (int i = 2; i <= K; ++i)
    {
        best[i] = best[i - 1]; from[i] = -1;
        for (int a = 0; a <= i - 2; ++a)
            if (const double c = best[a] + v[a][i - 1]; c > best[i] and v[a][i - 1] > 0.0) { best[i] = c; from[i] = a; }
    }
    std::vector<std::pair<int, int>> segs;
    for (int i = K; i >= 2;)
    {
        if (from[i] < 0) { --i; continue; }
        segs.emplace_back(from[i], i - 1);
        i = from[i];
    }
    std::ranges::reverse(segs);

    // A pair touching the newest keyframe may be a turn still in progress: keep it, unless forced.
    std::vector<std::pair<int, int>> chosen;
    for (const auto& s : segs) if (final_flush or s.second < K - 1) chosen.push_back(s);
    int erase_to = -1;                                    // last consumed index
    if (chosen.empty() and not final_flush and over_budget())
    {   // over budget with only an open segment (or none): commit it / drop the oldest so the ring shrinks
        chosen = segs;
        erase_to = segs.empty() ? 0 : K - 1;
    }
    for (const auto& s : chosen)
    {
        const Pair& p = P[s.first][s.second];
        H_acc_ += p.info; ep_.info += p.info; ++ep_.pairs;
        out_.push_back(p);
        erase_to = std::max(erase_to, s.second);
    }
    if (final_flush and chosen.empty()) erase_to = K - 1;
    ring_.erase(ring_.begin(), ring_.begin() + (erase_to + 1));
}

std::vector<Pair> TrajectoryHarvester::take_pairs() { return std::exchange(out_, {}); }

EpisodeSummary TrajectoryHarvester::close_episode() { return std::exchange(ep_, {}); }

}   // namespace sc::calib
