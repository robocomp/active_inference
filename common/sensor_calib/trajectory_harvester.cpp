#include "trajectory_harvester.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
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

void TrajectoryHarvester::add_gyro(std::int64_t ts, double rate)
{
    if (not std::isfinite(rate)) return;
    GNode n;
    if (not gyro_.empty())
    {
        if (ts <= gyro_.back().ts) return;
        const double dt = double(ts - gyro_.back().ts) * 1e-3;
        n.angle = gyro_.back().angle + gyro_.back().rate * dt;
        n.abs   = gyro_.back().abs + std::abs(gyro_.back().rate) * dt;
    }
    n.ts = ts; n.rate = rate;
    gyro_.push_back(n);
    const std::int64_t keep = std::int64_t(cfg_.history_s * 1000.0);
    while (gyro_.size() > 2 and gyro_[1].ts < ts - keep) gyro_.pop_front();
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
    if (not gyro_.empty() and stamp >= gyro_.front().ts)
    {
        const auto g = std::upper_bound(gyro_.begin(), gyro_.end(), stamp, [](std::int64_t t, const GNode& q) { return t < q.ts; }) - 1;
        const double dt = double(stamp - g->ts) * 1e-3;
        k.gth = g->angle + g->rate * dt; k.gabs = g->abs + std::abs(g->rate) * dt; k.has_gyro = true;
    }
    ring_.push_back(k);
    last_kf_ms_ = stamp; have_kf_ = true;
    if (ep_.keyframes == 0) ep_.t_begin_ms = stamp;
    ep_.t_end_ms = stamp;
    ++ep_.keyframes;
    while (over_budget()) commit(false);
    return true;
}

std::optional<std::int64_t> TrajectoryHarvester::oldest_pending_stamp_ms() const
{
    std::optional<std::int64_t> m;
    if (not ring_.empty()) m = ring_.front().stamp_ms;
    for (const auto& p : out_) m = m ? std::min(*m, p.stamp_a_ms) : p.stamp_a_ms;
    return m;
}

void TrajectoryHarvester::finish() { while (ring_.size() >= 2) commit(true); ring_.clear(); }

// Wheel speed integrated along the heading fused PER ODOMETRY INTERVAL (precision weighting of wheel and gyro
// rotation over that interval), in the frame of the heading at t0. ok=false when the odometry history does not cover it.
TrajectoryHarvester::Chord TrajectoryHarvester::integrate_chord(std::int64_t t0, std::int64_t t1) const
{
    Chord c;
    if (hist_.empty() or t0 < hist_.front().ts or t1 <= t0) return c;
    auto gyro_angle = [this](std::int64_t t) -> std::optional<double>
    {
        if (gyro_.empty() or t < gyro_.front().ts) return std::nullopt;
        const auto g = std::upper_bound(gyro_.begin(), gyro_.end(), t, [](std::int64_t q, const GNode& n) { return q < n.ts; }) - 1;
        return g->angle + g->rate * double(t - g->ts) * 1e-3;
    };
    auto it = std::upper_bound(hist_.begin(), hist_.end(), t0, [](std::int64_t t, const Node& n) { return t < n.ts; }) - 1;
    double th = 0.0;
    for (std::int64_t s0 = t0; s0 < t1; ++it)
    {
        const std::int64_t s1 = (it + 1 == hist_.end()) ? t1 : std::min(t1, (it + 1)->ts);
        const double dt = double(s1 - s0) * 1e-3;
        if (dt > 0.0)
        {
            const double dw = it->rot * dt;
            double d = dw;
            const auto g0 = gyro_angle(s0), g1 = gyro_angle(s1);
            if (g0 and g1)
            {
                const double dg = *g1 - *g0;
                const double vw = std::pow(cfg_.wheel_rot_sigma_rel * dw, 2) + cfg_.wheel_rot_sigma_floor * cfg_.wheel_rot_sigma_floor * dt;
                const double vg = std::pow(cfg_.gyro_sigma_rel * dg, 2) + cfg_.gyro_sigma_floor * cfg_.gyro_sigma_floor * dt;
                const double wg = 1.0 / std::max(vg, 1e-12), ww = 1.0 / std::max(vw, 1e-12);
                d = (dg * wg + dw * ww) / (wg + ww);
            }
            const double thm = th + 0.5 * d;
            c.cx += (it->adv * std::cos(thm) - it->side * std::sin(thm)) * dt;
            c.cy += (it->adv * std::sin(thm) + it->side * std::cos(thm)) * dt;
            th += d;
        }
        s0 = s1;
        if (it + 1 == hist_.end()) break;
    }
    c.dth = th; c.ok = true;
    return c;
}

Pair TrajectoryHarvester::pair_of(const Keyframe& a, const Keyframe& b) const
{
    const double dtw = b.th - a.th, rot_w = b.rot - a.rot, dt_s = double(b.stamp_ms - a.stamp_ms) * 1e-3;
    const double vw = std::pow(cfg_.wheel_rot_sigma_rel * rot_w, 2) + cfg_.wheel_rot_sigma_floor * cfg_.wheel_rot_sigma_floor * dt_s;
    double dth = dtw, var_th = vw, rotated = rot_w, dtg = dtw;
    if (a.has_gyro and b.has_gyro)
    {   // precision-weighted fusion of the two heading factors
        dtg = b.gth - a.gth; rotated = b.gabs - a.gabs;
        const double vg = std::pow(cfg_.gyro_sigma_rel * rotated, 2) + cfg_.gyro_sigma_floor * cfg_.gyro_sigma_floor * dt_s;
        const double wg = 1.0 / std::max(vg, 1e-12), ww = 1.0 / std::max(vw, 1e-12);
        dth = (dtg * wg + dtw * ww) / (wg + ww);
        var_th = 1.0 / (wg + ww);
    }
    // chord along the FUSED heading (frame of a), decomposed on the fused midpoint heading; wheel fallback if unresolved
    Chord ch = integrate_chord(a.stamp_ms, b.stamp_ms);
    double thm = 0.5 * ch.dth, cx = ch.cx, cy = ch.cy;
    if (not ch.ok)
    {
        thm = a.th + 0.5 * dtw; cx = b.x - a.x; cy = b.y - a.y;
    }
    Pair p = make_pair_from_motion(dth, cx * std::cos(thm) + cy * std::sin(thm), -cx * std::sin(thm) + cy * std::cos(thm),
                                   b.path - a.path, rotated, cfg_);
    p.chord_sigma = std::hypot(cx, cy) * std::sqrt(var_th);
    p.dth_sigma = std::sqrt(var_th); p.dth_wheel = dtw; p.dth_gyro = dtg;
    p.id_a = a.id; p.id_b = b.id; p.stamp_a_ms = a.stamp_ms; p.stamp_b_ms = b.stamp_ms;
    return p;
}

void TrajectoryHarvester::commit(bool final_flush)
{
    const int K = int(ring_.size());
    if (K < 2) { ring_.clear(); return; }
    const Eigen::Matrix3d Sigma = (H_acc_ + H_prior_).inverse();

    // v(a,b) = tr(Sigma I_ab): prior-scaled information trace; additive over disjoint pairs => segmentation by DP.
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
