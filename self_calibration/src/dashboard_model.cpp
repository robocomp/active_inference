#include "dashboard_model.h"

#include <algorithm>
#include <cstdio>

namespace sc::dash
{

Label classify(const Eigen::Matrix3d& info, const Priors& pr)
{
    const double lever = (info(0, 0) + info(1, 1)) * pr.var_xy;
    const double yaw   = info(2, 2) * pr.var_yaw;
    return lever > yaw ? Label::lever : Label::yaw;
}

// Wall clock only. A backward step (NTP, manual set) cannot be placed on the old axis: restart the timeline.
void DashboardModel::advance(std::int64_t wall_ms)
{
    if (wall_ms < newest_ms_)
    {
        frags_.clear();
        stalls_.clear();
        stall_open_ = false;
        ++resets_;
        std::fprintf(stderr, "[Dashboard] wall clock stepped back %lld ms: timeline reset\n", static_cast<long long>(newest_ms_ - wall_ms));
    }
    newest_ms_ = wall_ms;
}

void DashboardModel::add_pair(const calib::Pair& p, std::int64_t wall_ms)
{
    advance(wall_ms);
    const Label l = classify(p.info, pr_);
    ++total_[idx(l)];
    info_diag_ += p.info.diagonal();
    // helios -> wall with the offset current NOW; without a sweep yet, assume the pair ends at "now".
    const std::int64_t off = have_offset_ ? offset_ms_ : std::max(p.stamp_a_ms, p.stamp_b_ms) - wall_ms;
    frags_.push_back({std::min(p.stamp_a_ms, p.stamp_b_ms) - off, std::max(p.stamp_a_ms, p.stamp_b_ms) - off, l});
    prune();
}

long DashboardModel::pairs_recent(Label l) const
{
    return std::ranges::count_if(frags_, [&](const Fragment& f) { return f.label == l and f.t1_ms >= newest_ms_ - recent_ms(); });
}

std::vector<StreamRow> StreamRates::update(std::int64_t wall_ms, const std::vector<StreamSample>& samples)
{
    const double dt = window_start_ms_ < 0 ? 0.0 : double(wall_ms - window_start_ms_) / 1000.0;
    const bool roll = window_start_ms_ < 0 or dt >= 1.0;
    std::vector<StreamRow> rows;
    for (const auto& s : samples)
    {
        if (roll)
        {
            const auto it = totals_.find(s.key);
            rates_[s.key] = (it != totals_.end() and dt > 0.0) ? double(s.total - it->second) / dt : 0.0;
            totals_[s.key] = s.total;
        }
        rows.push_back({s.key, rates_[s.key],
                        not s.ever_seen ? StreamStatus::waiting : (s.stalled ? StreamStatus::stalled : StreamStatus::live), s.show_rate});
    }
    if (roll)
        window_start_ms_ = wall_ms;
    return rows;
}

void DashboardModel::note_keyframe(bool captured, bool stalled)
{
    if (captured)
        ++kf_;
    else if (stalled)
        ++kf_dropped_;
}

void DashboardModel::note_lidar_stamp(std::int64_t stamp_ms, std::int64_t wall_ms)
{
    offset_ms_ = stamp_ms - wall_ms;
    have_offset_ = true;
}

void DashboardModel::sample_stall(std::int64_t wall_ms, bool stalled)
{
    const std::int64_t t = wall_ms;
    advance(t);
    if (stalled)
    {
        if (stall_open_)
            stalls_.back().t1_ms = t;
        else
        {
            stalls_.push_back({t, t});
            stall_open_ = true;
        }
    }
    else
        stall_open_ = false;
    prune();
}

std::array<double, 3> DashboardModel::ratios() const
{
    return {info_diag_(0) * pr_.var_xy, info_diag_(1) * pr_.var_xy, info_diag_(2) * pr_.var_yaw};
}

void DashboardModel::prune()
{
    const std::int64_t cut = newest_ms_ - window_ms_;
    while (not frags_.empty() and frags_.front().t1_ms < cut)
        frags_.pop_front();
    while (not stalls_.empty() and stalls_.front().t1_ms < cut)
    {
        if (stall_open_ and stalls_.size() == 1)
            break;   // never drop the interval still being extended
        stalls_.pop_front();
    }
}

}   // namespace sc::dash
