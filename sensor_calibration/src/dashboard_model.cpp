#include "dashboard_model.h"

#include <algorithm>

namespace sc::dash
{

Label classify(const Eigen::Matrix3d& info, const Priors& pr)
{
    const double lever = (info(0, 0) + info(1, 1)) * pr.var_xy;
    const double yaw   = info(2, 2) * pr.var_yaw;
    return lever > yaw ? Label::lever : Label::yaw;
}

void DashboardModel::advance(std::int64_t t_ms) { newest_ms_ = std::max(newest_ms_, t_ms); }

void DashboardModel::add_pair(const calib::Pair& p)
{
    const Label l = classify(p.info, pr_);
    ++total_[idx(l)];
    ++episode_[idx(l)];
    info_diag_ += p.info.diagonal();
    frags_.push_back({std::min(p.stamp_a_ms, p.stamp_b_ms), std::max(p.stamp_a_ms, p.stamp_b_ms), l});
    advance(frags_.back().t1_ms);
    prune();
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
    advance(stamp_ms);
}

void DashboardModel::sample_stall(std::int64_t wall_ms, bool stalled)
{
    if (not have_offset_)
        return;
    const std::int64_t t = wall_ms + offset_ms_;
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

void DashboardModel::close_episode() { episode_[0] = episode_[1] = 0; }

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
