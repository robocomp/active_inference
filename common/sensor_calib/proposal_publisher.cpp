#include "proposal_publisher.h"

#include <algorithm>
#include <cmath>

namespace sc::calib
{

MountPosterior MountPosterior::prior(double sigma_xyz, double sigma_rpy)
{
    MountPosterior m;
    for (std::size_t i = 0; i < kMountDoF; ++i)
    {
        const double s = i < 3 ? sigma_xyz : sigma_rpy;
        m.cov[i * kMountDoF + i] = s * s;
        m.prior_precision[i] = 1.0 / (s * s);
    }
    return m;
}

double MountPosterior::sigma(std::size_t i) const { return std::sqrt(std::max(0.0, cov[i * kMountDoF + i])); }

double MountPosterior::informed(std::size_t i) const
{
    return prior_precision[i] > 0.0 ? data_precision[i] / prior_precision[i] : 0.0;
}

std::vector<float> pack_proposal(const MountPosteriors& p)
{
    std::vector<float> v;
    v.reserve(kProposalFloats);
    for (const auto& m : p)
    {
        for (const double d : m.mean) v.push_back(static_cast<float>(d));
        for (const double c : m.cov) v.push_back(static_cast<float>(c));
    }
    return v;
}

std::optional<MountPosteriors> unpack_proposal(const std::vector<float>& v)
{
    if (v.size() != kProposalFloats)
        return std::nullopt;
    MountPosteriors p{};
    for (std::size_t s = 0; s < kMountSensors; ++s)
    {
        const float* f = v.data() + s * kFloatsPerSensor;
        for (std::size_t i = 0; i < kMountDoF; ++i) p[s].mean[i] = f[i];
        for (std::size_t i = 0; i < kMountDoF * kMountDoF; ++i) p[s].cov[i] = f[kMountDoF + i];
    }
    return p;
}

std::optional<ProposalPayload> ProposalPublisher::offer(const MountPosteriors& p, std::int64_t now_ms)
{
    if (last_)
    {
        if (now_ms - last_ms_ < cfg_.min_interval_ms)
            return std::nullopt;
        bool moved = false;
        for (std::size_t s = 0; s < kMountSensors and not moved; ++s)
            for (std::size_t i = 0; i < kMountDoF; ++i)
                if (std::abs(p[s].mean[i] - (*last_)[s].mean[i]) > cfg_.move_sigma * p[s].sigma(i))
                {
                    moved = true;
                    break;
                }
        if (not moved)
            return std::nullopt;
    }
    last_ = p;
    last_ms_ = now_ms;
    return ProposalPayload{pack_proposal(p), ++version_, false};
}

}   // namespace sc::calib
