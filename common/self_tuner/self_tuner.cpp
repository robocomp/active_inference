#include "self_tuner.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <print>
#include <sstream>
#include <utility>

namespace rc::tune
{

SelfTuner::SelfTuner(std::vector<Parameter> params, TunerConfig cfg)
    : params_(std::move(params)), cfg_(cfg),
      best_obj_(cfg.goal == Goal::Minimize ? std::numeric_limits<double>::infinity()
                                           : -std::numeric_limits<double>::infinity()),
      radius_(cfg.init_radius),
      rng_(cfg.seed != 0 ? cfg.seed : std::random_device{}())
{
    best_x_.reserve(params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i)
        best_x_.push_back(clip(i, params_[i].initial));
    cand_x_ = best_x_;

    if (not cfg_.state_path.empty())
        load();   // resume a prior incumbent if one was persisted
}

bool SelfTuner::better(double a, double b) const
{
    return cfg_.goal == Goal::Minimize ? a < b : a > b;
}

double SelfTuner::clip(std::size_t i, double v) const
{
    return std::clamp(v, params_[i].min, params_[i].max);
}

std::optional<std::size_t> SelfTuner::index_of(const std::string& name) const
{
    for (std::size_t i = 0; i < params_.size(); ++i)
        if (params_[i].name == name)
            return i;
    return std::nullopt;
}

void SelfTuner::roll_candidate()
{
    for (std::size_t i = 0; i < params_.size(); ++i)
    {
        const double sigma = radius_ * (params_[i].max - params_[i].min);
        std::normal_distribution<double> step(0.0, std::max(sigma, 0.0));
        cand_x_[i] = clip(i, best_x_[i] + step(rng_));
    }
}

void SelfTuner::propose()
{
    // Keep the current candidate while it is still collecting its repeats.
    if (candidate_live_ and obj_count_ > 0 and obj_count_ < cfg_.repeats)
        return;

    if (cfg_.explore and allowed_)
        roll_candidate();
    else
        cand_x_ = best_x_;   // exploitation / coordination-gated: run the incumbent

    candidate_live_ = true;
    obj_sum_   = 0.0;
    obj_count_ = 0;
}

double SelfTuner::value(const std::string& name, double fallback) const
{
    if (auto i = index_of(name))
        return cand_x_[*i];
    std::print(stderr, "[self_tuner] unknown parameter '{}' — using fallback {}\n", name, fallback);
    return fallback;
}

double SelfTuner::value(std::size_t idx) const
{
    return idx < cand_x_.size() ? cand_x_[idx] : 0.0;
}

void SelfTuner::report(double objective)
{
    ++evals_;

    // A non-finite objective = a crashed/aborted episode. Don't average it in
    // (it would poison the candidate's mean); finalize the candidate as a miss.
    const bool aborted = not std::isfinite(objective);
    if (not aborted)
    {
        obj_sum_ += objective;
        ++obj_count_;
        if (obj_count_ < cfg_.repeats)
            return;   // still collecting repeats for this candidate
    }

    const double mean = obj_count_ > 0 ? obj_sum_ / obj_count_
                                       : (cfg_.goal == Goal::Minimize
                                              ? std::numeric_limits<double>::infinity()
                                              : -std::numeric_limits<double>::infinity());

    const bool exploring = cfg_.explore and allowed_;
    if (not exploring)
    {
        // Serving the incumbent: just refresh its objective estimate (handles slow
        // drift in inputs), no move, no radius change.
        if (not aborted)
            best_obj_ = mean;
    }
    else if (not aborted and better(mean, best_obj_))
    {
        best_x_   = cand_x_;
        best_obj_ = mean;
        radius_   = std::min(cfg_.max_radius, radius_ * cfg_.radius_grow);
        if (not cfg_.state_path.empty())
            save();
    }
    else
    {
        radius_ = std::max(cfg_.min_radius, radius_ * cfg_.radius_shrink);
    }

    candidate_live_ = false;
    obj_sum_   = 0.0;
    obj_count_ = 0;
}

void SelfTuner::set_exploration_allowed(bool allowed)
{
    allowed_ = allowed;
}

std::vector<std::pair<std::string, double>> SelfTuner::incumbent_named() const
{
    std::vector<std::pair<std::string, double>> out;
    out.reserve(params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i)
        out.emplace_back(params_[i].name, best_x_[i]);
    return out;
}

bool SelfTuner::save() const
{
    std::ofstream f(cfg_.state_path, std::ios::trunc);
    if (not f)
        return false;
    f << "# rc::tune::SelfTuner state v1\n";
    f << "best_objective " << best_obj_ << "\n";
    f << "evaluations "    << evals_    << "\n";
    f << "radius "         << radius_   << "\n";
    for (std::size_t i = 0; i < params_.size(); ++i)
        f << params_[i].name << ' ' << best_x_[i] << '\n';
    return static_cast<bool>(f);
}

bool SelfTuner::load()
{
    std::ifstream f(cfg_.state_path);
    if (not f)
        return false;
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() or line[0] == '#')
            continue;
        std::istringstream is(line);
        std::string key;
        is >> key;
        if (key == "best_objective") { is >> best_obj_; continue; }
        if (key == "evaluations")    { is >> evals_;    continue; }
        if (key == "radius")         { is >> radius_;   continue; }
        // Otherwise key is a parameter name → restore its incumbent value (clipped
        // to the CURRENT range, so a tightened range can't load an unsafe value).
        double v;
        if (is >> v)
            if (auto i = index_of(key))
                best_x_[*i] = clip(*i, v);
    }
    cand_x_ = best_x_;
    return true;
}

std::optional<Parameter> parse_parameter(std::string_view name, std::string_view spec)
{
    std::istringstream is{std::string(spec)};
    Parameter p;
    p.name = std::string(name);
    if (not (is >> p.min >> p.max))
        return std::nullopt;
    if (p.max < p.min)
        return std::nullopt;
    if (not (is >> p.initial))
        p.initial = 0.5 * (p.min + p.max);   // default: midpoint of the range
    p.initial = std::clamp(p.initial, p.min, p.max);
    return p;
}

}  // namespace rc::tune
