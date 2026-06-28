/*
 * human_fitter.cpp — per-person active-inference fit wrapping the cpp/core estimator.
 */

#include "human_fitter.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <print>

#include "human_controller.h"

namespace rc {

namespace
{
// Parse the track id from a "person_<id>" node name; fallback 0.
int track_id_from_name(const std::string& name)
{
    const auto us = name.rfind('_');
    if (us == std::string::npos or us + 1 >= name.size())
        return 0;
    return std::atoi(name.c_str() + us + 1);
}

int count_valid(const human::KpArray& kp)
{
    int n = 0;
    for (int i = 0; i < human::NUM_KP; ++i)
        if (std::isfinite(kp(i, 0)) and std::isfinite(kp(i, 1)) and std::isfinite(kp(i, 2)))
            ++n;
    return n;
}
}  // namespace

HumanFitter::HumanFitter(std::shared_ptr<DSR::DSRGraph> graph, HumanConfig& cfg)
    : G_(std::move(graph)), cfg_(cfg),
      model_(human::lengths_from_standard(human::standard_template()))
{}

human::InferenceConfig HumanFitter::make_infer_config() const
{
    human::InferenceConfig c;
    c.anchors   = cfg_.anchors;
    c.sigma_obs = cfg_.sigma_obs;
    c.sigma_dyn = cfg_.sigma_dyn;
    c.sigma_min = cfg_.sigma_min;
    c.sigma_max = cfg_.sigma_max;
    c.w_limits  = cfg_.w_limits;
    c.w_sym     = cfg_.w_sym;
    c.w_cross         = cfg_.w_cross;
    c.arm_cross_margin = cfg_.arm_cross_margin;
    c.w_neutral       = cfg_.w_neutral;
    c.max_innovation  = cfg_.max_innovation;
    c.gn_steps  = cfg_.gn_steps;
    c.damping   = cfg_.damping;
    c.dt        = cfg_.dt;
    c.w_vel     = cfg_.w_vel;
    c.w_acc     = cfg_.w_acc;
    c.omega_max = cfg_.omega_max;
    c.alpha_max = cfg_.alpha_max;
    c.vlin_max  = cfg_.vlin_max;
    c.alin_max  = cfg_.alin_max;
    return c;
}

bool HumanFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_node_id)
{
    room_node_id_ = room_node_id;
    if (instances_.contains(node.id()))
        return false;

    HumanInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();
    if (const auto pid = G_->get_attrib_by_name<person_id_att>(node); pid.has_value())
        inst.track_id = pid.value();
    else
        inst.track_id = track_id_from_name(node.name());
    inst.parent_id   = room_node_id;
    inst.parent_name = "room";
    inst.estimator   = std::make_unique<human::AInfLaplacePoseEstimator>(model_, make_infer_config());
    inst.affordance.init(G_, node.id(), node.name());

    instances_.emplace(node.id(), std::move(inst));
    std::print("human_concept: instance for '{}' id={} track={}\n",
               node.name(), node.id(), instances_.at(node.id()).track_id);
    return true;
}

HumanFitter::HumanObservation HumanFitter::observe(HumanInstance& inst, const DSR::Node& /*node*/)
{
    HumanObservation obs;

    const auto it = std::find_if(frame_.begin(), frame_.end(),
                                 [&](const SkeletonBody& b) { return b.id == inst.track_id; });
    if (it == frame_.end())
        return obs;

    const int valid = count_valid(it->kp);
    if (valid < 3)
        return obs;   // not enough to anchor a fit

    obs.has_fresh_data = true;
    obs.kp           = it->kp;
    obs.conf         = it->conf;
    obs.valid_count  = valid;
    if (it->conf.has_value())
    {
        float sum = 0.0f; int n = 0;
        for (int i = 0; i < human::NUM_KP; ++i)
            if (std::isfinite(it->kp(i, 0))) { sum += (*it->conf)[i]; ++n; }
        obs.mean_conf = n ? sum / n : 0.0f;
    }
    else
        obs.mean_conf = 100.0f * static_cast<float>(valid) / human::NUM_KP;   // proxy from coverage
    return obs;
}

float HumanFitter::run_inference(HumanInstance& inst, const HumanObservation& obs)
{
    ++inst.frames_since_detection;

    float fe = inst.prev_free_energy;
    if (obs.has_fresh_data and inst.estimator)
    {
        // Measured seconds since this person's previous fresh fit (clamped) — the real dt for the
        // speed/accel limits. The compute loop may run faster than the skeleton stream, so this is
        // NOT Period.Compute; falls back to cfg_.dt on the first fit.
        const auto now = std::chrono::steady_clock::now();
        float dt = cfg_.dt;
        if (inst.last_fit_time)
            dt = std::clamp(std::chrono::duration<float>(now - *inst.last_fit_time).count(), 0.01f, 0.5f);
        inst.last_fit_time = now;

        inst.last_result = inst.estimator->infer(obs.kp, obs.conf, dt);
        inst.has_result  = true;
        ++inst.matched_frames;
        inst.frames_since_detection = 0;
        inst.last_mask_confidence   = obs.mean_conf;

        // Output controller — drive the published command angles toward the belief target (theta* =
        // last_result.mu) under velocity/accel limits, then render the SMOOTHED room-frame pose. ONLY
        // on a valid fit: a degenerate frame (too few visible joints → NaN prediction / Kabsch with
        // <3 anchors) is SKIPPED so the model HOLDS its last pose under occlusion instead of vanishing.
        if (inst.last_result.kp_pred_aligned.allFinite())
        {
            const auto& target = inst.last_result.mu;
            if (not inst.cmd_init)
            {
                inst.theta_cmd = target;        // snap to the first fit, then track
                inst.theta_vel.setZero();
                inst.cmd_init = true;
            }
            const human::RateLimits lim{cfg_.omega_max, cfg_.alpha_max, cfg_.vlin_max, cfg_.alin_max};
            const auto [vsat, asat] = human::track_angles(inst.theta_cmd, inst.theta_vel, target, dt, lim);
            if (const human::KpArray cand = inst.estimator->predict_aligned_kp(inst.theta_cmd, obs.kp);
                cand.allFinite())
            {
                inst.cmd_kp  = cand;
                inst.has_cmd = true;
            }
            inst.track_err = (target - inst.theta_cmd).cwiseAbs().mean();
            inst.last_result.vel_clamped = vsat;   // repurposed: controller speed-saturated DOFs
            inst.last_result.acc_clamped = asat;   // controller accel-saturated DOFs
        }

        // Drive the shared stabiliser's posterior precision from the estimator's Laplace precision
        // (diagonal of Lambda), so posterior_std_milli + the worst-DOF epistemic gain are live.
        for (int j = 0; j < 11; ++j)
            inst.stab.fisher_info_raw[j] = inst.last_result.Lambda(j, j);

        if (std::isfinite(inst.last_result.mean_l2))
            fe = inst.last_result.mean_l2;
    }

    inst.detection_alive = inst.frames_since_detection < 5;
    return fe;
}

bool HumanFitter::should_log(const HumanInstance& inst) const
{
    return (inst.processed_cycles % 30) == 0;
}

}  // namespace rc
