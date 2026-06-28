/*
 * human_config.cpp — fill HumanConfig from a RoboComp ConfigLoader.
 */

#include "human_config.h"

#include <algorithm>
#include <print>
#include <sstream>

#include <genericworker.h>   // ConfigLoader

namespace rc {

namespace
{
std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (not tok.empty()) out.push_back(std::atoi(tok.c_str()));
    return out;
}
}  // namespace

HumanConfig load_human_config(const ConfigLoader& cfg)
{
    HumanConfig out;

    auto getf = [&](const std::string& k, float def) -> float {
        return cfg.exists(k) ? static_cast<float>(cfg.get<double>(k)) : def;
    };
    auto geti = [&](const std::string& k, int def) -> int {
        return cfg.exists(k) ? cfg.get<int>(k) : def;
    };
    auto gets = [&](const std::string& k, std::string def) -> std::string {
        return cfg.exists(k) ? cfg.get<std::string>(k) : def;
    };
    auto getb = [&](const std::string& k, bool def) -> bool {
        return cfg.exists(k) ? cfg.get<bool>(k) : def;
    };

    out.priors_path = gets("HumanConcept.PriorsPath", "etc/object_priors.toml");

    out.source_kind = gets("HumanConcept.SourceKind", "replay");
    out.replay_path = gets("HumanConcept.ReplayPath", "");
    out.replay_loop = getb("HumanConcept.ReplayLoop", true);

    if (const auto a = gets("HumanModel.Anchors", ""); not a.empty())
        out.anchors = parse_int_list(a);
    out.sigma_obs = getf("HumanModel.SigmaObs", 0.06f);
    out.sigma_dyn = getf("HumanModel.SigmaDyn", 0.25f);
    out.sigma_min = getf("HumanModel.SigmaMin", 0.02f);
    out.sigma_max = getf("HumanModel.SigmaMax", 0.15f);
    out.w_limits  = getf("HumanModel.WLimits", 5.0f);
    out.w_sym     = getf("HumanModel.WSym", 1.0f);
    out.gn_steps  = geti("HumanModel.GnSteps", 2);
    out.damping   = getf("HumanModel.Damping", 1e-3f);

    out.w_cross          = getf("HumanModel.WCross", 3.0f);
    out.arm_cross_margin = getf("HumanModel.ArmCrossMargin", 0.05f);
    out.w_neutral        = getf("HumanModel.WNeutral", 1.0f);
    out.max_innovation   = getf("HumanModel.MaxInnovation", 1.0f);

    // Speed/accel limits (applied by the output controller). dt fallback derives from Period.Compute;
    // the runtime uses the measured inter-fit interval. In-fit penalties default OFF (controller owns).
    out.dt        = getf("HumanModel.Dt", std::max(1, geti("Period.Compute", 50)) / 1000.0f);
    out.w_vel     = getf("HumanModel.WVel", 0.0f);
    out.w_acc     = getf("HumanModel.WAcc", 0.0f);
    out.omega_max = getf("HumanModel.OmegaMax", 8.0f);
    out.alpha_max = getf("HumanModel.AlphaMax", 80.0f);
    out.vlin_max  = getf("HumanModel.VLinMax", 3.0f);
    out.alin_max  = getf("HumanModel.ALinMax", 30.0f);
    out.fit_csv_path = gets("HumanModel.FitCsvPath", "");

    out.death_frames       = geti("HumanConcept.DeathFrames", 60);
    out.min_valid          = geti("HumanConcept.MinValid", 12);
    out.uncertainty_thresh = getf("HumanConcept.UncertaintyThresh", 0.05f);

    out.epistemic_obs_distance    = getf("Epistemic.ObsDistance", 1.5f);
    out.epistemic_view_info       = getf("Epistemic.ViewInfo", 50.0f);
    out.epistemic_cooldown_cycles = geti("Epistemic.CooldownCycles", 120);
    out.epistemic_csv_path        = gets("Epistemic.CsvPath", "");

    out.still_vel   = getf("Epistemic.StillVel", 0.10f);
    out.still_omega = getf("Epistemic.StillOmega", 0.15f);

    out.fisher_csv_path = gets("HumanModel.FisherCsvPath", "");

    out.pose_cov_scale = getf("HumanConcept.PoseCovScale", 0.02f);

    std::print("human_concept: configuration loaded (source='{}', replay='{}').\n",
               out.source_kind, out.replay_path);
    return out;
}

}  // namespace rc
