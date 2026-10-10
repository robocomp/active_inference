/*
 * human_config.cpp — fill HumanConfig from a RoboComp ConfigLoader.
 */

#include "human_config.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <algorithm>
#include <cstdlib>
#include <print>

#include "../../common/concept_manifest/concept_manifest.h"   // rc::manifest (SHARED)
#include <sstream>

#include <genericworker.h>   // ConfigLoader

#include "csv_parse.h"       // locale-independent number parsing (see the header)

namespace rc {

namespace
{
std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (not tok.empty()) out.push_back(csv::parse_int(tok));
    return out;
}
}  // namespace

HumanConfig load_human_config(const ConfigLoader& cfg)
{
    HumanConfig out;


    // The one place this path is written. Relative to the agent's CWD (<agent>/), not to src/ — the same
    // string was right in one place and wrong in the other for a week, and the manifest was inert the whole time.
    static constexpr const char* kManifestPath = "../common/concept_manifest/human.concept.toml";
    // ★★AN INHERITED WORLD FACT IS FATAL — rc::manifest::provenance_ok. `from = "inherited"` means the number
    // arrived by a rename and nobody chose it for THIS object; hood shipped ten such defects in a week with
    // several of them declared, in writing, in its manifest. A note stopped nothing, so this stops the agent.
    if (not rc::manifest::provenance_ok(kManifestPath, "human"))
        std::exit(EXIT_FAILURE);

    // Every read below registers its key, its CODE DEFAULT and a one-line description,
    // so the startup table can say where each value came from - not just what it is.
    // (The four local lambdas now forward to the shared registry; the call sites are
    // unchanged except for that description.)
    rc::cfg::Reader reader(cfg, "human_concept");
    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };

    out.priors_path = gets("HumanConcept.PriorsPath", "etc/object_priors.toml",
            "");

    out.source_kind = gets("HumanConcept.SourceKind", "replay",
            "'replay' (CSV) | 'live' (future: ZED / media plane)");
    out.replay_path = gets("HumanConcept.ReplayPath", "",
            "CSV of keypoints (frame[,id],54 kp[,18 conf]); empty = none");
    out.replay_loop = getb("HumanConcept.ReplayLoop", true,
            "wrap to the first frame at EOF");

    if (const auto a = gets("HumanModel.Anchors", "",
            "neck, shoulders, hips"); not a.empty())
        out.anchors = parse_int_list(a);
    out.sigma_obs = getf("HumanModel.SigmaObs", 0.06f,
            "");
    out.sigma_dyn = getf("HumanModel.SigmaDyn", 0.25f,
            "");
    out.sigma_min = getf("HumanModel.SigmaMin", 0.02f,
            "");
    out.sigma_max = getf("HumanModel.SigmaMax", 0.15f,
            "");
    out.min_kp_conf = getf("HumanModel.MinKpConf", 15.0f,
            "[0,100] hard floor: drop keypoints below this from fit + calibration");
    out.w_limits  = getf("HumanModel.WLimits", 5.0f,
            "");
    out.w_sym     = getf("HumanModel.WSym", 1.0f,
            "");
    out.gn_steps  = geti("HumanModel.GnSteps", 2,
            "");
    out.damping   = getf("HumanModel.Damping", 1e-3f,
            "");

    out.w_cross          = getf("HumanModel.WCross", 3.0f,
            "Anti arm-cross (sidedness) prior — filters YOLO L/R swaps");
    out.arm_cross_margin = getf("HumanModel.ArmCrossMargin", 0.05f,
            "m past the opposite shoulder before it penalises");
    out.w_neutral        = getf("HumanModel.WNeutral", 1.0f,
            "Neutral-pose prior — weak pull of arm DOFs toward rest; pins under-observed DOFs (stops drift)");
    out.max_innovation   = getf("HumanModel.MaxInnovation", 1.0f,
            "Innovation gate (rad): reject + hold on a per-frame angle jump bigger than this (glitch filter)");

    // Speed/accel limits (applied by the output controller). dt fallback derives from Period.Compute;
    // the runtime uses the measured inter-fit interval. In-fit penalties default OFF (controller owns).
    out.dt        = getf("HumanModel.Dt", std::max(1, geti("Period.Compute", 50,
            "20 Hz — also drives the kinematic speed/accel-limit dt (HumanModel.dt)")) / 1000.0f,
            "fallback fit interval (s); the runtime uses the MEASURED inter-fit interval, and this "
            "only stands in until there is one");
    out.w_vel     = getf("HumanModel.WVel", 0.0f,
            "in-fit velocity penalty (OFF — controller owns this)");
    out.w_acc     = getf("HumanModel.WAcc", 0.0f,
            "in-fit acceleration penalty (OFF)");
    out.omega_max = getf("HumanModel.OmegaMax", 8.0f,
            "rad/s (angle DOFs) — natural limb speed");
    out.alpha_max = getf("HumanModel.AlphaMax", 80.0f,
            "rad/s² (angle DOFs) — gentle ease-in/out");
    out.vlin_max  = getf("HumanModel.VLinMax", 3.0f,
            "m/s (lb_x, lb_z)");
    out.alin_max  = getf("HumanModel.ALinMax", 30.0f,
            "m/s² (lb_x, lb_z)");
    out.pose_smooth = getf("HumanModel.PoseSmooth", 0.2f,
            "global-pose EMA weight on the new Kabsch R,t (lower = steadier facing)");
    out.calibrate_bones = getb("HumanModel.CalibrateBones", true,
            "online per-person bone-length calibration (matches proportions → lower FE)");
    out.calib_smooth    = getf("HumanModel.CalibSmooth", 0.05f,
            "EMA weight on each new measured segment length (lower = steadier)");
    out.fit_csv_path = gets("HumanModel.FitCsvPath", "",
            "non-empty → per-cycle fit-diagnostics CSV (gate)");

    out.death_frames       = geti("HumanConcept.DeathFrames", 60,
            "cycles a person may go unseen before its node is removed (~3 s @20 Hz)");
    out.min_valid          = geti("HumanConcept.MinValid", 12,
            "fewer valid joints ⇒ raise the look affordance");
    out.uncertainty_thresh = getf("HumanConcept.UncertaintyThresh", 0.05f,
            "tr(cov) above this ⇒ raise the look affordance");

    out.epistemic_obs_distance    = getf("Epistemic.ObsDistance", 1.5f,
            "stand-off (m) from the person at the look viewpoint");
    out.epistemic_view_info       = getf("Epistemic.ViewInfo", 50.0f,
            "Fisher precision a clearer view is expected to add (ΔH scale)");
    out.epistemic_cooldown_cycles = geti("Epistemic.CooldownCycles", 120,
            "post-completion hold (cycles) during which the gain is suppressed");
    out.epistemic_csv_path        = gets("Epistemic.CsvPath", "",
            "non-empty → per-cycle epistemic/affordance CSV");

    out.still_vel   = getf("Epistemic.StillVel", 0.10f,
            "m/s");
    out.still_omega = getf("Epistemic.StillOmega", 0.15f,
            "rad/s");

    out.fisher_csv_path = gets("HumanModel.FisherCsvPath", "",
            "non-empty → per-cycle belief CSV (note: currently no writer)");

    out.pose_cov_scale = getf("HumanConcept.PoseCovScale", 0.02f,
            "Pose covariance written on the room→person RT edge: a diagonal scaled by tr(cov) (m²)");

    std::print("human_concept: configuration loaded (source='{}', replay='{}').\n",
               out.source_kind, out.replay_path);
    return out;
}

}  // namespace rc
