/*
 * kitchen_config.cpp — fill KitchenConfig from a RoboComp ConfigLoader.
 */

#include "kitchen_config.h"

#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

KitchenConfig load_kitchen_config(const ConfigLoader& cfg)
{
    KitchenConfig out;

    // ConfigLoader::get has no default overload; guard every key with exists().
    // ★A key written as an int (`Foo = 2`) read via get<double> comes back SILENTLY defaulted, so
    //  every float key below must be written with a decimal point in config.toml.
    auto geti = [&](const std::string& k, int def) -> int {
        return cfg.exists(k) ? cfg.get<int>(k) : def;
    };
    auto gets = [&](const std::string& k, std::string def) -> std::string {
        return cfg.exists(k) ? cfg.get<std::string>(k) : def;
    };
    auto getf = [&](const std::string& k, float def) -> float {
        return cfg.exists(k) ? static_cast<float>(cfg.get<double>(k)) : def;
    };
    auto getb = [&](const std::string& k, bool def) -> bool {
        return cfg.exists(k) ? cfg.get<bool>(k) : def;
    };

    // ★An EMPTY array in the toml (`MemberClasses = []`) throws inside ConfigLoader before main gets
    //  a chance to report it, so an empty result is treated as "key absent" and the default stands.
    if (cfg.exists("KitchenMetaconcept.MemberClasses"))
        if (auto v = cfg.get<std::vector<std::string>>("KitchenMetaconcept.MemberClasses"); not v.empty())
            out.member_classes = std::move(v);

    // Parallel class-evidence array. Sized to member_classes and zero-filled: a class with no stated
    // log-odds is UNINFORMATIVE (contributes nothing), which is the safe direction — a typo in the
    // toml weakens the class channel rather than silently inventing evidence for a class.
    out.member_class_logodds.assign(out.member_classes.size(), 0.0f);
    if (cfg.exists("KitchenMetaconcept.MemberClassLogOdds"))
        if (auto v = cfg.get<std::vector<double>>("KitchenMetaconcept.MemberClassLogOdds"); not v.empty())
        {
            if (v.size() != out.member_classes.size())
                std::print("kitchen_metaconcept: ⚠ MemberClassLogOdds has {} entries for {} classes — "
                           "the surplus is ignored and any shortfall stays uninformative (0).\n",
                           v.size(), out.member_classes.size());
            for (std::size_t i = 0; i < out.member_classes.size() and i < v.size(); ++i)
                out.member_class_logodds[i] = static_cast<float>(v[i]);
        }


    out.member_class_tiers.assign(out.member_classes.size(), "auto");
    if (cfg.exists("KitchenMetaconcept.MemberClassTiers"))
        if (auto v = cfg.get<std::vector<std::string>>("KitchenMetaconcept.MemberClassTiers"); not v.empty())
        {
            if (v.size() != out.member_classes.size())
                std::print("kitchen_metaconcept: ⚠ MemberClassTiers has {} entries for {} classes — "
                           "any shortfall stays \"auto\" (resolved from geometry).\n",
                           v.size(), out.member_classes.size());
            for (std::size_t i = 0; i < out.member_classes.size() and i < v.size(); ++i)
                out.member_class_tiers[i] = v[i];
        }

    out.node_subtype = gets("KitchenMetaconcept.NodeSubtype", "kitchen");
    out.node_prefix  = gets("KitchenMetaconcept.NodePrefix",  "kitchen_");

    out.axis_model_std_deg  = getf("KitchenMetaconcept.AxisModelStdDeg",  1.5f);
    out.worktop_model_std_m = getf("KitchenMetaconcept.WorktopModelStdM", 0.02f);
    out.depth_model_std_m   = getf("KitchenMetaconcept.DepthModelStdM",   0.03f);
    out.worktop_meas_std_m  = getf("KitchenMetaconcept.WorktopMeasStdM",  0.06f);
    out.depth_meas_std_m    = getf("KitchenMetaconcept.DepthMeasStdM",    0.05f);
    out.clutter_frac        = getf("KitchenMetaconcept.ClutterFrac",      0.20f);
    out.evidence_ema_alpha  = getf("KitchenMetaconcept.EvidenceEmaAlpha", 0.05f);

    out.axis_from_pinned_only  = getb("KitchenMetaconcept.AxisFromPinnedOnly", true);
    out.pinned_yaw_std_max_deg = getf("KitchenMetaconcept.PinnedYawStdMaxDeg", 3.5f);

    out.log_period_frames = geti("KitchenMetaconcept.LogPeriodFrames", 25);
    out.csv_path          = gets("KitchenMetaconcept.CsvPath", "");
    out.fit_csv_path      = gets("KitchenMetaconcept.FitCsvPath", "");

    std::string classes;
    for (const auto& c : out.member_classes) { if (not classes.empty()) classes += ","; classes += c; }
    std::print("kitchen_metaconcept: configuration loaded (members=[{}] owns='{}*' axis_from_pinned={} csv='{}').\n",
               classes, out.node_prefix, out.axis_from_pinned_only,
               out.csv_path.empty() ? "<disabled>" : out.csv_path);
    return out;
}

}  // namespace rc
