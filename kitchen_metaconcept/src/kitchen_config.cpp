/*
 * kitchen_config.cpp — fill KitchenConfig from a RoboComp ConfigLoader.
 */

#include "kitchen_config.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

KitchenConfig load_kitchen_config(const ConfigLoader& cfg)
{
    KitchenConfig out;

    // ConfigLoader::get has no default overload; guard every key with exists().
    // ★A key written as an int (`Foo = 2`) read via get<double> comes back SILENTLY defaulted, so
    //  every float key below must be written with a decimal point in config.toml.
    // Every read below registers its key, its CODE DEFAULT and a one-line description,
    // so the startup table can say where each value came from - not just what it is.
    // (The four local lambdas now forward to the shared registry; the call sites are
    // unchanged except for that description.)
    rc::cfg::Reader reader(cfg, "kitchen_metaconcept");
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };
    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };

    // ★An EMPTY array in the toml (`MemberClasses = []`) throws inside ConfigLoader before main gets
    //  a chance to report it, so an empty result is treated as "key absent" and the default stands.
    // ConfigLoader THROWS on an empty array, so absent and empty arrive the same way: v()
    // hands back the default for both and the exists()+non-empty guard collapses into one read.
    if (auto v = reader.v<std::string>("KitchenMetaconcept.MemberClasses", {}, "object classes counted as kitchen members"); not v.empty())
            out.member_classes = std::move(v);

    // Parallel class-evidence array. Sized to member_classes and zero-filled: a class with no stated
    // log-odds is UNINFORMATIVE (contributes nothing), which is the safe direction — a typo in the
    // toml weakens the class channel rather than silently inventing evidence for a class.
    out.member_class_logodds.assign(out.member_classes.size(), 0.0f);
    // ConfigLoader THROWS on an empty array, so absent and empty arrive the same way: v()
    // hands back the default for both and the exists()+non-empty guard collapses into one read.
    if (auto v = reader.v<double>("KitchenMetaconcept.MemberClassLogOdds", {}, "per-class prior log-odds of membership"); not v.empty())
        {
            if (v.size() != out.member_classes.size())
                std::print("kitchen_metaconcept: ⚠ MemberClassLogOdds has {} entries for {} classes — "
                           "the surplus is ignored and any shortfall stays uninformative (0).\n",
                           v.size(), out.member_classes.size());
            for (std::size_t i = 0; i < out.member_classes.size() and i < v.size(); ++i)
                out.member_class_logodds[i] = static_cast<float>(v[i]);
        }


    out.member_class_tiers.assign(out.member_classes.size(), "auto");
    // ConfigLoader THROWS on an empty array, so absent and empty arrive the same way: v()
    // hands back the default for both and the exists()+non-empty guard collapses into one read.
    if (auto v = reader.v<std::string>("KitchenMetaconcept.MemberClassTiers", {}, "which tier each member class belongs to"); not v.empty())
        {
            if (v.size() != out.member_classes.size())
                std::print("kitchen_metaconcept: ⚠ MemberClassTiers has {} entries for {} classes — "
                           "any shortfall stays \"auto\" (resolved from geometry).\n",
                           v.size(), out.member_classes.size());
            for (std::size_t i = 0; i < out.member_classes.size() and i < v.size(); ++i)
                out.member_class_tiers[i] = v[i];
        }

    out.node_subtype = gets("KitchenMetaconcept.NodeSubtype", "kitchen",
            "DSR type `metaconcept`, NOT `object`: a frame is a belief about a RELATION among nodes, so it must stay out of everyone's get_nodes_by_type('object')…");
    out.node_prefix  = gets("KitchenMetaconcept.NodePrefix", "kitchen_",
            "owned-node NAME prefix — the stale-sweep / cleanup key");

    out.axis_model_std_deg  = getf("KitchenMetaconcept.AxisModelStdDeg", 1.5f,
            "Intrinsic model spread per shared DOF: how much genuine variation the hypothesis 'one kitchen' allows");
    out.axis_common_mode_std_deg = getf("KitchenMetaconcept.AxisCommonModeStdDeg", 0.5f,
            "★Shared (room-polygon) orientation error — a FLOOR on the fused axis σ, not a model spread");
    out.worktop_model_std_m = getf("KitchenMetaconcept.WorktopModelStdM", 0.02f,
            "spread of base-unit worktop heights about one plane");
    out.depth_model_std_m   = getf("KitchenMetaconcept.DepthModelStdM", 0.03f,
            "spread of carcass depths about one nominal");
    out.worktop_meas_std_m  = getf("KitchenMetaconcept.WorktopMeasStdM", 0.06f,
            "at 1 m of run; scaled by 1/length below");
    out.depth_meas_std_m    = getf("KitchenMetaconcept.DepthMeasStdM", 0.05f,
            "");
    out.clutter_frac        = getf("KitchenMetaconcept.ClutterFrac", 0.20f,
            "prior mass on 'this member belongs to no frame'");
    out.evidence_ema_alpha  = getf("KitchenMetaconcept.EvidenceEmaAlpha", 0.05f,
            "low-pass on frame-vs-null (NOT an accumulator)");

    out.axis_from_pinned_only  = getb("KitchenMetaconcept.AxisFromPinnedOnly", true,
            "Estimate the shared axis ONLY from members whose yaw is pinned by the room polygon");
    out.pinned_yaw_std_max_deg = getf("KitchenMetaconcept.PinnedYawStdMaxDeg", 3.5f,
            "collinear-merge tolerance cabinet_concept publishes for wall-anchored runs");

    out.log_period_frames = geti("KitchenMetaconcept.LogPeriodFrames", 25,
            "per-cycle stdout log throttle for the graph-reader");
    out.csv_path          = gets("KitchenMetaconcept.CsvPath", "",
            "per-member snapshot; empty disables");
    out.fit_csv_path      = gets("KitchenMetaconcept.FitCsvPath", "",
            "frame fit + the down-prior it WOULD publish; empty disables");
    out.outline_csv_path  = gets("KitchenMetaconcept.OutlineCsvPath", "",
            "per-joint gaps/overlaps of the continuous outline; empty disables");
    out.publish           = getb("KitchenMetaconcept.Publish", false,
            "★THE SWITCH. False keeps the agent read-only: it fits, measures and logs, and writes nothing. Everything downstream of this was validated against…");

    std::string classes;
    for (const auto& c : out.member_classes) { if (not classes.empty()) classes += ","; classes += c; }
    std::print("kitchen_metaconcept: configuration loaded (members=[{}] owns='{}*' axis_from_pinned={} csv='{}').\n",
               classes, out.node_prefix, out.axis_from_pinned_only,
               out.csv_path.empty() ? "<disabled>" : out.csv_path);
    return out;
}

}  // namespace rc
