/*
 * viewer_config.cpp — fill ViewerConfig from a RoboComp ConfigLoader.
 */

#include "viewer_config.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <algorithm>
#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

ViewerConfig load_viewer_config(const ConfigLoader& cfg)
{
    ViewerConfig out;

    // ConfigLoader::get has no default overload; guard every key with exists().
    // Every read below registers its key, its CODE DEFAULT and a one-line description,
    // so the startup table can say where each value came from - not just what it is.
    // (The four local lambdas now forward to the shared registry; the call sites are
    // unchanged except for that description.)
    rc::cfg::Reader reader(cfg, "scene_graph_viewer");
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };

    out.refresh_ms       = geti("SceneGraphViewer.RefreshMs", out.refresh_ms,
            "Full scene rebuild period, ms");
    out.stratum_gap      = getf("SceneGraphViewer.StratumGap", out.stratum_gap,
            "Vertical spacing between abstraction strata, metres");
    out.show_affordances = getb("SceneGraphViewer.ShowAffordances", out.show_affordances,
            "The containment ladder — robot → room → instances → meta-1 → meta-2 → … — is always drawn: it IS the view");
    out.show_agents      = getb("SceneGraphViewer.ShowAgents", out.show_agents,
            "");
    out.floor_alpha      = std::clamp(getf("SceneGraphViewer.FloorAlpha", out.floor_alpha,
            "it hides the robot standing a rung below it and every drop-line that crosses it. 1.0 = opaque"), 0.0f, 1.0f);
    out.robot_mesh_path  = gets("SceneGraphViewer.RobotMeshPath", out.robot_mesh_path,
            "Components-root-relative path to the robot's display mesh");
    out.light_background = getb("SceneGraphViewer.LightBackground", out.light_background,
            "Paper theme: white background, with the whole scene re-inked for it (the renderer compresses lightness, it does not swap palettes)");
    out.window_delay_ms  = geti("SceneGraphViewer.WindowDelayMs", out.window_delay_ms,
            "Delay before the window is built, ms");
    out.remember_geometry = getb("SceneGraphViewer.RememberGeometry", out.remember_geometry,
            "Persist window geometry across runs (QSettings 'RoboComp'/<agent name>)");
    out.log_period_frames = geti("SceneGraphViewer.LogPeriodFrames", out.log_period_frames,
            "throttle for the per-cycle scene-size line; 0 disables");

    // Lives under [Presence], not [SceneGraphViewer]: it is the SAME display horizon
    // robot_concept's 2D agent-status overlay uses, and the two must agree.
    out.stale_after_ms   = geti("Presence.stale_display_ms", out.stale_after_ms,
            "Display-only staleness horizon for agent nodes, ms");

    std::print("scene_graph_viewer: configuration loaded (refresh={} ms, stratum_gap={:.2f} m, "
               "stale={} ms, affordances={}, agents={}).\n",
               out.refresh_ms, out.stratum_gap, out.stale_after_ms,
               out.show_affordances, out.show_agents);
    return out;
}

}   // namespace rc
