/*
 * viewer_config.cpp — fill ViewerConfig from a RoboComp ConfigLoader.
 */

#include "viewer_config.h"

#include <algorithm>
#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

ViewerConfig load_viewer_config(const ConfigLoader& cfg)
{
    ViewerConfig out;

    // ConfigLoader::get has no default overload; guard every key with exists().
    auto geti = [&](const std::string& k, int def) -> int {
        return cfg.exists(k) ? cfg.get<int>(k) : def;
    };
    auto getf = [&](const std::string& k, float def) -> float {
        return cfg.exists(k) ? static_cast<float>(cfg.get<double>(k)) : def;
    };
    auto getb = [&](const std::string& k, bool def) -> bool {
        return cfg.exists(k) ? cfg.get<bool>(k) : def;
    };
    auto gets = [&](const std::string& k, std::string def) -> std::string {
        return cfg.exists(k) ? cfg.get<std::string>(k) : std::move(def);
    };

    out.refresh_ms       = geti("SceneGraphViewer.RefreshMs", out.refresh_ms);
    out.stratum_gap      = getf("SceneGraphViewer.StratumGap", out.stratum_gap);
    out.show_affordances = getb("SceneGraphViewer.ShowAffordances", out.show_affordances);
    out.show_agents      = getb("SceneGraphViewer.ShowAgents", out.show_agents);
    out.floor_alpha      = std::clamp(getf("SceneGraphViewer.FloorAlpha", out.floor_alpha), 0.0f, 1.0f);
    out.robot_mesh_path  = gets("SceneGraphViewer.RobotMeshPath", out.robot_mesh_path);
    out.window_delay_ms  = geti("SceneGraphViewer.WindowDelayMs", out.window_delay_ms);
    out.remember_geometry = getb("SceneGraphViewer.RememberGeometry", out.remember_geometry);
    out.log_period_frames = geti("SceneGraphViewer.LogPeriodFrames", out.log_period_frames);

    // Lives under [Presence], not [SceneGraphViewer]: it is the SAME display horizon
    // robot_concept's 2D agent-status overlay uses, and the two must agree.
    out.stale_after_ms   = geti("Presence.stale_display_ms", out.stale_after_ms);

    std::print("scene_graph_viewer: configuration loaded (refresh={} ms, stratum_gap={:.2f} m, "
               "stale={} ms, affordances={}, agents={}).\n",
               out.refresh_ms, out.stratum_gap, out.stale_after_ms,
               out.show_affordances, out.show_agents);
    return out;
}

}   // namespace rc
