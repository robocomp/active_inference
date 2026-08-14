/*
 * viewer_config.cpp — fill ViewerConfig from a RoboComp ConfigLoader.
 */

#include "viewer_config.h"

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
    auto getb = [&](const std::string& k, bool def) -> bool {
        return cfg.exists(k) ? cfg.get<bool>(k) : def;
    };
    auto gets = [&](const std::string& k, std::string def) -> std::string {
        return cfg.exists(k) ? cfg.get<std::string>(k) : std::move(def);
    };

    out.refresh_ms        = geti("Viewer3D.RefreshMs",        out.refresh_ms);
    out.robot_mesh_path   = gets("Viewer3D.RobotMeshPath",    out.robot_mesh_path);
    out.window_delay_ms   = geti("Viewer3D.WindowDelayMs",    out.window_delay_ms);
    out.remember_geometry = getb("Viewer3D.RememberGeometry", out.remember_geometry);
    out.log_period_frames = geti("Viewer3D.LogPeriodFrames",  out.log_period_frames);
    out.lidar_topic       = gets("Viewer3D.LidarTopic",       out.lidar_topic);
    out.verbose           = getb("Component.Debug.Verbose",   out.verbose);

    std::print("viewer3d config: refresh={} ms, robot mesh='{}', lidar topic='{}'.\n",
               out.refresh_ms, out.robot_mesh_path, out.lidar_topic);
    return out;
}

}   // namespace rc
