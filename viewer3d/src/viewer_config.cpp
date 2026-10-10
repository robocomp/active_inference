/*
 * viewer_config.cpp — fill ViewerConfig from a RoboComp ConfigLoader.
 */

#include "viewer_config.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

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
    rc::cfg::Reader reader(cfg, "viewer3d");
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };

    out.refresh_ms        = geti("Viewer3D.RefreshMs", out.refresh_ms,
            "Full scene rebuild period, ms");
    out.robot_mesh_path   = gets("Viewer3D.RobotMeshPath", out.robot_mesh_path,
            "Components-root-relative path to the robot's display mesh");
    out.window_delay_ms   = geti("Viewer3D.WindowDelayMs", out.window_delay_ms,
            "Delay before the window is built, ms");
    out.remember_geometry = getb("Viewer3D.RememberGeometry", out.remember_geometry,
            "Persist window geometry across runs (QSettings 'RoboComp'/<agent name>)");
    out.log_period_frames = geti("Viewer3D.LogPeriodFrames", out.log_period_frames,
            "throttle for the per-cycle scene-size line; 0 disables");
    out.lidar_topic       = gets("Viewer3D.LidarTopic", out.lidar_topic,
            "LiDAR media-plane topic");
    out.verbose           = getb("Component.Debug.Verbose", out.verbose,
            "Verbose feed logging");

    std::print("viewer3d config: refresh={} ms, robot mesh='{}', lidar topic='{}'.\n",
               out.refresh_ms, out.robot_mesh_path, out.lidar_topic);
    return out;
}

}   // namespace rc
