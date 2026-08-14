/*
 * viewer_config.h
 *
 * Plain-data configuration for the scene_graph_viewer agent. Mirrors the per-agent config
 * pattern used everywhere else here (ring_config / bottle_config / table_config): a POD struct
 * plus a loader that fills it from a RoboComp ConfigLoader, so initialize() stays free of
 * configLoader.exists(...) ternaries.
 *
 * Everything here is DISPLAY policy. This agent infers nothing and writes nothing to the graph
 * beyond its own presence node, so none of these keys is a model parameter — changing them
 * cannot change anybody's belief.
 */

#pragma once

#include <string>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct ViewerConfig
{
    // ── Refresh ──────────────────────────────────────────────────────────────────
    // Full scene rebuild period, ms. The rebuild walks ~50 nodes and is cheap; 5 Hz is well
    // below anything the eye notices missing on a graph whose topology changes at human pace.
    int refresh_ms = 200;

    // Components-root-relative path to the robot's display mesh. The robot node's own `path` attr
    // is relative to robot_concept's run dir instead, so it cannot be resolved the way every other
    // agent's mesh_path is — hence naming it here. Empty ⇒ fall back to the synthetic robot glyph.
    std::string robot_mesh_path = "robot_concept/meshes/shadow.obj";

    // ── Window ───────────────────────────────────────────────────────────────────
    // Delay before the window is built, ms. Lets the graph finish syncing from the persistent
    // server so the first frame is the real scene rather than a near-empty one that snaps.
    int  window_delay_ms  = 1500;
    // Persist window geometry across runs (QSettings "RoboComp"/<agent name>). A viewer gets
    // restarted constantly; having it come back where you left it is most of its ergonomics.
    bool remember_geometry = true;

    // ── Diagnostics ──────────────────────────────────────────────────────────────
    // LiDAR media-plane topic. Domain and QoS come from the descriptor robot_concept authors on the
    // sensor node — never from here (CLAUDE.md, media plane); this only names the stream.
    std::string lidar_topic = "rc/lidar3d/points";
    // Verbose feed logging.
    bool verbose = false;
    int log_period_frames = 50;   // throttle for the per-cycle scene-size line; 0 disables
};

ViewerConfig load_viewer_config(const ConfigLoader& cfg);

}   // namespace rc
