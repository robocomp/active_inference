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

    // ── Scene layout (forwarded to rc::graph3d::SceneBuilder::Config) ─────────────
    // Vertical spacing between abstraction strata, metres. Sized against a typical apartment
    // room (~5-6 m across) so the stack is legibly tall without dwarfing the floor plan.
    float stratum_gap = 1.20f;
    // Display-only staleness horizon for agent nodes, ms. Deliberately DECOUPLED from
    // Presence.heartbeat_timeout_ms: that one is long because acting on a short silence proved
    // hypersensitive, whereas greying a node early costs nothing and is undone the moment the
    // heartbeat resumes. Read from Presence.stale_display_ms so it matches robot_concept's 2D
    // overlay — an agent should grey out in both views at the same moment.
    int stale_after_ms = 3000;
    // The containment ladder — robot → room → instances → meta-1 → meta-2 → … — is always drawn:
    // it IS the view. These two are a second, orthogonal story (what can be done with a thing,
    // and who is doing the believing) that doubles the node count, so they stay off by default.
    bool show_affordances = false;
    bool show_agents      = false;
    // Floor opacity, 0..1. The floor mesh spans the whole room on the ROOM plane; drawn opaque it
    // becomes a lid over the robot a rung below and over every drop-line crossing it. 1.0 = opaque.
    float floor_alpha = 0.35f;
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
    int log_period_frames = 50;   // throttle for the per-cycle scene-size line; 0 disables
};

ViewerConfig load_viewer_config(const ConfigLoader& cfg);

}   // namespace rc
