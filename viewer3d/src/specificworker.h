/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * scene_graph_viewer — the stratified 3D view of the shared DSR graph, as its own agent.
 *
 * WHY THIS VIEW EXISTS. The planar GraphViewer draws room→table→aff_table_1 with exactly the
 * same weight as mind→agent, so neither AGENT OWNERSHIP nor META-CONCEPT GROUPING is visible in
 * it — and group_member, being a non-RT edge, has no transform for a spatial force layout to work
 * with at all. This view splits the two roles: z carries the ABSTRACTION STRATUM (robot → room →
 * instances → meta-concepts → agents), x/y carries the TRUE METRIC POSE. Hierarchy reads
 * vertically while the floor plan still reads horizontally.
 *
 * WHY ITS OWN AGENT. It used to live inside robot_concept, docked into that agent's cortex
 * DSRViewer behind the otherwise-dead `Agent.3d` config key — an accident of robot_concept
 * happening to be the agent with a graph window up. Nothing about the view is robot-specific:
 * the feed needs only a DSRGraph + the LiDAR media plane, and the renderer is
 * DSR-blind. Split out, the view survives a robot_concept restart, can be started and stopped
 * on its own while debugging, and its heavy QOpenGLWidget stops sharing a backing store with
 * cortex's GraphViewer (see dsr_gui.h: that sharing is a stack-address free in paintAndFlush
 * when a peer joins).
 *
 * WHAT IT IS NOT. A pure observer: it infers nothing and writes nothing to the graph beyond its
 * own presence node. It therefore declares NO required peers — a viewer must render whatever
 * exists and must never tear itself down because a peer is absent.
 *
 * THREADING. rc::SceneFeed::refresh() walks InnerEigenAPI's ts==0 transform cache,
 * which is unlocked and therefore main-thread-only (CLAUDE.md). The refresh is driven by a
 * QTimer on the main thread, and NONE of the update_node/update_edge signals are connected —
 * those fire on the FastDDS reader threads.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <genericworker.h>
#include <fps/fps.h>

#include "viewer_config.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"

class QMainWindow;
class QWidget;
class QPushButton;
class QTimer;

// Forward-declared on purpose: voxel_opengl_viewer.h drags in the whole Qt OpenGL widget stack,
// and it is needed by exactly one TU.
namespace rc { class VoxelOpenGLViewer; class SceneFeed; }

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
    ~SpecificWorker();
    bool is_shutting_down() const noexcept { return shutting_down_.load(); }

public slots:
    void initialize();
    void compute();
    void emergency();
    void restore();
    int  startup_check();

private:
    // ── The view ──────────────────────────────────────────────────────────────────
    // Build the top-level window and start the refresh timer. Deferred by
    // cfg_.window_delay_ms so the first frame shows the synced graph, not a near-empty one.
    void build_viewer_window();
    // Layer-toggle bar + the GL widget, as one central widget. Returns a panel the window takes ownership of.
    QWidget* build_layer_panel();
    // One full refresh: DSR graph + LiDAR plane → the metric 3D view. MAIN THREAD ONLY.
    void refresh_scene();
    // ── World-frame hold ──────────────────────────────────────────────────────────
    // Everything this agent draws is expressed in the room frame and posed by room←robot. With
    // either node gone there is nothing to place anything against, so the agent HOLDS in Waiting and
    // says so on the canvas rather than rendering a confident picture of a world it cannot locate.
    // Called from refresh_scene (the timer, which runs in EVERY state — that is what lets Waiting
    // ever end). Takes the two names the feed just resolved (empty ⇒ that node is not in the graph);
    // plain strings so this header keeps SceneFeed forward-declared and the DSR headers out.
    void update_world_frame_state(const std::string& room_name, const std::string& robot_name);
    // Pick handler. The 3D view is an inspection tool, so a click answers "what is this?" in the
    // log. (In robot_concept this opened the node's live media-plane stream viewer; that handler
    // stays there — it needs media_transport, a domain-7 participant and the stream widgets, and
    // two agents subscribing the same streams buys nothing.)
    void restore_window_geometry();
    void save_window_geometry() const;

    // ── Presence protocol (copied from the level-1 agents; see CLAUDE.md) ──────────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void request_shutdown();
    // Crash-free exit (request_shutdown + DDS reset + _Exit), bypassing the Ice/static teardown abort.
    void terminal_shutdown();
    // Grace before a required-peer loss is treated as terminal (debounce transient presence flaps).
    // Kept even though this agent declares no required peers: it costs nothing and the day someone
    // adds one, the debounce is already there.
    static constexpr int REQUIRED_LOSS_GRACE_MS = 3000;
    void on_optional_peer_lost(const std::string& name, std::uint32_t id);
    void on_optional_peer_ready(const std::string& name, std::uint32_t id);

    // ── Members ───────────────────────────────────────────────────────────────────
    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    rc::ViewerConfig cfg_;

    // window_ OWNS the viewer via setCentralWidget's reparenting, so this stays a raw observer —
    // a unique_ptr here would double-delete. (robot_concept could hold one because
    // add_custom_widget_to_dock does not take ownership.)
    std::unique_ptr<QMainWindow>  window_;
    rc::VoxelOpenGLViewer*        viewer_ = nullptr;
    std::unique_ptr<rc::SceneFeed> feed_;
    bool                          lidar_ready_ = false;
    QTimer*                       refresh_timer_ = nullptr;
    // Last observed world-frame readiness. UNSET until the first refresh, so the very first
    // observation always counts as a change and an agent that comes up into a room-less graph says so
    // instead of holding mutely — the failure this whole path exists to make visible.
    std::optional<bool>           world_frames_ready_;

    std::uint64_t cycle_ = 0;
    // Liveness + COST of the view. A 3D viewer is the one agent on the graph that can quietly burn a
    // core (GL redraw + a full scene rebuild every refresh), so the throttled line reports process CPU
    // and RSS alongside the cycle rate. FPSCounter::get_cpu_use() is a DELTA since its own previous
    // call, so it must only ever be read through print() — see compute().
    FPSCounter fps_counter_;

signals:
    void presenceReady();
    void presenceLost();
    // Operating → Waiting when the room/robot frames vanish. A SEPARATE signal from presenceLost on
    // purpose: that one lands in Degraded, which schedules a shutdown. A missing room is not a reason
    // for a pure observer to die — it is a reason to hold and be visible about it.
    void worldFramesLost();
};

#endif   // SPECIFICWORKER_H
