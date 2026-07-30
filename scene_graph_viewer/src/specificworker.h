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
 * the builder (common/graph3d) needs only a DSRGraph, and the renderer (common/viewers) is
 * DSR-blind. Split out, the view survives a robot_concept restart, can be started and stopped
 * on its own while debugging, and its heavy QOpenGLWidget stops sharing a backing store with
 * cortex's GraphViewer (see dsr_gui.h: that sharing is a stack-address free in paintAndFlush
 * when a peer joins).
 *
 * WHAT IT IS NOT. A pure observer: it infers nothing and writes nothing to the graph beyond its
 * own presence node. It therefore declares NO required peers — a viewer must render whatever
 * exists and must never tear itself down because a peer is absent.
 *
 * THREADING. rc::graph3d::SceneBuilder::build() walks InnerEigenAPI's ts==0 transform cache,
 * which is unlocked and therefore main-thread-only (CLAUDE.md). The refresh is driven by a
 * QTimer on the main thread, and NONE of the update_node/update_edge signals are connected —
 * those fire on the FastDDS reader threads.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <genericworker.h>

#include "viewer_config.h"
#include "../../common/graph3d/graph3d_scene_builder.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"

class QMainWindow;
class QTimer;

// Forward-declared on purpose: gl_graph3d_viewer.h drags in the whole Qt OpenGL widget stack,
// and it is needed by exactly one TU.
namespace rc::viewers { class GLGraph3DViewer; }

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
    // One full rebuild: DSR graph → rc::graph3d::Scene → renderer. MAIN THREAD ONLY.
    void refresh_scene();
    // Pick handler. The 3D view is an inspection tool, so a click answers "what is this?" in the
    // log. (In robot_concept this opened the node's live media-plane stream viewer; that handler
    // stays there — it needs media_transport, a domain-7 participant and the stream widgets, and
    // two agents subscribing the same streams buys nothing.)
    void log_picked_node(std::uint64_t node_id, const std::string& type);
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

    rc::graph3d::SceneBuilder graph3d_builder_;
    // window_ OWNS the viewer via setCentralWidget's reparenting, so this stays a raw observer —
    // a unique_ptr here would double-delete. (robot_concept could hold one because
    // add_custom_widget_to_dock does not take ownership.)
    std::unique_ptr<QMainWindow>  window_;
    rc::viewers::GLGraph3DViewer* viewer_ = nullptr;
    QTimer*                       refresh_timer_ = nullptr;

    std::uint64_t cycle_ = 0;
    std::size_t   last_node_count_ = 0;
    std::size_t   last_wall_count_ = 0;
    std::size_t   last_wall_meshed_ = 0;
    std::size_t   last_wall_two_sided_ = 0;
    bool          last_ground_valid_ = false;

signals:
    void presenceReady();
    void presenceLost();
};

#endif   // SPECIFICWORKER_H
