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
 * ring_metaconcept — level-2 "concept of concepts" agent (CONCEPT_AGENT_RECIPE.md §6).
 *
 * A meta-concept whose latent (a radial "ring" arrangement) generates the poses of
 * already-inferred objects and pushes empirical priors back DOWN to the per-object
 * agents. The dining_set (a table anchor + a ring of chairs) is its first instance.
 *
 * Milestone 1 (this scaffold): the SAME chassis as the level-1 concept agents (presence
 * protocol, GRAFCET state machine, crash-free shutdown), but with the mask/media front
 * end REPLACED by a graph-reader — it polls the constituent member nodes (table + chair)
 * directly from the DSR graph, on the MAIN thread, connecting NONE of the update_node/
 * update_edge signals (§6 Delta 1: reacting to those from the FastDDS reader threads is
 * the DirectConnection heap-corruption crash). No RingBelief, no births, no down-prior
 * yet — just reach Operating and read the members.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <genericworker.h>
#include <fps/fps.h>

#include "ring_config.h"      // rc::RingConfig + load_ring_config
#include "ring_belief.h"      // rc::RingBelief — the level-2 arrangement latent
#include "ring_scene_graph.h" // rc::RingSceneGraph — every DSR write lives there
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"

// ─── One constituent member, as the graph-reader sees it ─────────────────────
//
// This is the meta-concept's MEASUREMENT (§6 Delta 1): not a mask point but a peer
// concept's published posterior — its room-frame pose together with the covariance that
// peer itself published on the room→member RT edge. Feeding the peer's own Σ in as the
// measurement noise is what makes a poorly-seen chair contribute weakly to the
// arrangement fit by construction, with no confidence gate anywhere.
struct RingMember
{
    std::uint64_t   id = 0;
    std::string     name;
    std::string     cls;                                    // "table" / "chair" (object_subtype, else name prefix)
    Eigen::Vector2f xy    = Eigen::Vector2f::Zero();        // room-frame position (m)
    float           yaw   = 0.0f;                           // room-frame heading (rad)
    Eigen::Matrix2f Sigma_xy = Eigen::Matrix2f::Identity(); // published position covariance (m²)
    float           var_yaw  = 0.0f;                        // published yaw variance (rad²)
    bool            has_cov  = false;                       // false ⇒ Sigma_xy/var_yaw are fallbacks, not published
};

// ─── SpecificWorker ──────────────────────────────────────────────────────────

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
    // ── Orchestration ─────────────────────────────────────────────────────────
    void load_config(const ConfigLoader& cfg);
    // Graph-reader front end (§6 Delta 1): poll the member nodes on the MAIN thread.
    // Returns this cycle's constituents — pose + published Σ, the meta-concept's measurements.
    struct MemberSnapshot
    {
        std::vector<RingMember> anchors;   // class == cfg_.anchor_class (the table)
        std::vector<RingMember> ring;      // class == cfg_.ring_class   (the chairs)
        bool empty() const { return anchors.empty() and ring.empty(); }

        // ── Why a node did NOT become a member ──────────────────────────────
        // Zero members is ambiguous on its own: it means either "no dining furniture in the scene"
        // (correct) or "the reader is broken" (not). These counters separate the two, so an empty
        // poll is never mistaken for a healthy one or vice versa.
        std::size_t              objects_total  = 0;   // type()=="object" nodes seen
        std::size_t              skipped_class  = 0;   // present, but not an anchor/ring class
        std::size_t              skipped_no_rt  = 0;   // right class, but no usable room→member RT pose
        std::vector<std::string> unclassified;         // a few names, to show WHAT is out there
    };
    MemberSnapshot poll_members();
    // Class of a generic "object" node: object_subtype when it names a known class, else the
    // NAME prefix. Mirrors voxelizer/src/scene_processor.cpp's object_class_of — for a table
    // object_subtype may instead carry the SHAPE (round/square), so the prefix is the fallback.
    std::string object_class_of(const DSR::Node& node) const;
    // Read one member's room-frame pose + published covariance off the room→member RT edge.
    // Returns nullopt when the edge or its pose is missing (the member is not room-parented yet).
    std::optional<RingMember> read_member(const DSR::Node& node, const std::string& cls);

    // ── Arrangement belief (M2, READ-ONLY: fits and logs, writes nothing to the graph) ──
    // Build one poll's evidence: ring members become "points" carrying their OWN published variance,
    // the anchor becomes the centre prior consumed by RingBelief::accumulate_extra.
    rc::RingFrame build_ring_frame(const MemberSnapshot& members) const;
    // First-time seed. GN on a circle is not forgiving, so start from the anchor-implied centre, the
    // mean member radius and a phase that puts slot 0 on the first member.
    void          seed_ring_belief(const MemberSnapshot& members, const rc::RingFrame& frame);
    void          step_ring_belief(const MemberSnapshot& members);

    // ── Cavity / leave-one-out (§6 Delta 3, the one hard requirement) ──────────
    // The message sent DOWN to member i must come from an arrangement fitted WITHOUT member i.
    // Otherwise the rig folds i's own contribution into the centre it then tells i to face — the
    // belief is fed its own output and manufactures agreement. In message-passing terms: never send
    // a node the message it sent up. Here the fit is position-only, so what leaks is i's POSITION
    // pulling the centre toward itself; `shift` records how much that was worth.
    struct CavityPrior { float yaw = 0.0f; float std_dev = 0.0f; float shift = 0.0f; };
    void          compute_cavity_priors(const MemberSnapshot& members, const rc::RingFrame& frame);

    // Per-slot observability ∈ [0,1] from the live room→zed pose. Gates the EMPTY-seat evidence: a
    // seat the robot has never looked at is uninformative, not proof the seat is vacant. Continuous
    // (soft FoV/range edges) so a seat at the rim of the frame is partially observed, not a step.
    std::vector<float> compute_slot_visibility(const rc::RingBeliefState& s, int n_slots) const;

    // ── DSR writes (M3) ────────────────────────────────────────────────────────
    // Birth/refresh the dining_set node and push each member's CAVITY prior onto its group_member
    // edge — or retire the rig when the arrangement stops out-explaining the independent-objects null.
    void          publish_rig(const MemberSnapshot& members);
    // Area of the room polygon (shoelace), m². The independent-objects NULL is a uniform over the
    // room, so its density is 1/area — taking that from the actual room instead of a constant is what
    // keeps the ring-vs-null log-Bayes factor from being scaled by an arbitrary number.
    std::optional<float> room_area_m2() const;

    // ── Diagnostics ────────────────────────────────────────────────────────────
    void log_members_csv(const MemberSnapshot& members);
    void log_ring_csv(const MemberSnapshot& members);

    // ── Presence protocol (copied from the level-1 agents; see CLAUDE.md) ───────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void remove_owned_dining_set_nodes();   // startup stale-sweep of leftover "dining_set" nodes
    void request_shutdown();
    // Crash-free exit (request_shutdown + DDS reset + _Exit), bypassing the Ice/static teardown abort.
    void terminal_shutdown();
    // Grace before a required-peer loss is treated as terminal (debounce transient presence flaps).
    static constexpr int REQUIRED_LOSS_GRACE_MS = 3000;
    void on_optional_peer_lost(const std::string& name, std::uint32_t id);
    void on_optional_peer_ready(const std::string& name, std::uint32_t id);

    // ── Members ─────────────────────────────────────────────────────────────────
    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    rc::RingConfig  cfg_;
    std::uint64_t   room_node_id_ = 0;
    FPSCounter      fps_counter_;     // overall compute()-cycle rate

    // RT_API reads the room→member edge pose + covariance. It resolves the edge's timestamp ring
    // buffer for us, which reading rt_translation_att raw does NOT (those attrs are ring-buffered).
    std::unique_ptr<DSR::RT_API> rt_api_;
    // room→zed for slot visibility. ts==0 uses InnerEigenAPI's UNLOCKED cache, so this instance is
    // touched from the MAIN thread only (CLAUDE.md) — this agent has no worker threads.
    std::unique_ptr<DSR::InnerEigenAPI>  inner_eigen_;
    std::unique_ptr<rc::RingSceneGraph>  scene_graph_;

    rc::RingBelief  ring_belief_;
    bool            ring_seeded_ = false;
    std::unordered_map<std::uint64_t, CavityPrior> cavity_;   // per-member leave-one-out down-prior

    std::ofstream   csv_;             // per-member snapshot (the diagnostics channel that gets read)
    std::ofstream   ring_csv_;        // per-cycle arrangement fit + the down-prior it WOULD publish
    std::uint64_t   cycle_ = 0;

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
