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
 * kitchen_metaconcept — level-2 "concept of concepts" agent (CONCEPT_AGENT_RECIPE.md §6).
 *
 * SCHEMA = RECTILINEAR: the members of one kitchen share a 90°-periodic axis grid, a worktop plane
 * and a carcass depth. This is the second schema the meta-concept chassis carries (ring was the
 * first), and it is the case ring_metaconcept/SCHEMA_GENERALITY_TODO.md §4 Step 4 nominated as the
 * right one to validate the abstraction against: "a run of cabinets. ★cabinet_concept already
 * implements a wall-anchored RUN model independently — that IS this schema, built separately."
 *
 * THE MOTIVATING DEFECT (live, cabinet_concept): a wall-anchored run takes its yaw from the room
 * polygon and lands on the kitchen grid (measured 0.28° / 89.01° / 89.60°). The free-standing island
 * instead DERIVES its chart axis from its own points once, at birth, and freezes it — and because
 * yaw is not a DOF of WallRunState, no later evidence can revise it. Across one run the island was
 * born three times, at 91.75°, 83.21° and 78.87°: up to 10.8° off the grid its neighbours agree on,
 * permanently. The information that fixes it does not exist in the island's own mask; it is in the
 * arrangement, which is exactly what a level-2 concept is for.
 *
 * ★MILESTONE 2 — THIS AGENT IS READ-ONLY. It polls the members, fits the frame, and writes two CSVs
 * including the down-prior it WOULD publish. It creates no node, writes no group_member edge and
 * pushes no attribute, so nothing it does can move a cabinet. The premise gets checked against live
 * data before anything is allowed to steer.
 *
 * ★SAFETY (CLAUDE.md / §6 Delta 1): this agent's primary input is the graph, which is precisely when
 * one reaches for update_node_signal / update_edge_signal — the DirectConnection heap-corruption
 * crash, since those fire on the FastDDS reader threads. It connects NONE of them and polls on the
 * MAIN thread from compute() instead.
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

#include "kitchen_config.h"   // rc::KitchenConfig + load_kitchen_config
#include "kitchen_belief.h"   // rc::KitchenBelief — the level-2 rectilinear-frame latent
#include "kitchen_outline.h"  // rc::KitchenOutline — the kitchen as ONE continuous shape
#include "kitchen_scene_graph.h"  // rc::KitchenSceneGraph — every DSR write lives there
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"

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
    struct MemberSnapshot
    {
        std::vector<rc::KitchenMember> members;   // everything of a member class with a usable pose
        std::vector<rc::KitchenMember> pinned;    // the subset whose yaw is polygon-pinned (axis evidence)

        // ── Why a node did NOT become a member ──────────────────────────────
        // Zero members is ambiguous on its own: it means either "no kitchen in the scene" (correct)
        // or "the reader is broken" (not). These counters separate the two, so an empty poll is never
        // mistaken for a healthy one or vice versa.
        std::size_t              objects_total = 0;   // type()=="object" nodes seen
        std::size_t              skipped_class = 0;   // present, but not a member class
        std::size_t              skipped_no_rt = 0;   // right class, but no usable room→member RT pose
        std::size_t              no_cov        = 0;   // right class, but the producer published no Σ
        std::vector<std::string> unclassified;        // a few names, to show WHAT is out there
    };
    MemberSnapshot poll_members();
    // Class of a generic "object" node: object_subtype when it names a known class, else the NAME
    // prefix. Mirrors ring_metaconcept::object_class_of / voxelizer's object_class_of.
    std::string object_class_of(const DSR::Node& node) const;
    // Class-evidence lookup: log[P(class | kitchen unit) / P(class | not)]; 0 for an unknown class.
    float class_logodds_of(const std::string& cls) const;
    // Which TIER a member is: the class decides where it can (refrigerator → tall, hood → wall),
    // geometry otherwise. Decides WHICH shared quantities apply to it, not merely whether it belongs.
    rc::KitchenTier resolve_tier(const std::string& cls, float z0, float top) const;
    // Read one member's room-frame pose + the covariance ITS OWN agent published on the room→member
    // RT edge. Returns nullopt when the edge or its pose is missing (not room-parented yet).
    std::optional<rc::KitchenMember> read_member(const DSR::Node& node, const std::string& cls);

    // ── Frame belief (M2, READ-ONLY: fits and logs, writes nothing to the graph) ──
    void step_kitchen_belief(const MemberSnapshot& s);

    // ── Cavity / leave-one-out (§6 Delta 3, the one hard requirement) ──────────
    // The message sent DOWN to member i must come from a frame fitted WITHOUT member i, or the frame
    // is fed its own output and manufactures agreement.
    //
    // ★This schema gets the down-date STRUCTURALLY, which the ring never could: the shared axis is
    // estimated from POLYGON-PINNED members only, and a pinned member's yaw comes from the room
    // polygon rather than from any fit — so it cannot have adopted a prior from us, and scoring it
    // cannot close the self-confirmation loop SCHEMA_GENERALITY_TODO.md §2.5 warns about. The
    // per-member refit below is still done for the unpinned members (the island), whose value we
    // would otherwise fold into the message we send it.
    void compute_cavity_priors(const MemberSnapshot& s);

    // ── The OUTLINE (step 1: READ-ONLY — measures joints, corrects nothing) ────
    // The seams between adjacent carcasses are absent from the sensor data, so the split of one
    // continuous front surface into runs cannot be recovered from below — it must be imposed here.
    // This builds the chain and reports, per joint, how far each run's end is from the shared vertex.
    // Positive = a hole, negative = an overlap. Floor units (base + tall) and wall units form
    // SEPARATE outlines: they are at different heights and must not be joined to each other.
    void step_outline(const MemberSnapshot& s);
    // STEP 2: birth/refresh the kitchen node and push each member's prior — the grid axis from the
    // cavity fit, and the END targets from the continuous outline. Gated on cfg_.publish.
    void publish_priors(const MemberSnapshot& s);
    // Room-polygon centroid, used only to orient each run's front face INTO the room.
    std::optional<Eigen::Vector2f> room_interior() const;

    // ── Diagnostics ────────────────────────────────────────────────────────────
    void log_outline_csv();
    void log_members_csv(const MemberSnapshot& s);
    void log_fit_csv(const MemberSnapshot& s);

    // ── Presence protocol (copied from the level-1 agents; see CLAUDE.md) ───────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void remove_owned_kitchen_nodes();   // startup stale-sweep of leftover "kitchen_*" nodes
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

    rc::KitchenConfig cfg_;
    std::uint64_t     room_node_id_ = 0;
    FPSCounter        fps_counter_;

    // RT_API reads the room→member edge pose + covariance. It resolves the edge's timestamp ring
    // buffer for us, which reading rt_translation_att raw does NOT (those attrs are ring-buffered).
    std::unique_ptr<DSR::RT_API> rt_api_;

    rc::KitchenBelief kitchen_belief_;
    // Per-member leave-one-out down-prior — what M3 would publish on each group_member edge.
    std::unordered_map<std::uint64_t, rc::KitchenBelief::MemberPrior> cavity_;

    // ── OUTSTANDING MESSAGES (the EP cavity store) ────────────────────────────
    // What this frame last told each member, per DOF. The frame is the SINGLE WRITER, so it can do
    // the whole down-date on the reading side: subtract this from a member's published value before
    // using it as evidence, and our own message can never circulate back as corroboration.
    // ★Empty while the agent stays read-only — and it must be in place BEFORE the first push, not
    // after, or the first thing the loop learns is its own voice.
    struct SentMessage
    {
        std::uint64_t stamp_ms = 0;
        rc::SentScalar yaw, worktop, depth;
    };
    std::unordered_map<std::uint64_t, SentMessage> sent_;
    // Subtract our outstanding message from a member's published values. Returns false if any DOF's
    // subtraction was refused (see rc::down_date) — reported, never silently clamped.
    bool apply_down_date(rc::KitchenMember& m) const;
    std::size_t down_date_refused_ = 0;   // per-cycle count, surfaced in the log

    std::unique_ptr<rc::KitchenSceneGraph> scene_graph_;
    rc::KitchenOutline outline_floor_;   // base + tall units: the floor-level chain
    rc::KitchenOutline outline_wall_;     // wall units: the upper chain
    std::ofstream outline_csv_;           // per-joint record; empty path disables

    std::ofstream csv_;       // per-member snapshot (pose + published Σ)
    std::ofstream fit_csv_;   // per-cycle frame fit + the down-prior it WOULD publish
    std::uint64_t cycle_ = 0;

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
