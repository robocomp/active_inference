/*
 * interaction_anchor.h — while an affordance on an object is being executed, the object's OWNING
 * agent hangs a shallow copy of it from the robot node. The RT edge robot→copy carries the
 * robot-relative pose (x, y, yaw) and its covariance, stamped; the affordance node names the copy
 * through `interaction_node_id`. The controller then executes against that edge instead of a
 * room-frame target, so a room-frame jump (relocalisation, a proto-room promotion) cannot move the
 * target under it.
 *
 * Who writes what (one writer per node, see affordance_manager.h on update_node):
 *   - the anchor node and its RT edge: the object agent (create / write / relink / remove);
 *   - affordance.interaction_node_id: the object agent too — it owns the affordance node;
 *   - NEVER the controller→affordance `executing` edge: that is the consumer's.
 *
 * The link back to the original object is `interaction_node_id` ON THE ANCHOR, not `parent`:
 * `parent` is the RT tree's own attribute, and RT_API rewrites it to the robot's id on every
 * insert_or_assign_edge_RT (cortex api/dsr_rt_api.cpp, and check_RT_tree repairs it the same way).
 * A copy's id in `parent` would be clobbered on the first write.
 *
 * Task 12 — THE DOOR'S SWUNG LEAF ON THE ANCHOR (existing attributes only; full contract in anchor_leaf.h). In ANCHOR
 * coordinates (origin = the opening's centre on the wall's mid-plane, x along the wall, +y the normal, the robot on −y at
 * creation): delimiting_polygon_x/_y = 3 vertices [the hinge on the leaf's centre line (its x = ±w_clear/2 names the
 * hinge jamb), the free edge at the MAP φ (its y sign names the swing side: −y toward the robot, +y away), the free edge one
 * σ_φ further open]; door_phi_rad = φ. Written by the owner (Anchor::set_leaf), read by read_anchor_leaf. No leaf known ⇒
 * the attributes are ABSENT (never a sentinel). Leaf half-thickness: kLeafHalfThickM, the model's, shared by both sides.
 *
 * is_anchor() is header-inline so a pure reader (ltsm, episodic, chat, graph3d) needs only this
 * include. The Anchor class lives in interaction_anchor.cpp, added to the writers' SOURCES.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>

#include "anchor_leaf.h"
#include "relative_pose.h"

namespace rc::interaction
{

inline constexpr std::string_view kAnchorSuffix = "@robot";

// READER predicate. An anchor: a node of type "object" that carries `interaction_node_id` (its
// original) and EITHER hangs from a node of type "robot" OR is named "<original>@robot". The type and
// the attribute are both needed: the robot also parents its sensor mounts and `body` (all with
// `parent`), and affordances carry `interaction_node_id` too (but are not type "object"). The name arm
// covers the window between insert_node and the first RT write, and a copy whose edge never landed:
// such a node has no RT parent yet, and a reader must still not take it for an object.
[[nodiscard]] inline bool is_anchor(DSR::DSRGraph& G, const DSR::Node& n)
{
    if (n.type() != "object") return false;
    if (not G.get_attrib_by_name<interaction_node_id_att>(n).has_value()) return false;
    if (n.name().ends_with(kAnchorSuffix)) return true;
    const auto p = G.get_attrib_by_name<parent_att>(n);
    if (not p.has_value()) return false;
    const auto pn = G.get_node(p.value());
    return pn.has_value() and pn->type() == "robot";
}

// OWNER predicate, edge-independent: this agent's copy, whether or not its RT edge exists. Used to
// reclaim (create, sweep_stale) — a copy that leaked before its edge was written must still be ours
// to delete, or the object could never be anchored again.
[[nodiscard]] inline bool is_own_anchor_node(DSR::DSRGraph& G, const DSR::Node& n, int agent_id)
{
    if (n.type() != "object" or not n.name().ends_with(kAnchorSuffix)) return false;
    if (not G.get_attrib_by_name<interaction_node_id_att>(n).has_value()) return false;
    const auto owner = G.get_attrib_by_name<agent_id_att>(n);
    return owner.has_value() and owner.value() == static_cast<std::uint32_t>(agent_id);
}

class Anchor
{
public:
    // Shallow-copies `object` under the robot node; sets affordance.interaction_node_id to the copy.
    // nullopt (and nothing left in the graph) if the object, the affordance or the robot is missing,
    // or any write fails.
    static std::optional<Anchor> create(DSR::DSRGraph& G, std::uint64_t object_id, std::uint64_t affordance_id,
                                        int agent_id, const RelPose& init);
    bool write(DSR::DSRGraph& G, const RelPose& r);              // the stamped RT edge Shadow → anchor
    bool relink(DSR::DSRGraph& G, std::uint64_t new_object_id);  // the original, after re-identification
    // Name this anchor on `affordance_id` (fresh read, whole-node update, written only when the link is
    // missing or names something else). If it differs from the affordance held so far, that one's link is
    // dropped first (only if it still names us) and this becomes the affordance remove() unlinks. Two uses:
    // ONE anchor serving successive affordances on the same object (door: approach → cross), and re-asserting
    // the link every cycle — a consumer's whole-node update of the affordance can erase an attribute it never
    // read. false if the affordance node is missing or the write fails.
    bool link_affordance(DSR::DSRGraph& G, std::uint64_t affordance_id);
    // The anchor's width_m (Task 9 fix round 1: the owner's CLEAR width, not the object's). Written only on a change.
    bool set_width(DSR::DSRGraph& G, float width_m);
    // How long ago the owner last SAW what the edge's pose describes (unseen_time, ms; 0 = seen this cycle). Written only on
    // a change. See kAnchorPredictMs.
    bool set_unseen_ms(DSR::DSRGraph& G, int unseen_ms);
    // Task 12 fix A: the door's swung leaf in ANCHOR coordinates (anchor_leaf.h: delimiting_polygon_x/_y = hinge, free edge,
    // free edge one σ further open; door_phi_rad = φ). nullopt REMOVES both (no leaf known — never a sentinel). Written
    // only on a change.
    bool set_leaf(DSR::DSRGraph& G, const std::optional<AnchorLeaf>& leaf);
    void remove(DSR::DSRGraph& G);                               // node, edge, and the affordance's link
    [[nodiscard]] std::uint64_t id() const { return id_; }
    [[nodiscard]] std::uint64_t object_id() const { return object_id_; }
    [[nodiscard]] std::uint64_t affordance_id() const { return affordance_id_; }

private:
    Anchor() = default;
    std::uint64_t id_ = 0, object_id_ = 0, affordance_id_ = 0, robot_id_ = 0;
    std::shared_ptr<DSR::RT_API> rt_;                            // shared: Anchor stays copyable
};

// Startup: delete this agent's leaked anchors (a SIGKILLed previous run) and the links its own
// affordances still hold to them.
void sweep_stale(DSR::DSRGraph& G, int agent_id);

// ★ A STALE EDGE MEANS ITS OWNER DIED, NOT A WEAK MEASUREMENT. The owner rewrites the edge every
// cycle while it executes; an edge older than this is not "uncertain", it has no writer, so the
// reader refuses it instead of widening it. FLAGGED decision level (no model term can stand in for
// a dead producer).
inline constexpr std::int64_t kAnchorStaleMs = 1000;

// ⚑ FLAGGED decision level (Task 9 fix round 1, user-approved): while the anchored door's CROSS is claimed and the anchor
// has been SEEN in this passage, the owner keeps publishing its ODOMETRY-PREDICTED pose (and the predicted covariance,
// growing with the motion) for up to this long after the last frame it saw the door in — the doorway band, where the jambs
// leave the dense sampling, is exactly where the strict rule stopped the robot. The anchor node's unseen_time (ms) says
// how old the last sighting is: 0 = seen, > 0 = predicted. Past this, nothing is published (the strict rule).
inline constexpr std::int64_t kAnchorPredictMs = 3000;

// Controller side: follows affordance.interaction_node_id to the anchor and reads the newest block of
// the robot→anchor RT edge. Returns {pose, anchor's width_m (0 if absent)}; nullopt if there is no
// link, the node or edge is missing, the edge carries no covariance, or its newest stamp is older than
// kAnchorStaleMs relative to now_ms.
std::optional<std::pair<RelPose, float>> read_affordance_anchor(DSR::DSRGraph& G, const DSR::Node& affordance,
                                                                std::int64_t now_ms);

// The same read, by the anchor's own id — for a consumer that latched it from read_affordance_anchor and must keep
// executing after the affordance node is gone (door_concept deletes it at a room switch). Same refusals, same
// kAnchorStaleMs rule; nullopt once the owner removes the anchor.
std::optional<std::pair<RelPose, float>> read_anchor(DSR::DSRGraph& G, std::uint64_t anchor_id, std::int64_t now_ms);

// The same read WITHOUT the kAnchorStaleMs refusal (Task 9): the anchor's newest published block, whatever its age
// (RelPose::stamp_ms says when). For a consumer that must know an anchor EXISTS and where it last was — to WAIT for it
// and turn toward it — never to drive on it: a stale pose is a pose nobody is looking at. Same other refusals.
std::optional<std::pair<RelPose, float>> read_anchor_latest(DSR::DSRGraph& G, std::uint64_t anchor_id);

// Task 12 fix A: the leaf the owner publishes on the anchor (anchor_leaf.h), or nullopt (none known / malformed).
std::optional<AnchorLeaf> read_anchor_leaf(DSR::DSRGraph& G, std::uint64_t anchor_id);

}  // namespace rc::interaction
