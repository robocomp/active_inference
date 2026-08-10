/*
 * table_affordance.cpp  —  affordance-node lifecycle + controller-protocol state machine.
 *
 * Creates/refreshes/removes the per-table affordance DSR node, writes the epistemic target + the shared
 * execution contract, and drives the idle→pending→executing→satisfied transitions off the controller-owned
 * active_att / epistemic_pending_att flags observed on the node.
 */

#include "object_affordance.h"

#include <algorithm>
#include <array>
#include <format>
#include <string>
#include <QtGlobal>
#include <cmath>

#include "../../common/affordance_protocol/affordance_protocol.h"
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

// ─── Public API ───────────────────────────────────────────────────────────────


namespace rc {
void ObjectAffordance::init(std::shared_ptr<DSR::DSRGraph> G,
                            uint64_t    parent_node_id,
                            std::string parent_node_name,
                           std::string object_type)
{
    G_               = std::move(G);
    parent_node_id_   = parent_node_id;
    parent_node_name_ = std::move(parent_node_name);
    object_type_      = std::move(object_type);
}

void ObjectAffordance::update(const AffordanceTarget& prop, bool orient_mode)
{
    orient_mode_ = orient_mode;
    if (not G_) return;
    if (not prop.valid or not prop.is_finite())
    {
        if (prop.valid)
            qWarning() << "[affordance] rejecting non-finite epistemic proposal for"
                       << parent_node_name_.c_str();
        return;
    }

    if (not node_created_)
        create_node(prop);
    else
        update_node(prop);
}

void ObjectAffordance::hold_offered()
{
    if (not G_ or not node_created_ or affordance_node_id_ == 0)
        return;
    auto node_opt = G_->get_node(affordance_node_id_);
    if (not node_opt.has_value())
        return;
    auto& n = node_opt.value();

    const bool active = G_->get_attrib_by_name<active_att>(n).value_or(false);
    const bool pending = G_->get_attrib_by_name<epistemic_pending_att>(n).value_or(true);
    if (active and pending)
        return;                         // Executing: the controller owns it, do not touch the flags
    if (not active and pending)
        return;                         // already Offered — nothing to repair

    // Completed (or Invalid) with no proposal to publish: put it back on offer, and publish a gain of
    // ZERO. Not the last one — a stale 5 nats would send the robot across the room on the strength of a
    // number we can no longer justify. Zero is the honest floor: the affordance stays in the contest, so
    // it is visible and selectable when nothing else bids, and it starts attracting travel again the
    // moment a real proposal returns.
    G_->add_or_modify_attrib_local<active_att>           (n, false);
    G_->add_or_modify_attrib_local<epistemic_pending_att>(n, true);
    G_->add_or_modify_attrib_local<epistemic_gain_att>   (n, 0.0f);
    // Hypothesis→located (or back): swap the Orient↔Servo contract when the mode changed.
    if (orient_mode_ != contract_is_orient_)
        write_policy_contract(n);

    state_ = State::pending;
    G_->update_node(n);
    refresh_edge();
    std::print("[affordance] '{}' re-offered with gain 0 — no proposal this cycle "
               "(it was left Completed, which no producer ever intended)\n", parent_node_name_);
}

void ObjectAffordance::remove()
{
    if (not G_ or not node_created_) return;

    G_->delete_node(affordance_node_id_);
    std::print("[affordance] removed node for '{}' (state={})\n",
               parent_node_name_, state_name(state_));
    reset();   // returns to State::idle so a new cycle can start if the model degrades later
}

void ObjectAffordance::on_node_modified(uint64_t id)
{
    if (not node_created_ or id != affordance_node_id_) return;

    auto node_opt = G_->get_node(affordance_node_id_);
    if (not node_opt.has_value()) return;

    const bool active = G_->get_attrib_by_name<active_att>(node_opt.value()).value_or(false);
    const bool pending = G_->get_attrib_by_name<epistemic_pending_att>(node_opt.value()).value_or(true);

    if (active and pending)
    {
        if (state_ != State::executing)
        {
            state_ = State::executing;
            std::print("[affordance] '{}' → executing (controller claimed)\n",
                       parent_node_name_);
        }
        return;
    }

    if (not active and pending)
    {
        state_ = State::pending;
        return;
    }

    if (not active and not pending)
    {
        if (state_ != State::satisfied)
        {
            state_ = State::satisfied;
            std::print("[affordance] '{}' → satisfied (controller completed)\n",
                       parent_node_name_);
        }
        return;
    }

    if (state_ != State::aborted)
    {
        state_ = State::aborted;
        std::print("[affordance] '{}' → aborted (invalid protocol state active=1 pending=0)\n",
                   parent_node_name_);
    }
}

void ObjectAffordance::on_node_deleted(uint64_t id)
{
    if (not node_created_ or id != affordance_node_id_) return;

    std::print("[affordance] '{}' node deleted externally (state={}) → idle\n",
               parent_node_name_, state_name(state_));
    reset();
}

std::string_view ObjectAffordance::state_name(State s)
{
    switch (s)
    {
        case State::idle:      return "idle";
        case State::pending:   return "pending";
        case State::executing: return "executing";
        case State::satisfied: return "satisfied";
        case State::aborted:   return "aborted";
    }
    return "unknown";
}

// ─── Private helpers ─────────────────────────────────────────────────────────

// Build the object-relative viewpoint constraint (the authoritative "where to look") from a proposal:
// all four table faces in object frame + their adequacy-bounded gains + the stand-off band + framing +
// Σ*. The controller resolves this into a collision-free reachable pose. Face order = [+x,-x,+y,-y].
static rc::affordance::ViewpointConstraint make_viewpoint(const AffordanceTarget& prop)
{
    rc::affordance::ViewpointConstraint v;
    v.object_relative = true;
    // ★FACES ARE AGENT-DEPENDENT. A box enumerates four; a bottle is a cylinder with none; a door is
    // judged on its panel. Publish the canonical face names only for as many gains as the planner
    // actually produced, so an agent that enumerates nothing publishes nothing rather than four zeros
    // the controller would treat as four equally-bad viewpoints.
    static constexpr std::array<const char*, 4> kFaceNames{"+x", "-x", "+y", "-y"};
    const std::size_t nf = std::min(prop.face_gains.size(), kFaceNames.size());
    for (std::size_t i = 0; i < nf; ++i)
    {
        v.faces.emplace_back(kFaceNames[i]);
        v.face_gains.push_back(prop.face_gains[i]);
    }
    v.standoff_min_m = prop.standoff_min_m;
    v.standoff_max_m = prop.standoff_max_m;
    v.framing_fill   = prop.framing_fill;
    v.sigma_star.assign(prop.sigma_star.begin(), prop.sigma_star.end());
    return v;
}

void ObjectAffordance::create_node(const AffordanceTarget& prop)
{
    const std::string aff_name = "aff_" + parent_node_name_;

    DSR::Node aff_node = DSR::Node::create<affordance_node_type>(aff_name);
    G_->add_or_modify_attrib_local<level_att>                   (aff_node, 4);
    G_->add_or_modify_attrib_local<parent_att>                  (aff_node, parent_node_id_);
    G_->add_or_modify_attrib_local<pos_x_att>                   (aff_node, 300.f);
    G_->add_or_modify_attrib_local<pos_y_att>                   (aff_node, 200.f);
    G_->add_or_modify_attrib_local<active_att>                  (aff_node, false);
    G_->add_or_modify_attrib_local<epistemic_target_x_m_att>    (aff_node, prop.x_m);
    G_->add_or_modify_attrib_local<epistemic_target_y_m_att>    (aff_node, prop.y_m);
    G_->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(aff_node, prop.yaw_rad);
    G_->add_or_modify_attrib_local<epistemic_gain_att>          (aff_node, prop.gain);
    G_->add_or_modify_attrib_local<epistemic_pending_att>       (aff_node, true);

    // Declare the execution contract: how the controller should complete this affordance (Servo
    // lock-on bound to the table's projected-ROI / detection feedback attributes + completion
    // predicate). Uses the shared type-level default; producers can override per node here.
    write_policy_contract(aff_node);
    // Object-relative viewpoint constraint (the authoritative epistemic target the controller resolves).
    rc::affordance::write_viewpoint(*G_, aff_node, make_viewpoint(prop));

    rc::provenance::stamp_creation(*G_, aff_node);   // birth stamp: epoch ms + local ISO-8601
    const auto id_opt = G_->insert_node(aff_node);
    if (not id_opt.has_value())
    {
        qWarning() << "[affordance] failed to insert DSR node for" << aff_name.c_str();
        return;
    }
    affordance_node_id_ = id_opt.value();
    node_created_       = true;
    state_              = State::pending;

    refresh_edge();

    std::print("[affordance] created '{}' id={} target=({:.2f},{:.2f}) "
               "yaw={:.2f} gain={:.4f}\n",
               aff_name, affordance_node_id_,
               prop.x_m, prop.y_m,
               prop.yaw_rad, prop.gain);

    // Self-check (producer side): read the object-relative viewpoint constraint back off the node we just
    // inserted and validate the round-trip + value sanity — faces↔gains parity (4 for a table), finite
    // non-negative gains, Σ* of size 6, ordered stand-off band. Fires once per affordance birth; a MISMATCH
    // means a serialization or value bug at the source (before it ever crosses the DSR boundary).
    if (auto rb = G_->get_node(affordance_node_id_); rb.has_value())
    {
        const auto vc = rc::affordance::read_viewpoint(rb.value());
        bool gains_ok = true;   // no faces is legitimate (a cylinder has none)
        for (const float g : vc.face_gains)
            if (not (std::isfinite(g) and g >= 0.0f)) gains_ok = false;
        // Shape checks are RELATIONAL, not absolute: faces must match gains (whatever the count), and the
        // band must be ordered. Asserting "4 faces and 6 sigma*" was table's own geometry hard-coded into
        // what is now shared code — a bottle has 5 DOF and no faces, and would have failed a healthy publish.
        const bool ok = vc.object_relative and vc.faces.size() == vc.face_gains.size()
                        and gains_ok and vc.standoff_min_m <= vc.standoff_max_m;
        if (ok)
        {
            std::string gains_s, star_s;
            for (std::size_t i = 0; i < vc.face_gains.size(); ++i)
                gains_s += std::format("{}{:.2f}", i ? "," : "", vc.face_gains[i]);
            for (std::size_t i = 0; i < vc.sigma_star.size(); ++i)
                star_s += std::format("{}{:.3f}", i ? "," : "", vc.sigma_star[i]);
            std::print("[affordance-selfcheck] '{}' viewpoint OK: faces={} gains=[{}] standoff=[{:.2f},{:.2f}] "
                       "fill={:.2f} σ*=[{}]\n",
                       parent_node_name_, vc.faces.size(), gains_s,
                       vc.standoff_min_m, vc.standoff_max_m, vc.framing_fill, star_s);
        }
        else
            std::print("[affordance-selfcheck] '{}' viewpoint MISMATCH: object_relative={} faces={} gains={} "
                       "σ*={} standoff=[{:.2f},{:.2f}]\n",
                       parent_node_name_, vc.object_relative, vc.faces.size(), vc.face_gains.size(),
                       vc.sigma_star.size(), vc.standoff_min_m, vc.standoff_max_m);
    }
}

void ObjectAffordance::update_node(const AffordanceTarget& prop)
{
    auto node_opt = G_->get_node(affordance_node_id_);
    if (not node_opt.has_value())
    {
        // Node was deleted externally — recreate on next cycle
        std::print("[affordance] '{}' node missing, will recreate\n", parent_node_name_);
        reset();
        return;
    }
    auto& n = node_opt.value();
    const bool active = G_->get_attrib_by_name<active_att>(n).value_or(false);
    const bool pending = G_->get_attrib_by_name<epistemic_pending_att>(n).value_or(true);

    if (active and pending)
    {
        state_ = State::executing;
        // Refresh ONLY the epistemic value while the controller owns the claim, so the grounded EFE
        // selection sees the belief decay during a long execution (ΔH→0 as evidence accumulates). Do
        // NOT touch the target pose or active/pending flags. Dead-band the write to avoid per-cycle
        // graph churn.
        const float cur_gain = G_->get_attrib_by_name<epistemic_gain_att>(n).value_or(0.f);
        // 0.1 nat dead-band: write-suppression only (avoids per-cycle DSR churn on sub-perceptible gain
        // changes), NOT a belief threshold — it gates a graph WRITE, never the fit or a decision. Left
        // hardcoded; a deployer never needs to tune graph-write hysteresis.
        if (std::abs(cur_gain - prop.gain) > 0.1f)
        {
            G_->add_or_modify_attrib_local<epistemic_gain_att>(n, prop.gain);
            G_->update_node(n);
        }
        refresh_edge();
        return;
    }

    G_->add_or_modify_attrib_local<epistemic_target_x_m_att>    (n, prop.x_m);
    G_->add_or_modify_attrib_local<epistemic_target_y_m_att>    (n, prop.y_m);
    G_->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(n, prop.yaw_rad);
    G_->add_or_modify_attrib_local<epistemic_gain_att>          (n, prop.gain);
    // Refresh the object-relative viewpoint constraint alongside the hint pose (offered/pending path only;
    // like the pose, it is held stable while the controller owns an executing claim).
    rc::affordance::write_viewpoint(*G_, n, make_viewpoint(prop));

    if (not pending or active)
    {
        G_->add_or_modify_attrib_local<active_att>              (n, false);
        G_->add_or_modify_attrib_local<epistemic_pending_att>   (n, true);
    }

    state_ = State::pending;
    G_->update_node(n);

    refresh_edge();
}

// Orient vs Servo. The completion feedback attributes follow the fleet's `<object>_*` convention
// (chair_roi_offset, door_detection_alive, …), so the contract is built from object_type_ rather than
// hard-coded per agent — the two copies this was taken from differed only in that prefix.
void ObjectAffordance::write_policy_contract(DSR::Node& node)
{
    using namespace rc::affordance;
    if (orient_mode_)
        // Rotate in place toward the target yaw (the bearing), fine-centre on the object's ROI once it
        // enters the zed view, and complete when a real DEPTH detection fires — which is exactly the
        // event that promotes a bearing-only hypothesis into a located instance.
        write_contract(*G_, node, Contract::orient()
            .center(object_type_ + "_roi_offset")
            .valid (object_type_ + "_roi_valid")
            .until (object_type_ + "_detection_alive",      CompareOp::GE, 0.5f)
            .and_  (object_type_ + "_detection_confidence", CompareOp::GE, 0.20f)
            .still(0.0f, 0.15f).stable(2).timeout_s(8).on_fail(OnFail::Consume));
    else
        write_contract(*G_, node, default_contract_for(object_type_));
    contract_is_orient_ = orient_mode_;
}

void ObjectAffordance::refresh_edge()
{
    auto edge = DSR::Edge::create<has_intention_edge_type>(parent_node_id_, affordance_node_id_);
    G_->insert_or_assign_edge(edge);
}

void ObjectAffordance::reset()
{
    affordance_node_id_ = 0;
    node_created_       = false;
    state_              = State::idle;
}

}  // namespace rc
