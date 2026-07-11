/*
 * table_affordance.cpp  —  affordance-node lifecycle + controller-protocol state machine.
 *
 * Creates/refreshes/removes the per-table affordance DSR node, writes the epistemic target + the shared
 * execution contract, and drives the idle→pending→executing→satisfied transitions off the controller-owned
 * active_att / epistemic_pending_att flags observed on the node.
 */

#include "table_affordance.h"
#include <QtGlobal>
#include <cmath>

#include "../../common/affordance_protocol/affordance_protocol.h"

// ─── Public API ───────────────────────────────────────────────────────────────


namespace rc {
void TableAffordance::init(std::shared_ptr<DSR::DSRGraph> G,
                            uint64_t    table_node_id,
                            std::string table_node_name)
{
    G_               = std::move(G);
    table_node_id_   = table_node_id;
    table_node_name_ = std::move(table_node_name);
}

void TableAffordance::update(const EpistemicProposal& prop)
{
    if (!G_) return;
    if (!prop.valid || !prop.is_finite())
    {
        if (prop.valid)
            qWarning() << "[affordance] rejecting non-finite epistemic proposal for"
                       << table_node_name_.c_str();
        return;
    }

    if (!node_created_)
        create_node(prop);
    else
        update_node(prop);
}

void TableAffordance::remove()
{
    if (!G_ || !node_created_) return;

    G_->delete_node(affordance_node_id_);
    std::print("[affordance] removed node for '{}' (state={})\n",
               table_node_name_, state_name(state_));
    reset();
    state_ = State::satisfied;
    // Reset to idle so a new cycle can start if the model degrades later
    state_ = State::idle;
}

void TableAffordance::on_node_modified(uint64_t id)
{
    if (!node_created_ || id != affordance_node_id_) return;

    auto node_opt = G_->get_node(affordance_node_id_);
    if (!node_opt.has_value()) return;

    const bool active = G_->get_attrib_by_name<active_att>(node_opt.value()).value_or(false);
    const bool pending = G_->get_attrib_by_name<epistemic_pending_att>(node_opt.value()).value_or(true);

    if (active && pending)
    {
        if (state_ != State::executing)
        {
            state_ = State::executing;
            std::print("[affordance] '{}' → executing (controller claimed)\n",
                       table_node_name_);
        }
        return;
    }

    if (!active && pending)
    {
        state_ = State::pending;
        return;
    }

    if (!active && !pending)
    {
        if (state_ != State::satisfied)
        {
            state_ = State::satisfied;
            std::print("[affordance] '{}' → satisfied (controller completed)\n",
                       table_node_name_);
        }
        return;
    }

    if (state_ != State::aborted)
    {
        state_ = State::aborted;
        std::print("[affordance] '{}' → aborted (invalid protocol state active=1 pending=0)\n",
                   table_node_name_);
    }
}

void TableAffordance::on_node_deleted(uint64_t id)
{
    if (!node_created_ || id != affordance_node_id_) return;

    std::print("[affordance] '{}' node deleted externally (state={}) → idle\n",
               table_node_name_, state_name(state_));
    reset();
}

std::string_view TableAffordance::state_name(State s)
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
static rc::affordance::ViewpointConstraint make_viewpoint(const EpistemicProposal& prop)
{
    rc::affordance::ViewpointConstraint v;
    v.object_relative = true;
    v.faces        = {"+x", "-x", "+y", "-y"};
    v.face_gains   = {prop.face_gains[0], prop.face_gains[1], prop.face_gains[2], prop.face_gains[3]};
    v.standoff_min_m = prop.standoff_min_m;
    v.standoff_max_m = prop.standoff_max_m;
    v.framing_fill   = prop.framing_fill;
    v.sigma_star.assign(prop.sigma_star.begin(), prop.sigma_star.end());
    return v;
}

void TableAffordance::create_node(const EpistemicProposal& prop)
{
    const std::string aff_name = "aff_" + table_node_name_;

    DSR::Node aff_node = DSR::Node::create<affordance_node_type>(aff_name);
    G_->add_or_modify_attrib_local<level_att>                   (aff_node, 4);
    G_->add_or_modify_attrib_local<parent_att>                  (aff_node, table_node_id_);
    G_->add_or_modify_attrib_local<pos_x_att>                   (aff_node, 300.f);
    G_->add_or_modify_attrib_local<pos_y_att>                   (aff_node, 200.f);
    G_->add_or_modify_attrib_local<active_att>                  (aff_node, false);
    G_->add_or_modify_attrib_local<epistemic_target_x_m_att>    (aff_node, prop.epistemic_target_x_m);
    G_->add_or_modify_attrib_local<epistemic_target_y_m_att>    (aff_node, prop.epistemic_target_y_m);
    G_->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(aff_node, prop.epistemic_target_yaw_rad);
    G_->add_or_modify_attrib_local<epistemic_gain_att>          (aff_node, prop.epistemic_gain);
    G_->add_or_modify_attrib_local<epistemic_pending_att>       (aff_node, true);

    // Declare the execution contract: how the controller should complete this affordance (Servo
    // lock-on bound to the table's projected-ROI / detection feedback attributes + completion
    // predicate). Uses the shared type-level default; producers can override per node here.
    rc::affordance::write_contract(*G_, aff_node, rc::affordance::default_contract_for("table"));
    // Object-relative viewpoint constraint (the authoritative epistemic target the controller resolves).
    rc::affordance::write_viewpoint(*G_, aff_node, make_viewpoint(prop));

    const auto id_opt = G_->insert_node(aff_node);
    if (!id_opt.has_value())
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
               prop.epistemic_target_x_m, prop.epistemic_target_y_m,
               prop.epistemic_target_yaw_rad, prop.epistemic_gain);
}

void TableAffordance::update_node(const EpistemicProposal& prop)
{
    auto node_opt = G_->get_node(affordance_node_id_);
    if (!node_opt.has_value())
    {
        // Node was deleted externally — recreate on next cycle
        std::print("[affordance] '{}' node missing, will recreate\n", table_node_name_);
        reset();
        return;
    }
    auto& n = node_opt.value();
    const bool active = G_->get_attrib_by_name<active_att>(n).value_or(false);
    const bool pending = G_->get_attrib_by_name<epistemic_pending_att>(n).value_or(true);

    if (active && pending)
    {
        state_ = State::executing;
        // Refresh ONLY the epistemic value while the controller owns the claim, so the grounded EFE
        // selection sees the belief decay during a long execution (ΔH→0 as evidence accumulates). Do
        // NOT touch the target pose or active/pending flags. Dead-band the write to avoid per-cycle
        // graph churn.
        const float cur_gain = G_->get_attrib_by_name<epistemic_gain_att>(n).value_or(0.f);
        if (std::abs(cur_gain - prop.epistemic_gain) > 0.1f)
        {
            G_->add_or_modify_attrib_local<epistemic_gain_att>(n, prop.epistemic_gain);
            G_->update_node(n);
        }
        refresh_edge();
        return;
    }

    G_->add_or_modify_attrib_local<epistemic_target_x_m_att>    (n, prop.epistemic_target_x_m);
    G_->add_or_modify_attrib_local<epistemic_target_y_m_att>    (n, prop.epistemic_target_y_m);
    G_->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(n, prop.epistemic_target_yaw_rad);
    G_->add_or_modify_attrib_local<epistemic_gain_att>          (n, prop.epistemic_gain);
    // Refresh the object-relative viewpoint constraint alongside the hint pose (offered/pending path only;
    // like the pose, it is held stable while the controller owns an executing claim).
    rc::affordance::write_viewpoint(*G_, n, make_viewpoint(prop));

    if (!pending || active)
    {
        G_->add_or_modify_attrib_local<active_att>              (n, false);
        G_->add_or_modify_attrib_local<epistemic_pending_att>   (n, true);
    }

    state_ = State::pending;
    G_->update_node(n);

    refresh_edge();
}

void TableAffordance::refresh_edge()
{
    auto edge = DSR::Edge::create<has_intention_edge_type>(table_node_id_, affordance_node_id_);
    G_->insert_or_assign_edge(edge);
}

void TableAffordance::reset()
{
    affordance_node_id_ = 0;
    node_created_       = false;
    state_              = State::idle;
}

}  // namespace rc
