/*
 * cabinet_affordance.cpp  —  affordance-node lifecycle + controller-protocol state machine.
 *
 * Creates/refreshes/removes the per-cabinet affordance DSR node, writes the epistemic target + the shared
 * execution contract, and drives the idle→pending→executing→satisfied transitions off the controller-owned
 * active_att / epistemic_pending_att flags observed on the node.
 */

#include "cabinet_affordance.h"
#include <QtGlobal>
#include <cmath>

#include "../../common/affordance_protocol/affordance_protocol.h"
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

// ─── Public API ───────────────────────────────────────────────────────────────


namespace rc {
void CabinetAffordance::init(std::shared_ptr<DSR::DSRGraph> G,
                            uint64_t    cabinet_node_id,
                            std::string cabinet_node_name)
{
    G_               = std::move(G);
    cabinet_node_id_   = cabinet_node_id;
    cabinet_node_name_ = std::move(cabinet_node_name);
}

void CabinetAffordance::update(const EpistemicProposal& prop)
{
    if (!G_) return;
    if (!prop.valid || !prop.is_finite())
    {
        if (prop.valid)
            qWarning() << "[affordance] rejecting non-finite epistemic proposal for"
                       << cabinet_node_name_.c_str();
        return;
    }

    if (!node_created_)
        create_node(prop);
    else
        update_node(prop);
}

void CabinetAffordance::remove()
{
    if (!G_ || !node_created_) return;

    G_->delete_node(affordance_node_id_);
    std::print("[affordance] removed node for '{}' (state={})\n",
               cabinet_node_name_, state_name(state_));
    reset();
    state_ = State::satisfied;
    // Reset to idle so a new cycle can start if the model degrades later
    state_ = State::idle;
}

void CabinetAffordance::on_node_modified(uint64_t id)
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
                       cabinet_node_name_);
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
                       cabinet_node_name_);
        }
        return;
    }

    if (state_ != State::aborted)
    {
        state_ = State::aborted;
        std::print("[affordance] '{}' → aborted (invalid protocol state active=1 pending=0)\n",
                   cabinet_node_name_);
    }
}

void CabinetAffordance::on_node_deleted(uint64_t id)
{
    if (!node_created_ || id != affordance_node_id_) return;

    std::print("[affordance] '{}' node deleted externally (state={}) → idle\n",
               cabinet_node_name_, state_name(state_));
    reset();
}

std::string_view CabinetAffordance::state_name(State s)
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
// all four cabinet faces in object frame + their adequacy-bounded gains + the stand-off band + framing +
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

void CabinetAffordance::create_node(const EpistemicProposal& prop)
{
    const std::string aff_name = "aff_" + cabinet_node_name_;

    DSR::Node aff_node = DSR::Node::create<affordance_node_type>(aff_name);
    G_->add_or_modify_attrib_local<level_att>                   (aff_node, 4);
    G_->add_or_modify_attrib_local<parent_att>                  (aff_node, cabinet_node_id_);
    G_->add_or_modify_attrib_local<pos_x_att>                   (aff_node, 300.f);
    G_->add_or_modify_attrib_local<pos_y_att>                   (aff_node, 200.f);
    G_->add_or_modify_attrib_local<active_att>                  (aff_node, false);
    G_->add_or_modify_attrib_local<epistemic_target_x_m_att>    (aff_node, prop.epistemic_target_x_m);
    G_->add_or_modify_attrib_local<epistemic_target_y_m_att>    (aff_node, prop.epistemic_target_y_m);
    G_->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(aff_node, prop.epistemic_target_yaw_rad);
    G_->add_or_modify_attrib_local<epistemic_gain_att>          (aff_node, prop.epistemic_gain);
    G_->add_or_modify_attrib_local<epistemic_pending_att>       (aff_node, true);

    // Declare the execution contract: how the controller should complete this affordance (Servo
    // lock-on bound to the cabinet's projected-ROI / detection feedback attributes + completion
    // predicate). Uses the shared type-level default; producers can override per node here.
    rc::affordance::write_contract(*G_, aff_node, rc::affordance::default_contract_for("cabinet"));
    // Object-relative viewpoint constraint (the authoritative epistemic target the controller resolves).
    rc::affordance::write_viewpoint(*G_, aff_node, make_viewpoint(prop));

    rc::provenance::stamp_creation(*G_, aff_node);   // birth stamp: epoch ms + local ISO-8601
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

    // Self-check (producer side): read the object-relative viewpoint constraint back off the node we just
    // inserted and validate the round-trip + value sanity — faces↔gains parity (4 for a cabinet), finite
    // non-negative gains, Σ* of size 6, ordered stand-off band. Fires once per affordance birth; a MISMATCH
    // means a serialization or value bug at the source (before it ever crosses the DSR boundary).
    if (auto rb = G_->get_node(affordance_node_id_); rb.has_value())
    {
        const auto vc = rc::affordance::read_viewpoint(rb.value());
        bool gains_ok = not vc.face_gains.empty();
        for (const float g : vc.face_gains)
            if (not (std::isfinite(g) and g >= 0.0f)) gains_ok = false;
        const bool ok = vc.object_relative and vc.faces.size() == 4
                    and vc.face_gains.size() == 4 and gains_ok
                    and vc.sigma_star.size() == 7 and vc.standoff_min_m <= vc.standoff_max_m;
        if (ok)
            std::print("[affordance-selfcheck] '{}' viewpoint OK: gains(end+,end-,front,back)=[{:.2f},{:.2f},{:.2f},{:.2f}] "
                       "standoff=[{:.2f},{:.2f}] fill={:.2f} σ*=[{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]\n",
                       cabinet_node_name_,
                       vc.face_gains[0], vc.face_gains[1], vc.face_gains[2], vc.face_gains[3],
                       vc.standoff_min_m, vc.standoff_max_m, vc.framing_fill,
                       vc.sigma_star[0], vc.sigma_star[1], vc.sigma_star[2], vc.sigma_star[3],
                       vc.sigma_star[4], vc.sigma_star[5], vc.sigma_star[6]);
        else
            std::print("[affordance-selfcheck] '{}' viewpoint MISMATCH: object_relative={} faces={} gains={} "
                       "σ*={} standoff=[{:.2f},{:.2f}]\n",
                       cabinet_node_name_, vc.object_relative, vc.faces.size(), vc.face_gains.size(),
                       vc.sigma_star.size(), vc.standoff_min_m, vc.standoff_max_m);
    }
}

void CabinetAffordance::update_node(const EpistemicProposal& prop)
{
    auto node_opt = G_->get_node(affordance_node_id_);
    if (!node_opt.has_value())
    {
        // Node was deleted externally — recreate on next cycle
        std::print("[affordance] '{}' node missing, will recreate\n", cabinet_node_name_);
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

void CabinetAffordance::refresh_edge()
{
    auto edge = DSR::Edge::create<has_intention_edge_type>(cabinet_node_id_, affordance_node_id_);
    G_->insert_or_assign_edge(edge);
}

void CabinetAffordance::reset()
{
    affordance_node_id_ = 0;
    node_created_       = false;
    state_              = State::idle;
}

}  // namespace rc
