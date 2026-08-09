/*
 * human_affordance.cpp — object-agnostic affordance mechanics (mirrors bottle_affordance.cpp);
 * parent object type is "person", contract from default_contract_for("person").
 */

#include "human_affordance.h"

#include <QtGlobal>
#include <cmath>

#include "../../common/affordance_protocol/affordance_protocol.h"
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

namespace rc {

void HumanAffordance::init(std::shared_ptr<DSR::DSRGraph> G,
                           std::uint64_t person_node_id,
                           std::string   person_node_name)
{
    G_                = std::move(G);
    person_node_id_   = person_node_id;
    person_node_name_ = std::move(person_node_name);
}

void HumanAffordance::update(const EpistemicProposal& prop)
{
    if (not G_) return;
    if (not prop.valid or not prop.is_finite())
    {
        if (prop.valid)
            qWarning() << "[affordance] rejecting non-finite epistemic proposal for"
                       << person_node_name_.c_str();
        return;
    }

    if (not node_created_)
        create_node(prop);
    else
        update_node(prop);
}

void HumanAffordance::remove()
{
    if (not G_ or not node_created_) return;
    G_->delete_node(affordance_node_id_);
    std::print("[affordance] removed node for '{}' (state={})\n",
               person_node_name_, state_name(state_));
    reset();
}

void HumanAffordance::on_node_modified(std::uint64_t id)
{
    if (not node_created_ or id != affordance_node_id_) return;
    auto node_opt = G_->get_node(affordance_node_id_);
    if (not node_opt.has_value()) return;

    const bool active  = G_->get_attrib_by_name<active_att>(node_opt.value()).value_or(false);
    const bool pending = G_->get_attrib_by_name<epistemic_pending_att>(node_opt.value()).value_or(true);

    if (active and pending)
    {
        if (state_ != State::executing)
        {
            state_ = State::executing;
            std::print("[affordance] '{}' → executing (controller claimed)\n", person_node_name_);
        }
        return;
    }
    if (not active and pending) { state_ = State::pending; return; }
    if (not active and not pending)
    {
        if (state_ != State::satisfied)
        {
            state_ = State::satisfied;
            std::print("[affordance] '{}' → satisfied (controller completed)\n", person_node_name_);
        }
        return;
    }
    if (state_ != State::aborted)
    {
        state_ = State::aborted;
        std::print("[affordance] '{}' → aborted (invalid protocol state active=1 pending=0)\n",
                   person_node_name_);
    }
}

void HumanAffordance::on_node_deleted(std::uint64_t id)
{
    if (not node_created_ or id != affordance_node_id_) return;
    std::print("[affordance] '{}' node deleted externally (state={}) → idle\n",
               person_node_name_, state_name(state_));
    reset();
}

std::string_view HumanAffordance::state_name(State s)
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

void HumanAffordance::create_node(const EpistemicProposal& prop)
{
    const std::string aff_name = "aff_" + person_node_name_;

    DSR::Node aff_node = DSR::Node::create<affordance_node_type>(aff_name);
    G_->add_or_modify_attrib_local<level_att>                   (aff_node, 4);
    G_->add_or_modify_attrib_local<parent_att>                  (aff_node, person_node_id_);
    G_->add_or_modify_attrib_local<pos_x_att>                   (aff_node, 300.f);
    G_->add_or_modify_attrib_local<pos_y_att>                   (aff_node, 200.f);
    G_->add_or_modify_attrib_local<active_att>                  (aff_node, false);
    G_->add_or_modify_attrib_local<epistemic_target_x_m_att>    (aff_node, prop.epistemic_target_x_m);
    G_->add_or_modify_attrib_local<epistemic_target_y_m_att>    (aff_node, prop.epistemic_target_y_m);
    G_->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(aff_node, prop.epistemic_target_yaw_rad);
    G_->add_or_modify_attrib_local<epistemic_gain_att>          (aff_node, prop.epistemic_gain);
    G_->add_or_modify_attrib_local<epistemic_pending_att>       (aff_node, true);

    // Execution contract (Servo lock-on + completion predicate + .still dwell) from the shared
    // type-level default for a person.
    rc::affordance::write_contract(*G_, aff_node, rc::affordance::default_contract_for("person"));

    rc::provenance::stamp_creation(*G_, aff_node);   // birth stamp: epoch ms + local ISO-8601
    const auto id_opt = G_->insert_node(aff_node);
    if (not id_opt.has_value())
    {
        qWarning() << "[affordance] failed to insert DSR node for" << aff_name.c_str();
        return;
    }
    affordance_node_id_ = id_opt.value();
    node_created_       = true;
    state_             = State::pending;

    refresh_edge();

    std::print("[affordance] created '{}' id={} target=({:.2f},{:.2f}) yaw={:.2f} gain={:.4f}\n",
               aff_name, affordance_node_id_,
               prop.epistemic_target_x_m, prop.epistemic_target_y_m,
               prop.epistemic_target_yaw_rad, prop.epistemic_gain);
}

void HumanAffordance::update_node(const EpistemicProposal& prop)
{
    auto node_opt = G_->get_node(affordance_node_id_);
    if (not node_opt.has_value())
    {
        std::print("[affordance] '{}' node missing, will recreate\n", person_node_name_);
        reset();
        return;
    }
    auto& n = node_opt.value();
    const bool active  = G_->get_attrib_by_name<active_att>(n).value_or(false);
    const bool pending = G_->get_attrib_by_name<epistemic_pending_att>(n).value_or(true);

    if (active and pending)
    {
        state_ = State::executing;
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

    if (not pending or active)
    {
        G_->add_or_modify_attrib_local<active_att>           (n, false);
        G_->add_or_modify_attrib_local<epistemic_pending_att>(n, true);
    }

    state_ = State::pending;
    G_->update_node(n);
    refresh_edge();
}

void HumanAffordance::refresh_edge()
{
    auto edge = DSR::Edge::create<has_intention_edge_type>(person_node_id_, affordance_node_id_);
    G_->insert_or_assign_edge(edge);
}

void HumanAffordance::reset()
{
    affordance_node_id_ = 0;
    node_created_       = false;
    state_             = State::idle;
}

}  // namespace rc
