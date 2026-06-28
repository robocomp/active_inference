/*
 * human_affordance.h
 *
 * Lifecycle of an "affordance" DSR node hanging from a person node via a "has_intention" edge. It
 * advertises the robot pose that maximises epistemic value for this person — the reduce-occlusion
 * viewpoint from EpistemicPlanner — and tracks the controller's progress through a small state
 * machine. Object-agnostic mechanics mirror bottle_affordance.h / table_affordance.h; only the parent
 * node type differs (person).
 */

#pragma once

#include <cstdint>
#include <memory>
#include <print>
#include <string>

#include <dsr/api/dsr_api.h>

#include "epistemic_planner.h"

namespace rc {

class HumanAffordance
{
public:
    enum class State { idle, pending, executing, satisfied, aborted };

    HumanAffordance() = default;

    void init(std::shared_ptr<DSR::DSRGraph> G,
              std::uint64_t person_node_id,
              std::string   person_node_name);

    void update(const EpistemicProposal& prop);
    void remove();

    void on_node_modified(std::uint64_t id);
    void on_node_deleted(std::uint64_t id);

    State         state()     const { return state_; }
    bool          is_active() const { return state_ != State::idle; }
    std::uint64_t node_id()   const { return affordance_node_id_; }

    static std::string_view state_name(State s);

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    std::uint64_t person_node_id_ = 0;
    std::string   person_node_name_;

    std::uint64_t affordance_node_id_ = 0;
    bool          node_created_       = false;
    State         state_              = State::idle;

    void create_node(const EpistemicProposal& prop);
    void update_node(const EpistemicProposal& prop);
    void refresh_edge();
    void reset();
};

}  // namespace rc
