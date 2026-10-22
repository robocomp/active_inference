#pragma once
/*
 * controller_approach_order.h — APPROACH ORDER: `approach <object> [from <side>]` (2026-10-18, plan
 * docs/superpowers/plans/2026-10-18-object-approach-affordance.md).
 *
 * Every object offers ONE pragmatic affordance, aff_<object>_approach (common/pragmatic_affordance/object_approach.h),
 * with the sides it can be approached from listed INSIDE it (aff_option_names, the default first). The controller
 * only activates what objects propose: this class names the node and the option the selector may take
 * (AffordanceManager::set_required_name(name, option)), and ends when the producer completes it. The object stays the
 * authority on where to stand and on which sides exist; nothing here computes a standpoint.
 *
 * A ONE-STEP ORDER. It is DoorPassage's skeleton without the door's rules (skip to a later step, back to `open` when
 * shut, the crossing corridor, the room change, the close deadline), kept apart from it on purpose: the door passage
 * is a live-critical path with its own test, and sharing a base class would change it for no gain to either.
 *
 * Any terminal outcome ends the order (Satisfied, or the consumer could not get there): the robot is held, not sent to
 * try for ever — the person asks again. A completion read BEFORE our own claim was executed is an earlier approach's
 * and is ignored. The node disappearing under the claim (the object retired) ends it too.
 *
 * AFTER: a single command HOLDS position (user, 2026-10-18): the caller leaves the base in Command mode with no order,
 * where it does not forage. A mission's step RETURNS to the mission, which runs its next step.
 *
 * Header-only; no Qt. update() from the control thread; start/stop may come from the GUI thread, hence the mutex.
 */

#include <cstdint>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <dsr/api/dsr_api.h>

#include "../../common/affordance_protocol/affordance_protocol.h"

namespace rc
{

class ApproachOrder
{
public:
    enum class After { Hold, Return };

    /// The affordance node an object's approach lives on.
    [[nodiscard]] static std::string node_name(const std::string& object) { return "aff_" + object + "_approach"; }

    /// Why `approach <object> [from <option>]` cannot be ordered now, or nullopt when it can: the object must offer
    /// its approach, and a named side must be one it lists.
    [[nodiscard]] static std::optional<std::string> check(DSR::DSRGraph& G, const std::string& object,
                                                          const std::string& option)
    {
        const auto n = G.get_node(node_name(object));
        if (not n.has_value())
            return std::format("'{}' offers no approach (no {} in the graph)", object, node_name(object));
        if (option.empty()) return std::nullopt;
        const std::string listed = G.get_attrib_by_name<aff_option_names_att>(n.value()).value_or(std::string{});
        std::size_t a = 0;
        while (a <= listed.size() and not listed.empty())
        {
            const auto b = std::min(listed.find(',', a), listed.size());
            if (listed.compare(a, b - a, option) == 0) return std::nullopt;
            a = b + 1;
        }
        return std::format("'{}' cannot be approached from '{}' now; it offers: {}", object, option,
                           listed.empty() ? std::string("its default side only") : listed);
    }

    void start(std::string object, std::string option, After after)
    {
        std::scoped_lock lk(m_);
        object_ = std::move(object);
        option_ = std::move(option);
        after_ = after;
        ++id_;
        active_ = true;
        seen_executing_ = false;
        finished_ = false;
        just_started_ = true;
        outcome_.clear();
        state_ = "not offered";
        last_event_ = std::format("[approach] {}{}: started", object_, option_.empty() ? "" : " from " + option_);
    }
    void stop()
    {
        std::scoped_lock lk(m_);
        if (active_) last_event_ = std::format("[approach] {}: stopped", object_);
        active_ = false;
    }
    [[nodiscard]] bool active() const
    {
        std::scoped_lock lk(m_);
        return active_;
    }
    /// True once when the order ends by itself (not on stop()): the caller applies `after`.
    [[nodiscard]] bool take_finished()
    {
        std::scoped_lock lk(m_);
        return std::exchange(finished_, false);
    }
    /// True once per start(): the caller releases whatever it was executing, so the claim moves over.
    [[nodiscard]] bool take_just_started()
    {
        std::scoped_lock lk(m_);
        return std::exchange(just_started_, false);
    }

    struct Out
    {
        std::string required;               ///< "" = no restriction (idle or done)
        std::string option;                 ///< the side asked for ("" = the object's default)
        std::optional<std::string> event;   ///< a one-line log of a transition this cycle
    };

    Out update(DSR::DSRGraph& G)
    {
        std::scoped_lock lk(m_);
        Out out;
        if (not active_) return out;
        const std::string name = node_name(object_);
        const auto n = G.get_node(name);
        if (not n.has_value())
        {
            if (seen_executing_)
            {
                outcome_ = "gone";
                return end(out, std::format("[approach] {}: its approach is gone (the object was removed) — order DONE",
                                            object_));
            }
            state_ = "not offered";
        }
        else
        {
            const bool a = G.get_attrib_by_name<active_att>(n.value()).value_or(false);
            const bool p = G.get_attrib_by_name<epistemic_pending_att>(n.value()).value_or(false);
            if (p and a)
            {
                seen_executing_ = true;
                state_ = "executing";
            }
            else if (p)
                state_ = "offered";
            else if (seen_executing_)   // completed — ours, since we saw it executing
            {
                outcome_ = std::string(rc::affordance::to_string(rc::affordance::read_outcome(n.value())));
                return end(out, std::format("[approach] {}{}: {} — order DONE, {}", object_,
                                            option_.empty() ? "" : " from " + option_,
                                            outcome_.empty() ? std::string("ended") : outcome_,
                                            after_ == After::Hold ? "holding position" : "back to the mission"));
            }
            else
                state_ = "completed (an earlier one)";
        }
        out.required = name;
        out.option = option_;
        return out;
    }

    struct View
    {
        std::string object;    ///< "" = never started
        std::string option;
        std::uint64_t id = 0;  ///< which order (one per start), so a report can tell "superseded" from "done"
        After after = After::Hold;
        bool active = false;
        std::string state;     ///< of the node: not offered / offered / executing / …
        std::string outcome;   ///< how the last order ended ("" while running)
        std::string last_event;
    };
    [[nodiscard]] View view() const
    {
        std::scoped_lock lk(m_);
        return {.object = object_, .option = option_, .id = id_, .after = after_, .active = active_, .state = state_,
                .outcome = outcome_, .last_event = last_event_};
    }

private:
    Out& end(Out& out, std::string event)
    {
        active_ = false;
        finished_ = true;
        last_event_ = event;
        out.event = std::move(event);
        return out;
    }

    mutable std::mutex m_;
    std::string object_, option_;
    After after_ = After::Hold;
    std::uint64_t id_ = 0;
    bool active_ = false;
    bool seen_executing_ = false;
    bool finished_ = false;
    bool just_started_ = false;
    std::string state_, outcome_, last_event_;
};

}   // namespace rc
