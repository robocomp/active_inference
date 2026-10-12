#pragma once
/*
 * controller_door_passage.h — DOOR PASSAGE: take the robot through one door by sequencing door_concept's three
 * pragmatic affordances (2026-10-11, plan room_concept/docs/superpowers/plans/2026-10-11-cross-door-into-adjacent-room.md).
 *
 *   aff_<door>_approach  →  aff_<door>_open  →  aff_<door>_cross
 *
 * The controller already knows how to execute ONE affordance: claim it, plan to its standpoint, hold it until the
 * producer completes it. What it lacked is a reason to pick these three, in this order, over the explorer's
 * afford_room. This class supplies only that: the name of the affordance the selector may take now
 * (AffordanceManager::set_required_name), advanced when the producer completes the current one with Satisfied.
 * door_concept stays the authority on each step's precondition — "open" is offered when the door is believed shut
 * and within reach, "cross" when it is believed passable — so nothing here judges the door.
 *
 * A LATER step that is offered first wins: a door already open is never offered "open", and the robot may already
 * stand where "approach" would take it. The sequence therefore jumps to the latest offered step whenever the current
 * one is not being executed.
 *
 * THE CROSS STEP NEEDS SPACE THE ROOM DOES NOT HAVE. Its standpoint lies past the doorway, outside the room polygon
 * the planner treats as the world, and the door's own node is a box across the opening. While "cross" is required,
 * passage() returns a corridor through the doorway (door centre → cross standpoint, door-wide) that the grid, the
 * MPPI boundary and the outside-room check accept as free, and exempt() names the door node so it is not an obstacle.
 * Both end when the step ends. The open leaf still reaches the planner as residual LiDAR cells.
 *
 * Header-only; no Qt. Driven from the control thread (update() before the selector runs); start/stop may come from
 * the GUI thread, hence the mutex.
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "../../common/affordance_protocol/affordance_protocol.h"
#include "door_passage_corridor.h"

namespace rc
{

class DoorPassage
{
public:
    enum class Step { Idle, Approach, Open, Cross, Done };

    /// Begin a passage through `door` (the door node's name, e.g. "door_1").
    void start(std::string door)
    {
        std::scoped_lock lk(m_);
        door_ = std::move(door);
        step_ = Step::Approach;
        seen_executing_ = false;
        attempts_ = 0;
        cross_room_.clear();
        just_started_ = true;
        preempted_ = false;
        note_ = "started";
        step_state_.clear();
        last_event_ = std::format("[door-passage] {}: started", door_);
    }
    void stop()
    {
        std::scoped_lock lk(m_);
        step_ = Step::Idle;
        note_ = "stopped";
        last_event_ = std::format("[door-passage] {}: stopped", door_);
    }
    [[nodiscard]] bool active() const
    {
        std::scoped_lock lk(m_);
        return step_ == Step::Approach or step_ == Step::Open or step_ == Step::Cross;
    }
    /// True once per start(): the caller releases whatever affordance it was executing, so the claim moves over.
    [[nodiscard]] bool take_just_started()
    {
        std::scoped_lock lk(m_);
        return std::exchange(just_started_, false);
    }

    /// True once per pre-emption: the step being executed was superseded by a later one on offer. The caller
    /// releases its claim and drops the lock-on/orient it was running, so the new step is selected next cycle.
    [[nodiscard]] bool take_preempted()
    {
        std::scoped_lock lk(m_);
        return std::exchange(preempted_, false);
    }

    struct Out
    {
        std::string required;                               ///< "" = no restriction (passage idle or done)
        std::vector<std::vector<Eigen::Vector2f>> passages; ///< free corridor while crossing (room frame)
        std::unordered_set<std::string> exempt;             ///< node names that are not obstacles now
        std::optional<std::string> event;                   ///< a one-line log of a transition this cycle
    };

    /// One control cycle. `room_frame` = the controller's current room node name (the frame the affordances and
    /// the planner live in).
    Out update(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner, const std::string& room_frame)
    {
        std::scoped_lock lk(m_);
        Out out;
        if (step_ == Step::Idle or step_ == Step::Done) return out;

        // Jump to the latest step its producer is offering — EVEN WHILE THE CURRENT ONE IS BEING EXECUTED
        // (2026-10-12). A later step on offer is door_concept saying the current step's purpose is already met:
        // `open` is offered once P(in reach)·P(shut) clears its offer band, `cross` once P(passable) does. Waiting
        // for the current step's own, stricter completion added nothing but standing still — live try16: the
        // approach servo held the robot ~20 s while P(reach) crept 0.70→0.94, with `open` already on offer.
        // The caller releases the dropped claim (take_preempted()), so the producer sees it Offered again.
        const auto st_cur = state_of(G, name_of(step_));
        for (Step later : {Step::Cross, Step::Open})
        {
            if (static_cast<int>(later) <= static_cast<int>(step_)) break;
            // A later step skips an earlier one only while the earlier one is NOT on offer: `cross` over a door
            // door_concept also offers to `open` is two contradictory beliefs, and the shut door wins (live try20:
            // a transient P(passable) on a SHUT door sent the robot to cross it, `open` was offered a moment later).
            const bool open_wanted = later == Step::Cross and state_of(G, name_of(Step::Open)) == State::Offered;
            if (state_of(G, name_of(later)) == State::Offered and not open_wanted)
            {
                const bool was_executing = st_cur == State::Executing;
                out.event = std::format("[door-passage] {}: {} is offered — {} to it", door_, name_of(later),
                                        was_executing ? std::format("pre-empting {} and continuing", name_of(step_))
                                                      : std::string("skipping"));
                if (was_executing) preempted_ = true;
                step_ = later;
                seen_executing_ = false;
                break;
            }
        }

        // ★ A STEP THAT IS NO LONGER ON OFFER FALLS BACK to the latest earlier one that is. door_concept withdraws
        //   `cross` when the door is believed shut and `open` when out of reach; `approach` is always offered for a
        //   located door. Without this the passage waited for a withdrawn step for ever (live try21: `cross` taken on
        //   a transient P(passable) of the SHUT door from across the room, then withdrawn — robot parked, door shut).
        if (step_ == Step::Cross or step_ == Step::Open)
        {
            const State st_now = state_of(G, name_of(step_));
            if (st_now != State::Offered and st_now != State::Executing and not seen_executing_)
                for (Step earlier : {Step::Open, Step::Approach})
                {
                    if (static_cast<int>(earlier) >= static_cast<int>(step_)) continue;
                    if (state_of(G, name_of(earlier)) == State::Offered)
                    {
                        out.event = std::format("[door-passage] {}: {} is not offered — back to {}", door_,
                                                name_of(step_), name_of(earlier));
                        step_ = earlier;
                        break;
                    }
                }
        }

        // ★ BACK TO `open` WHEN THE DOOR TURNS OUT SHUT. `cross` may have been taken on a transient P(passable) while
        //   the robot was still out of reach (so `open` could not be offered yet). Once door_concept offers `open` —
        //   P(in reach)·P(shut) above its band — the door is shut in front of us: open it, whatever `cross` is doing.
        if (step_ == Step::Cross and cross_room_ == room_frame and state_of(G, name_of(Step::Open)) == State::Offered)
        {
            const bool was_executing = state_of(G, name_of(Step::Cross)) == State::Executing;
            out.event = std::format("[door-passage] {}: {} is offered — the door is shut: back from {} to it", door_,
                                    name_of(Step::Open), name_of(Step::Cross));
            if (was_executing) preempted_ = true;
            step_ = Step::Open;
            seen_executing_ = false;
        }

        // ★ THE CROSSING IS ROOM_CONCEPT'S TO DECLARE. Its crossing evidence (body wholly past the doorway line)
        //   births the new room and moves the robot's RT there, and the controller's room follows (room_resolve
        //   rule 2a) — usually before door_concept completes "cross" (0.8 m past the plane). From then the cross
        //   standpoint is a pose in the room just left, read in the wrong frame: the passage is over.
        if (step_ == Step::Cross and not cross_room_.empty() and room_frame != cross_room_)
        {
            step_ = Step::Done;
            note_ = "done";
            out.event = std::format("[door-passage] {}: the room changed {} -> {} while crossing — passage DONE",
                                    door_, cross_room_, room_frame);
            last_event_ = *out.event;
            return out;
        }
        if (step_ == Step::Cross and cross_room_.empty()) cross_room_ = room_frame;
        if (step_ != Step::Cross) cross_room_.clear();

        const std::string req = name_of(step_);
        const auto node = G.get_node(req);
        const State st = state_of(G, req);
        if (st == State::Executing) seen_executing_ = true;
        if (seen_executing_ and st == State::Completed and node.has_value())
        {
            const auto outcome = rc::affordance::read_outcome(node.value());
            seen_executing_ = false;
            if (outcome == rc::affordance::Outcome::Satisfied)
            {
                const Step done = step_;
                step_ = done == Step::Approach ? Step::Open : done == Step::Open ? Step::Cross : Step::Done;
                attempts_ = 0;
                out.event = std::format("[door-passage] {}: {} SATISFIED -> {}", door_, req,
                                        step_ == Step::Done ? std::string("passage DONE") : name_of(step_));
            }
            else
            {
                ++attempts_;
                out.event = std::format("[door-passage] {}: {} ended {} (attempt {}) — waiting for it to be offered again",
                                        door_, req, rc::affordance::to_string(outcome), attempts_);
            }
        }
        step_state_ = state_name(state_of(G, name_of(step_)));
        note_ = std::format("{} ({})", name_of(step_), step_state_);
        if (out.event.has_value()) last_event_ = *out.event;
        if (step_ == Step::Done) { note_ = "done"; return out; }

        out.required = name_of(step_);
        if (step_ == Step::Cross)
        {
            out.exempt = {door_};
            if (auto c = corridor(G, inner, room_frame); c.has_value()) out.passages.push_back(std::move(*c));
        }
        return out;
    }

    [[nodiscard]] std::string status() const
    {
        std::scoped_lock lk(m_);
        return door_.empty() ? std::string("idle") : door_ + ": " + note_;
    }

    /// What the controller window shows (controller_door_passage_bar.h). A copy, so the GUI never holds the lock.
    struct View
    {
        std::string door;          ///< "" = never started
        Step step = Step::Idle;
        std::string step_state;    ///< the current step's affordance: offered / executing / completed / not offered
        int attempts = 0;          ///< failed completions of the current step
        std::string last_event;    ///< the latest [door-passage] transition, verbatim
    };
    [[nodiscard]] View view() const
    {
        std::scoped_lock lk(m_);
        return {.door = door_, .step = step_, .step_state = step_state_, .attempts = attempts_, .last_event = last_event_};
    }
    [[nodiscard]] static const char* step_name(Step s)
    {
        switch (s)
        {
            case Step::Idle: return "idle";
            case Step::Approach: return "approach";
            case Step::Open: return "open";
            case Step::Cross: return "cross";
            case Step::Done: return "done";
        }
        return "?";
    }
    /// The doors a passage can be started through: door_concept's beliefs, type "object" named door_* (its own sweep's
    /// predicate, door_concept specificworker.cpp). Read from the control thread, which owns graph access here.
    [[nodiscard]] static std::vector<std::string> list_doors(DSR::DSRGraph& G)
    {
        std::vector<std::string> names;
        for (const auto& n : G.get_nodes_by_type("object"))
            if (n.name().starts_with("door")) names.push_back(n.name());
        std::ranges::sort(names);
        return names;
    }

private:
    enum class State { Missing, Offered, Executing, Completed };

    [[nodiscard]] std::string name_of(Step s) const
    {
        const char* kind = s == Step::Approach ? "approach" : s == Step::Open ? "open" : "cross";
        return "aff_" + door_ + "_" + kind;
    }
    static const char* state_name(State s)
    {
        return s == State::Offered ? "offered" : s == State::Executing ? "executing"
             : s == State::Completed ? "completed" : "not offered";
    }
    // The affordance protocol's two flags (affordance_manager.h decode_protocol_state): pending and not active =
    // Offered, both = Executing, neither = Completed.
    static State state_of(DSR::DSRGraph& G, const std::string& name)
    {
        const auto n = G.get_node(name);
        if (not n.has_value()) return State::Missing;
        const bool a = G.get_attrib_by_name<active_att>(n.value()).value_or(false);
        const bool p = G.get_attrib_by_name<epistemic_pending_att>(n.value()).value_or(false);
        if (p and not a) return State::Offered;
        if (p and a) return State::Executing;
        if (not p and not a) return State::Completed;
        return State::Missing;
    }

    /// The doorway as a rectangle in the room frame: along the cross standpoint's heading (door_concept's through-pose,
    /// door_pragmatics.h: c - side·standoff·n, heading = the travel direction, +x at 0), door-wide, from well inside the
    /// room to one body length past the standpoint. ★ NOT FROM THE DOOR NODE: its pose is the LEAF's, and an open leaf
    /// has swung beside the doorway — a corridor aimed from it ran diagonally into the wall (live 2026-10-11: no route,
    /// 72k expansions). nullopt while the standpoint is unknown (no corridor = the room alone, the safe failure).
    std::optional<std::vector<Eigen::Vector2f>> corridor(DSR::DSRGraph& G, DSR::InnerEigenAPI*, const std::string&) const
    {
        const auto cross = G.get_node(name_of(Step::Cross));
        const auto door  = G.get_node(door_);
        if (not cross.has_value() or not door.has_value()) return std::nullopt;
        const auto tx  = G.get_attrib_by_name<epistemic_target_x_m_att>(cross.value());
        const auto ty  = G.get_attrib_by_name<epistemic_target_y_m_att>(cross.value());
        const auto yaw = G.get_attrib_by_name<epistemic_target_yaw_rad_att>(cross.value());
        if (not tx.has_value() or not ty.has_value() or not yaw.has_value()) return std::nullopt;
        const float w = G.get_attrib_by_name<width_m_att>(door.value()).value_or(1.f);
        // Back from the standpoint through the aperture (door_concept's CrossStandoffM, 1.2 m) and one more metre into
        // the room, which the room polygon already holds — the two regions overlap rather than touch; forward one
        // body length so the robot fits wholly inside at the standpoint. Geometry in door_passage_corridor.h.
        return door_passage_corridor({tx.value(), ty.value()}, yaw.value(), w, 2.2f, 0.8f);
    }

    mutable std::mutex m_;
    std::string door_;
    Step step_ = Step::Idle;
    bool seen_executing_ = false;
    bool just_started_ = false;
    bool preempted_ = false;
    int attempts_ = 0;
    std::string cross_room_;   // the room the cross step began in
    std::string note_ = "idle";
    std::string step_state_;   // view()
    std::string last_event_;   // view()
};

}   // namespace rc
