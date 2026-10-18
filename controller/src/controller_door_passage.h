#pragma once
/*
 * controller_door_passage.h — DOOR PASSAGE: take the robot through one door by sequencing door_concept's three
 * pragmatic affordances (2026-10-11, plan room_concept/docs/superpowers/plans/2026-10-11-cross-door-into-adjacent-room.md).
 *
 *   aff_<door>_approach  →  aff_<door>_open  →  aff_<door>_cross  [→  aff_<door>_close]
 *
 * CLOSE BEHIND (2026-10-18, user): a passage asked for by a single command (chat, the panel, remote_cmd.txt) goes on,
 * once in the new room, to door_concept's `close` — offered while the door is open and the robot clear of its swing —
 * and then hands the base back to the affordances (take_finished() + then_affordances: the caller sets
 * AffordancesOnly). A mission's `cross` stops after Cross as before: its next step owns the base. Where no `close` is
 * ever offered (no door actuator in this world) nothing in the graph tells "not yet" from "never", so the step has a
 * deadline (set_close_wait_ms), stated in the log, like every contract's own timeout_s; a failed close ends it too —
 * leaving a door open is not worth holding the robot for.
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
#include <chrono>
#include <cmath>
#include <functional>
#include <format>
#include <limits>
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
#include "../../common/world_query/door_query.h"
#include "door_passage_corridor.h"

namespace rc
{

class DoorPassage
{
public:
    enum class Step { Idle, Approach, Open, Cross, Close, Done };

    /// Begin a passage through `door` (the door node's name, e.g. "door_1"). `stop_after` = the last step to run:
    /// Cross (the whole passage) or Open (a mission's `open door`: the passage is Done once the door is open, and
    /// a `cross` on offer — door_concept believing it passable — counts as open already).
    void start(std::string door, Step stop_after = Step::Cross, bool then_affordances = false)
    {
        std::scoped_lock lk(m_);
        door_ = std::move(door);
        stop_after_ = stop_after;
        then_affordances_ = then_affordances;
        finished_ = false;
        last_outcome_.clear();
        step_ = Step::Approach;
        seen_executing_ = false;
        attempts_ = 0;
        cross_room_.clear();
        just_started_ = true;
        preempted_ = false;
        reached_door_ = false;
        opened_ = false;
        note_ = "started";
        step_state_.clear();
        cross_unreachable_ = 0;
        failed_ = false;
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
        return step_ == Step::Approach or step_ == Step::Open or step_ == Step::Cross or step_ == Step::Close;
    }
    /// True once when the passage reaches Done (not on stop()): the caller hands the base back if then_affordances.
    [[nodiscard]] bool take_finished()
    {
        std::scoped_lock lk(m_);
        return std::exchange(finished_, false);
    }
    /// How long the Close step waits for door_concept to OFFER `close` (see the header comment).
    void set_close_wait_ms(std::uint64_t ms)
    {
        std::scoped_lock lk(m_);
        close_wait_ms_ = ms;
    }
    /// Steady-clock milliseconds, injectable for tests.
    void set_clock(std::function<std::uint64_t()> clock)
    {
        std::scoped_lock lk(m_);
        clock_ = std::move(clock);
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
            const bool open_wanted = later == Step::Cross and not opened_ and state_of(G, name_of(Step::Open)) == State::Offered;
            // ★ NEVER FROM APPROACH STRAIGHT TO CROSS (2026-10-12). From across the room door_concept can rate a SHUT
            //   door passable (try20/21/26: P(passable)=1.00 at phi=0.44 rad, clear span 0.10 m); `cross` then fails
            //   unreachable, stays offered, and the robot never goes to the door. Approach is cheap and ends within
            //   reach, facing the door — where `open` vs `cross` is decided on the beliefs that matter.
            if (later == Step::Cross and step_ == Step::Approach) continue;
            if (state_of(G, name_of(later)) == State::Offered and not open_wanted)
            {
                const bool was_executing = st_cur == State::Executing;
                out.event = std::format("[door-passage] {}: {} is offered — {} to it", door_, name_of(later),
                                        was_executing ? std::format("pre-empting {} and continuing", name_of(step_))
                                                      : std::string("skipping"));
                if (was_executing) preempted_ = true;
                if (step_ == Step::Approach) reached_door_ = true;   // `open`/`cross` offered ⇒ the door is within reach
                step_ = later;
                seen_executing_ = false;
                break;
            }
        }

        // ★ A STEP THAT IS NO LONGER ON OFFER FALLS BACK to the latest earlier one that is. door_concept withdraws
        //   `cross` when the door is believed shut and `open` when out of reach; `approach` is always offered for a
        //   located door. Without this the passage waited for a withdrawn step for ever (live try21: `cross` taken on
        //   a transient P(passable) of the SHUT door from across the room, then withdrawn — robot parked, door shut).
        // …but NOT once the robot has reached the door (2026-10-13, try36/37/40): a `cross` withdrawn for a few cycles by
        // a flickering P(passable) sent the robot back to `approach` while it stood AT the door, inside the band the
        // approach servo needs, and the servo gave up for ever. At the door the passage waits for `cross` instead.
        if ((step_ == Step::Cross or step_ == Step::Open) and not reached_door_)
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
        // Not once `open` has succeeded in this passage: a re-offer of `open` after that is the flicker of a door we just
        // opened (try38: 223 open/cross alternations), not news that it shut.
        if (step_ == Step::Cross and not opened_ and cross_room_ == room_frame
            and state_of(G, name_of(Step::Open)) == State::Offered)
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
            if (static_cast<int>(stop_after_) >= static_cast<int>(Step::Close))
            {
                step_ = Step::Close;
                close_since_ms_ = now_ms();
                seen_executing_ = false;
                attempts_ = 0;
                out.event = std::format("[door-passage] {}: the room changed {} -> {} while crossing — now closing the "
                                        "door behind", door_, cross_room_, room_frame);
            }
            else
            {
                finish();
                out.event = std::format("[door-passage] {}: the room changed {} -> {} while crossing — passage DONE",
                                        door_, cross_room_, room_frame);
                last_event_ = *out.event;
                return out;
            }
        }
        // ★ CLOSE: offered → claimed → completed by door_concept once the door is perceived shut. Never offered within
        //   close_wait_ms_ (no actuator), or ended without success: leave it open and finish — see the header.
        if (step_ == Step::Close)
        {
            const State st = state_of(G, name_of(Step::Close));
            if (st == State::Executing) seen_executing_ = true;
            std::optional<std::string> why_done;
            if (seen_executing_ and st == State::Completed)
            {
                const auto node = G.get_node(name_of(Step::Close));
                const auto outcome = node.has_value() ? rc::affordance::read_outcome(node.value())
                                                      : rc::affordance::Outcome::None;
                why_done = outcome == rc::affordance::Outcome::Satisfied
                               ? std::format("{} SATISFIED — door closed behind, passage DONE", name_of(Step::Close))
                               : std::format("{} ended {} — door left open, passage DONE", name_of(Step::Close),
                                             rc::affordance::to_string(outcome));
            }
            else if (not seen_executing_ and st != State::Offered and now_ms() - close_since_ms_ >= close_wait_ms_)
                why_done = std::format("{} not offered within {:.1f} s (no door actuator?) — door left open, passage "
                                       "DONE", name_of(Step::Close), double(close_wait_ms_) / 1000.0);
            if (why_done.has_value())
            {
                finish();
                out.event = std::format("[door-passage] {}: {}", door_, *why_done);
                last_event_ = *out.event;
                return out;
            }
            step_state_ = state_name(st);
            note_ = std::format("{} ({})", name_of(step_), step_state_);
            if (out.event.has_value()) last_event_ = *out.event;
            out.required = name_of(Step::Close);
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
                if (done == Step::Approach or done == Step::Open) reached_door_ = true;
                if (done == Step::Open) opened_ = true;
                step_ = done == Step::Approach ? Step::Open : done == Step::Open ? Step::Cross : Step::Done;
                if (step_ == Step::Done) finished_ = true;
                attempts_ = 0;
                out.event = std::format("[door-passage] {}: {} SATISFIED -> {}", door_, req,
                                        step_ == Step::Done ? std::string("passage DONE") : name_of(step_));
            }
            else
            {
                ++attempts_;
                last_outcome_ = rc::affordance::to_string(outcome);
                out.event = std::format("[door-passage] {}: {} ended {} (attempt {}) — waiting for it to be offered again",
                                        door_, req, rc::affordance::to_string(outcome), attempts_);
                // ⚑ Task 9 fix round 1 — a FLAGGED decision level (controller ruling): kMaxCrossUnreachable Unreachable crosses
                //   per passage, then the passage reports failure. Every attempt now waits on the anchor up to the cross
                //   contract's timeout, so without a cap a door the robot cannot get through held it for ever.
                if (step_ == Step::Cross and outcome == rc::affordance::Outcome::Unreachable
                    and ++cross_unreachable_ >= kMaxCrossUnreachable)
                {
                    finish();
                    failed_ = true;
                    last_outcome_ = "failed";
                    out.event = std::format("[door-passage] {}: {} ended unreachable {} times in this passage — passage FAILED "
                                            "(the cap kMaxCrossUnreachable)", door_, req, cross_unreachable_);
                    last_event_ = *out.event;
                    return out;
                }
            }
        }
        if (step_ != Step::Done and static_cast<int>(step_) > static_cast<int>(stop_after_))
        {
            finish();
            out.event = std::format("[door-passage] {}: {} reached — passage DONE (it was asked to stop after {})", door_,
                                    step_name(stop_after_), step_name(stop_after_));
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

    /// What the controller window shows (its status line, controller_status_line.h). A copy, so the GUI never holds the lock.
    struct View
    {
        std::string door;          ///< "" = never started
        Step step = Step::Idle;
        std::string step_state;    ///< the current step's affordance: offered / executing / completed / not offered
        int attempts = 0;          ///< failed completions of the current step
        std::string last_event;    ///< the latest [door-passage] transition, verbatim
        std::string last_outcome;  ///< aff_outcome of the latest failed completion ("" = none in this passage)
        bool then_affordances = false;   ///< started by a single command: hand back to the affordances when Done
        bool failed = false;             ///< Done by the Unreachable-cross cap (kMaxCrossUnreachable), not by crossing
    };
    [[nodiscard]] View view() const
    {
        std::scoped_lock lk(m_);
        return {.door = door_, .step = step_, .step_state = step_state_, .attempts = attempts_, .last_event = last_event_,
                .last_outcome = last_outcome_, .then_affordances = then_affordances_, .failed = failed_};
    }
    [[nodiscard]] static const char* step_name(Step s)
    {
        switch (s)
        {
            case Step::Idle: return "idle";
            case Step::Approach: return "approach";
            case Step::Open: return "open";
            case Step::Cross: return "cross";
            case Step::Close: return "close";
            case Step::Done: return "done";
        }
        return "?";
    }
    /// The doors a passage can be started through, and the one "cross door" means. Both live in the shared
    /// common/world_query/door_query.h (2026-10-12) so chat_agent validates a request with the same predicate the
    /// controller executes it with; these wrappers keep every existing call site unchanged.
    [[nodiscard]] static std::vector<std::string> list_doors(DSR::DSRGraph& G) { return rc::world::list_doors(G); }
    [[nodiscard]] static std::optional<std::string> nearest_door(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner,
                                                                 const std::string& room, const std::string& robot)
    { return rc::world::nearest_door(G, inner, room, robot); }

private:
    enum class State { Missing, Offered, Executing, Completed };

    [[nodiscard]] std::string name_of(Step s) const
    {
        const char* kind = s == Step::Approach ? "approach" : s == Step::Open ? "open" : s == Step::Close ? "close" : "cross";
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

    void finish()   // under m_
    {
        step_ = Step::Done;
        note_ = "done";
        finished_ = true;
    }
    [[nodiscard]] std::uint64_t now_ms() const   // under m_
    {
        if (clock_) return clock_();
        return std::uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    mutable std::mutex m_;
    std::string door_;
    bool then_affordances_ = false;   ///< a single command's passage: hand the base back to the affordances when Done
    bool finished_ = false;           ///< reached Done since the last take_finished()
    std::uint64_t close_wait_ms_ = 5000;
    std::uint64_t close_since_ms_ = 0;
    std::function<std::uint64_t()> clock_;
    Step step_ = Step::Idle;
    Step stop_after_ = Step::Cross;
    bool seen_executing_ = false;
    bool just_started_ = false;
    bool preempted_ = false;
    bool reached_door_ = false;   ///< approach satisfied (or a later step taken from it) in this passage
    bool opened_ = false;         ///< `open` satisfied in this passage
    int attempts_ = 0;
    std::string cross_room_;   // the room the cross step began in
    std::string note_ = "idle";
    std::string step_state_;   // view()
    std::string last_event_;   // view()
    std::string last_outcome_; // view()
    int  cross_unreachable_ = 0;   // Unreachable cross completions in this passage (kMaxCrossUnreachable)
    bool failed_ = false;          // finished by the cap, not by crossing
public:
    static constexpr int kMaxCrossUnreachable = 3;   // ⚑ flagged decision level (Task 9 fix round 1)
};

}   // namespace rc
