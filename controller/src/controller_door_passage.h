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
#include <print>
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
#include "controller_anchor_servo.h"   // rc::anchor_cross_end (Task 12 Fix E)

namespace rc
{

class DoorPassage
{
public:
    // Look (Live fix 3): the epistemic look at a REMEMBERED door that is not in the graph (door_concept's
    // `aff_door_look_<door>`, door_look.h) — the passage's first step until the door is re-born.
    enum class Step { Idle, Look, Approach, Open, Cross, Close, Done };

    /// Begin a passage through `door` (the door node's name, e.g. "door_1"). `stop_after` = the last step to run:
    /// Cross (the whole passage) or Open (a mission's `open door`: the passage is Done once the door is open, and
    /// a `cross` on offer — door_concept believing it passable — counts as open already).
    void start(std::string door, Step stop_after = Step::Cross, bool then_affordances = false, bool look_first = false)
    {
        std::scoped_lock lk(m_);
        door_ = std::move(door);
        stop_after_ = stop_after;
        then_affordances_ = then_affordances;
        finished_ = false;
        last_outcome_.clear();
        step_ = look_first ? Step::Look : Step::Approach;
        seen_executing_ = false;
        attempts_ = 0;
        cross_room_.clear();
        just_started_ = true;
        look_chosen_ = false;
        preempted_ = false;
        reached_door_ = false;
        opened_ = false;
        note_ = "started";
        step_state_.clear();
        cross_unreachable_ = 0;
        failed_ = false;
        clearing_ = false;
        anchored_ = false;
        through_ = false;
        p_through_ = -1.0;
        target_ignored_logged_.clear();
        last_open_sp_.reset();
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
        return step_ == Step::Look or step_ == Step::Approach or step_ == Step::Open or step_ == Step::Cross or step_ == Step::Close;
    }
    /// Live fix 3: is the epistemic look at a remembered-but-absent `door` on offer (or being executed)? Then a passage can
    /// start with it (start(..., look_first = true)) instead of failing "not in the graph".
    [[nodiscard]] static bool look_available(DSR::DSRGraph& G, const std::string& door)
    {
        const State st = state_of(G, look_name(door));
        return st == State::Offered or st == State::Executing;
    }
    [[nodiscard]] static std::string look_name(const std::string& door) { return "aff_door_look_" + door; }
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
    /// The price of distance in the passage's choice between its step and the door's epistemic look: the SAME λ (nats per
    /// metre) and commitment margin (nats) the affordance selector uses (Controller.AffordanceLambdaCost / SwitchMargin).
    void set_efe(float lambda_cost, float switch_margin)
    {
        std::scoped_lock lk(m_);
        efe_lambda_ = std::max(0.0f, lambda_cost);
        efe_margin_ = std::max(0.0f, switch_margin);
    }
    /// Gate for the look-vs-step choice below: the [Policy] chain's (Policy.Enabled). The old chain keeps its old behaviour.
    void set_look_enabled(bool on)
    {
        std::scoped_lock lk(m_);
        look_enabled_ = on;
    }
    /// ★Task E — THE PASSAGE'S CHOICE BETWEEN ITS GOAL-DIRECTED STEP AND AN EPISTEMIC LOOK, on ONE scale (nats, the selector's
    /// score  −G = value − λ·distance). Each side's `value` is what its producer publishes on the node's epistemic_gain: the
    /// look's is the expected information gain on the door state (D-optimal on [s, w, h], detection-weighted), the step's is
    /// its preference in nats times the probability it can be completed under the door belief's CURRENT σ — so a poor door
    /// estimate lowers the step's value and raises the look's, and the choice falls out of the comparison: no "if degraded
    /// go to…" rule. Doing nothing is a policy with score 0, which both must beat; the incumbent (whichever was chosen last
    /// cycle) carries the selector's switch margin so the choice does not flap. nullopt value = not on offer.
    struct Choice { bool look = false; float step_score = 0.f, look_score = 0.f; };
    [[nodiscard]] static Choice choose_look_or_step(std::optional<float> step_value, float step_dist,
                                                    std::optional<float> look_value, float look_dist,
                                                    float lambda, float margin, bool look_is_incumbent)
    {
        Choice c;
        constexpr float kStay = 0.0f;   // doing nothing: G = 0
        c.step_score = step_value.has_value() ? *step_value - lambda * step_dist + (look_is_incumbent ? 0.f : margin) : kStay;
        c.look_score = look_value.has_value() and *look_value > 0.f
                           ? *look_value - lambda * look_dist + (look_is_incumbent ? margin : 0.f) : kStay;
        c.look = c.look_score > std::max(c.step_score, kStay);
        return c;
    }
    /// Steady-clock milliseconds, injectable for tests.
    void set_clock(std::function<std::uint64_t()> clock)
    {
        std::scoped_lock lk(m_);
        clock_ = std::move(clock);
    }
    /// Task 12 Fix E — the session's word on the cross, each cycle before update(): is it being driven on an interaction
    /// anchor (engaged, not lost), and has the servo reached `through` (s ≥ AnchorCrossStandoffM)?
    /// Live fix 1 (review): the robot's position (room frame of the affordances; nullopt = unknown) and the arrival tolerance
    /// the approach completes with — the EVIDENCE for falling back to `approach` once at the door (see update()).
    void note_robot(std::optional<Eigen::Vector2f> pos, float arrival_tol_m)
    {
        std::scoped_lock lk(m_);
        robot_ = pos;
        arrival_tol_ = arrival_tol_m;
    }
    /// Live fix 4 ruling 4: the session handed the cross back because the anchor stays unpublished from where the robot
    /// stands — go back to `approach` (door_concept's standpoint) to change the viewpoint. The cross's late Unreachable
    /// completion lands on a step the passage has left, so it costs no attempt.
    void reposition(const std::string& why)
    {
        std::scoped_lock lk(m_);
        if (step_ != Step::Cross and step_ != Step::Open) return;
        last_event_ = std::format("[door-passage] {}: {} — back to {} to change the viewpoint", door_, why, name_of(Step::Approach));
        std::println("{}", last_event_);
        step_ = Step::Approach;
        reached_door_ = false;
        seen_executing_ = false;
    }
    void note_anchor(bool engaged, bool through)
    {
        std::scoped_lock lk(m_);
        anchored_ = engaged;
        through_ = through;
    }
    /// Task 3f r1 (M-1): the session's posterior P(through) while clearing ([Policy]; never set by the old chain). A clearing
    /// that ends without `through` and with this ≤ ½ may leave the body straddling the doorway: the passage does not close.
    void note_through_p(double p)
    {
        std::scoped_lock lk(m_);
        p_through_ = p;
    }
    /// True while the cross goes on past a room switch on the anchor: the session drives the servo on the latched anchor and
    /// selects nothing.
    [[nodiscard]] bool clearing() const
    {
        std::scoped_lock lk(m_);
        return clearing_ and step_ == Step::Cross;
    }
    static constexpr const char* kClearingSuffix = "#through";   // appended to the cross's name: required, matched by none

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

        // ★ Live fix 3 — THE LOOK PRE-STEP (tourP7 leg 1: `cross door_2` with door_2 not in the graph, door_concept offering
        //   aff_door_look_door_2). While the door node is absent the passage requires the look (driven like any affordance,
        //   through the policy); the door RE-BORN (its node appears) moves it on to `approach` and the rest as today; the look
        //   gone without the door (door_concept RULED OUT this visit) fails the passage — the "not in the graph" failure.
        if (step_ == Step::Look)
        {
            const std::string look = look_name(door_);
            if (G.get_node(door_).has_value())
            {
                step_ = Step::Approach;
                seen_executing_ = false;
                out.event = std::format("[door-passage] {}: the door is re-born (in the graph again) — {} done, on to {}", door_, look,
                                        name_of(Step::Approach));
                last_event_ = *out.event;
            }
            else if (G.get_node(look).has_value())
            {
                step_state_ = state_name(state_of(G, look));
                note_ = std::format("{} ({})", look, step_state_);
                out.required = look;
                return out;
            }
            else
            {
                finish();
                failed_ = true;
                last_outcome_ = "look_ruled_out";
                out.event = std::format("[door-passage] {}: {} withdrawn and the door is not in the graph (ruled out this visit) — "
                                        "passage FAILED", door_, look);
                last_event_ = *out.event;
                return out;
            }
        }

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
            // FROM APPROACH STRAIGHT TO CROSS when door_concept offers it. This used to be forbidden (2026-10-12) because
            // door_concept then rated a SHUT door passable from across the room (try20/21/26: P(passable)=1.00 at phi=0.44
            // rad, clear span 0.10 m). That belief was the defect, and it is fixed at its source (10-19: the leaf model
            // scores every return and explains the room's walls, so a shut door reads shut from either side). Keeping
            // the rule made an OPEN, passable door wait for the approach servo's own completion — try99 leg 2: cross on
            // offer at P(passable)=1.00, phi 1.57 rad, while the approach servo micro-searched until approach_timeout.
            // `open_wanted` still makes a shut door win over a contradictory `cross`.
            // ★ R2: `cross` is skipped to ONLY when its own pre: clause holds (P(cross executable from here) — at the door x the run
            //   clear of the leaf x passable), and NEVER while the approach's hand-back is being waited out (attempts_ > 0 on the
            //   approach: door_concept is re-siting the standpoint). Otherwise the passage stays on / re-enters the approach.
            const bool approach_handed_back = step_ == Step::Approach and attempts_ > 0;
            const bool pre_blocks = later == Step::Cross and (approach_handed_back or cross_pre(G) == Pre::Fails);
            if (state_of(G, name_of(later)) == State::Offered and not open_wanted and not pre_blocks)
            {
                const bool was_executing = st_cur == State::Executing;
                out.event = std::format("[door-passage] {}: {} is offered — {} to it", door_, name_of(later),
                                        was_executing ? std::format("pre-empting {} and continuing", name_of(step_))
                                                      : std::string("skipping"));
                if (was_executing) preempted_ = true;
                if (step_ == Step::Approach) reached_door_ = true;   // `open`/`cross` offered ⇒ the door is within reach
                // ★ Live fix 4 ruling 3: an EXECUTING `open` pre-empted by `cross` whose door is MEASURED open (door_concept's
                //   door_open_prob known — not its -1 sentinel — and more likely open than not) counts as opened: the released
                //   open re-offered for a cycle is not news that the door shut (tourP8: cross -> open -> cross within 1 s).
                if (step_ == Step::Open and later == Step::Cross and was_executing)
                    if (const auto dn = G.get_node(door_); dn.has_value())
                        if (const auto po = G.get_attrib_by_name<door_open_prob_att>(*dn); po.has_value() and *po >= 0.f and *po > 0.5f)
                        { opened_ = true; reached_door_ = true; }
                step_ = later;
                seen_executing_ = false;
                attempts_ = 0;   // ★ R3: a step change starts the new step clean — no attempt carried over to fail it
                break;
            }
        }

        // ★ Task 3e (5), rtD1h leg 5 — AN `open` WITHDRAWN BECAUSE ITS GOAL HOLDS IS SATISFIED. door_concept offers `open` while
        //   P(in band)·P(shut) is up, and withdraws it when the door is OPEN — "withdrawn because satisfied". The passage had
        //   entered `open` from a satisfied approach and waited for that offer 7 min (until the mission timeout), with the door
        //   measured open (P(shut) 0.00, phi 1.57). Not on offer (neither Offered nor being executed by us) while the door is
        //   MEASURED more likely open than shut (door_open_prob known — not its -1 sentinel — and > 1/2: the MAP of the door's
        //   own open/shut belief, the same evidence ruling 3 uses above) ⇒ the step's goal holds: on to `cross`. An
        //   `open` that IS on offer is door_concept believing the door shut, and it is taken (unchanged). A protocol fix:
        //   both drivers take it.
        // ★ Fix round 1 (I-3): the skip does NOT set opened_ — nothing was opened. opened_ makes a re-offered `open` "the
        //   flicker of a door we just opened" and ignores it; here a re-offer is door_concept's own MAP that the door is SHUT
        //   (pushed to), so the "back to open when the door turns out shut" rule below must stay armed.
        // ★ M-4: the door node carries no stamp, so a door_open_prob could be a dead owner's last word. Its owner is LIVE while it
        //   holds a standing offer for this door (approach or cross offered / being executed) — the same agent writes both.
        const auto owner_live = [&]
        {
            for (Step s : {Step::Approach, Step::Cross})
                if (const State st = state_of(G, name_of(s)); st == State::Offered or st == State::Executing) return true;
            return false;
        };
        // ★ F2 (D3 fix round 2, gold14): an `open` that is COMPLETED (Satisfied) and WITHDRAWN is a door already open — the passage
        //   sat on it for ~10 min waiting for it to come back. It advances to `cross` whatever cross's pre says: the pre then decides
        //   cross vs re-approach (an offered cross whose run meets the leaf falls back to the approach below). The same for an
        //   `open` that is not offered while the door is MEASURED open (the rule above), cross's pre deciding what comes next.
        const auto open_done = [&]
        {
            const auto on = G.get_node(name_of(Step::Open));
            return on.has_value() and state_of(G, name_of(Step::Open)) == State::Completed
                   and rc::affordance::read_outcome(on.value()) == rc::affordance::Outcome::Satisfied;
        };
        if (step_ == Step::Open and not seen_executing_)
            if (const State so = state_of(G, name_of(Step::Open)); so != State::Offered and so != State::Executing)
            {
                std::optional<float> po;
                if (owner_live())
                    if (const auto dn = G.get_node(door_); dn.has_value())
                        if (const auto v = G.get_attrib_by_name<door_open_prob_att>(*dn); v.has_value() and *v >= 0.f and *v > 0.5f) po = *v;
                if (po.has_value() or open_done())
                {
                    reached_door_ = true;   // as a satisfied `open`: the robot is at the door (no falling back to `open`)
                    step_ = Step::Cross;
                    seen_executing_ = false;
                    attempts_ = 0;
                    out.event = std::format("[door-passage] {}: {} is not offered and the door is {} — its goal holds: SATISFIED -> {}",
                                            door_, name_of(Step::Open),
                                            po.has_value() ? std::format("measured open (P {:.2f})", *po) : std::string("already opened (completed, satisfied)"),
                                            name_of(Step::Cross));
                }
            }

        // ★ A STEP THAT IS NO LONGER ON OFFER FALLS BACK to the latest earlier one that is. door_concept withdraws
        //   `cross` when the door is believed shut and `open` when out of reach; `approach` is always offered for a
        //   located door. Without this the passage waited for a withdrawn step for ever (live try21: `cross` taken on
        //   a transient P(passable) of the SHUT door from across the room, then withdrawn — robot parked, door shut).
        // …but NOT once the robot has reached the door (2026-10-13, try36/37/40): a `cross` withdrawn for a few cycles by
        // a flickering P(passable) sent the robot back to `approach` while it stood AT the door, inside the band the
        // approach servo needs, and the servo gave up for ever. At the door the passage waits for `cross` instead.
        // ★ Live fix 1 (controller ruling, tourP1 leg 1, 2026-10-20): …EXCEPT back to `approach` itself. After `open` the
        //   local policy's release moved the robot 0.26 m back, out of the 2.0 m reach band: door_concept withheld `cross`
        //   (P(reach) 0.09) and kept offering `approach`, and the passage waited 9 min for a `cross` that could not appear.
        //   A robot OUTSIDE the band must get back into it: when the current step is not on offer and `approach` IS, the
        //   passage returns to `approach` even after reaching the door (its completion re-arms reached_door_; `opened_` is
        //   kept, so a re-offer of `open` after it is still the flicker of the door just opened, not news that it shut).
        //   Falling back to `open` stays forbidden once at the door (try36-40). With the policy driving, an approach
        //   standpoint the robot already stands on completes at once (instant completion), so a flicker costs one cycle.
        // ★ REVIEW (Live fix 1): at the door the fall back needs EVIDENCE the robot is out of position — its measured distance
        //   from the reference standpoint (door_concept's approach standpoint as offered now, else the `open` standpoint the
        //   passage last saw) beyond the arrival tolerance the approach completes with. Inside it the passage waits for
        //   door_concept as before (a one-cycle `cross` flicker at the door is not a reason to walk away; and with the robot
        //   at the approach standpoint, `open`/`cross` absent cannot cycle Approach <-> Open). And only while the last
        //   outcome is unset or satisfied: after a FAILED step the retry path ("waiting for it to be offered again",
        //   kMaxCrossUnreachable) stands unchanged.
        if (const auto sp = standpoint_of(G, Step::Open); sp.has_value()) last_open_sp_ = sp;
        const auto ref_sp = [&]() -> std::optional<Eigen::Vector2f> {
            if (const auto a = standpoint_of(G, Step::Approach); a.has_value()) return a;
            return last_open_sp_;
        }();
        const bool out_of_position = robot_.has_value() and ref_sp.has_value() and (*robot_ - *ref_sp).norm() > arrival_tol_;
        const bool outcome_ok = last_outcome_.empty() or last_outcome_ == "satisfied";
        if (step_ == Step::Cross or step_ == Step::Open)
        {
            const State st_now = state_of(G, name_of(step_));
            // ★ R2: a `cross` that is OFFERED but whose pre: clause fails from here (the run to the far side meets the leaf) is not
            //   executable: re-enter the approach, which re-sites the robot, exactly as for a withdrawn step. Never one being executed.
            const bool cross_not_executable = step_ == Step::Cross and st_now == State::Offered and not seen_executing_
                                              and cross_pre(G) == Pre::Fails;
            if ((st_now != State::Offered and st_now != State::Executing and not seen_executing_) or cross_not_executable)
                for (Step earlier : {Step::Open, Step::Approach})
                {
                    if (static_cast<int>(earlier) >= static_cast<int>(step_)) continue;
                    if (reached_door_ and (earlier != Step::Approach or not out_of_position or not outcome_ok)) continue;
                    if (state_of(G, name_of(earlier)) == State::Offered)
                    {
                        out.event = std::format("[door-passage] {}: {} is not offered — back to {}{}", door_,
                                                name_of(step_), name_of(earlier),
                                                reached_door_ ? " (the robot left the door's reach band)" : "");
                        if (earlier == Step::Approach) reached_door_ = false;   // re-armed by approach's completion
                        step_ = earlier;
                        attempts_ = 0;   // ★ R3
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
            attempts_ = 0;   // ★ R3
        }

        // ★ THE CROSSING ENDS WHEN THE ROBOT IS THROUGH (Task 12 Fix E, user 10-18: "keep going inside so it can localise in
        //   the room"). room_concept's crossing evidence moves the robot's RT to the new room with the body only ~0.3 m past
        //   the plane (try71: close-behind then "ended infeasible", and exploration wedged the robot beside the 97° leaf).
        //   While the cross is driven on the interaction anchor — robot-relative, so the switch does not move it — the switch
        //   alone no longer ends it: the session keeps servoing the same anchor (clearing()) until the servo's `through`
        //   (s ≥ AnchorCrossStandoffM along its normal). Without an anchor (the room route), or if the anchor is lost while
        //   clearing, the old rule: from the switch on the cross standpoint is a pose in the room just left — the passage
        //   moves on. rc::anchor_cross_end is the decision, pure (tests/test_anchor_servo.cpp).
        // Task 3f r1 (M-8): a switch counts only INTO a room this door leads to (target_rooms; unknown ⇒ any other room). A
        // relocaliser jump into a third room (rtD1e: 4.4 m / 179° at a hand-over) is not the robot through this door.
        bool switched = step_ == Step::Cross and not cross_room_.empty() and room_frame != cross_room_;
        if (switched)
        {
            const auto targets = target_rooms(G);
            if (not targets.empty() and std::ranges::find(targets, room_frame) == targets.end())
            {
                switched = false;
                if (target_ignored_logged_ != room_frame)
                {
                    target_ignored_logged_ = room_frame;
                    std::println("[door-passage] {}: the room changed {} -> {}, which this door does not lead to ({}) — not a "
                                 "switch through it", door_, cross_room_, room_frame, targets.front());
                }
            }
        }
        if (step_ == Step::Cross and (switched or clearing_)
            and rc::anchor_cross_end(anchored_, switched or clearing_, through_) == rc::AnchorCrossEnd::Continue)
        {
            if (not clearing_)
                out.event = std::format("[door-passage] {}: the room changed {} -> {} while crossing on the anchor — going on "
                                        "until through", door_, cross_room_, room_frame);
            clearing_ = true;
            note_ = "cross (through the doorway, on the anchor)";
            step_state_ = "clearing";
            if (out.event.has_value()) last_event_ = *out.event;
            out.required = name_of(Step::Cross) + kClearingSuffix;   // a NAME no affordance carries: nothing is selectable
            return out;
        }
        if (step_ == Step::Cross and (switched or clearing_))
        {
            const bool was_through = clearing_ and through_;
            // Task 3f r1 (M-1): a clearing GIVEN UP (no route / Unreachable / anchor gone) with the session's own posterior
            // saying the body may still straddle the doorway: not through — never close the door on it; the passage ends here
            const bool may_straddle = clearing_ and not through_ and p_through_ >= 0.0 and not (p_through_ > 0.5);
            clearing_ = false;
            if (may_straddle)
            {
                finish();
                out.event = std::format("[door-passage] {}: the clearing was given up past the room switch {} -> {} — NOT through "
                                        "(P(through) {:.3f}: the body may straddle the doorway) — door left open, passage DONE",
                                        door_, cross_room_, room_frame, p_through_);
                last_event_ = *out.event;
                return out;
            }
            if (static_cast<int>(stop_after_) >= static_cast<int>(Step::Close))
            {
                step_ = Step::Close;
                close_since_ms_ = now_ms();
                seen_executing_ = false;
                attempts_ = 0;
                out.event = was_through
                    ? std::format("[door-passage] {}: through on the anchor (AnchorCrossStandoffM past the doorway) — now "
                                  "closing the door behind", door_)
                    : std::format("[door-passage] {}: the room changed {} -> {} while crossing — now closing the "
                                  "door behind", door_, cross_room_, room_frame);
            }
            else
            {
                finish();
                out.event = was_through
                    ? std::format("[door-passage] {}: through on the anchor — passage DONE", door_)
                    : std::format("[door-passage] {}: the room changed {} -> {} while crossing — passage DONE",
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

        // ★Task E — a step that is not clearly the better policy yields to the door's epistemic look (aff_<door>, door_concept's
        //   NBV on its [s, w, h] belief). Only BEFORE the robot is in the doorway: an approach, or an open/cross that is not on
        //   offer (withdrawn). An executing open/cross is never interrupted — standing in a swing or a doorway is not a place
        //   to go and look from.
        const std::string rq = name_of(step_);   // the step as it stands NOW (the completion above may have advanced it)
        const State sq = state_of(G, rq);
        if (look_enabled_ and robot_.has_value()   // no robot pose ⇒ no distances ⇒ no comparison (never a bias toward the look)
            and (step_ == Step::Approach or ((step_ == Step::Open or step_ == Step::Cross) and sq != State::Offered and sq != State::Executing)))
        {
            const std::string look = "aff_" + door_;
            const State sl = state_of(G, look);
            const auto gain_of = [&](const std::string& n) -> std::optional<float>
            {
                const auto nd = G.get_node(n);
                if (not nd.has_value()) return std::nullopt;
                const auto g = G.get_attrib_by_name<epistemic_gain_att>(nd.value());
                return g.has_value() and std::isfinite(*g) ? *g : 0.0f;
            };
            const auto dist_to = [&](const std::string& n) -> float
            {
                const auto nd = G.get_node(n);
                if (not nd.has_value()) return 0.0f;
                const auto x = G.get_attrib_by_name<epistemic_target_x_m_att>(nd.value());
                const auto y = G.get_attrib_by_name<epistemic_target_y_m_att>(nd.value());
                if (not x.has_value() or not y.has_value() or not std::isfinite(*x) or not std::isfinite(*y)) return 0.0f;
                return (*robot_ - Eigen::Vector2f(*x, *y)).norm();
            };
            const bool look_on = sl == State::Offered or sl == State::Executing;
            const bool step_on = sq == State::Offered or sq == State::Executing;
            const auto c = choose_look_or_step(step_on ? gain_of(rq) : std::nullopt, step_on ? dist_to(rq) : 0.0f,
                                               look_on ? gain_of(look) : std::nullopt, look_on ? dist_to(look) : 0.0f,
                                               efe_lambda_, efe_margin_, look_chosen_);
            if (c.look != look_chosen_)
            {
                out.event = std::format("[door-passage] {}: {} {} (EFE: look {:+.3f} vs {} {:+.3f} nats) — {}", door_,
                                        c.look ? "the epistemic look beats" : "the step beats the look", rq, c.look_score,
                                        rq, c.step_score,
                                        c.look ? "taking the look" : "back to the step");
                last_event_ = *out.event;
                // the claim on the one that lost is released by the caller, so the producer sees it Offered again
                if (c.look ? sq == State::Executing : sl == State::Executing) preempted_ = true;
                look_chosen_ = c.look;
            }
            if (c.look)
            {
                step_state_ = state_name(sl);
                note_ = std::format("{} ({}) — EFE look", look, step_state_);
                out.required = look;
                return out;
            }
        }
        else
            look_chosen_ = false;

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
        // For the motion verdict (motion_verdict.h): which node it waits on, and the two rules that keep it waiting.
        std::string required;            ///< the current step's affordance name ("" = idle / done)
        bool reached_door = false;       ///< no fallback to earlier steps (update(): "…but NOT once the robot has reached")
        bool opened = false;             ///< `open` succeeded: a re-offered `open` is the flicker, ignored
        bool clearing = false;           ///< crossing on past the room switch, on the anchor
    };
    [[nodiscard]] View view() const
    {
        std::scoped_lock lk(m_);
        const bool live = step_ == Step::Look or step_ == Step::Approach or step_ == Step::Open or step_ == Step::Cross or step_ == Step::Close;
        return {.door = door_, .step = step_, .step_state = step_state_, .attempts = attempts_, .last_event = last_event_,
                .last_outcome = last_outcome_, .then_affordances = then_affordances_, .failed = failed_,
                .required = live ? name_of(step_) : std::string{}, .reached_door = reached_door_, .opened = opened_,
                .clearing = clearing_ and step_ == Step::Cross};
    }
    [[nodiscard]] static const char* step_name(Step s)
    {
        switch (s)
        {
            case Step::Idle: return "idle";
            case Step::Look: return "look";
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
        if (s == Step::Look) return look_name(door_);
        const char* kind = s == Step::Approach ? "approach" : s == Step::Open ? "open" : s == Step::Close ? "close" : "cross";
        return "aff_" + door_ + "_" + kind;
    }
    /// ★ R2 (D3 fix round 1) — the cross's OWN `pre:` clauses, read from ITS contract as the Place/Orient completion reads its
    ///   own (contract.pre against the door node's attribute of that name). Unstated = the producer declares none (an older
    ///   door_concept): the legacy behaviour, offered means executable. Fails when the probability is below the clause's level OR
    ///   not measured (the -1 sentinel is no evidence, never a pass).
    enum class Pre { Unstated, Holds, Fails };
    [[nodiscard]] Pre cross_pre(DSR::DSRGraph& G) const
    {
        const auto an = G.get_node(name_of(Step::Cross));
        if (not an.has_value()) return Pre::Unstated;
        const auto contract = rc::affordance::read_contract(an.value());
        bool stated = false;
        for (const auto& cl : contract.pre)
        {
            if (cl.attr != "door_cross_exec_prob") continue;
            stated = true;
            const auto dn = G.get_node(door_);
            const auto v = dn.has_value() ? G.get_attrib_by_name<door_cross_exec_prob_att>(*dn) : std::nullopt;
            if (not v.has_value() or not std::isfinite(*v) or *v < 0.f) return Pre::Fails;
            const bool ok = cl.op == rc::affordance::CompareOp::LE ? *v <= cl.value : *v >= cl.value;
            if (not ok) return Pre::Fails;
        }
        return stated ? Pre::Holds : Pre::Unstated;
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
    float efe_lambda_ = 0.2f, efe_margin_ = 0.5f;   ///< set_efe: the selector's λ (nats/m) and switch margin (nats)
    bool look_enabled_ = false;   ///< set_look_enabled
    bool look_chosen_ = false;    ///< the last EFE choice was the epistemic look, not the step (the incumbent)
    bool reached_door_ = false;   ///< approach satisfied (or a later step taken from it) in this passage
    bool opened_ = false;         ///< `open` satisfied in this passage
    int attempts_ = 0;
    std::string cross_room_;   // the room the cross step began in
    std::string note_ = "idle";
    std::string step_state_;   // view()
    std::string last_event_;   // view()
    std::string last_outcome_; // view()
    std::optional<Eigen::Vector2f> robot_;          // note_robot (Live fix 1 review)
    float arrival_tol_ = 0.25f;
    std::optional<Eigen::Vector2f> last_open_sp_;   // the `open` standpoint as last seen
    // An affordance's standpoint (epistemic_target_x/y_m) as its producer writes it; nullopt = node or attribute absent.
    std::optional<Eigen::Vector2f> standpoint_of(DSR::DSRGraph& G, Step s) const
    {
        const auto n = G.get_node(name_of(s));
        if (not n.has_value()) return std::nullopt;
        const auto x = G.get_attrib_by_name<epistemic_target_x_m_att>(*n);
        const auto y = G.get_attrib_by_name<epistemic_target_y_m_att>(*n);
        if (not x.has_value() or not y.has_value() or not std::isfinite(*x) or not std::isfinite(*y)) return std::nullopt;
        return Eigen::Vector2f(*x, *y);
    }
    int  cross_unreachable_ = 0;   // Unreachable cross completions in this passage (kMaxCrossUnreachable)
    bool failed_ = false;          // finished by the cap, not by crossing
    bool clearing_ = false;        // Task 12 Fix E: crossing on past the room switch, on the anchor
    bool anchored_ = false;        // the session: the cross is being driven on an anchor
    bool through_ = false;         // the session: the servo reached `through`
    double p_through_ = -1.0;      // Task 3f r1 (M-1): the session's P(through) while clearing; < 0 = not computed (old chain)
    std::string target_ignored_logged_;   // Task 3f r1 (M-8): the non-target room a switch was last ignored for (log once)
    // ★ Task 3f r1 (M-8): THE ROOMS THIS DOOR LEADS TO. ltsm's passage node (`metaconcept`, passage_live.h) holds a `match` edge
    //   to each of the doorway's two faces, one door node per room; the faces other than this door hang (RT chain) from the
    //   rooms on the far side. Empty = no passage node names this door yet (the far face not born): every other room counts,
    //   as before — the controller holds no other map of which rooms a door joins.
    std::vector<std::string> target_rooms(DSR::DSRGraph& G) const
    {
        std::vector<std::string> rooms;
        const auto door = G.get_node(door_);
        if (not door.has_value()) return rooms;
        for (const auto& in : G.get_edges_to_id(door->id()))
        {
            if (in.type() != "match") continue;
            const auto passage = G.get_node(in.from());
            if (not passage.has_value()) continue;
            for (const auto& e : DSR::DSRGraph::get_node_edges_by_type(passage.value(), "match"))
            {
                if (e.to() == door->id()) continue;
                const auto rid = rc::world::room_of(G, e.to());
                if (rid == 0) continue;
                if (const auto room = G.get_node(rid); room.has_value() and std::ranges::find(rooms, room->name()) == rooms.end())
                    rooms.push_back(room->name());
            }
        }
        return rooms;
    }
public:
    static constexpr int kMaxCrossUnreachable = 3;   // ⚑ flagged decision level (Task 9 fix round 1)
};

}   // namespace rc
