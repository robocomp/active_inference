#pragma once
// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE AFFORDANCE PROTOCOL AS PURE RULES — no DSR, no Qt, no robot.
//
// Why this exists: on 2026-08-19 five distinct protocol failures were each diagnosed by a ~20 minute
// run on the real robot, and four of the fixes were wrong and had to be reverted. Every one of those
// failures is decidable in a few hundred simulated cycles. The rules were untestable only because they
// were tangled with DSR attribute access, so the only way to exercise them was to drive a robot.
//
// The policy flags below are REAL code paths, and each defaulted the way the live system behaved.
// protocol_bench.cpp turns each of the day's failures into a scenario that fails without its fix.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cstdint>
#include <optional>
#include <map>
#include <set>
#include <vector>

namespace rc::affordance::core
{

enum class NodeState { Offered, Executing, Completed };
enum class Outcome   { None, Satisfied, Refused };

// What actually lives on the shared graph node. `cell` is the standpoint's identity — the thing the
// old TLA+ model lacked, and therefore the thing that made "the producer will propose a different
// cell" an assumption rather than a property.
struct Wire
{
    NodeState state   = NodeState::Completed;
    int       cell    = -1;
    Outcome   outcome = Outcome::None;
    bool      claimed = false;      // consumer has taken this arming
};

struct ProducerPolicy
{
    // ★★★RULE 3: EVERY CLAIM IS A LEASE. The offer timeout only covered Offered-and-unclaimed. If the
    // consumer CLAIMS and then stops — planner finds no route, it drops target and plan but never
    // releases the claim — the node sits Executing for ever and the producer has no escape at all.
    // Confirmed three ways on 2026-08-19: 12% of live records were `Executing` with the consumer
    // holding no target; TLA+ with a stuttering consumer VIOLATES ProducerLive; and an independent
    // review named it before seeing either. A lease expires on the PRODUCER's own clock and does not
    // ask the consumer's permission.
    int execution_lease_cycles = 200;    // <=0 disables
    // ★★★RULE 5: BELIEFS CHANGE ONLY THROUGH SENSING. Feeding a protocol event (Refused) into the
    // value function is what made the 104/min oscillator: a completion path that costs no physical
    // time, wired straight into what to do next. Reachability failures may inform a COST model; they
    // must never touch the information-gain term. With this false, the only thing that moves the
    // argmax is an observation, and observations cost seconds by construction.
    bool belief_from_protocol_events = true;
    // ★"an unchanged proposal is not news". Live until 2026-08-19 15:00. AffordanceSelection.tla:
    // DEADLOCK in all 8 combinations of the other choices when this is false.
    bool rearm_unchanged = true;
    // ★does de-prioritising a refused cell actually change which cell wins? Measured live: gain fell
    // 0.171 -> 0.0004 and the cell still won, because score (~0.28) is not dominated by gain.
    bool refusal_moves_argmax = true;
    // ★an offer nobody claims must expire, or the producer waits on a consumer that already declined.
    // Measured: 100.6 s, then 191.6 s, holding one cell while the robot stood still.
    int offer_timeout_cycles = 100;     // <=0 disables
};

struct ConsumerPolicy
{
    // Simulate the measured failure: claim, then stop for ever (no route, or a crash).
    int die_after_claims = 0;            // 0 = never dies
    // ★a refused standpoint is un-takeable for a while. Without it: ~104 completions/min, robot still.
    int refusal_hold_cycles = 60;       // <=0 disables
    // ★"not the one that just finished" — right, but it was the only thing clearing last_completed,
    // so when its yield was blocked the selector starved.
    bool suppress_just_completed = true;
    // ★"already at this standpoint on the first cycle => REFUSE". Introduced 9b3e5a4, removed the same
    // day: it made an early arrival into a producer-visible refusal, which is what livelocked the pair.
    bool instant_arrival_is_refusal = false;
};

// The world the pair is acting in: which standpoints can actually be stood at, and where the robot is.
struct World
{
    std::set<int> reachable;
    int  robot_at = -1;                 // cell the robot currently occupies (-1 = elsewhere)
    int  observations = 0;              // ← the only thing that means the system is working
    std::set<int> observed;
};

class Producer
{
public:
    ProducerPolicy p;
    explicit Producer(ProducerPolicy pol, std::vector<int> cells) : p(pol), cells_(std::move(cells)) {}

    // Deterministic argmax: the same state yields the same cell. Modelling this as a free choice is
    // what made the first three versions of the TLA+ model declare everything sound.
    [[nodiscard]] int pick() const
    {
        if (p.refusal_moves_argmax)
            for (int c : cells_)
                if (not deprio_.contains(c)) return c;
        return cells_.empty() ? -1 : cells_.front();
    }

    void tick(Wire &w, std::uint64_t cycle)
    {
        // ★THE LEASE COVERS THE CLAIMED CASE TOO. Unconditional, on our clock.
        if (w.state == NodeState::Executing and p.execution_lease_cycles > 0
            and cycle - armed_at_ >= static_cast<std::uint64_t>(p.execution_lease_cycles))
        {
            w.state = NodeState::Completed;
            w.outcome = Outcome::None;
            w.claimed = false;
            return;
        }
        if (w.state == NodeState::Offered and not w.claimed and p.offer_timeout_cycles > 0
            and cycle - armed_at_ >= static_cast<std::uint64_t>(p.offer_timeout_cycles))
        {
            deprio_.insert(w.cell);                 // nobody is coming; look elsewhere
            w.state = NodeState::Completed;
            w.outcome = Outcome::None;
            return;
        }
        if (w.state != NodeState::Completed) return;
        if (w.outcome != Outcome::None)             // read the result before proposing again
        {
            // ★RULE 5 in one line: a refusal is a fact about REACHABILITY, not about information.
            if (w.outcome == Outcome::Refused and p.belief_from_protocol_events)
                deprio_.insert(w.cell);
            w.outcome = Outcome::None;
            return;
        }
        const int c = pick();
        if (c < 0) return;
        if (not p.rearm_unchanged and c == w.cell) return;   // ← the deadlock, when the argmax is stuck
        w.cell = c;
        w.state = NodeState::Offered;
        w.claimed = false;
        armed_at_ = cycle;
    }
private:
    std::vector<int> cells_;
    std::set<int> deprio_;
    std::uint64_t armed_at_ = 0;
};

class Consumer
{
public:
    ConsumerPolicy p;
    explicit Consumer(ConsumerPolicy pol) : p(pol) {}

    void tick(Wire &w, World &world, std::uint64_t cycle)
    {
        if (p.die_after_claims > 0 and claims_ >= p.die_after_claims) return;   // stuttering consumer
        if (w.state == NodeState::Offered)
        {
            if (p.suppress_just_completed and w.cell == just_completed_
                and cycle - completed_at_ < 2) return;              // not twice in a row
            if (p.refusal_hold_cycles > 0)
                if (const auto it = held_.find(w.cell); it != held_.end())
                    if (cycle - it->second < static_cast<std::uint64_t>(p.refusal_hold_cycles)) return;
            w.state = NodeState::Executing;
            w.claimed = true;
            ++claims_;
            started_at_ = cycle;
            arrived_instantly_ = (world.robot_at == w.cell);
            return;
        }
        if (w.state != NodeState::Executing) return;

        // Drive. Reaching a cell takes time unless the robot is already standing on it.
        const bool arrived = arrived_instantly_ or (cycle - started_at_) >= 5;
        if (not arrived) return;

        just_completed_ = w.cell;
        completed_at_ = cycle;
        if (arrived_instantly_ and p.instant_arrival_is_refusal)
        {   // 9b3e5a4: report an early arrival as a refusal. This is the livelock.
            held_[w.cell] = cycle;
            w.outcome = Outcome::Refused;
        }
        else if (world.reachable.contains(w.cell))
        {
            world.robot_at = w.cell;
            world.observed.insert(w.cell);
            ++world.observations;
            w.outcome = Outcome::Satisfied;
        }
        else
        {
            held_[w.cell] = cycle;
            w.outcome = Outcome::Refused;
        }
        w.state = NodeState::Completed;
        w.claimed = false;
    }
private:
    std::map<int, std::uint64_t> held_;
    int claims_ = 0;
    int just_completed_ = -1;
    std::uint64_t completed_at_ = 0, started_at_ = 0;
    bool arrived_instantly_ = false;
};

// Run the pair and report what the robot achieved. THE metric is observations, not message traffic:
// every failure of 2026-08-19 kept the wire busy while observing nothing.
struct Result
{
    int observations = 0;
    int completions = 0;
    std::uint64_t cycles = 0;
    bool deadlocked = false;        // wire stopped changing entirely
};

inline Result run(Producer &prod, Consumer &cons, World &world, std::uint64_t cycles)
{
    Wire w;
    Result r;
    Wire last = w;
    std::uint64_t unchanged = 0;
    for (std::uint64_t c = 0; c < cycles; ++c)
    {
        const Outcome before = w.outcome;
        prod.tick(w, c);
        cons.tick(w, world, c);
        if (w.outcome != Outcome::None and before == Outcome::None) ++r.completions;
        if (w.state == last.state and w.cell == last.cell and w.outcome == last.outcome) ++unchanged;
        else unchanged = 0;
        last = w;
        if (unchanged > 200) { r.deadlocked = true; break; }
        r.cycles = c + 1;
    }
    r.observations = world.observations;
    return r;
}

}  // namespace rc::affordance::core
