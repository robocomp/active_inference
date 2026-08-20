#pragma once
// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE AFFORDANCE PROTOCOL WITH N PRODUCERS — pure rules, no DSR, no Qt, no robot.
//
// protocol_core.h models ONE producer against ONE consumer, which is the pair that failed on
// 2026-08-19. The system it has to become has TEN agents offering standpoints into one consumer
// (room, table, chair, bottle, cabinet, fridge, hood, door, kitchen, dining-set), and three of the
// properties that matter cannot even be STATED with one producer:
//
//   · FAIRNESS   — with one producer, "nobody starves" is trivially true.
//   · PREEMPTION — with one producer, there is nothing to switch to mid-drive.
//   · ISOLATION  — with one producer, a badly-behaved agent cannot hold the consumer away from
//                  the other nine, which is the failure mode that scales worst.
//
// Everything here is a REAL code path in controller_session.cpp / affordance_manager.cpp /
// room_scene_graph.cpp, and every policy flag defaults to what the live system does today, so a
// scenario that fails here fails on the robot.
//
// The metric is never message traffic. It is OBSERVATIONS: a standpoint the robot physically stood
// at, from which the producer is entitled to update its beliefs. Every failure of 2026-08-19 kept
// the wire busy at 20 Hz while observing nothing.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <string>
#include <vector>

namespace rc::affordance::multi
{

// ── THE WIRE ────────────────────────────────────────────────────────────────────────────────────
enum class NodeState { Offered, Executing, Completed };

// The full vocabulary, as it now exists on the wire (affordance_goal_parse.h). Only Satisfied
// licenses a belief update; every other word — including one a future consumer invents that this
// producer has never heard of — must read as "nothing was observed".
enum class Outcome { None, Satisfied, Timeout, Refused, Abandoned, Infeasible, Unreachable,
                     OutsideRoom };

[[nodiscard]] inline bool observation_happened(Outcome o) { return o == Outcome::Satisfied; }

[[nodiscard]] inline const char *name(Outcome o)
{
    switch (o)
    {
        case Outcome::Satisfied:   return "satisfied";
        case Outcome::Timeout:     return "timeout";
        case Outcome::Refused:     return "refused";
        case Outcome::Abandoned:   return "abandoned";
        case Outcome::Infeasible:  return "infeasible";
        case Outcome::Unreachable: return "unreachable";
        case Outcome::OutsideRoom: return "outside_room";
        default:                   return "none";
    }
}

struct Wire
{
    NodeState state   = NodeState::Completed;
    int       cell    = -1;
    float     gain    = 0.f;
    Outcome   outcome = Outcome::None;
    bool      claimed = false;
};

// ── THE WORLD ───────────────────────────────────────────────────────────────────────────────────
// What is true of a standpoint, independent of what any agent believes about it. These are exactly
// the three questions the consumer can answer and the producer cannot.
enum class Standable
{
    Yes,          // the body fits at the published point
    OffCentre,    // it does not fit at the point but does inside the producer's 0.5 m cell
    Never         // nothing in the cell admits the body  → infeasible
};

enum class Routable
{
    Yes,          // a route exists from wherever the robot is
    CloserExists, // no route, but there is a reachable pose nearer the cell → approach
    // ★THE APPROACH THAT IS NOT A DRIVE. The closest reachable pose is inside the arrival band of
    // where the robot already stands, so "approaching" it covers no ground: the consumer would arrive
    // on the cycle it set off. Live on 2026-08-19 this alternated with the arrival branch and stopped
    // the base 47 times in one burst at controller_session.cpp:3725 (plan present, follower inactive).
    // The consumer must recognise it as the sealed case and report at once rather than pretend to move.
    CloserWithinBand,
    Sealed        // no route and no closer pose → the robot is in a pocket
};

struct Cell
{
    int       id       = -1;
    float     x        = 0.f;      // one-dimensional world is enough: only distance matters here
    Standable standable = Standable::Yes;
    Routable  routable  = Routable::Yes;
    bool      detects   = true;    // does the acquisition the contract asks for actually arrive?
    bool      wedges    = false;   // does the drive physically wedge on the way?
    // ★NOT IN THE ROOM AT ALL. A producer bug, not a navigation one: the cell is outside the layout
    // that same producer published. The consumer can see it and must say so in its own words, or a
    // grid-extent error upstream reads downstream as "the body does not fit".
    bool      outside_room = false;
};

struct World
{
    std::vector<Cell> cells;
    float robot_x   = 0.f;
    int   robot_at  = -1;

    // ★THE ONLY LEDGER THAT COUNTS. `stood_at` is physical truth — where the body actually was.
    // `observed` is what producers were told. A protocol is correct only if observed ⊆ stood_at;
    // the phantom-arrival bug was exactly the moment those two sets diverged (140 of 163 arrivals).
    // Bumped whenever the obstacle picture changes. A map-only verdict survives exactly as long as
    // this does not move — that is what makes the cache a statement about the world rather than a
    // timer standing in for one.
    int map_version = 0;
    std::multiset<int> stood_at;
    std::multiset<int> observed;
    int observations = 0;

    [[nodiscard]] const Cell &cell(int id) const
    {
        for (const auto &c : cells) if (c.id == id) return c;
        return cells.front();
    }
};

// ── PRODUCER ────────────────────────────────────────────────────────────────────────────────────
struct ProducerPolicy
{
    int  execution_lease_cycles = 200;   // rule 3: a claim is a lease. <=0 disables
    int  offer_timeout_cycles   = 100;   // an offer nobody takes must expire
    bool rearm_unchanged        = true;  // rule 2: a re-offer IS news; false deadlocks the pair
    // ★RULE 5. When true, a protocol event (a refusal, an unreachable) suppresses the cell's
    // INFORMATION term — which is the 104/min oscillator: a completion path costing no physical time
    // wired straight into the value function. When false, only an observation moves the belief, and
    // attempts decay a separate COST term instead.
    bool belief_from_protocol_events = false;
    int  attempt_decay_cycles   = 150;   // how long an attempted cell stays de-prioritised
    // ★NEGLECT GROWS BACK. A cell observed once is not finished for ever — its information decays and
    // it becomes worth revisiting, which is what makes the exploration a continuing process rather
    // than a queue that drains. Without this the world EXHAUSTS after one pass, and "no observations
    // in the last third" reports correct behaviour as a failure: the first version of this bench
    // failed six scenarios that way, which is the instrument measuring the wrong thing.
    int  revisit_cycles         = 600;
    // ★NO BACKOFF TIMER HERE. An earlier version of this file gave the producer a growing quiet
    // period after each non-observation; it cut scenario 5 from 922 completions to 302 and left a
    // 46-completion barren run, because with ten agents the consumer just turns to the next one's
    // offer — a per-agent limit cannot bound a global resource. The consumer's verdict cache does it
    // with no constant at all. Recorded so it is not reintroduced.
};

class Producer
{
public:
    ProducerPolicy p;
    std::string    label;

    Producer(std::string name, ProducerPolicy pol, std::vector<int> cells)
        : p(pol), label(std::move(name)), cells_(std::move(cells)) {}

    [[nodiscard]] const std::vector<int> &cells() const { return cells_; }
    [[nodiscard]] int served() const { return served_; }
    // What this agent BELIEVES it has explored. The phantom check on the consumer's side cannot see
    // damage here: rule 5 pollutes the producer's beliefs directly, without any false report ever
    // crossing the wire, and a belief nobody can audit is how "already looked there" becomes
    // permanent for a place the robot never reached.
    [[nodiscard]] std::vector<int> believed() const
    {
        std::vector<int> v;
        for (const auto &[cell, when] : observed_at_) v.push_back(cell);
        return v;
    }
    [[nodiscard]] int completions() const { return completions_; }

    // Deterministic argmax over gain, with attempted cells suppressed by a DECAYING term. Modelling
    // this as a free choice is what let three versions of the TLA+ model declare everything sound:
    // "the producer will pick a different one" was an assumption inside the abstraction, not a
    // property anyone had checked.
    [[nodiscard]] int pick(std::uint64_t cycle) const
    {
        int best = -1;
        float best_score = -1e9f;
        for (int c : cells_)
        {
            if (const auto it = observed_at_.find(c); it != observed_at_.end())
                if (cycle - it->second < static_cast<std::uint64_t>(p.revisit_cycles))
                    continue;                                // still fresh: no information to gain
            float score = neglect_gain(c, cycle);
            if (const auto it = attempted_.find(c); it != attempted_.end())
            {
                // ★THE COST OF A FAILED ATTEMPT ACCUMULATES. A flat decaying suppressor lets a cell
                // that has failed twenty times return at full strength every decay window, which is
                // what produced 112 completions in a row with nothing observed: the pair is honest,
                // busy, and useless. Repeated failure is evidence about the cell, and evidence adds.
                const auto age = cycle - it->second.when;
                const float k = static_cast<float>(p.attempt_decay_cycles) * static_cast<float>(it->second.hits);
                score *= (static_cast<float>(age) >= k) ? 1.f : static_cast<float>(age) / k;
            }
            if (score > best_score) { best_score = score; best = c; }
        }
        return best;
    }

    // The neglect term of the real planner: log(1 + a/T), monotone and unbounded in the age `a`
    // since the cell was last actually observed. A never-observed cell always eventually outranks a
    // fresher one, which is what stops any agent from starving behind a closer neighbour.
    [[nodiscard]] float neglect_gain(int c, std::uint64_t cycle) const
    {
        const auto it = observed_at_.find(c);
        const auto age = (it == observed_at_.end()) ? cycle + static_cast<std::uint64_t>(p.revisit_cycles)
                                                    : cycle - it->second;
        return std::log1p(static_cast<float>(age) / static_cast<float>(p.revisit_cycles));
    }

    void tick(Wire &w, std::uint64_t cycle)
    {
        if (w.state == NodeState::Executing and p.execution_lease_cycles > 0
            and cycle - armed_at_ >= static_cast<std::uint64_t>(p.execution_lease_cycles))
        {
            w.state = NodeState::Completed; w.outcome = Outcome::None; w.claimed = false;
            return;                                   // reclaimed on OUR clock, no permission asked
        }
        if (w.state == NodeState::Offered and not w.claimed and p.offer_timeout_cycles > 0
            and cycle - armed_at_ >= static_cast<std::uint64_t>(p.offer_timeout_cycles))
        {
            { auto &a = attempted_[w.cell]; a.when = cycle; ++a.hits; }
            w.state = NodeState::Completed; w.outcome = Outcome::None;
            return;
        }
        if (w.state != NodeState::Completed) return;

        if (w.outcome != Outcome::None)
        {
            ++completions_;
            if (observation_happened(w.outcome))
            {
                observed_at_[w.cell] = cycle;         // belief updated — and ONLY here
                ++served_;
            }
            else
            {
                // Every non-Satisfied word means the same thing to the value function: nothing was
                // seen, so try elsewhere for a while. It is a COST, and it decays.
                { auto &a = attempted_[w.cell]; a.when = cycle; ++a.hits; }

                if (p.belief_from_protocol_events)
                    observed_at_[w.cell] = cycle;     // ← rule-5 violation: a protocol event, believed
            }
            w.outcome = Outcome::None;
            return;
        }

        const int c = pick(cycle);
        if (c < 0) return;                            // nothing left worth looking at
        if (not p.rearm_unchanged and c == w.cell) return;
        w.cell = c;
        w.gain = neglect_gain(c, cycle);
        w.state = NodeState::Offered;
        w.claimed = false;
        armed_at_ = cycle;
    }

private:
    struct Attempt { std::uint64_t when = 0; int hits = 0; };
    std::vector<int> cells_;
    std::map<int, Attempt> attempted_;
    std::map<int, std::uint64_t> observed_at_;
    std::uint64_t armed_at_ = 0;
    int served_ = 0, completions_ = 0;
};

// ── CONSUMER ────────────────────────────────────────────────────────────────────────────────────
struct ConsumerPolicy
{
    int  die_after_claims        = 0;      // 0 = never; simulates a crashed or wedged consumer
    int  refusal_hold_cycles     = 60;     // a spot just refused is un-takeable for a while
    bool suppress_just_completed = true;
    // ★THE BUG OF 2026-08-19. When true, an unroutable standpoint is silently relocated to the
    // closest reachable pose — often the robot's own position — and the arrival is reported as
    // SATISFIED against the PRODUCER's cell. Measured: 140 of 163 accepted arrivals a median 2.86 m
    // from the published cell; the producer marked all of them observed.
    bool substitute_and_claim_satisfied = false;
    // With the fix: still approach the closest reachable pose (real work, changes what is visible)
    // but report `unreachable`, so nothing about the producer's cell is marked observed.
    bool approach_reports_unreachable = true;
    // A local veto that does not tell the producer. Two minutes per cell, no message: the producer
    // keeps a cell on offer that the consumer has quietly vetoed and both wait.
    bool silent_local_veto = false;
    // ★★★A VERDICT READ OFF AN UNCHANGED MAP IS STILL THE SAME VERDICT — so remember it instead of
    // re-deciding it. This replaces two timers (a per-producer backoff and a global quiet period)
    // with the invariant they were both approximating badly. The timers bounded the churn and starved
    // the useful work with it: measured, the global one took scenario 3 from 120 observations to 42
    // and scenario 5 to zero in its last third, because a mute is indiscriminate — it silences the
    // servable offers along with the impossible ones.
    // ★The cache has no duration and no constant. A map-only verdict (the body does not fit; there is
    // no route and nowhere closer) is a function of the map and the robot's pose, so it stays true
    // until the robot moves — and the robot moving is exactly what costs physical time. Every cell can
    // therefore produce at most ONE such verdict per pose, and a new one requires a drive. That is the
    // no-zero-cost-edge property as a mechanism rather than a hope, and it blocks nothing that could
    // have succeeded.
    bool cache_map_verdicts    = true;
    int  drive_cycles_per_unit = 2;        // physical time. No zero-cost edge may exist.
    int  dwell_cycles          = 3;
    int  dwell_max_cycles      = 12;       // bounded: a detection that never comes ends as Timeout
};

class Consumer
{
public:
    ConsumerPolicy p;
    explicit Consumer(ConsumerPolicy pol) : p(pol) {}

    [[nodiscard]] int claims() const { return claims_; }

    // Select ONE offer across every producer's node: the robot has one body. Ranked by gain minus
    // travel cost, exactly as AffordanceManager::neg_efe does.
    [[nodiscard]] int select(const std::vector<Wire> &wires, const World &world,
                             std::uint64_t cycle) const
    {
        int best = -1;
        float best_score = -1e9f;
        for (std::size_t i = 0; i < wires.size(); ++i)
        {
            const auto &w = wires[i];
            if (w.state != NodeState::Offered) continue;
            if (p.suppress_just_completed and w.cell == just_completed_ and cycle - completed_at_ < 4)
                continue;
            if (p.refusal_hold_cycles > 0)
                if (const auto it = held_.find(w.cell); it != held_.end())
                    if (cycle - it->second < static_cast<std::uint64_t>(p.refusal_hold_cycles)) continue;
            // ★KEYED ON (cell, pose, map), NOT ON "since the last move". Clearing the cache whenever
            // the robot moved sounded conservative and re-decided 173 verdicts in one scenario: the
            // robot returns to poses it has already been at, and each return re-asked a question whose
            // answer is a function of exactly these three things. A verdict is invalidated by the MAP
            // changing, which is the only thing that can change it.
            if (p.cache_map_verdicts and world.map_version == verdict_map_version_)
                if (decided_at.contains(key_of(w.cell, world.robot_x))) continue;
            if (p.silent_local_veto)
                if (const auto it = vetoed_.find(w.cell); it != vetoed_.end())
                    if (cycle - it->second < 240) continue;          // 120 s at 2 Hz, and no message
            const float d = std::fabs(world.cell(w.cell).x - world.robot_x);
            const float score = w.gain - 0.05f * d;
            if (score > best_score) { best_score = score; best = static_cast<int>(i); }
        }
        return best;
    }

    void tick(std::vector<Wire> &wires, World &world, std::uint64_t cycle)
    {
        if (p.die_after_claims > 0 and claims_ >= p.die_after_claims) return;   // stuttering consumer

        if (active_ >= 0)
        {
            step_active(wires, world, cycle);
            return;
        }
        const int idx = select(wires, world, cycle);
        if (idx < 0) return;
        auto &w = wires[static_cast<std::size_t>(idx)];
        const Cell &c = world.cell(w.cell);

        w.state = NodeState::Executing;
        w.claimed = true;
        ++claims_;
        active_ = idx;
        started_at_ = cycle;
        approach_only_ = false;

        // ── The two questions only this side can answer ────────────────────────────────────────
        if (c.outside_room)
        {
            note_map_only_verdict(w.cell, world.robot_x, world.map_version);
            finish(w, world, Outcome::OutsideRoom, cycle, /*stood=*/false);
            return;
        }
        if (c.standable == Standable::Never)
        {
            note_map_only_verdict(w.cell, world.robot_x, world.map_version);
            finish(w, world, Outcome::Infeasible, cycle, /*stood=*/false);
            return;
        }
        if (c.routable == Routable::Sealed or c.routable == Routable::CloserWithinBand)
        {
            if (p.substitute_and_claim_satisfied)
            {   // THE PHANTOM: relocate onto the robot, arrive instantly, call it the producer's cell
                finish(w, world, Outcome::Satisfied, cycle, /*stood=*/false);
                return;
            }
            note_map_only_verdict(w.cell, world.robot_x, world.map_version);
            finish(w, world, Outcome::Unreachable, cycle, /*stood=*/false);
            return;
        }
        if (c.routable == Routable::CloserExists)
        {
            if (p.substitute_and_claim_satisfied)
            {
                finish(w, world, Outcome::Satisfied, cycle, /*stood=*/false);
                return;
            }
            approach_only_ = p.approach_reports_unreachable;
        }
        travel_ = static_cast<int>(std::fabs(c.x - world.robot_x)) * p.drive_cycles_per_unit + 1;
    }

private:
    // Stamped with the pose it was decided from; the drive below drops the whole cache, because a
    // robot that has moved is looking at a different map.
    static std::pair<int, int> key_of(int cell, float robot_x)
    {
        return {cell, static_cast<int>(robot_x * 100.f)};
    }
    void note_map_only_verdict(int cell, float robot_x, int map_version)
    {
        if (map_version != verdict_map_version_) { decided_at.clear(); verdict_map_version_ = map_version; }
        const auto key = key_of(cell, robot_x);
        if (decided_at.contains(key)) ++redecided_;
        decided_at.insert(key);
    }
public:
    [[nodiscard]] int redecided() const { return redecided_; }
private:
    int redecided_ = 0;

    void step_active(std::vector<Wire> &wires, World &world, std::uint64_t cycle)
    {
        auto &w = wires[static_cast<std::size_t>(active_)];
        if (w.state != NodeState::Executing)          // the producer's lease reclaimed it under us
        {
            active_ = -1;
            return;
        }
        const Cell &c = world.cell(w.cell);
        const auto elapsed = static_cast<int>(cycle - started_at_);
        if (elapsed < travel_) return;                 // driving: physical time, by construction

        if (c.wedges)
        {
            vetoed_[w.cell] = cycle;                   // a DRIVE failure is worth remembering locally
            held_[w.cell] = cycle;
            finish(w, world, Outcome::Abandoned, cycle, /*stood=*/false);
            return;
        }

        // The robot is physically there. This is the ONLY place `stood_at` may grow — and the moment
        // the map can have changed, so the quiet period is over.
        world.robot_x = c.x;
        world.robot_at = c.id;
        world.stood_at.insert(c.id);

        if (approach_only_)
        {   // We got as close as the map allows — real work, and honestly reported as not-the-cell.
            finish(w, world, Outcome::Unreachable, cycle, /*stood=*/true);
            return;
        }
        if (not c.detects)
        {
            if (elapsed < travel_ + p.dwell_max_cycles) return;    // bounded wait, then say so
            finish(w, world, Outcome::Timeout, cycle, /*stood=*/true);
            return;
        }
        if (elapsed < travel_ + p.dwell_cycles) return;            // acquisition costs wall-clock
        finish(w, world, Outcome::Satisfied, cycle, /*stood=*/true);
    }

    void finish(Wire &w, World &world, Outcome o, std::uint64_t cycle, bool stood)
    {
        if (observation_happened(o))
        {
            world.observed.insert(w.cell);
            ++world.observations;
            (void)stood;                              // the checker compares observed against stood_at
        }
        if (o == Outcome::Refused or o == Outcome::Unreachable or o == Outcome::Infeasible)
            held_[w.cell] = cycle;
        just_completed_ = w.cell;
        completed_at_ = cycle;
        w.outcome = o;
        w.state = NodeState::Completed;
        w.claimed = false;
        active_ = -1;
    }

    std::map<int, std::uint64_t> held_, vetoed_;
    int verdict_map_version_ = 0;
public:
    // (cell, pose) pairs already decided — the checker's view of "did we re-decide anything".
    std::set<std::pair<int, int>> decided_at;
private:
    int claims_ = 0;
    int active_ = -1;
    int travel_ = 0;
    int just_completed_ = -1;
    bool approach_only_ = false;
    std::uint64_t completed_at_ = 0, started_at_ = 0;
};

// ── RUN + PROPERTIES ────────────────────────────────────────────────────────────────────────────
struct Result
{
    int observations = 0;
    int completions  = 0;
    std::uint64_t cycles = 0;

    // P1 no phantom: every cell a producer was told about was one the robot actually stood at.
    bool phantom = false;
    int  phantom_count = 0;
    // P1b belief integrity: and every cell a producer BELIEVES explored was stood at too. The two
    // differ exactly when a producer updates from a protocol event instead of from sensing.
    int  believed_unvisited = 0;
    // P2 progress: observations keep happening — checked on the LAST THIRD of the run, because
    // "at least one" is satisfied by one arrival followed by an hour of silence, which is exactly
    // what the viewer showed.
    int  observations_last_third = 0;
    // P3 fairness: every producer holding a servable cell was served at least once.
    std::vector<std::string> starved;
    // P4 no livelock. ★NOT "completions without observations" — that conflates honest work with
    // churn: a drive to a standpoint whose detection never arrives costs the robot seconds and ends
    // in a Timeout, and a run of those is the system working. The failure is completions that cost
    // the robot NOTHING, decided from an unchanged map at loop rate. Measured live 2026-08-19:
    // 9677 `unreachable` facts in 20 minutes, ~8 per second, base commanded to zero throughout.
    int  longest_free_run = 0;           // consecutive completions with no motion between them
    // ★THE EXACT PROPERTY, with no constant in it: a verdict read off the map from a given pose must
    // never be re-decided from that same pose. A run of free completions is FINE as long as each one
    // decides a different cell — that run is bounded by the number of cells, and the next one costs a
    // drive. Comparing the run length against an arbitrary 12 was measuring the world's size.
    int  redecided = 0;
    bool deadlocked = false;
};

inline Result run(std::vector<Producer> &producers, Consumer &cons, World &world,
                  std::uint64_t cycles)
{
    std::vector<Wire> wires(producers.size());
    Result r;
    int obs_at_two_thirds = 0;
    int barren = 0;
    std::uint64_t unchanged = 0;
    std::vector<Wire> last = wires;

    for (std::uint64_t c = 0; c < cycles; ++c)
    {
        std::vector<Outcome> before(wires.size());
        for (std::size_t i = 0; i < wires.size(); ++i) before[i] = wires[i].outcome;
        const auto stood_before = world.stood_at.size();

        cons.tick(wires, world, c);
        // ★COUNTED BEFORE THE PRODUCER READS IT. The producer consumes the outcome (sets it back to
        // None) in the same cycle, so counting after both ticks read zero completions in every
        // scenario — a counter whose only possible value was 0, reported as if it were data.
        for (std::size_t i = 0; i < wires.size(); ++i)
            if (before[i] == Outcome::None and wires[i].outcome != Outcome::None)
            {
                ++r.completions;
                if (world.stood_at.size() != stood_before) barren = 0;      // the body moved: real work
                else { ++barren; r.longest_free_run = std::max(r.longest_free_run, barren); }
            }
        for (std::size_t i = 0; i < producers.size(); ++i) producers[i].tick(wires[i], c);

        if (c == cycles * 2 / 3) obs_at_two_thirds = world.observations;

        bool same = true;
        for (std::size_t i = 0; i < wires.size(); ++i)
            if (wires[i].state != last[i].state or wires[i].cell != last[i].cell
                or wires[i].outcome != last[i].outcome) { same = false; break; }
        unchanged = same ? unchanged + 1 : 0;
        last = wires;
        if (unchanged > 400) { r.deadlocked = true; r.cycles = c + 1; break; }
        r.cycles = c + 1;
    }

    r.redecided = cons.redecided();
    r.observations = world.observations;
    r.observations_last_third = world.observations - obs_at_two_thirds;

    // ★THE CHECK THAT WOULD HAVE CAUGHT THE PHANTOM ON THE FIRST RUN. Anything a producer booked as
    // observed must appear in the ledger of places the body actually was, with multiplicity.
    auto stood = world.stood_at;
    for (int cell : world.observed)
    {
        const auto it = stood.find(cell);
        if (it == stood.end()) { r.phantom = true; ++r.phantom_count; }
        else stood.erase(it);
    }

    for (const auto &prod : producers)
        for (int cell : prod.believed())
            if (not world.stood_at.contains(cell)) ++r.believed_unvisited;

    for (const auto &prod : producers)
    {
        bool has_servable = false;
        for (int cid : prod.cells())
        {
            const auto &cell = world.cell(cid);
            // ★SERVABLE MEANS OBSERVABLE, not merely approachable. A cell whose only access is an
            // approach to a closer pose can never be stood at, so it can never be observed — counting
            // it made "chair starved" a checker bug rather than a protocol failure. A property that
            // is wrong in the direction of alarm still costs a debugging round.
            if (cell.standable != Standable::Never and cell.routable == Routable::Yes
                and not cell.wedges and cell.detects and not cell.outside_room)
                has_servable = true;
        }
        if (has_servable and prod.served() == 0) r.starved.push_back(prod.label);
    }
    return r;
}

}  // namespace rc::affordance::multi
