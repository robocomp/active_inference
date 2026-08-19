/*
 *  affordance_goal_parse.h — the parts of the affordance contract that are PURE.
 *
 *  Split out of affordance_protocol.h for one reason: everything here is a decision about the
 *  contract's own consistency, it needs no DSR type to make that decision, and it was previously
 *  unreachable by a test because it sat inline in a header that pulls in the whole graph API.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────────────
 *  WHY REJECTION, NOT REPAIR
 *
 *  The completion predicate is what stops an affordance. Every way of silently mis-parsing it makes
 *  it WEAKER or INVERTS it, and both mean the affordance reports success without the thing having
 *  happened. Two live examples from the wire format as it stood:
 *
 *    · the clause zip ran `for (i < names.size() && i < vals.size())`, so a producer writing three
 *      attribute names and two values silently lost the third CLAUSE. Fewer clauses = easier to
 *      satisfy = completes early.
 *    · a clause whose operator was missing fell back to GE. So `sigma LE 0.01` — "wait until the
 *      uncertainty is SMALL" — silently became `sigma GE 0.01`, which is true exactly when the
 *      uncertainty is LARGE. The predicate does not weaken there, it INVERTS, and the affordance
 *      completes immediately on the condition it was written to wait out.
 *
 *  Neither announced itself. A truncated or inverted predicate looks identical to a satisfied one
 *  from the outside — the affordance simply completes, and the producer books an observation it
 *  never got. So the rule here is: a clause set that does not round-trip EXACTLY is refused whole,
 *  the caller keeps the conservative type default, and the reason is reported. Refusing is safe
 *  (nothing completes that should not); repairing is not.
 */
#pragma once

#include <cmath>
#include <string>
#include <string_view>
#include <vector>

namespace rc::affordance
{

enum class CompareOp { GE, LE, EQ, NE };

struct GoalClause
{
    std::string attr;                 // attribute name on the feedback node
    CompareOp   op    = CompareOp::GE;
    float       value = 0.0f;
};

// The wire packs clauses as three parallel arrays joined by '|'. That is the format; this is the
// only place allowed to interpret it.
inline constexpr char kClauseDelim = '|';

inline std::string_view to_string(CompareOp o)
{
    switch (o) { case CompareOp::LE: return "le"; case CompareOp::EQ: return "eq"; case CompareOp::NE: return "ne"; default: return "ge"; }
}
// Strict: an unrecognised operator is NOT silently GE — see the file header. Returns false on a
// token this format does not define, so the caller can refuse the whole clause set.
inline bool op_from_strict(std::string_view s, CompareOp& out)
{
    if (s == "ge") { out = CompareOp::GE; return true; }
    if (s == "le") { out = CompareOp::LE; return true; }
    if (s == "eq") { out = CompareOp::EQ; return true; }
    if (s == "ne") { out = CompareOp::NE; return true; }
    return false;
}

struct ClauseParse
{
    std::vector<GoalClause>  clauses;
    std::vector<std::string> problems;   // empty ⇒ the set round-trips exactly
    [[nodiscard]] bool ok() const { return problems.empty(); }
};

/// Zip the three wire arrays into clauses, refusing anything that does not correspond exactly.
///
/// `ops` may be EMPTY, meaning "every clause is GE" — that is a legal encoding and the only
/// abbreviation allowed. A PARTIAL ops list is refused: there is no way to tell which clauses it was
/// meant to cover, and guessing is what inverted the predicate above.
[[nodiscard]] inline ClauseParse parse_goal_clauses(const std::vector<std::string>& names,
                                                    const std::vector<std::string>& ops,
                                                    const std::vector<float>&       values)
{
    ClauseParse out;
    const auto n = names.size();

    if (n != values.size())
        out.problems.push_back("clause count mismatch: " + std::to_string(n) + " attribute name(s) but "
                               + std::to_string(values.size()) + " value(s)");
    if (not ops.empty() and ops.size() != n)
        out.problems.push_back("operator count mismatch: " + std::to_string(ops.size()) + " operator(s) for "
                               + std::to_string(n) + " clause(s) — a partial list cannot be assigned");
    if (not out.ok())
        return out;                      // sizes are wrong; indexing below would be meaningless

    for (std::size_t i = 0; i < n; ++i)
    {
        if (names[i].empty())
            out.problems.push_back("clause " + std::to_string(i) + " has an empty attribute name");
        else if (names[i].find(kClauseDelim) != std::string::npos)
            out.problems.push_back("clause " + std::to_string(i) + " attribute '" + names[i] + "' contains '"
                                   + kClauseDelim + "', the unescaped field delimiter — it would desynchronise "
                                   "every clause after it");
        if (not std::isfinite(values[i]))
            out.problems.push_back("clause " + std::to_string(i) + " ('" + names[i] + "') has a non-finite value");

        CompareOp op = CompareOp::GE;
        if (not ops.empty() and not op_from_strict(ops[i], op))
            out.problems.push_back("clause " + std::to_string(i) + " ('" + names[i] + "') has unknown operator '"
                                   + ops[i] + "' — refusing rather than defaulting to 'ge', which would invert "
                                   "an 'le' clause");
        out.clauses.push_back({names[i], op, values[i]});
    }
    if (not out.ok())
        out.clauses.clear();
    return out;
}

/// Parity rule for any pair of arrays the wire carries as "parallel". Used for the viewpoint
/// constraint's faces/face_gains, which a consumer zips to pick the best face: a length mismatch
/// silently re-pairs every face with the WRONG gain, so the argmax names a face nobody scored.
[[nodiscard]] inline bool parallel_ok(std::size_t a, std::size_t b) { return a == b or b == 0; }


// ─── contract-level validation ────────────────────────────────────────────────
// Structural mistakes that make a contract behave in a way the author did not intend, checked at
// AUTHORING time. Reported, never fatal: this runs on the graph-publish path, where aborting costs
// more than a bad contract. Kept pure (no Contract type) so it can be tested and so the protocol
// header stays the only place that knows about DSR.
struct ContractShape
{
    bool  is_reach          = false;   // Reach ignores the servo bindings entirely
    bool  has_servo_binding = false;   // err_vec_attr or scalar_attr set
    int   stable_n    = 1;
    float timeout_ms  = 0.0f;
    float max_observe_omega = 0.0f;
    bool  is_orient   = false;
};

[[nodiscard]] inline std::vector<std::string>
validate_contract(const ContractShape& shape, const std::vector<GoalClause>& goal)
{
    std::vector<std::string> problems;

    if (not shape.is_reach and goal.empty())
        problems.emplace_back("servo/orient contract has no completion clause — it can never complete");
    if (shape.is_reach and shape.has_servo_binding)
        problems.emplace_back("Reach contract carries servo bindings (center/advance) — the executor "
                              "ignores them, so they are a silent no-op rather than the behaviour authored");
    if (shape.stable_n < 1)
        problems.emplace_back("stable_n < 1: the predicate is never required to hold, so the first "
                              "cycle that happens to satisfy it completes the affordance");
    if (shape.timeout_ms <= 0.0f)
        problems.emplace_back("timeout_ms <= 0: nothing bounds the wait, so a predicate that can never "
                              "be satisfied parks the robot on this affordance indefinitely");

    // ★ Two clauses on the SAME attribute can be jointly unsatisfiable, and a contract that can never
    // complete is indistinguishable at runtime from one that is merely slow — it just runs to timeout,
    // every time, and the producer books a failure it cannot explain. Cheap to detect here.
    for (std::size_t i = 0; i < goal.size(); ++i)
        for (std::size_t j = i + 1; j < goal.size(); ++j)
        {
            if (goal[i].attr != goal[j].attr) continue;
            const auto& a = goal[i]; const auto& b = goal[j];
            const bool lo_hi = (a.op == CompareOp::GE and b.op == CompareOp::LE and b.value < a.value)
                            or (a.op == CompareOp::LE and b.op == CompareOp::GE and a.value < b.value);
            const bool eq_ne = (a.op == CompareOp::EQ and b.op == CompareOp::NE and a.value == b.value)
                            or (a.op == CompareOp::NE and b.op == CompareOp::EQ and a.value == b.value);
            const bool eq_eq = (a.op == CompareOp::EQ and b.op == CompareOp::EQ and a.value != b.value);
            if (lo_hi or eq_ne or eq_eq)
                problems.push_back("clauses " + std::to_string(i) + " and " + std::to_string(j)
                                   + " on '" + a.attr + "' are jointly unsatisfiable — this contract "
                                     "can only ever end in timeout");
        }
    return problems;
}


// ─── how an affordance ENDED ──────────────────────────────────────────────────
// Stamped by the executor at the terminal transition (aff_outcome), read by the producer.
// ★ Completed alone conflates outcomes that mean OPPOSITE things to a producer: "I looked and saw"
// against "I gave up". A producer that cannot tell them apart books an observation it never got and
// retires the very affordance that would have gone back for it. Three agents made that conflation,
// and they made it because the wire offered no way to distinguish the cases.
enum class Outcome
{
    None,        // not terminal yet (attribute absent or empty)
    Satisfied,   // the completion predicate held for stable_n cycles — an OBSERVATION happened
    Timeout,     // the predicate never held within aff_timeout_ms — nothing was observed
    Refused,     // the consumer could not get there (see epistemic_refused_*) — never attempted
    Abandoned,   // an operator or a higher-priority interrupt ended it
    // ── FACTS THE CONSUMER OWNS AND THE PRODUCER CANNOT SEE (added 2026-08-19) ───────────────────
    // ★★★THE CELL IS THE PRODUCER'S DECISION VARIABLE. It chose that standpoint by maximising expected
    // information gain over its own grid; the consumer knows nothing about that gain and everything
    // about reachability. Before these existed the consumer had no way to say "I cannot", so when a
    // standpoint was unreachable it silently MOVED the goal to somewhere it could reach and reported
    // Satisfied — measured 2026-08-19: 140 of 163 accepted arrivals were a median 2.86 m from the
    // published cell, each one telling room a place had been observed that nothing ever looked at.
    // ★These are FACTS, not verdicts: mechanical statements about the approach that the consumer alone
    // can make. Whether the epistemic goal was served remains the producer's to compute, privately.
    Infeasible,  // the body does not fit at that pose (or cannot turn there) — nothing was observed
    Unreachable  // the pose is fine but no route exists from where the robot is — never attempted
};

inline std::string_view to_string(Outcome o)
{
    switch (o)
    {
        case Outcome::Satisfied: return "satisfied";
        case Outcome::Timeout:   return "timeout";
        case Outcome::Refused:     return "refused";
        case Outcome::Abandoned:   return "abandoned";
        case Outcome::Infeasible:  return "infeasible";
        case Outcome::Unreachable: return "unreachable";
        default:                 return "";
    }
}
inline Outcome outcome_from(std::string_view s)
{
    if (s == "satisfied") return Outcome::Satisfied;
    if (s == "timeout")   return Outcome::Timeout;
    if (s == "refused")   return Outcome::Refused;
    if (s == "abandoned") return Outcome::Abandoned;
    if (s == "infeasible")  return Outcome::Infeasible;
    if (s == "unreachable") return Outcome::Unreachable;
    // ★An UNKNOWN word reads as None, and observation_happened(None) is false — a producer running
    // against a newer consumer that reports a fact it has never heard of treats it as "nothing was
    // observed", which at worst sends the robot back to look again. That is the safe direction, and it
    // is why these words could be added without a lock-step upgrade of both agents.
    return Outcome::None;
}

/// Did anything get OBSERVED? The single question a producer asks, and the reason this enum exists.
/// ★ Deliberately not "did it complete": every non-Satisfied outcome, INCLUDING an unrecognised one
/// (Outcome::None from a future or malformed value), answers false. A producer running against a
/// newer executor that stamps an outcome it does not know must not treat that as evidence — the safe
/// reading of "I do not understand what happened" is "nothing was observed", which at worst sends the
/// robot back to look again.
[[nodiscard]] inline bool observation_happened(Outcome o) { return o == Outcome::Satisfied; }

}  // namespace rc::affordance
