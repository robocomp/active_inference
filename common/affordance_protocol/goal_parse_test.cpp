// Standalone test for the PURE half of the affordance contract (affordance_goal_parse.h).
// No DSR, no Ice, no Qt — builds in a second:
//     g++ -std=c++23 -O1 goal_parse_test.cpp -o /tmp/goal_parse_test && /tmp/goal_parse_test
#include "affordance_goal_parse.h"

#include <cstdio>
#include <string>
#include <vector>

namespace
{
int failures = 0;
void check(const char* name, bool ok, const std::string& detail = {})
{
    std::printf("  %-58s %s%s%s\n", name, ok ? "PASS" : "FAIL",
                detail.empty() ? "" : "   ", detail.c_str());
    if (not ok) ++failures;
}
using namespace rc::affordance;
}

int main()
{
    std::printf("\naffordance goal-clause parsing\n");

    // ── the format as producers actually write it ──────────────────────────────
    {
        const auto p = parse_goal_clauses({"a", "b"}, {"ge", "le"}, {0.5f, 0.2f});
        check("well-formed set parses", p.ok() and p.clauses.size() == 2
              and p.clauses[1].op == CompareOp::LE);
    }
    {
        // Empty ops is the one legal abbreviation: every clause is GE.
        const auto p = parse_goal_clauses({"a", "b"}, {}, {0.5f, 0.2f});
        check("empty operator list means all-GE", p.ok() and p.clauses.size() == 2
              and p.clauses[0].op == CompareOp::GE and p.clauses[1].op == CompareOp::GE);
    }

    // ── the two live defects this file exists to stop ──────────────────────────
    {
        // WAS: zipped to the shorter array, silently losing clause 2. Fewer clauses = easier to
        // satisfy = the affordance completes without the thing having happened.
        const auto p = parse_goal_clauses({"a", "b", "c"}, {"ge", "ge", "ge"}, {0.5f, 0.2f});
        check("3 names / 2 values is REFUSED, not truncated", not p.ok() and p.clauses.empty(),
              p.problems.empty() ? "" : p.problems.front());
    }
    {
        // WAS: a missing operator defaulted to GE, so `sigma LE 0.01` ("wait until small") became
        // `sigma GE 0.01` ("proceed while large"). Not weaker — INVERTED.
        const auto p = parse_goal_clauses({"sigma", "b"}, {"le"}, {0.01f, 1.0f});
        check("partial operator list is REFUSED, never defaulted", not p.ok() and p.clauses.empty(),
              p.problems.empty() ? "" : p.problems.front());
    }
    {
        const auto p = parse_goal_clauses({"sigma"}, {"lt"}, {0.01f});
        check("unknown operator is REFUSED, not silently GE", not p.ok() and p.clauses.empty());
    }
    {
        // The delimiter is unescaped, so a name containing it desynchronises every later clause.
        const auto p = parse_goal_clauses({"a|b"}, {"ge"}, {1.0f});
        check("attribute containing the field delimiter is REFUSED", not p.ok());
    }
    {
        const auto p = parse_goal_clauses({"a"}, {"ge"}, {std::nanf("")});
        check("non-finite threshold is REFUSED", not p.ok());
    }

    // ── parallel-array parity (viewpoint faces vs their gains) ─────────────────
    check("faces/gains parity: equal lengths ok", parallel_ok(4, 4));
    check("faces/gains parity: empty gains is the legal 'unscored'", parallel_ok(4, 0));
    check("faces/gains parity: 4 faces vs 3 gains rejected", not parallel_ok(4, 3));

    std::printf("\ncontract validation\n");
    {
        ContractShape sh{.is_reach = false, .stable_n = 1, .timeout_ms = 1000.f};
        check("servo with no clause is reported", validate_contract(sh, {}).size() == 1);
    }
    {
        // ★A BEARING-ONLY ORIENT IS LEGAL. "Rotate in place to this yaw" is a complete instruction, and
        // it is what a calibration pivot publishes. This used to be warned about as "can never
        // complete" while the executor in fact completed it on cycle ONE, standing still — the warning
        // and the behaviour were wrong in opposite directions, which is why neither exposed the other.
        ContractShape sh{.is_reach = false, .stable_n = 1, .timeout_ms = 1000.f, .is_orient = true};
        check("orient with no clause is LEGAL — the rotation is the goal",
              validate_contract(sh, {}).empty());
    }
    {
        // …and everything else still applies to it: legal is not exempt.
        ContractShape sh{.is_reach = false, .stable_n = 1, .timeout_ms = 0.f, .is_orient = true};
        check("a bearing-only orient with no timeout is still reported",
              not validate_contract(sh, {}).empty());
    }
    {
        ContractShape sh{.is_reach = true, .has_servo_binding = true, .stable_n = 1, .timeout_ms = 1000.f};
        check("Reach carrying servo bindings is reported (silent no-op)",
              not validate_contract(sh, {{"a", CompareOp::GE, 1.f}}).empty());
    }
    {
        ContractShape sh{.is_reach = false, .stable_n = 0, .timeout_ms = 1000.f};
        check("stable_n < 1 is reported", not validate_contract(sh, {{"a", CompareOp::GE, 1.f}}).empty());
    }
    {
        ContractShape sh{.is_reach = false, .stable_n = 1, .timeout_ms = 0.f};
        check("no timeout is reported (parks the robot for ever)",
              not validate_contract(sh, {{"a", CompareOp::GE, 1.f}}).empty());
    }
    {
        // A contract that can never complete looks identical at runtime to one that is merely slow.
        ContractShape sh{.is_reach = false, .stable_n = 1, .timeout_ms = 1000.f};
        const std::vector<GoalClause> g{{"x", CompareOp::GE, 0.9f}, {"x", CompareOp::LE, 0.1f}};
        check("jointly unsatisfiable clauses on one attribute are reported",
              not validate_contract(sh, g).empty());
    }
    {
        ContractShape sh{.is_reach = false, .stable_n = 1, .timeout_ms = 1000.f};
        const std::vector<GoalClause> g{{"x", CompareOp::GE, 0.1f}, {"x", CompareOp::LE, 0.9f}};
        check("a legitimate band on one attribute is NOT reported", validate_contract(sh, g).empty());
    }

    std::printf("\noutcome vocabulary\n");
    {
        bool round_trip = true;
        for (auto o : {Outcome::Satisfied, Outcome::Timeout, Outcome::Refused, Outcome::Abandoned})
            round_trip = round_trip and outcome_from(to_string(o)) == o;
        check("every outcome round-trips through its wire string", round_trip);
    }
    check("an absent outcome reads as None", outcome_from("") == Outcome::None);
    // ★ The safe reading of "I do not understand what happened" is "nothing was observed": a producer
    // built against an older protocol must not treat an outcome it cannot parse as evidence.
    check("an UNKNOWN outcome reads as None, not as success",
          outcome_from("some_future_value") == Outcome::None);
    {
        bool only_satisfied = observation_happened(Outcome::Satisfied)
                          and not observation_happened(Outcome::Timeout)
                          and not observation_happened(Outcome::Refused)
                          and not observation_happened(Outcome::Abandoned)
                          and not observation_happened(Outcome::None);
        check("ONLY Satisfied counts as an observation", only_satisfied);
    }

    std::printf("\n%s (%d failure%s)\n\n", failures == 0 ? "ALL PASS" : "FAILURES",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
