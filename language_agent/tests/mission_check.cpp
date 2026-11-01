// mission_check — reads a planner reply on stdin, prints `OK` then one line per step in pre-order
// (`DEPTH op key=value … [bind=NAME] [on_fail=…]`), or `ERROR <message>`. Used by tests/check_llm_missions.py.
// Refs print as node:NAME · var:NAME · anchor:NAME · passage:NAME@ROOM · pose:ROOM; strings are quoted.
#include "../src/mission_plan.h"

#include <iostream>
#include <iterator>
#include <print>

namespace
{
using namespace rc::mission;

std::string ref_str(const Ref& r)
{
    switch (r.kind)
    {
        case Ref::Kind::Node: return "node:" + r.name;
        case Ref::Kind::Passage: return "passage:" + r.name + "@" + r.from;
        case Ref::Kind::Anchor: return "anchor:" + r.name;
        case Ref::Kind::Pose: return "pose:" + r.pose.room;
        case Ref::Kind::Var: return "var:" + r.name;
    }
    return "?";
}

std::string pred_str(const Pred& p)
{
    std::string s = p.name + "(";
    bool first = true;
    for (const auto& [k, r] : p.refs)
    {
        s += (first ? "" : ",") + ref_str(r);
        first = false;
    }
    s += ")";
    if (p.cmp.has_value()) s += std::format("{}{}", p.cmp->op == Comparison::Op::Ge ? ">=" : "<", p.cmp->value);
    return s;
}

void print_steps(const std::vector<Step>& list, int depth = 0)
{
    for (const auto& s : list)
    {
        std::string line = std::format("{} {}", depth, s.op);
        for (const auto& [k, v] : s.args)
        {
            line += " " + k + "=";
            line += std::visit(
                [](const auto& x) -> std::string
                {
                    using T = std::decay_t<decltype(x)>;
                    if constexpr (std::is_same_v<T, double> or std::is_same_v<T, std::int64_t>) return std::format("{}", x);
                    else if constexpr (std::is_same_v<T, std::string>) return "\"" + x + "\"";
                    else if constexpr (std::is_same_v<T, Ref>) return ref_str(x);
                    else return pred_str(x);
                },
                v);
        }
        if (s.bind.has_value()) line += " bind=" + *s.bind;
        if (s.on_fail.kind == OnFail::Kind::Skip) line += " on_fail=skip";
        if (s.on_fail.kind == OnFail::Kind::Retry) line += std::format(" retry={}", s.on_fail.retries);
        std::println("{}", line);
        print_steps(s.steps, depth + 1);
    }
}
}   // namespace

int main()
{
    const std::string text{std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()};
    const auto m = rc::chat::parse_planned(text);
    if (not m.has_value())
    {
        std::println("ERROR {}", m.error());
        return 0;
    }
    std::println("OK");
    print_steps(m->steps);
    return 0;
}
