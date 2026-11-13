#include "human.h"
#include "repertory.h"

#include <algorithm>
#include <ranges>

namespace ltom
{
std::optional<ClaimKind> parse_claim_kind(std::string_view s)
{
    if (s == "name") return ClaimKind::name;
    if (s == "owns") return ClaimKind::owns;
    if (s == "disowns") return ClaimKind::disowns;
    return std::nullopt;
}

const char* claim_kind_name(ClaimKind k)
{
    switch (k)
    {
        case ClaimKind::name: return "name";
        case ClaimKind::owns: return "owns";
        case ClaimKind::disowns: return "disowns";
    }
    return "name";
}

std::optional<std::string> HumanBook::name() const
{
    if (human_.names.empty()) return std::nullopt;
    return human_.names.back().value;
}

std::vector<const Ownership*> HumanBook::owned() const
{
    std::vector<const Ownership*> out;
    for (const auto& o : human_.things)
        if (not o.retracted.has_value()) out.push_back(&o);
    return out;
}

Answer HumanBook::apply(const Claim& c, const Repertory& r, std::int64_t now_ms)
{
    // The chat leaves its claim up until it sees the answer, so LTOM reads the same claim every cycle: same stamp,
    // same answer, nothing re-applied.
    if (last_ and last_->stamp == c.stamp) return *last_;

    const Said said{now_ms, c.words};
    auto answer = [&](std::string text) { last_ = Answer{c.stamp, std::move(text)}; return *last_; };
    auto live = std::ranges::find_if(human_.things, [&](const Ownership& o)
                                     { return o.object_uid == c.object_uid and not o.retracted.has_value(); });
    switch (c.kind)
    {
        case ClaimKind::name:
            if (c.value.empty()) return answer("rejected: no name");
            human_.names.push_back(NameRecord{c.value, said});
            dirty_ = true;
            return answer("recorded");
        case ClaimKind::owns:
        {
            const auto e = r.entries().find(c.object_uid);
            if (e == r.entries().end()) return answer("rejected: I have no memory of that object");
            if (live != human_.things.end()) return answer("recorded");
            human_.things.push_back(Ownership{c.object_uid, e->second.cls, e->second.name, said, std::nullopt});
            dirty_ = true;
            return answer("recorded");
        }
        case ClaimKind::disowns:
            if (live == human_.things.end()) return answer("rejected: it was not yours");
            live->retracted = said;
            dirty_ = true;
            return answer("recorded");
    }
    return answer("rejected: unknown claim");
}

void HumanBook::set(Human h)
{
    human_ = std::move(h);
    dirty_ = false;
}
}
