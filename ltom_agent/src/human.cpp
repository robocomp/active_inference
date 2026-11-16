#include "human.h"
#include "repertory.h"

#include <algorithm>
#include <charconv>
#include <ranges>

namespace ltom
{
std::optional<ClaimKind> parse_claim_kind(std::string_view s)
{
    if (s == "name") return ClaimKind::name;
    if (s == "owns") return ClaimKind::owns;
    if (s == "disowns") return ClaimKind::disowns;
    if (s == "moved") return ClaimKind::moved;
    if (s == "forget") return ClaimKind::forget;
    return std::nullopt;
}

const char* claim_kind_name(ClaimKind k)
{
    switch (k)
    {
        case ClaimKind::name: return "name";
        case ClaimKind::owns: return "owns";
        case ClaimKind::disowns: return "disowns";
        case ClaimKind::moved: return "moved";
        case ClaimKind::forget: return "forget";
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

Belief& HumanBook::belief(std::string_view key)
{
    const auto it = std::ranges::find(human_.beliefs, key, &Belief::key);
    if (it != human_.beliefs.end()) return *it;
    human_.beliefs.push_back(Belief{std::string(key), {}, {}});
    return human_.beliefs.back();
}

Answer HumanBook::apply(const Claim& c, Repertory& r, std::int64_t now_ms)
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
        case ClaimKind::moved:
        {
            const auto colon = c.value.find(':');
            const std::string_view verdict = std::string_view(c.value).substr(0, colon);
            std::uint64_t q = 0;
            if (colon == std::string::npos) return answer("rejected: the answer names no question");
            const char* first = c.value.data() + colon + 1;
            const char* last = c.value.data() + c.value.size();
            if (const auto [p, ec] = std::from_chars(first, last, q); ec != std::errc{} or p != last)
                return answer("rejected: the answer names no question");
            if (verdict != "yes" and verdict != "no" and verdict != "unknown") return answer("rejected: not a yes, a no or an I don't know");
            if (not r.close_question(c.object_uid, q)) return answer("rejected: that question is no longer open");
            if (verdict == "yes")
            {
                r.confirm_link(c.object_uid);
                belief("moves_things").records.push_back(BeliefRecord{said, true, c.object_uid});
                belief("gets_moved:" + std::to_string(c.object_uid)).records.push_back(BeliefRecord{said, true, c.object_uid});
            }
            else if (verdict == "no")
            {
                r.cut_link(c.object_uid, now_ms, c.words.empty() ? std::string("no") : c.words);
                belief("moves_things").records.push_back(BeliefRecord{said, false, c.object_uid});
            }
            dirty_ = true;
            return answer("recorded");
        }
        case ClaimKind::forget:
        {
            const auto it = std::ranges::find(human_.beliefs, c.value, &Belief::key);
            if (it == human_.beliefs.end()) return answer("rejected: I had not learned that");
            it->records.clear();
            it->forgotten.push_back(said);
            dirty_ = true;
            return answer("recorded");
        }
    }
    return answer("rejected: unknown claim");
}

void HumanBook::set(Human h)
{
    human_ = std::move(h);
    dirty_ = false;
}
}
