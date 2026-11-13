#pragma once
/*
 * human_claim.h — the chat side of the human-at-the-chat protocol (spec 2026-11-11-human-model), pure parts.
 * A claim (my name is …, that is my chair, that is not my chair) is shown to the person as a read-back card and
 * posted on chat_intention ONLY after their yes; ltom_agent records it and answers on human_1. Nothing here stores.
 */

#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <vector>

#include "object_facts.h"
#include "human_refs.h"

namespace rc::chat
{

/// How long the chat waits for LTOM's answer before telling the person nothing was stored. An OPERATIONAL timeout
/// (LTOM not running), not a modelling threshold — flagged as such in the spec (call 3).
constexpr std::int64_t kClaimAnswerWaitMs = 5000;

struct HumanClaim
{
    std::string kind;            ///< "name" | "owns" | "disowns"
    std::string value{};         ///< the name, for kind name
    std::uint64_t uid = 0;       ///< object_uid, for owns / disowns
    std::string words{};         ///< the person's own words
};

/// "chair_34, the magenta one ahead" / "chair_35, to the right" — the robot's own words for a perceived object.
inline std::string object_words(const ObjectFact& o)
{
    const std::string where = o.bearing_deg.has_value() ? bearing_words(*o.bearing_deg) : std::string{};
    if (not o.color_name.empty())
        return std::format("{}, the {} one{}", o.name, o.color_name, where.empty() ? "" : " " + where);
    return where.empty() ? o.name : std::format("{}, {}", o.name, where);
}

/// "chair_36, the yellow one I remember in room_2" — for an object only the long-term memory has.
inline std::string remembered_words(const RememberedObject& o)
{
    const std::string room = o.room.empty() ? std::string{} : " in " + o.room;
    return o.color_name.empty() ? std::format("{}, the one I remember{}", o.name, room)
                                : std::format("{}, the {} one I remember{}", o.name, o.color_name, room);
}

/// The card's question. Answered by a plain yes, or by anything else (which cancels it).
inline std::string claim_readback(const HumanClaim& c, const std::string& what)
{
    if (c.kind == "name") return std::format("Your name is {}?", c.value);
    if (c.kind == "disowns") return std::format("{}, is NOT yours?", what);
    return std::format("{}, is yours?", what);
}

/// What the chat says when LTOM answers "recorded".
inline std::string claim_recorded_words(const HumanClaim& c, const std::string& object_name)
{
    if (c.kind == "name") return std::format("I will remember that your name is {}.", c.value);
    return std::format("I will remember that {} is {}yours.", object_name, c.kind == "disowns" ? "not " : "");
}

/// "Which one do you mean: a, b, or c?"
inline std::string which_question(const std::vector<std::string>& described)
{
    std::string s = "Which one do you mean: ";
    for (std::size_t i = 0; i < described.size(); ++i)
        s += (i == 0 ? "" : (i + 1 == described.size() ? ", or " : ", ")) + described[i];
    return s + "?";
}

/// What a claim is about. owns: "that"/"this"/"that chair"/"the magenta one"/"my chair" all mean what the person sees
/// (in view, named lately, the last approach target), leaving out what they already own; a name ("chair_34") is that
/// object. disowns "my chair": their owned objects of the class. Every candidate is described by its UID. More than
/// one = the caller asks which (which_question); none = it says it cannot tell.
struct ClaimCandidate { std::string name; std::uint64_t uid = 0; std::string words; };
inline std::vector<ClaimCandidate> claim_candidates(const std::string& kind, std::string object, const HumanFacts& h,
                                                    const std::vector<ObjectFact>& live,
                                                    const std::vector<RememberedObject>& remembered,
                                                    std::span<const std::string> recent_names,
                                                    const std::optional<std::string>& last_target, double half_fov_deg)
{
    std::ranges::transform(object, object.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    while (not object.empty() and object.back() == ' ') object.pop_back();
    while (not object.empty() and object.front() == ' ') object.erase(object.begin());
    const auto describe = [&](std::uint64_t uid, bool is_live) -> std::optional<ClaimCandidate>
    {
        if (is_live)
            if (const auto f = std::ranges::find(live, uid, &ObjectFact::uid); f != live.end()) return ClaimCandidate{f->name, uid, object_words(*f)};
        if (const auto r = std::ranges::find(remembered, uid, &RememberedObject::uid); r != remembered.end())
            return ClaimCandidate{r->name, uid, remembered_words(*r)};
        return std::nullopt;
    };
    std::vector<ClaimCandidate> out;
    const auto cls = my_class(object);
    if (cls.has_value() and kind == "disowns")
    {
        for (const auto& c : my_candidates(*cls, h, live, remembered))
            if (auto d = describe(c.uid, c.live)) out.push_back(*d);
        return out;
    }
    if (not cls.has_value())   // a name: the person names what they see first, then what is remembered
    {
        if (const auto f = std::ranges::find(live, object, &ObjectFact::name); f != live.end())
            return {ClaimCandidate{f->name, f->uid, object_words(*f)}};
        if (const auto r = std::ranges::find(remembered, object, &RememberedObject::name); r != remembered.end())
            return {ClaimCandidate{r->name, r->uid, remembered_words(*r)}};
    }
    std::string rest = cls.has_value() ? *cls : object;
    for (const std::string_view p : {"that ", "this ", "the ", "esa ", "esta ", "ese ", "este "})
        if (rest.starts_with(p)) { rest = rest.substr(p.size()); break; }
    const auto [colour, what] = split_colour(rest);
    const bool any = what.empty() or what == "that" or what == "this" or what == "it" or what == "one" or what == "eso" or what == "esto";
    const std::string want = any ? std::string{} : normalize_subtype(what);
    std::vector<ObjectFact> pool;
    for (const auto& f : live)
        if ((want.empty() or f.subtype == want) and colour_may_be(f.color_name, colour)
            and not (kind == "owns" and f.uid != 0 and h.owned_uids.contains(f.uid)))
            pool.push_back(f);
    for (const auto& c : that_candidates(pool, recent_names, last_target, half_fov_deg))
        if (auto d = describe(c.uid, true)) out.push_back(*d);
    return out;
}

/// One claim per model round: a second human_claim in the same turn would replace the first card unseen (review I3).
constexpr bool second_claim_this_turn(std::optional<std::uint64_t> pending_card_turn, std::uint64_t user_turn)
{
    return pending_card_turn.has_value() and *pending_card_turn == user_turn;
}

/// A yes with a mission card AND a claim card pending goes to the NEWER card (review focus 1): never both.
enum class YesGoesTo { mission, claim };
constexpr YesGoesTo newer_card_takes_the_yes(std::uint64_t mission_serial, std::uint64_t claim_serial)
{
    return claim_serial > mission_serial ? YesGoesTo::claim : YesGoesTo::mission;
}

constexpr bool claim_answer_overdue(std::int64_t posted_ms, std::int64_t now_ms)
{
    return now_ms - posted_ms > kClaimAnswerWaitMs;
}

}   // namespace rc::chat
