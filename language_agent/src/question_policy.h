#pragma once
/*
 * question_policy.h — the chat side of the changed-object question (spec 2026-11-15-changed-object, owner's choice B).
 * LTOM opens a question when a live object is linked to a remembered one and the old spot was seen empty; the chat
 * asks it, or — once the person's answers make it likely they moved it — reports it. Pure, header-only.
 *
 * Ask or report: p = P(the person moved it) = the mean of Beta(1 + yes, 1 + no) over their dated answers. Report when
 * a wrong "I assume you moved it" costs less in expectation than a question: (1 − p)·cost_wrong < cost_ask.
 */

#include <cctype>
#include <charconv>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "fleet_control.h"
#include "place_words.h"

namespace rc::chat
{
struct OpenQuestion
{
    std::uint64_t stamp = 0, memory_uid = 0, live_uid = 0;
    std::string room;          ///< room node name, e.g. room_2
    std::string cls;           ///< e.g. chair
    std::string colour;        ///< the remembered colour name, "" = never measured
    std::string memory_name;   ///< e.g. chair_50
    P2 old_xy, new_xy;         ///< room frame
    bool again = false;        ///< the memory already holds a confirmed move
};
struct BeliefFacts { std::string key; int yes = 0, no = 0; std::vector<std::string> evidence; };

enum class QMode { ask, report };
[[nodiscard]] inline QMode choose_mode(int yes, int no, float cost_ask, float cost_wrong)
{
    const float p = static_cast<float>(1 + yes) / static_cast<float>(2 + yes + no);
    return (1.f - p) * cost_wrong < cost_ask ? QMode::report : QMode::ask;
}

enum class Reply { yes, no, unknown, other };
[[nodiscard]] inline Reply classify_reply(std::string_view text)
{
    std::string t = fleet::lower_trim(text);
    while (not t.empty() and (t.back() == '!' or t.back() == '.' or t.back() == ',' or t.back() == '?')) t.pop_back();
    const auto has = [&t](std::string_view w) { return t.find(w) != std::string::npos; };
    const auto starts = [&t](std::string_view w)
    { return t.starts_with(w) and (t.size() == w.size() or not std::isalpha(static_cast<unsigned char>(t[w.size()]))); };
    if (has("don't know") or has("dont know") or has("do not know") or has("no idea") or has("not sure") or
        starts("no sé") or starts("no se") or has("no lo sé") or has("ni idea"))
        return Reply::unknown;
    if (starts("no") or starts("nope") or has("i didn't") or has("i did not") or has("not me") or has("no lo moví")
        or has("no lo movi"))
        return Reply::no;
    if (fleet::is_yes(t) or starts("yes") or starts("yeah") or starts("yep") or starts("sí") or starts("si") or
        has("i moved it") or has("lo moví") or has("lo movi"))
        return Reply::yes;
    return Reply::other;
}

/// "room_2" -> "room 2".
[[nodiscard]] inline std::string room_words(std::string_view room)
{
    std::string r(room);
    for (char& c : r) if (c == '_') c = ' ';
    return r;
}

/// The question, in the spec's exact forms. owner_was_away: the second raise, after the person's next message.
[[nodiscard]] inline std::string question_text(const OpenQuestion& q, QMode mode, bool owner_was_away,
                                               std::string_view old_place, std::string_view new_place)
{
    const std::string what = q.colour.empty() ? "The " + q.cls : std::format("The {} {}", q.colour, q.cls);
    if (mode == QMode::report)
        return std::format("{} has moved{}, {}. I assume you moved it.", what, q.again ? " again" : "", new_place);
    if (owner_was_away)
    {
        std::string w = what;
        w[0] = 't';
        return std::format("By the way, while you were away {} in {} moved {}. I've assumed it's the same {}. Did you move it?",
                           w, room_words(q.room), new_place, q.cls);
    }
    return std::format("{} in {} is {} now; it was {}. Did you move it?", what, room_words(q.room), new_place, old_place);
}

/// The answer as LTOM reads it: human_claim_value of a `moved` claim.
[[nodiscard]] inline std::string moved_claim_value(Reply r, std::uint64_t stamp)
{
    const char* v = r == Reply::yes ? "yes" : r == Reply::no ? "no" : "unknown";
    return std::format("{}:{}", v, stamp);
}

/// A belief key in words: "moves_things" -> "you move things"; "gets_moved:<uid>" -> "<name> gets moved".
[[nodiscard]] inline std::string belief_words(std::string_view key, std::string_view object_name)
{
    if (key == "moves_things") return "you move things";
    if (key.starts_with("gets_moved:")) return std::format("{} gets moved", object_name.empty() ? "that object" : object_name);
    return std::string(key);
}

/// The what_i_learned tool's answer: each belief with its counts and dated evidence lines, never a bare number.
/// name_of(uid) names the object of a gets_moved belief.
template <typename NameOf>
[[nodiscard]] std::string learned_text(const std::vector<BeliefFacts>& beliefs, NameOf name_of)
{
    if (beliefs.empty()) return "I have not learned anything about you from our conversations yet.";
    std::string out;
    for (const auto& b : beliefs)
    {
        std::string name;
        if (b.key.starts_with("gets_moved:"))
        {
            std::uint64_t uid = 0;
            const char* first = b.key.data() + std::string_view("gets_moved:").size();
            std::from_chars(first, b.key.data() + b.key.size(), uid);
            name = name_of(uid);
        }
        out += std::format("Belief: {} (yes {}, no {}). Evidence, one dated line each (time | answer | the person's words | object):\n",
                           belief_words(b.key, name), b.yes, b.no);
        for (const auto& e : b.evidence)
            out += "  " + e + (e.contains("| forgotten |") ? "   (you asked me to forget the earlier answers)" : "") + "\n";
    }
    return out;
}

/// Raised on open (1), once more after the person's next message that is not an answer (2), then silent.
struct RaiseState { std::uint64_t stamp = 0; int raised = 0; QMode mode = QMode::ask; };
}   // namespace rc::chat
