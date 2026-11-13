// verbatim_reply.h -- a tool that REFUSES (a limitation the person must hear as it is) marks its result; the session then
// shows that text as the robot's reply instead of asking the model to word it. Live 2026-11-02: imagine_view said "I do not
// know which room I am in, so I cannot imagine it here." and the model's follow-up invented "I do not have a way to imagine
// things". The mark is set by the tool at its refusal site (never found by matching English in the result).
// Also an answer about REMEMBERED objects (where_is / find_objects with long-term memory, ChatSession::said_as_is): live
// 2026-11-10 the model's follow-up moved chair_39 to another room and dropped "it may have moved".
#pragma once

#include <optional>
#include <string>
#include <utility>

namespace rc::chat
{
class VerbatimReply
{
public:
    /// A tool's refusal text for this round. Several refusals in one round are all said, in order.
    void add(std::string text)
    {
        if (text.empty()) return;
        ++refusals_;
        text_ += (text_.empty() ? "" : "\n") + std::move(text);
    }
    [[nodiscard]] bool pending() const { return not text_.empty(); }
    /// The verbatim reply replaces the model's follow-up ONLY when every call of the round refused: a mixed round has an
    /// outcome (a movement, a picture, a draw to repair) the follow-up must still word.
    [[nodiscard]] bool all_refused(int calls_in_round) const { return calls_in_round > 0 and refusals_ == calls_in_round; }
    /// The text to show, and clears it for the next round.
    [[nodiscard]] std::optional<std::string> take()
    {
        if (text_.empty()) return std::nullopt;
        refusals_ = 0;
        return std::exchange(text_, {});
    }
private:
    std::string text_;
    int refusals_ = 0;
};
}   // namespace rc::chat
