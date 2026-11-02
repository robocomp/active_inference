// verbatim_reply.h -- a tool that REFUSES (a limitation the person must hear as it is) marks its result; the session then
// shows that text as the robot's reply instead of asking the model to word it. Live 2026-11-02: imagine_view said "I do not
// know which room I am in, so I cannot imagine it here." and the model's follow-up invented "I do not have a way to imagine
// things". The mark is set by the tool at its refusal site (never found by matching English in the result).
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
        text_ += (text_.empty() ? "" : "\n") + std::move(text);
    }
    [[nodiscard]] bool pending() const { return not text_.empty(); }
    /// The text to show, and clears it for the next round.
    [[nodiscard]] std::optional<std::string> take()
    {
        if (text_.empty()) return std::nullopt;
        return std::exchange(text_, {});
    }
private:
    std::string text_;
};
}   // namespace rc::chat
