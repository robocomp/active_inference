// Chat-history trimming that keeps ollama's prompt cache. ollama reuses the processed prompt up to the first token that
// differs from the previous request, so trimming ONE message off the front of a full history on every turn changes the
// prefix every turn and forces a full prefill (about 100 s for 100k tokens on the Orin). Trim in BLOCKS instead:
// when the history exceeds `max`, drop down to `trim_to`; the prefix then changes once every (max - trim_to) pushes.
// Pure functions over Qt Core JSON; tests/test_history_trim.cpp.
#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <functional>
#include <string>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace rc::chat
{
constexpr double kCharsPerToken = 3.5;     ///< text estimate
constexpr int kTokensPerImage = 400;       ///< measured gemma4 image prompts: 200-370 tokens

inline std::size_t estimate_tokens_chars(std::size_t chars) { return static_cast<std::size_t>(static_cast<double>(chars) / kCharsPerToken) + 1; }

/// Tokens of one history message: its JSON without the base64 images, plus kTokensPerImage per image.
inline std::size_t estimate_tokens(const QJsonObject& m)
{
    QJsonObject copy = m;
    const auto n_images = static_cast<std::size_t>(copy.value("images").toArray().size());
    copy.remove("images");
    const auto chars = static_cast<std::size_t>(QJsonDocument(copy).toJson(QJsonDocument::Compact).size());
    return estimate_tokens_chars(chars) + n_images * kTokensPerImage;
}

inline void drop_leading_tools(std::deque<QJsonObject>& h)
{
    while (not h.empty() and h.front().value("role").toString() == "tool") h.pop_front();
}

/// When h.size() > max: drop from the front down to trim_to (<= 0 or >= max ⇒ max/2), never leaving a leading tool
/// message. Returns true when it trimmed.
inline bool trim_history(std::deque<QJsonObject>& h, int max, int trim_to)
{
    if (static_cast<int>(h.size()) <= max) return false;
    const int target = (trim_to > 0 and trim_to < max) ? trim_to : max / 2;
    while (static_cast<int>(h.size()) > target) h.pop_front();
    drop_leading_tools(h);
    return true;
}

/// While fixed_tokens + history exceeds budget_tokens: drop a block (half of the history) from the front. The newest
/// message is always kept (a single message over the budget cannot be helped: `log` is told and the loop ends).
/// Returns the number of messages dropped.
inline int trim_to_budget(std::deque<QJsonObject>& h, std::size_t fixed_tokens, std::size_t budget_tokens,
                          const std::function<void(const std::string&)>& log)
{
    auto total = [&]
    {
        std::size_t t = fixed_tokens;
        for (const auto& m : h) t += estimate_tokens(m);
        return t;
    };
    const int before = static_cast<int>(h.size());
    std::size_t t = total();
    if (t <= budget_tokens) return 0;
    while (t > budget_tokens and h.size() > 1)
    {
        const std::size_t keep = h.size() / 2;
        while (h.size() > std::max<std::size_t>(keep, 1)) h.pop_front();
        while (h.size() > 1 and h.front().value("role").toString() == "tool") h.pop_front();
        t = total();
    }
    const int dropped = before - static_cast<int>(h.size());
    if (log)
        log("history over the token budget: dropped " + std::to_string(dropped) + " of " + std::to_string(before) +
            " messages; ~" + std::to_string(t) + " tokens now vs budget " + std::to_string(budget_tokens) +
            (t > budget_tokens ? " (the newest message alone exceeds it)" : ""));
    return dropped;
}
}   // namespace rc::chat
