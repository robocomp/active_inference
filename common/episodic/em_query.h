#pragma once
/*
 * em_query.h — the pure query library chat_agent answers from (2026-10-17): filters over today's records and the
 * all-time Stats. Header-only, no DSR; the DSR adapter `load_store` is in em_query_dsr.h. Absent is not zero: last() is
 * an optional, an empty result is an empty vector.
 */

#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "em_schema.h"

namespace rc::em
{

struct Store
{
    std::vector<Record> today;
    std::vector<Stats> stats;
};

/// since inclusive, until exclusive, both against t0. kind/target empty = no filter. Ordered by t0 (stable).
inline std::vector<Record> episodes(const Store& st, std::optional<std::uint64_t> since, std::optional<std::uint64_t> until,
                                    std::optional<Author> author, std::string_view kind = {}, std::string_view target = {})
{
    std::vector<Record> out;
    for (const Record& r : st.today)
    {
        if (since and r.t0 < *since) continue;
        if (until and r.t0 >= *until) continue;
        if (author and r.author != *author) continue;
        if (not kind.empty() and r.kind != kind) continue;
        if (not target.empty() and r.target != target) continue;
        out.push_back(r);
    }
    std::ranges::stable_sort(out, {}, &Record::t0);
    return out;
}

/// The record and all its descendants, pre-order (children by t0). Walks by `parent` id, not by time: a child may
/// outlive its parent's t1. Empty when the id is unknown.
inline std::vector<Record> episode(const Store& st, std::string_view id)
{
    std::vector<Record> out;
    const auto root = std::ranges::find(st.today, id, &Record::id);
    if (root == st.today.end()) return out;
    auto walk = [&](auto&& self, const Record& r) -> void
    {
        out.push_back(r);
        std::vector<const Record*> kids;
        for (const Record& c : st.today)
            if (c.parent == r.id and c.id != r.id) kids.push_back(&c);
        std::ranges::stable_sort(kids, [](const Record* x, const Record* y) { return x->t0 < y->t0; });
        for (const Record* c : kids) self(self, *c);
    };
    walk(walk, *root);
    return out;
}

/// kind "" = all kinds; identity "" = all identities; level ("mission"|"step"|"claim") "" = all levels.
inline std::vector<Stats> stats(const Store& st, std::string_view kind, std::string_view identity = {},
                                std::string_view level = {})
{
    std::vector<Stats> out;
    for (const Stats& s : st.stats)
        if ((kind.empty() or s.key_kind == kind) and (identity.empty() or s.key_identity == identity)
            and (level.empty() or s.key_level == level))
            out.push_back(s);
    return out;
}

/// The newest-t0 record whose target is `target` or that carries a Fact (pre/post) on that node.
inline std::optional<Record> last(const Store& st, std::string_view target)
{
    const Record* best = nullptr;
    auto touches = [&](const Record& r)
    {
        if (r.target == target) return true;
        auto on = [&](const Fact& f) { return f.node == target; };
        return std::ranges::any_of(r.pre, on) or std::ranges::any_of(r.post, on);
    };
    for (const Record& r : st.today)
        if (touches(r) and (best == nullptr or r.t0 > best->t0)) best = &r;
    if (best == nullptr) return std::nullopt;
    return *best;
}

namespace detail
{
// A mission's one-line summary: its JSON `title` if action_json is an object with a string one, else the raw text
// cut to 60 characters (on a UTF-8 code point boundary).
inline std::string summary(const std::string& action_json)
{
    const QJsonDocument d = QJsonDocument::fromJson(QByteArray::fromStdString(action_json));
    if (d.isObject() and d.object()["title"].isString()) return d.object()["title"].toString().toStdString();
    if (action_json.size() <= 60) return action_json;
    std::size_t cut = 60;
    while (cut > 0 and (static_cast<unsigned char>(action_json[cut]) & 0xC0) == 0x80) --cut;   // not mid code point
    return action_json.substr(0, cut);
}
}   // namespace detail

/// One line for the chat greeting. Counts level-Mission records authored by a Human:
///   "<N> mission[s] (<K> succeeded); last: <summary>[ → <room>]"   ("1 mission", "2 missions")
/// K = outcome "succeeded" or "completed" (the controller's interpreter uses both). "last:" is the most recent
/// (by t0) CLOSED Human mission; summary = the `title` of its action_json when that is a JSON object with one, else
/// the raw text truncated to 60 chars; " → room" is appended when its room is non-empty. The "; last:" part is
/// omitted when no Human mission has closed. No Human missions at all: exactly "no missions today".
inline std::string today_line(const Store& st)
{
    int n = 0, ok = 0;
    const Record* lastc = nullptr;
    for (const Record& r : st.today)
    {
        if (r.level != Level::Mission or r.author != Author::Human) continue;
        ++n;
        if (r.outcome == "succeeded" or r.outcome == "completed") ++ok;
        if (r.t1 and (lastc == nullptr or r.t0 > lastc->t0)) lastc = &r;
    }
    if (n == 0) return "no missions today";
    std::string line = std::format("{} mission{} ({} succeeded)", n, n == 1 ? "" : "s", ok);
    if (lastc != nullptr)
    {
        line += "; last: " + detail::summary(lastc->action_json);
        if (not lastc->room.empty()) line += " → " + lastc->room;
    }
    return line;
}

}   // namespace rc::em
