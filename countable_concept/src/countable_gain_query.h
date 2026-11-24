/*
 * countable_gain_query.h — "what would a look from THIS standpoint deliver?" asked from outside the agent (shared-engine plan
 * rev 2, P1 follow-up: the A/B/A visit's promise at each scripted claim). Pure: no DSR, no Qt; gain_query_test drives it.
 *
 * The request file is <etc>/gain_query.txt (beside offers.csv), written by room_concept/estimation_scripts/scripts/aba_visit.sh,
 * one request per line:
 *     <id> <node|*> <x> <y> <yaw>          id: a positive integer, increasing; x, y, yaw: the published room frame (m, rad)
 * The agent polls the file's mtime once per tick and answers every line whose id is above the last one it answered, for the
 * named instance (or every instance with `*`), in <etc>/gain_query.csv. Numbers are parsed with std::from_chars (locale-free).
 */
#pragma once

#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace rc {

struct GainQuery
{
    long        id = 0;
    std::string node;           // an instance's node name, or "*" for every instance
    float       x = 0.0f, y = 0.0f, yaw = 0.0f;
};

namespace gain_query_detail {
inline std::vector<std::string_view> split_ws(std::string_view s)
{
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < s.size())
    {
        while (i < s.size() and (s[i] == ' ' or s[i] == '\t' or s[i] == '\r')) ++i;
        std::size_t j = i;
        while (j < s.size() and not (s[j] == ' ' or s[j] == '\t' or s[j] == '\r')) ++j;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}
template <class T>
bool parse_whole(std::string_view s, T& out)
{
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
    return ec == std::errc{} and p == s.data() + s.size();
}
}  // namespace gain_query_detail

// One request line ⇒ the query; nullopt on a malformed line (wrong field count, a number that does not parse whole, id ≤ 0).
inline std::optional<GainQuery> parse_gain_query(std::string_view line)
{
    const auto f = gain_query_detail::split_ws(line);
    if (f.size() != 5) return std::nullopt;
    GainQuery q;
    q.node = std::string(f[1]);
    if (not gain_query_detail::parse_whole(f[0], q.id) or q.id <= 0) return std::nullopt;
    if (not gain_query_detail::parse_whole(f[2], q.x) or not gain_query_detail::parse_whole(f[3], q.y) or not gain_query_detail::parse_whole(f[4], q.yaw))
        return std::nullopt;
    return q;
}

// Every well-formed request of a file's text whose id is above `last_id`, in file order.
inline std::vector<GainQuery> new_gain_queries(std::string_view text, long last_id)
{
    std::vector<GainQuery> out;
    std::size_t i = 0;
    while (i <= text.size())
    {
        const std::size_t j = text.find('\n', i);
        const std::string_view line = text.substr(i, j == std::string_view::npos ? std::string_view::npos : j - i);
        if (const auto q = parse_gain_query(line); q and q->id > last_id) out.push_back(*q);
        if (j == std::string_view::npos) break;
        i = j + 1;
    }
    return out;
}

}  // namespace rc
