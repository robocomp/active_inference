#pragma once
/*
 * mission_time.h — clock times in the mission language (docs/superpowers/specs/2026-11-14-tense-in-commands-design.md §1).
 *
 * `wait at=…` carries an ABSOLUTE local date-time with its UTC offset, "2026-11-14T18:00:00+01:00": grounded, so the
 * controller never guesses a date, a time zone or "the next 18:00". The planner writes a bare clock time (`at=18:00`);
 * language_agent grounds it (ground_clock) to the next occurrence before the card is made.
 *
 * Pure: <charconv> (never strtol — CLAUDE.md, the machines run a comma-decimal locale) and <ctime> for the local offset.
 */

#include <charconv>
#include <cstdint>
#include <ctime>
#include <format>
#include <optional>
#include <string>
#include <string_view>

namespace rc::mission
{

namespace time_detail
{
inline bool digits(std::string_view s, std::size_t at, std::size_t n, int& out)
{
    if (at + n > s.size()) return false;
    const char* b = s.data() + at;
    const auto [p, ec] = std::from_chars(b, b + n, out);
    return ec == std::errc{} and p == b + n;
}
/// Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant's days_from_civil).
constexpr std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d)
{
    y -= m <= 2;
    const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}
}   // namespace time_detail

/// A clock time "HH:MM" or "HH:MM:SS" → seconds after midnight. nullopt = not a clock time.
[[nodiscard]] inline std::optional<int> parse_clock(std::string_view s)
{
    using time_detail::digits;
    int h = 0, m = 0, sec = 0;
    const bool hm = (s.size() == 5 or s.size() == 8) and digits(s, 0, 2, h) and s[2] == ':' and digits(s, 3, 2, m);
    if (not hm) return std::nullopt;
    if (s.size() == 8 and not (s[5] == ':' and digits(s, 6, 2, sec))) return std::nullopt;
    if (h > 23 or m > 59 or sec > 59) return std::nullopt;
    return h * 3600 + m * 60 + sec;
}

/// "YYYY-MM-DDTHH:MM:SS±HH:MM" (or a trailing Z) → epoch milliseconds. nullopt = not that form (a bare "18:00" too).
[[nodiscard]] inline std::optional<std::int64_t> parse_iso(std::string_view s)
{
    using time_detail::digits;
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, se = 0;
    if (s.size() < 20 or not digits(s, 0, 4, y) or s[4] != '-' or not digits(s, 5, 2, mo) or s[7] != '-'
        or not digits(s, 8, 2, d) or s[10] != 'T' or not digits(s, 11, 2, h) or s[13] != ':' or not digits(s, 14, 2, mi)
        or s[16] != ':' or not digits(s, 17, 2, se))
        return std::nullopt;
    if (mo < 1 or mo > 12 or d < 1 or d > 31 or h > 23 or mi > 59 or se > 59) return std::nullopt;
    int off_s = 0;
    const std::string_view tz = s.substr(19);
    if (tz == "Z") off_s = 0;
    else
    {
        int oh = 0, om = 0;
        if (tz.size() != 6 or (tz[0] != '+' and tz[0] != '-') or not digits(tz, 1, 2, oh) or tz[3] != ':'
            or not digits(tz, 4, 2, om) or oh > 14 or om > 59)
            return std::nullopt;
        off_s = (tz[0] == '-' ? -1 : 1) * (oh * 3600 + om * 60);
    }
    const std::int64_t days = time_detail::days_from_civil(y, unsigned(mo), unsigned(d));
    return (days * 86400 + h * 3600 + mi * 60 + se - off_s) * 1000;
}

/// The "HH:MM" of an ISO value as written (its own local time), for people. "" = not ISO.
[[nodiscard]] inline std::string iso_hhmm(std::string_view s)
{
    return parse_iso(s).has_value() ? std::string(s.substr(11, 5)) : std::string{};
}

/// epoch ms → "YYYY-MM-DDTHH:MM:SS±HH:MM" in this machine's local time.
[[nodiscard]] inline std::string format_iso_local(std::int64_t epoch_ms)
{
    const std::time_t t = static_cast<std::time_t>(epoch_ms / 1000);
    std::tm lt{};
    localtime_r(&t, &lt);
    const long off = lt.tm_gmtoff;
    const long a = off < 0 ? -off : off;
    return std::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}{:02}:{:02}", lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday,
                       lt.tm_hour, lt.tm_min, lt.tm_sec, off < 0 ? '-' : '+', a / 3600, (a % 3600) / 60);
}

/// A bare clock time → its NEXT occurrence in local time, as ISO: today if it is still ahead of `now_ms`, else
/// tomorrow. A value that is already ISO is returned unchanged; anything else → nullopt.
[[nodiscard]] inline std::optional<std::string> ground_clock(std::string_view value, std::int64_t now_ms)
{
    if (parse_iso(value).has_value()) return std::string(value);
    const auto secs = parse_clock(value);
    if (not secs.has_value()) return std::nullopt;
    const std::time_t t = static_cast<std::time_t>(now_ms / 1000);
    for (int add_days = 0; add_days < 3; ++add_days)
    {
        std::tm lt{};
        localtime_r(&t, &lt);
        lt.tm_mday += add_days;
        lt.tm_hour = *secs / 3600;
        lt.tm_min = (*secs % 3600) / 60;
        lt.tm_sec = *secs % 60;
        lt.tm_isdst = -1;   // mktime decides (a DST change between now and then)
        const std::time_t when = std::mktime(&lt);
        if (when != std::time_t(-1) and std::int64_t(when) * 1000 > now_ms) return format_iso_local(std::int64_t(when) * 1000);
    }
    return std::nullopt;
}

}   // namespace rc::mission
