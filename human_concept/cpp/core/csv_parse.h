#pragma once
/*
 * csv_parse.h — LOCALE-INDEPENDENT number parsing for every human_concept text input
 * (replay CSVs, the parity fixtures, harness CLI args).
 *
 * ★DO NOT use strtof/atof/stof/atoi or an istringstream on this data. These machines run
 * LANG=es_ES.UTF-8, where the decimal separator is a COMMA, and Qt calls setlocale(LC_ALL, "")
 * at startup, which activates that locale for the C library. The CSVs are WRITTEN by
 * std::ofstream, which formats through the C++ global locale (still "C" ⇒ decimal POINT), so
 * strtof stops dead at the '.' and returns the integer part — silently, with no error flag.
 * Measured 2026-08-03 on this component's own fixtures: 2822/2880 keypoint fields in
 * etc/skeleton_replay.csv and cpp/test/inputs.csv truncated ("0.626452" → 0), and EVERY value
 * in cpp/test/reference.csv collapsed to 0.000000 — i.e. the C++/Python parity check was
 * comparing zeros against zeros. See ../../CLAUDE.md ("Parsing numbers from files").
 *
 * std::from_chars is locale-independent BY DEFINITION, allocates nothing, and reports failure
 * instead of guessing — the only property that makes a data file portable across whatever
 * locale the process happens to boot into.
 *
 * Note for harnesses: a standalone binary never calls setlocale, so it stays in "C" and the bug
 * VANISHES — a test and the agent would disagree about the same file. Harnesses here call
 * std::setlocale(LC_ALL, "") in main() so they answer the same question the agent does.
 */

#include <charconv>
#include <limits>
#include <string_view>

namespace rc::csv
{

// Trim ASCII whitespace from both ends. CSV cells arrive with stray spaces / a trailing '\r'.
inline std::string_view trim(std::string_view s) noexcept
{
    const auto ws = [](char c) { return c == ' ' or c == '\t' or c == '\r' or c == '\n'; };
    while (not s.empty() and ws(s.front())) s.remove_prefix(1);
    while (not s.empty() and ws(s.back()))  s.remove_suffix(1);
    return s;
}

// Parse one number; `fallback` on empty/garbage. from_chars rejects a leading '+' (strtof accepts
// it), so drop it first — the only strtof behaviour worth keeping.
template <typename T>
inline T parse_number(std::string_view s, T fallback = T{}) noexcept
{
    s = trim(s);
    if (not s.empty() and s.front() == '+') s.remove_prefix(1);
    if (s.empty()) return fallback;
    T out{};
    const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return (r.ec == std::errc{}) ? out : fallback;
}

// Float fields: an empty / unparsable / "nan" cell means MISSING, and missing must stay NaN so the
// estimator's validity mask drops it rather than fitting a 0-metre keypoint at the origin.
inline float parse_float(std::string_view s) noexcept
{
    return parse_number<float>(s, std::numeric_limits<float>::quiet_NaN());
}

inline int parse_int(std::string_view s, int fallback = 0) noexcept
{
    return parse_number<int>(s, fallback);
}

inline long parse_long(std::string_view s, long fallback = 0) noexcept
{
    return parse_number<long>(s, fallback);
}

}   // namespace rc::csv
