// flat_json.h -- the minimal FLAT-JSON scanner and locale-independent float I/O the media descriptors use,
// factored out of media_transport.cpp VERBATIM so other modules (perception_card) share ONE facility instead
// of growing a second JSON parser. Header-only. See CLAUDE.md "Parsing numbers": from_chars/to_chars only.
#pragma once
#include <cctype>
#include <charconv>
#include <map>
#include <string>

namespace rc::media::flat
{
// Minimal scanner for a FLAT JSON object: {"k": "str", "k2": 3, "k3": true}.
// No nesting/arrays (the descriptor schema is deliberately flat), so this stays
// small and robust. Every value is returned as its raw string token; typed
// fields are converted by the caller.
inline std::map<std::string, std::string> parse_flat_json(const std::string& s)
{
    std::map<std::string, std::string> out;
    const std::size_t n = s.size();
    std::size_t i = 0;
    auto skip_ws = [&] { while (i < n and std::isspace(static_cast<unsigned char>(s[i]))) ++i; };
    auto parse_string = [&](std::string& dst) -> bool
    {
        if (i >= n or s[i] != '"') return false;
        ++i;
        std::string r;
        while (i < n and s[i] != '"')
        {
            if (s[i] == '\\' and i + 1 < n)
            {
                ++i;
                const char c = s[i];
                r += (c == 'n') ? '\n' : (c == 't') ? '\t' : c;   // covers \" \\ too
            }
            else
                r += s[i];
            ++i;
        }
        if (i >= n) return false;
        ++i;                                                       // closing quote
        dst = std::move(r);
        return true;
    };
    skip_ws();
    if (i < n and s[i] == '{') ++i;
    while (true)
    {
        skip_ws();
        if (i >= n or s[i] == '}') break;
        std::string key;
        if (not parse_string(key)) break;
        skip_ws();
        if (i < n and s[i] == ':') ++i; else break;
        skip_ws();
        std::string val;
        if (i < n and s[i] == '"')
        {
            if (not parse_string(val)) break;
        }
        else if (i < n and s[i] == '[')                            // JSON array, kept whole
        {
            // The scan below stops at ',' or '}', so an array would truncate at its first
            // element. Capture it wholesale instead; as_float_list() splits it afterwards.
            // No nesting: these are flat numeric arrays (per-ring elevations), nothing deeper.
            const std::size_t start = i;
            while (i < n and s[i] != ']') ++i;
            if (i < n) ++i;                                        // closing bracket
            val = s.substr(start, i - start);
        }
        else                                                       // number / bool / null token
        {
            const std::size_t start = i;
            while (i < n and s[i] != ',' and s[i] != '}') ++i;
            val = s.substr(start, i - start);
            while (not val.empty() and std::isspace(static_cast<unsigned char>(val.back()))) val.pop_back();
        }
        out.emplace(std::move(key), std::move(val));
        skip_ws();
        if (i < n and s[i] == ',') { ++i; continue; }
    }
    return out;
}

inline float as_float(const std::string& v, float def)
{
    const char* b = v.data();
    const char* e = b + v.size();
    while (b < e and std::isspace(static_cast<unsigned char>(*b))) ++b;
    float out{};
    const auto [ptr, ec] = std::from_chars(b, e, out);
    return ec == std::errc{} ? out : def;
}

inline
// Shortest round-trippable form, never locale-formatted.
std::string fmt_float(float v)
{
    char buf[64];
    const auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v);
    return ec == std::errc{} ? std::string(buf, ptr) : std::string("0");
}
} // namespace rc::media::flat
