// toml_peek.h -- test helper: read ONE numeric `Key = value` from an agent's etc/config.toml with from_chars
// (locale-independent). Skips comment lines; first match wins; returns fallback when the key is absent.
#pragma once
#include <charconv>
#include <fstream>
#include <string>

inline float toml_peek(const std::string& path, const std::string& key, float fallback)
{
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line))
    {
        std::size_t i = line.find_first_not_of(" \t");
        if (i == std::string::npos or line[i] == '#') continue;
        if (line.compare(i, key.size(), key) != 0) continue;
        i += key.size();
        while (i < line.size() and (line[i] == ' ' or line[i] == '\t')) ++i;
        if (i >= line.size() or line[i] != '=') continue;
        ++i;
        while (i < line.size() and (line[i] == ' ' or line[i] == '\t')) ++i;
        float v = 0;
        const auto r = std::from_chars(line.data() + i, line.data() + line.size(), v);
        if (r.ec == std::errc{}) return v;
    }
    return fallback;
}
