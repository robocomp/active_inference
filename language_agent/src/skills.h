#pragma once
/*
 * skills.h — markdown notes the model reads only when it needs them (2026-10-23, spec §5.4). The system prompt lists each
 * skill as one line (`- name: description`); the tool use_skill(name) returns the body as a tool result, which lands at
 * the END of the conversation so the cached prefix survives. Pure: no Qt, no graph.
 *
 *   etc/skills/<name>.md :   ---\n description: <one line>\n---\n <body>
 */

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace rc::chat
{
struct Skill
{
    std::string name{};          ///< the file stem
    std::string description{};   ///< one line, from the front matter
    std::string body{};          ///< everything after the front matter
};

/// One Skill per *.md in `dir`, sorted by name. A file without a `description:` front-matter line is skipped, with a warning.
[[nodiscard]] inline std::vector<Skill> load_skills(const std::filesystem::path& dir)
{
    namespace fs = std::filesystem;
    std::vector<Skill> out;
    std::error_code ec;
    if (not fs::is_directory(dir, ec))
    {
        std::println("[language_agent] WARNING: skills directory '{}' not found; no skills", dir.string());
        return out;
    }
    for (const auto& e : fs::directory_iterator(dir, ec))
    {
        if (not e.is_regular_file() or e.path().extension() != ".md") continue;
        std::ifstream f(e.path());
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string text = ss.str();
        Skill s{.name = e.path().stem().string()};
        bool ok = false;
        if (text.starts_with("---\n"))
        {
            const std::size_t close = text.find("\n---\n", 3);
            if (close != std::string::npos)
            {
                std::istringstream head(text.substr(4, close - 4));
                for (std::string line; std::getline(head, line);)
                    if (line.starts_with("description:"))
                    {
                        s.description = line.substr(12);
                        s.description.erase(0, s.description.find_first_not_of(' '));
                        while (not s.description.empty() and (s.description.back() == ' ' or s.description.back() == '\r'))
                            s.description.pop_back();
                    }
                s.body = text.substr(close + 5);
                ok = not s.description.empty();
            }
        }
        if (not ok)
        {
            std::println("[language_agent] WARNING: skill '{}' has no `description:` front matter; skipped", e.path().string());
            continue;
        }
        out.push_back(std::move(s));
    }
    std::ranges::sort(out, {}, &Skill::name);
    return out;
}

/// The system prompt's list: "- <name>: <description>\n" per skill.
[[nodiscard]] inline std::string skills_index(const std::vector<Skill>& skills)
{
    std::string out;
    for (const auto& s : skills) out += "- " + s.name + ": " + s.description + "\n";
    return out;
}
}   // namespace rc::chat
