// card_io.h -- perception cards as FILES: a concept agent writes <agent_dir>/etc/perception_card.json next to
// itself, chat_agent reads them all. Header-only. Atomic write (temp file in the same dir + rename) so a reader
// never sees a half-written card. Locale-safe: output is imbued with the classic locale, parsing is card.h's.
#pragma once

#include "perception_card/card.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <locale>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace rc::card
{
inline std::filesystem::path card_file_path(const std::filesystem::path& agent_dir)
{
    return agent_dir / "etc" / "perception_card.json";
}

inline bool write_card_file(const std::filesystem::path& agent_dir, const Card& card_in)
{
    namespace fs = std::filesystem;
    // blind_spot_map_path must be ABSOLUTE in a card: the reader (chat_agent) runs from another directory. The
    // agents' config paths are relative to their own working dir (= agent_dir), so resolve them here, once, for all.
    Card c = card_in;
    if (not c.blind_spot_map_path.empty() and fs::path(c.blind_spot_map_path).is_relative())
        c.blind_spot_map_path = (fs::absolute(agent_dir) / c.blind_spot_map_path).lexically_normal().string();
    const fs::path target = card_file_path(agent_dir);
    fs::path tmp = target;  tmp += ".tmp";
    std::error_code ec;
    fs::create_directories(target.parent_path(), ec);
    if (ec) { std::println(stderr, "[card] cannot create {}: {}", target.parent_path().string(), ec.message()); return false; }
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.imbue(std::locale::classic());
        f << to_json(c);
        f.flush();
        if (not f) { std::println(stderr, "[card] cannot write {}", tmp.string()); return false; }
    }
    fs::rename(tmp, target, ec);
    if (ec) { std::println(stderr, "[card] cannot rename {} -> {}: {}", tmp.string(), target.string(), ec.message());
              fs::remove(tmp, ec); return false; }
    return true;
}

// nullopt when the file is absent (silent) or corrupt (logged with the parse error).
inline std::optional<Card> read_card_file(const std::filesystem::path& agent_dir)
{
    const auto path = card_file_path(agent_dir);
    std::ifstream f(path, std::ios::binary);
    if (not f) return std::nullopt;
    f.imbue(std::locale::classic());
    std::ostringstream ss;  ss << f.rdbuf();
    auto r = card_from_json(ss.str());
    if (not r) { std::println(stderr, "[card] corrupt {}: {}", path.string(), r.error()); return std::nullopt; }
    return std::move(*r);
}

// Scans <root>/*_concept/etc/perception_card.json; cards sorted by concept name.
inline std::vector<Card> read_all_cards(const std::filesystem::path& root)
{
    namespace fs = std::filesystem;
    std::vector<Card> out;
    std::error_code ec;
    for (fs::directory_iterator it(root, ec), end; not ec and it != end; it.increment(ec))
    {
        const auto name = it->path().filename().string();
        if (not name.ends_with("_concept") or not it->is_directory(ec)) continue;
        if (auto c = read_card_file(it->path())) out.push_back(std::move(*c));
    }
    std::ranges::sort(out, {}, &Card::concept_);
    return out;
}
}
