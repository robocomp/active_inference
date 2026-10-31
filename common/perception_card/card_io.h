// card_io.h -- perception cards as FILES: a concept agent writes <agent_dir>/etc/perception_card.json next to
// itself, chat_agent reads them all. Header-only. Atomic write (temp file in the same dir + rename) so a reader
// never sees a half-written card. Locale-safe: output is imbued with the classic locale, parsing is card.h's.
#pragma once

#include "perception_card/card.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <locale>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace rc::card
{
inline std::filesystem::path card_file_path(const std::filesystem::path& agent_dir)
{
    return agent_dir / "etc" / "perception_card.json";
}

// blind_spot_map_path must be ABSOLUTE in a card: the reader (chat_agent) runs from another directory. The
// agents' config paths are relative to their own working dir (= agent_dir), so resolve them here, once, for all
// (file AND graph publication).
inline Card with_absolute_paths(const std::filesystem::path& agent_dir, Card c)
{
    namespace fs = std::filesystem;
    if (not c.blind_spot_map_path.empty() and fs::path(c.blind_spot_map_path).is_relative())
        c.blind_spot_map_path = (fs::absolute(agent_dir) / c.blind_spot_map_path).lexically_normal().string();
    return c;
}

inline bool write_card_file(const std::filesystem::path& agent_dir, const Card& card_in)
{
    namespace fs = std::filesystem;
    const Card c = with_absolute_paths(agent_dir, card_in);
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

// ---- the same card through the shared DSR graph -------------------------------------------------------------
// Header stays free of DSR (as rc::media::sensor_model_from_graph): `Att` is the cortex attribute type
// (`perception_card_att`) and `G` the graph; both are template parameters. Main thread only.

// Writes to_json(card) into the agent node's perception_card attribute, only when it differs from the current
// value. Returns false when the node does not exist yet (the agent node is created lazily by the DSR heartbeat:
// the caller retries); true when the attribute is up to date.
template <class Att, class G>
bool publish_card(G& graph, std::uint64_t agent_node_id, const Card& card)
{
    auto node = graph.get_node(agent_node_id);
    if (not node.has_value()) return false;
    const std::string json = to_json(card);
    const auto cur = graph.template get_attrib_by_name<Att>(*node);
    if (cur.has_value() and std::string(cur.value()) == json) return true;
    graph.template add_or_modify_attrib_local<Att>(*node, json);
    graph.update_node(*node);
    return true;
}

// The card of `concept_name`: from the graph (any "agent" node whose perception_card parses and names that
// concept), else from <root>/<concept>_concept/etc/perception_card.json. `from_graph` (optional) reports which.
template <class G>
std::optional<Card> read_card(G& graph, std::string_view concept_name, const std::filesystem::path& root,
                              bool* from_graph = nullptr)
{
    if (from_graph) *from_graph = false;
    for (const auto& n : graph.get_nodes_by_type("agent"))
    {
        const auto& attrs = n.attrs();
        const auto it = attrs.find("perception_card");
        if (it == attrs.end()) continue;
        auto r = card_from_json(it->second.str());
        if (r and r->concept_ == concept_name)
        {
            if (from_graph) *from_graph = true;
            return std::move(*r);
        }
    }
    return read_card_file(root / (std::string(concept_name) + "_concept"));
}

// Every concept's card: the graph's agent nodes first, then the files for the concepts the graph did not give.
// Sorted by concept name. `n_graph` (optional) = how many came from the graph.
template <class G>
std::vector<Card> read_all_cards(G& graph, const std::filesystem::path& root, std::size_t* n_graph = nullptr)
{
    std::vector<Card> out;
    for (const auto& n : graph.get_nodes_by_type("agent"))
    {
        const auto it = n.attrs().find("perception_card");
        if (it == n.attrs().end()) continue;
        if (auto r = card_from_json(it->second.str()))
            if (std::ranges::find(out, r->concept_, &Card::concept_) == out.end()) out.push_back(std::move(*r));
    }
    if (n_graph) *n_graph = out.size();
    for (auto& c : read_all_cards(root))
        if (std::ranges::find(out, c.concept_, &Card::concept_) == out.end()) out.push_back(std::move(c));
    std::ranges::sort(out, {}, &Card::concept_);
    return out;
}
}
