// card_io.h -- perception cards as FILES: a concept agent writes <agent_dir>/etc/perception_card.json next to
// itself, language_agent reads them all. Header-only. Atomic write (temp file in the same dir + rename) so a reader
// never sees a half-written card. Locale-safe: output is imbued with the classic locale, parsing is card.h's.
#pragma once

#include "perception_card/card.h"

#include <algorithm>
#include <cstdint>
#include <utility>
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

// blind_spot_map_path must be ABSOLUTE in a card: the reader (language_agent) runs from another directory. The
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
template <class Att, class G, class Node>
bool publish_card_json(G& graph, Node& node_ref, const std::string& json)
{
    auto* node = &node_ref;
    const auto cur = graph.template get_attrib_by_name<Att>(*node);
    if (cur.has_value() and std::string(cur.value()) == json) return true;
    graph.template add_or_modify_attrib_local<Att>(*node, json);
    graph.update_node(*node);
    return true;
}

template <class Att, class G>
bool publish_card(G& graph, std::uint64_t agent_node_id, const Card& card)
{
    auto node = graph.get_node(agent_node_id);
    if (not node.has_value()) return false;
    return publish_card_json<Att>(graph, *node, to_json(card));
}

// Graph-side card selection. `A` names the attributes (a struct of aliases, defined where the DSR is visible):
//   A::card (perception_card_att), A::presence (agent_presence_state_att), A::stamp (timestamp_agent_att).
// Two agent nodes can claim one concept (a restarted agent whose old node lingers): the live one wins
// (presence == "Operating"), then the most recent heartbeat.
namespace detail
{
struct GraphCard { Card card; bool live = false; std::uint64_t stamp = 0; };

template <class A, class G>
std::vector<GraphCard> graph_cards(G& graph)
{
    std::vector<GraphCard> out;
    for (auto& n : graph.get_nodes_by_type("agent"))
    {
        const auto js = graph.template get_attrib_by_name<typename A::card>(n);
        if (not js.has_value()) continue;
        auto r = card_from_json(std::string(js.value()));
        if (not r) continue;
        GraphCard gc{std::move(*r), false, 0};
        if (const auto pr = graph.template get_attrib_by_name<typename A::presence>(n); pr.has_value())
            gc.live = std::string(pr.value()) == "Operating";
        if (const auto ts = graph.template get_attrib_by_name<typename A::stamp>(n); ts.has_value())
            gc.stamp = static_cast<std::uint64_t>(ts.value());
        if (const auto dup = std::ranges::find(out, gc.card.concept_, [](const GraphCard& g) { return g.card.concept_; });
            dup == out.end()) out.push_back(std::move(gc));
        else if (std::pair(gc.live, gc.stamp) > std::pair(dup->live, dup->stamp)) *dup = std::move(gc);
    }
    return out;
}
}

// The card of `concept_name`: from the graph, else from <root>/<concept>_concept/etc/perception_card.json.
// `from_graph` (optional) reports which.
template <class A, class G>
std::optional<Card> read_card(G& graph, std::string_view concept_name, const std::filesystem::path& root,
                              bool* from_graph = nullptr)
{
    if (from_graph) *from_graph = false;
    for (auto& gc : detail::graph_cards<A>(graph))
        if (gc.card.concept_ == concept_name)
        {
            if (from_graph) *from_graph = true;
            return std::move(gc.card);
        }
    return read_card_file(root / (std::string(concept_name) + "_concept"));
}

// Every concept's card: the graph's first, then the files for the concepts the graph did not give; sorted by
// concept name. `n_graph` (optional) = how many came from the graph.
template <class A, class G>
std::vector<Card> read_all_cards(G& graph, const std::filesystem::path& root, std::size_t* n_graph = nullptr)
{
    std::vector<Card> out;
    for (auto& gc : detail::graph_cards<A>(graph)) out.push_back(std::move(gc.card));
    if (n_graph) *n_graph = out.size();
    for (auto& c : read_all_cards(root))
        if (std::ranges::find(out, c.concept_, &Card::concept_) == out.end()) out.push_back(std::move(c));
    std::ranges::sort(out, {}, &Card::concept_);
    return out;
}

// What an agent hands to its presence coordinator: the card as the graph copy (absolute paths), pre-serialised so
// the shared publisher never needs to know the card type.
inline std::string graph_json(const Card& c)
{
    return to_json(with_absolute_paths(std::filesystem::current_path(), c));
}
}
