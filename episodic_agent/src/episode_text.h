#pragma once
/*
 * episode_text.h — the text the em window shows for a domain-4 node (2026-10-18): left-click a node → "View data"
 * → a popup (SpecificWorker::show_node_text). Records are told in plain words (episode_story.h); the statistics keep
 * chat_agent's wording (common/episodic/em_answers.h). Pure over a Store, keyed by the node NAME
 * (em_graph.h names nodes by record id, EmGraph::stats_name, and "root"); tested in tests/dsr_adapters_test.cpp.
 *
 *   record node → its mission told as a story for a person (episode_story.h): start, what happened in order, end.
 *                 A step or claim tells its whole mission, its own lines marked.
 *   stats node  → its all-time line (counts per outcome, duration, information gain, first/last).
 *   root        → the missions of the newest day held, one plain line each, then the all-time statistics.
 */

#include "em_graph.h"
#include "episode_story.h"
#include "../../common/episodic/em_answers.h"

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>

namespace rc::em
{

[[nodiscard]] inline std::string node_text(const Store& st, std::string_view node)
{
    using namespace rc::chat::em_detail;
    if (auto story = rc::em::story::tell(st, node)) return std::move(*story);
    if (const auto s = std::ranges::find_if(st.stats, [&](const Stats& x) { return EmGraph::stats_name(x) == node; });
        s != st.stats.end())
        return stats_line(*s, stats_label(*s));
    if (node == "root")
    {
        if (st.today.empty() and st.stats.empty()) return "episodic memory is empty";
        std::string out;
        if (not st.today.empty())   // Store::today is sorted by t0 (load_store): back() is the newest day held
            out = rc::em::story::day_list(st, day_of(st.today.back().t0));
        const auto rows = all_time(st);
        if (not rows.empty())
        {
            out += (out.empty() ? "" : "\n") + std::string("all-time statistics:\n");
            for (const Stats& s : rows) out += "- " + stats_line(s, stats_label(s)) + "\n";
        }
        return out;
    }
    return "no episode record on node '" + std::string(node) + "'";
}

}   // namespace rc::em
