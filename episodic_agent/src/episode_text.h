#pragma once
/*
 * episode_text.h — the text the em window shows for a domain-4 node (2026-10-18): left-click a node → "View data"
 * → a popup (SpecificWorker::show_node_text). Same wording as chat_agent's episodic tools
 * (common/episodic/em_answers.h), so a person reads what the model reads. Pure over a Store, keyed by the node NAME
 * (em_graph.h names nodes by record id, EmGraph::stats_name, and "root"); tested in tests/dsr_adapters_test.cpp.
 *
 *   record node → episode(id): the record, everything under it indented, beliefs before/after, structural events.
 *                 "today" = the record's own day, so a record kept past midnight (not yet rolled over) still reads.
 *   stats node  → its all-time line (counts per outcome, duration, information gain, first/last).
 *   root        → the missions of the newest day held, then the all-time statistics (today folded in).
 */

#include "em_graph.h"
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
    if (const auto r = std::ranges::find(st.today, node, &Record::id); r != st.today.end())
        return rc::chat::answer_episode(st, r->id, day_of(r->t0));
    if (const auto s = std::ranges::find_if(st.stats, [&](const Stats& x) { return EmGraph::stats_name(x) == node; });
        s != st.stats.end())
        return stats_line(*s, stats_label(*s));
    if (node == "root")
    {
        if (st.today.empty() and st.stats.empty()) return "episodic memory is empty";
        std::string out;
        if (not st.today.empty())   // Store::today is sorted by t0 (load_store): back() is the newest day held
            out = rc::chat::answer_episodes(st, std::nullopt, std::nullopt, "", "", "", day_of(st.today.back().t0));
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
