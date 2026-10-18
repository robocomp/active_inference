#pragma once
/*
 * em_query_dsr.h — the DSR adapter of em_query.h (2026-10-17): reads the episodic-memory graph (domain 4, written
 * by episodic_agent/src/em_graph.h) back into a Store. Kept apart from em_query.h so the pure tests need no DSR.
 * A node whose em_record / em_stats does not parse is skipped (absent is not zero: nothing is invented for it).
 */

#include "em_query.h"
#include "em_schema.h"

#include <QByteArray>
#include <QJsonDocument>

#include <dsr/api/dsr_api.h>

namespace rc::em
{

inline Store load_store(DSR::DSRGraph& G)
{
    Store st;
    const auto object_of = [](const std::string& text) -> std::optional<QJsonObject>
    {
        const QJsonDocument d = QJsonDocument::fromJson(QByteArray(text.data(), qsizetype(text.size())));
        if (not d.isObject()) return std::nullopt;
        return d.object();
    };
    for (const auto& n : G.get_nodes_by_type("metaconcept"))
    {
        if (const auto rec = G.get_attrib_by_name<em_record_att>(n); rec.has_value())
        {
            if (const auto o = object_of(rec.value()))
                if (auto r = record_from_json(*o)) st.today.push_back(std::move(*r));
        }
        else if (const auto sta = G.get_attrib_by_name<em_stats_att>(n); sta.has_value())
        {
            if (const auto o = object_of(sta.value()))
                if (auto s = stats_from_json(*o)) st.stats.push_back(std::move(*s));
        }
    }
    std::ranges::stable_sort(st.today, {}, &Record::t0);
    return st;
}

}   // namespace rc::em
