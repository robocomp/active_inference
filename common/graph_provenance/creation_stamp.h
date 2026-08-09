/*
 * creation_stamp.h — stamp every node this fleet creates with WHEN it was created (header-only)
 *
 * WHY. A DSR node carries no birth time of its own: insert_node() assigns an id and publishes a
 * delta, and nothing records the instant. So "when did this cabinet appear?", "how long has this
 * person been tracked?", "did that door node survive the restart or is it a fresh one?" were all
 * unanswerable from the graph — you had to correlate log lines. The CRDT's internal per-attribute
 * timestamps do not help either: they move on every update, so they say when the node was last
 * WRITTEN, not when it was born.
 *
 * WHAT. Two attributes for one instant, written together immediately BEFORE insert_node:
 *
 *   timestamp_creation  uint64_t     ms since the Unix epoch (UTC). The machine-readable one —
 *                                    an age is `now_ms() - timestamp_creation`, a subtraction with
 *                                    no parsing and no timezone in the way.
 *   creation_datetime   std::string  the same instant as LOCAL civil time, ISO-8601 with the UTC
 *                                    offset: "2026-08-06T14:32:07.512+0200". The human-readable
 *                                    one — it is what the DSR graph viewer's attribute table
 *                                    shows, and a bare epoch count there tells nobody anything.
 *
 * Both, not one, because the two readers are different: code subtracts, people read. Deriving the
 * string on demand in every viewer would mean every viewer re-implements the offset handling.
 *
 * WHEN. Once, at birth, before the node exists in the graph — never refreshed. A re-acquired
 * object (one that died and was seen again) gets a NEW stamp, which is correct: it is a new node
 * with a new id, and only its NAME carried over. If you want "when did this object first appear,
 * across deaths", that is an existence-belief question, not a node-provenance one.
 *
 * LOCALE. The string is built with std::format and explicit integer fields, NOT strftime and not
 * an ostream: these machines run LANG=es_ES.UTF-8 and Qt calls setlocale(LC_ALL, "") at startup
 * (see CLAUDE.md, "Parsing numbers from files"). std::format is locale-independent unless an `L`
 * specifier asks otherwise, so this cannot acquire localised month names or separators. The
 * numeric fields are fixed-width and zero-padded, so the string is fixed-length and sorts
 * lexicographically within one offset.
 *
 * USE:
 *     #include "../../common/graph_provenance/creation_stamp.h"
 *     DSR::Node n = DSR::Node::create<object_node_type>(name);
 *     ...set the node's own attributes...
 *     rc::provenance::stamp_creation(*G_, n);      // ← last thing before the insert
 *     const auto id_opt = G_->insert_node(n);
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <format>
#include <string>

#include <dsr/api/dsr_api.h>

namespace rc::provenance
{

// Wall-clock ms since the Unix epoch. system_clock because this stamp is meant to be compared with
// other machines' and other runs' stamps — a steady_clock reading would be meaningless across both.
inline std::uint64_t now_ms()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

// Epoch ms → local civil time, ISO-8601 with the UTC offset ("2026-08-06T14:32:07.512+0200").
// localtime_r is thread-safe and gives us tm_gmtoff (glibc), so the offset needs no second call.
inline std::string iso8601_local(std::uint64_t ms)
{
    const std::time_t secs = static_cast<std::time_t>(ms / 1000);
    std::tm tmv{};
    if (::localtime_r(&secs, &tmv) == nullptr)      // only fails on an out-of-range time_t
        return std::format("epoch_ms:{}", ms);

    const long off  = tmv.tm_gmtoff;                // seconds east of UTC, DST already applied
    const char sign = off < 0 ? '-' : '+';
    const long aoff = off < 0 ? -off : off;
    return std::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}{}{:02}{:02}",
                       tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                       tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ms % 1000,
                       sign, aoff / 3600, (aoff % 3600) / 60);
}

// Stamp a node that has NOT been inserted yet. Both attributes describe one instant, so `ms` is
// taken once and shared; pass it explicitly when several nodes are born from the same observation
// and you want them to agree exactly (e.g. an object and its affordance children).
inline void stamp_creation(DSR::DSRGraph& G, DSR::Node& node, std::uint64_t ms)
{
    G.add_or_modify_attrib_local<timestamp_creation_att>(node, ms);
    G.add_or_modify_attrib_local<creation_datetime_att>(node, iso8601_local(ms));
}

inline void stamp_creation(DSR::DSRGraph& G, DSR::Node& node)
{
    stamp_creation(G, node, now_ms());
}

// Age of an already-inserted node, in seconds; nullopt if it carries no stamp (a node created
// before this existed, or by an agent outside this tree). Never negative — a clock step backwards
// clamps to 0 rather than reporting a node born in the future.
inline std::optional<double> age_s(DSR::DSRGraph& G, const DSR::Node& node)
{
    const auto t0 = G.get_attrib_by_name<timestamp_creation_att>(node);
    if (not t0.has_value() or t0.value() == 0)
        return {};
    const std::uint64_t now = now_ms();
    return now > t0.value() ? (now - t0.value()) / 1000.0 : 0.0;
}

}  // namespace rc::provenance
