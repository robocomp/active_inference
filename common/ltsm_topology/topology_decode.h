// topology_decode.h — the `ltsm_topology` summary ltsm publishes on the live graph (ltsm_agent/src/topology_digest.h),
// DECODED. Shared by language_agent (routes, answers) and the controller's exploration drive (crossed_doors). PURE (Qt Core
// JSON only, no graph). Moved from language_agent/src/topology.h (2026-11-05), which re-exports these names as rc::chat::.
#pragma once

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <expected>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace rc::topo
{
struct TopoRoomInfo { std::string name; bool live = false; };

struct TopoPassageInfo
{
    std::string id, room_a, face_a, room_b, face_b;
    int crossings = 0;
    double pass_alpha = 1, pass_beta = 1, open_alpha = 1, open_beta = 1;
    std::optional<std::string> live_door;
    double p_pass() const { return pass_alpha / (pass_alpha + pass_beta); }
};

struct Topology
{
    std::vector<TopoRoomInfo> rooms;
    std::vector<TopoPassageInfo> passages;
    std::optional<std::vector<std::uint64_t>> room_ids;   // every room_id in LTSM memory; nullopt = the publisher predates the field
};

inline std::expected<Topology, std::string> decode_topology(std::string_view json)
{
    QJsonParseError err;
    const auto doc = QJsonDocument::fromJson(QByteArray(json.data(), static_cast<qsizetype>(json.size())), &err);
    if (err.error != QJsonParseError::NoError or not doc.isObject())
        return std::unexpected("ltsm_topology: not a JSON object");
    const auto root = doc.object();
    const auto str = [](const QJsonObject& o, const char* k) { return o.value(k).toString().toStdString(); };
    Topology t;
    for (const auto& v : root.value("rooms").toArray())
    {
        const auto o = v.toObject();
        t.rooms.push_back({str(o, "name"), o.value("live").toBool()});
    }
    if (root.contains("room_ids"))
    {
        t.room_ids.emplace();
        for (const auto& v : root.value("room_ids").toArray()) t.room_ids->push_back(static_cast<std::uint64_t>(v.toDouble()));
    }
    for (const auto& v : root.value("passages").toArray())
    {
        const auto o = v.toObject();
        TopoPassageInfo p;
        p.id = str(o, "id");
        p.room_a = o.value("a").toObject().value("room").toString().toStdString();
        p.face_a = o.value("a").toObject().value("face").toString().toStdString();
        p.room_b = o.value("b").toObject().value("room").toString().toStdString();
        p.face_b = o.value("b").toObject().value("face").toString().toStdString();
        p.crossings = o.value("crossings").toInt();
        const auto pp = o.value("p_pass").toObject(), po = o.value("p_open").toObject();
        p.pass_alpha = pp.value("alpha").toDouble(1); p.pass_beta = pp.value("beta").toDouble(1);
        p.open_alpha = po.value("alpha").toDouble(1); p.open_beta = po.value("beta").toDouble(1);
        if (o.contains("live_door")) p.live_door = str(o, "live_door");
        if (p.id.empty()) return std::unexpected("ltsm_topology: passage without id");
        t.passages.push_back(std::move(p));
    }
    return t;
}

// Memory names a face r<digits>_<live name> (ltsm_agent topology_digest.h live_name): the live name is what follows.
[[nodiscard]] inline std::string face_live_name(const std::string& mem)
{
    const auto us = mem.find('_');
    if (us != std::string::npos and us > 1 and mem.starts_with('r')
        and std::all_of(mem.begin() + 1, mem.begin() + static_cast<std::ptrdiff_t>(us), [](unsigned char c) { return std::isdigit(c); }))
        return mem.substr(us + 1);
    return mem;
}

// The door names the LTSM remembers as CROSSED: both faces and the live door of every passage with crossings >= 1. Empty
// when the summary is absent or malformed (ltsm not up yet) — "nothing remembered", never an error that blocks a caller.
[[nodiscard]] inline std::set<std::string> crossed_doors(std::string_view json)
{
    std::set<std::string> out;
    const auto t = decode_topology(json);
    if (not t.has_value()) return out;
    for (const auto& p : t->passages)
    {
        if (p.crossings < 1) continue;
        for (const auto& f : {p.face_a, p.face_b}) if (not f.empty()) out.insert(face_live_name(f));
        if (p.live_door.has_value()) out.insert(*p.live_door);
    }
    return out;
}
// Passages of `room` the LTSM remembers, never crossed, whose door is not in view (no live_door): a re-entered room shows
// only the face it was entered by at first, and its other remembered doors are found again by the room's looks. 0 when the
// summary is absent or malformed.
[[nodiscard]] inline int remembered_uncrossed(std::string_view json, std::string_view room)
{
    const auto t = decode_topology(json);
    if (not t.has_value()) return 0;
    int n = 0;
    for (const auto& p : t->passages)
        if ((p.room_a == room or p.room_b == room) and p.crossings < 1 and not p.live_door.has_value()) ++n;
    return n;
}

// The live door now standing for the passage one of whose faces was called `name` (a door name goes stale when its room
// is left and re-entered): that passage's live_door, or nullopt when none is in view / no passage has such a face.
[[nodiscard]] inline std::optional<std::string> live_door_for(std::string_view json, const std::string& name)
{
    const auto t = decode_topology(json);
    if (not t.has_value()) return std::nullopt;
    for (const auto& p : t->passages)
        if (face_live_name(p.face_a) == name or face_live_name(p.face_b) == name or p.live_door == name)
            // never the name asked about (explore_tour6: right after a crossing live_door still named the crossed face)
            return p.live_door.has_value() and *p.live_door != name ? p.live_door : std::nullopt;
    return std::nullopt;
}
}  // namespace rc::topo
