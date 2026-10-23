// Remembered topology as chat_agent sees it: the `ltsm_topology` summary ltsm publishes (ltsm_agent/src/
// topology_digest.h), decoded, and the cheapest route over it. PURE (Qt Core JSON only, no graph).
// Edge cost is 1 / E[p_pass] = (alpha + beta) / alpha, so a route prefers passages that have worked, and parallel
// passages between the same rooms resolve to the more reliable one. Passages are undirected.
#pragma once

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <algorithm>
#include <expected>
#include <map>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <vector>

namespace rc::chat
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

// Passage ids in order from `from_room` to `to_room`; empty when unreachable (or from == to).
inline std::vector<std::string> route(const Topology& t, std::string_view from_room, std::string_view to_room)
{
    if (from_room == to_room) return {};
    struct Prev { std::string room; std::size_t passage; };
    std::map<std::string, double, std::less<>> dist;
    std::map<std::string, Prev> prev;
    using Item = std::pair<double, std::string>;
    std::priority_queue<Item, std::vector<Item>, std::greater<>> q;
    dist[std::string(from_room)] = 0;
    q.push({0, std::string(from_room)});
    while (not q.empty())
    {
        auto [d, r] = q.top(); q.pop();
        if (d > dist[r]) continue;
        if (r == to_room) break;
        for (std::size_t i = 0; i < t.passages.size(); ++i)
        {
            const auto& p = t.passages[i];
            const std::string* other = p.room_a == r ? &p.room_b : p.room_b == r ? &p.room_a : nullptr;
            if (not other) continue;
            const double nd = d + (p.pass_alpha + p.pass_beta) / p.pass_alpha;
            if (const auto it = dist.find(*other); it == dist.end() or nd < it->second)
            {
                dist[*other] = nd;
                prev[*other] = {r, i};
                q.push({nd, *other});
            }
        }
    }
    if (not dist.contains(std::string(to_room))) return {};
    std::vector<std::string> out;
    for (std::string r(to_room); r != from_room; r = prev.at(r).room)
        out.push_back(t.passages[prev.at(r).passage].id);
    std::ranges::reverse(out);
    return out;
}
}   // namespace rc::chat
