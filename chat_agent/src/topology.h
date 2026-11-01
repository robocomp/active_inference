// Remembered topology as chat_agent sees it: the `ltsm_topology` summary ltsm publishes (ltsm_agent/src/
// topology_digest.h), decoded, and the cheapest route over it. PURE (Qt Core JSON only, no graph).
// Edge cost is 1 / E[p_pass] = (alpha + beta) / alpha, so a route prefers passages that have worked, and parallel
// passages between the same rooms resolve to the more reliable one. Passages are undirected.
#pragma once

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <algorithm>
#include <cctype>
#include <expected>
#include <format>
#include <map>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <vector>

#include "name_norm.h"

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

// The edge cost of crossing a passage: 1 / E[p_pass].
inline double passage_cost(const TopoPassageInfo& p) { return (p.pass_alpha + p.pass_beta) / p.pass_alpha; }

// A room name as the person/model spelled it -> the topology's own spelling ("Room 2" -> "room_2"); unchanged when no
// remembered room matches under the one name rule (name_norm.h).
[[nodiscard]] inline std::string canon_room(const Topology& t, std::string_view name)
{
    const std::string k = norm_name(name);
    for (const auto& r : t.rooms) if (norm_name(r.name) == k) return r.name;
    for (const auto& p : t.passages)
    {
        if (norm_name(p.room_a) == k) return p.room_a;
        if (norm_name(p.room_b) == k) return p.room_b;
    }
    return std::string(name);
}

// Cheapest route as (cost, passage ids in order); nullopt when unreachable; from == to is (0, {}).
[[nodiscard]] inline std::optional<std::pair<double, std::vector<std::string>>>
route_cost(const Topology& t, std::string_view from_given, std::string_view to_given)
{
    const std::string from_room = canon_room(t, from_given), to_room = canon_room(t, to_given);
    if (from_room == to_room) return std::pair{0.0, std::vector<std::string>{}};
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
            const double nd = d + passage_cost(p);
            if (const auto it = dist.find(*other); it == dist.end() or nd < it->second)
            {
                dist[*other] = nd;
                prev[*other] = {r, i};
                q.push({nd, *other});
            }
        }
    }
    const auto end = dist.find(to_room);
    if (end == dist.end()) return std::nullopt;
    std::vector<std::string> out;
    for (std::string r(to_room); r != from_room; r = prev.at(r).room)
        out.push_back(t.passages[prev.at(r).passage].id);
    std::ranges::reverse(out);
    return std::pair{end->second, std::move(out)};
}

// Passage ids in order from `from_room` to `to_room`; empty when unreachable (or from == to).
inline std::vector<std::string> route(const Topology& t, std::string_view from_room, std::string_view to_room)
{
    if (auto r = route_cost(t, from_room, to_room)) return std::move(r->second);
    return {};
}

// The cheapest route that crosses `passage_id` (the imagined "go through that one instead"): from -> one end,
// the passage, the other end -> to; the cheaper of the two directions. nullopt when the id is unknown or neither
// direction is reachable.
[[nodiscard]] inline std::optional<std::vector<std::string>>
route_via(const Topology& t, std::string_view from_given, std::string_view to_given, std::string_view passage_id)
{
    const std::string from_room = canon_room(t, from_given), to_room = canon_room(t, to_given);
    const auto pit = std::ranges::find(t.passages, passage_id, &TopoPassageInfo::id);
    if (pit == t.passages.end()) return std::nullopt;
    std::optional<std::pair<double, std::vector<std::string>>> best;
    const auto side = [&](const std::string& near_room, const std::string& far_room)
    {
        const auto head = route_cost(t, from_room, near_room);
        const auto tail = route_cost(t, far_room, to_room);
        if (not head or not tail) return;
        const double cost = head->first + passage_cost(*pit) + tail->first;
        if (best and best->first <= cost) return;
        std::vector<std::string> ids = head->second;
        ids.push_back(pit->id);
        ids.insert(ids.end(), tail->second.begin(), tail->second.end());
        best = std::pair{cost, std::move(ids)};
    };
    side(pit->room_a, pit->room_b);
    side(pit->room_b, pit->room_a);
    if (not best) return std::nullopt;
    return std::move(best->second);
}

// A name the person/LLM uses -> the passage id: a passage id, or the live door a passage carries. The one name rule
// (name_norm.h): "the door 2", "Door-2" and "door_2" are the same.
[[nodiscard]] inline std::optional<std::string> resolve_passage(const Topology& t, std::string_view name)
{
    const std::string k = norm_name(name);
    for (const auto& p : t.passages) if (norm_name(p.id) == k) return p.id;
    for (const auto& p : t.passages) if (p.live_door and norm_name(*p.live_door) == k) return p.id;
    return std::nullopt;
}
// ── tool answers (text the model reads; PURE) ─────────────────────────────────────────────────────────────────
// The other end of a passage, as seen from `room`; empty when `room` is neither end.
inline std::string topo_other_room(const TopoPassageInfo& p, std::string_view room)
{
    return p.room_a == room ? p.room_b : p.room_b == room ? p.room_a : std::string{};
}
inline std::string topo_door_words(const TopoPassageInfo& p)
{
    return std::format("{} <-> {} ({}), crossed {}x, passable {:.0f}%{}", p.room_a, p.room_b, p.id, p.crossings,
                       100 * p.p_pass(), p.live_door ? std::format(", live door {}", *p.live_door) : std::string{});
}

// known_rooms(): every remembered room (marking the live ones) and every passage with its two rooms.
inline std::string answer_known_rooms(const Topology& t, std::string_view current)
{
    if (t.rooms.empty()) return "No rooms are remembered yet.";
    std::string o = "Remembered rooms:";
    for (const auto& r : t.rooms)
        o += std::format(" {}{}{};", r.name, r.name == current ? " (you are here)" : "", r.live and r.name != current ? " (live)" : "");
    o += t.passages.empty() ? "\nNo passages between them are remembered." : "\nPassages:";
    for (const auto& p : t.passages) o += "\n- " + topo_door_words(p);
    return o;
}

// route_to(room): the passages in order from the current room. from == to and unreachable are told apart HERE
// because route() returns {} for both.
inline std::string answer_route_to(const Topology& t, std::string_view current, std::string_view to)
{
    if (to == current) return std::format("You are already in {}.", to);
    if (std::ranges::none_of(t.rooms, [&](const auto& r) { return r.name == to; }))
        return std::format("No remembered room is called '{}'. Call known_rooms to see the names.", to);
    const auto ids = route(t, current, to);
    if (ids.empty())
        return std::format("No remembered route from {} to {}: no chain of passages joins them.", current, to);
    std::string o = std::format("Route from {} to {}, {} passage(s) in order:", current, to, ids.size());
    std::string at(current);
    for (const auto& id : ids)
    {
        const auto it = std::ranges::find(t.passages, id, &TopoPassageInfo::id);
        const auto next = topo_other_room(*it, at);
        o += std::format("\n- cross passage:{}@{} expect={}{}", id, at, next,
                         it->live_door ? std::format(" (live door there: {})", *it->live_door) : std::string{});
        at = next;
    }
    return o;
}

// The STATE line: `rooms in long-term memory (call route_to for the way): room_2 (behind door_1, crossed 3x), room_3 (...)`; empty when nothing is remembered
// besides the current room.
/// The head of the chat's long-term-memory STATE line, and the planner's copy of it (planner_state_text).
inline constexpr std::string_view kRememberedRoomsHead = "rooms in long-term memory (call route_to for the way): ";
inline constexpr std::string_view kRememberedRoomsPlannerHead = "rooms in long-term memory: ";

inline std::string remembered_rooms_line(const Topology& t, std::string_view current)
{
    std::string o;
    for (const auto& r : t.rooms)
    {
        if (r.name == current) continue;
        std::string how;
        if (const auto ids = route(t, current, r.name); not ids.empty())
        {
            const auto it = std::ranges::find(t.passages, ids.front(), &TopoPassageInfo::id);
            how = std::format("{}{}, crossed {}x", ids.size() > 1 ? std::format("{} passages away, first ", ids.size()) : std::string{},
                              it->live_door ? "behind " + *it->live_door : std::string("via ") + it->id, it->crossings);
        }
        else
            how = "no known route";
        o += std::format("{}{} ({})", o.empty() ? "" : ", ", r.name, how);
    }
    return o.empty() ? o : std::string(kRememberedRoomsHead) + o;
}
// A name a mission may use from the long-term memory: a remembered room (ltsm's recall re-creates it with the SAME name
// when the robot crosses into it, room_recall.h) or a passage id. Doors and faces are not: a door is a live name.
inline bool remembered_name(const Topology& t, std::string_view name)
{
    return std::ranges::any_of(t.rooms, [&](const auto& r) { return r.name == name; })
        or std::ranges::any_of(t.passages, [&](const auto& p) { return p.id == name; });
}

// The PLANNER's STATE line (plan_mission; not the chat's): for every remembered room reachable from `current`, the
// legs in order as mission steps — the first by its live door when it has one (a door of the current room), every
// later one as passage:ID@ROOM, ROOM being the room that leg starts in. Empty when no room is reachable.
// `ways from here (each leg in order): room_2 = cross door_1 expect=room_2; room_3 = cross door_1 expect=room_2, cross passage:p1@room_2 expect=room_3`
inline std::string ways_line(const Topology& t, std::string_view current)
{
    std::string o;
    for (const auto& r : t.rooms)
    {
        if (r.name == current) continue;
        const auto ids = route(t, current, r.name);
        if (ids.empty()) continue;
        std::string legs, at(current);
        for (const auto& id : ids)
        {
            const auto it = std::ranges::find(t.passages, id, &TopoPassageInfo::id);
            const std::string next = topo_other_room(*it, at);
            const std::string ref = (legs.empty() and it->live_door) ? *it->live_door : "passage:" + id + "@" + at;
            legs += std::format("{}cross {} expect={}", legs.empty() ? "" : ", ", ref, next);
            at = next;
        }
        o += std::format("{}{} = {}", o.empty() ? "" : "; ", r.name, legs);
    }
    return o.empty() ? o : "ways from here (each leg in order): " + o;
}

/// The planner's STATE from the chat's: ★ the planner has NO tools (final review I6), so the long-term-memory line loses
/// its "(call route_to for the way)" and the way itself is given instead — `ways` (ways_line), appended as its own line.
/// tests/check_llm_missions.py builds its remembered-room STATE the same way.
inline std::string planner_state_text(std::string_view chat_state, std::string_view ways)
{
    std::string st(chat_state);
    if (const auto at = st.find(kRememberedRoomsHead); at != std::string::npos)
        st.replace(at, kRememberedRoomsHead.size(), kRememberedRoomsPlannerHead);
    if (not ways.empty()) st += std::string(ways) + "\n";
    return st;
}
}   // namespace rc::chat
