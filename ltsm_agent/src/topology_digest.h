// The topology summary ltsm publishes on the live graph (node `ltsm_topology`, attribute ltsm_topology_att), because
// long-term memory lives on ltsm's own DSR domain and chat_agent cannot read it (10-13 design §1.5, §3.2(i)).
// PURE: no graph access; the caller turns memory records into TopoRoom / TopoPassage. Room names are the LIVE names
// (`room_1`, not the memory prefix `r<idx>_room_1`). Numbers go through std::format, which ignores LC_NUMERIC.
#pragma once

#include <format>
#include <optional>
#include <string>
#include <vector>

namespace rc::ltsm
{
struct TopoRoom { std::string name; bool live = false; };

struct TopoPassage
{
    std::string id, room_a, face_a, room_b, face_b;
    int crossings = 0;
    double pass_alpha = 1, pass_beta = 1, open_alpha = 1, open_beta = 1;
    std::optional<std::string> live_door;      // the door node currently standing for it, when it is in view
};

inline std::string topo_json_string(const std::string& s)
{
    std::string o = "\"";
    for (const char c : s)
    {
        if (c == '"' or c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) o += std::format("\\u{:04x}", static_cast<unsigned>(c));
        else o += c;
    }
    return o + "\"";
}

// {"rooms":[{"name","live"}],"passages":[{"id","a":{"room","face"},"b":{"room","face"},"crossings",
//  "p_pass":{"alpha","beta"},"p_open":{"alpha","beta"},"live_door"?}]}
inline std::string encode_topology(const std::vector<TopoRoom>& rooms, const std::vector<TopoPassage>& passages)
{
    std::string o = "{\"rooms\":[";
    for (std::size_t i = 0; i < rooms.size(); ++i)
        o += std::format("{}{{\"name\":{},\"live\":{}}}", i ? "," : "", topo_json_string(rooms[i].name), rooms[i].live);
    o += "],\"passages\":[";
    for (std::size_t i = 0; i < passages.size(); ++i)
    {
        const auto& p = passages[i];
        o += std::format("{}{{\"id\":{},\"a\":{{\"room\":{},\"face\":{}}},\"b\":{{\"room\":{},\"face\":{}}},"
                         "\"crossings\":{},\"p_pass\":{{\"alpha\":{},\"beta\":{}}},\"p_open\":{{\"alpha\":{},\"beta\":{}}}",
                         i ? "," : "", topo_json_string(p.id), topo_json_string(p.room_a), topo_json_string(p.face_a),
                         topo_json_string(p.room_b), topo_json_string(p.face_b), p.crossings,
                         p.pass_alpha, p.pass_beta, p.open_alpha, p.open_beta);
        if (p.live_door) o += std::format(",\"live_door\":{}", topo_json_string(*p.live_door));
        o += "}";
    }
    return o + "]}";
}
}   // namespace rc::ltsm
