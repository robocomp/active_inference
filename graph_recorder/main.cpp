// graph_recorder — a PASSIVE DSR participant that logs the structural history of the shared graph (2026-10-30).
//
// Why: the protocol mock (room_concept/estimation_scripts/scripts/protocol_mock.py) must mirror the real fleet. This
// records what the fleet ACTUALLY writes, with the writer's agent id and a wall-clock stamp, so that (1) the mock's
// timings are calibrated from measurements, (2) the mock's invariant checker can run on a real trace (a violation there
// is a real bug; a sequence the mock cannot produce is a mock bug), and (3) known live failures can be replayed.
//
// It never writes to the graph. Logged: node insert / rename / delete; the first appearance and the deletion of each
// edge (from, to, type) — per-frame RT updates are NOT logged (an RT edge that already exists is skipped); attribute
// changes only for a whitelist of protocol attributes (state, room_id, parent, recall_late), with their new value.
// A snapshot of every node and edge is written once the graph has synced.
// Signals are connected QUEUED to the main thread (CLAUDE.md: never Qt::DirectConnection).
// usage: graph_recorder <out.csv> [agent_id=28] [domain=0]      stop with SIGTERM / SIGINT (flushes and leaves).
#include <dsr/api/dsr_api.h>

#include <QCoreApplication>
#include <QTimer>

#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <locale>
#include <map>
#include <memory>
#include <print>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace
{
std::int64_t now_ms()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string csv(std::string s)   // names are plain identifiers; quote defensively anyway
{
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string o = "\"";
    for (char c : s) { if (c == '"') o += '"'; o += c; }
    return o + '"';
}

struct Recorder
{
    std::shared_ptr<DSR::DSRGraph> G;
    std::ofstream out;
    std::map<std::uint64_t, std::pair<std::string, std::string>> nodes;   // id -> (name, type)
    std::set<std::tuple<std::uint64_t, std::uint64_t, std::string>> edges;  // live edges seen

    // t_ms,event,agent,a_id,a_name,a_type,b_id,b_name,edge_type,attr,value
    void row(std::string_view ev, std::uint32_t agent, std::uint64_t a, std::uint64_t b = 0, std::string_view etype = {},
             std::string_view attr = {}, const std::string& value = {})
    {
        const auto na = nodes.contains(a) ? nodes[a] : std::pair<std::string, std::string>{};
        const auto nb = b != 0 and nodes.contains(b) ? nodes[b] : std::pair<std::string, std::string>{};
        out << now_ms() << ',' << ev << ',' << agent << ',' << a << ',' << csv(na.first) << ',' << na.second << ','
            << (b != 0 ? std::to_string(b) : std::string{}) << ',' << csv(nb.first) << ',' << etype << ',' << attr << ','
            << csv(value) << '\n';
    }

    void on_node(std::uint64_t id, const std::string& type, DSR::SignalInfo info)
    {
        const auto n = G->get_node(id);
        if (not n.has_value()) return;
        const auto it = nodes.find(id);
        if (it == nodes.end())
        {
            nodes[id] = {n->name(), type};
            row("node_insert", info.agent_id, id);
            // the protocol attributes it was born with
            for (const auto* a : {"state", "room_id", "parent", "recall_late"}) on_attr(id, a, info, true);
        }
        else if (it->second.first != n->name())
        {
            const std::string old = it->second.first;
            it->second.first = n->name();
            row("node_rename", info.agent_id, id, 0, {}, "old_name", old);
        }
    }

    void on_attr(std::uint64_t id, std::string_view a, DSR::SignalInfo info, bool at_birth = false)
    {
        const auto n = G->get_node(id);
        if (not n.has_value()) return;
        std::string v;
        bool has = false;
        if (a == "state")       { if (auto x = G->get_attrib_by_name<state_att>(n.value()); x) { v = x.value(); has = true; } }
        else if (a == "room_id"){ if (auto x = G->get_attrib_by_name<room_id_att>(n.value()); x) { v = std::to_string(x.value()); has = true; } }
        else if (a == "parent") { if (auto x = G->get_attrib_by_name<parent_att>(n.value()); x) { v = std::to_string(x.value()); has = true; } }
        else if (a == "recall_late") { if (auto x = G->get_attrib_by_name<recall_late_att>(n.value()); x) { v = x.value() ? "1" : "0"; has = true; } }
        if (has or not at_birth) row(at_birth ? "attr_birth" : "attr", info.agent_id, id, 0, {}, a, v);
    }

    void on_edge(std::uint64_t from, std::uint64_t to, const std::string& type, DSR::SignalInfo info)
    {
        if (edges.insert({from, to, type}).second) row("edge_insert", info.agent_id, from, to, type);
    }

    void on_edge_deleted(const DSR::Edge& e, DSR::SignalInfo info)
    {
        edges.erase({e.from(), e.to(), e.type()});
        row("edge_delete", info.agent_id, e.from(), e.to(), e.type());
    }

    void on_node_deleted(const DSR::Node& n, DSR::SignalInfo info)
    {
        nodes[n.id()] = {n.name(), n.type()};
        row("node_delete", info.agent_id, n.id());
        nodes.erase(n.id());
        std::erase_if(edges, [&](const auto& k) { return std::get<0>(k) == n.id() or std::get<1>(k) == n.id(); });
    }

    void snapshot()
    {
        for (const auto& n : G->get_nodes())
        {
            nodes[n.id()] = {n.name(), n.type()};
            row("snap_node", 0, n.id());
        }
        for (const auto& n : G->get_nodes())
            for (const auto& [k, e] : n.fano())
                if (edges.insert({e.from(), e.to(), e.type()}).second) row("snap_edge", 0, e.from(), e.to(), e.type());
        out.flush();
    }
};

void on_signal(int) { QCoreApplication::quit(); }
}   // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    if (argc < 2) { std::println(stderr, "usage: graph_recorder <out.csv> [agent_id=28] [domain=0]"); return 2; }
    std::uint32_t agent_id = 28;
    int domain = 0;
    const auto parse = [](const char* s, auto& v) { std::from_chars(s, s + std::strlen(s), v); };   // locale-free (CLAUDE.md)
    if (argc > 2) parse(argv[2], agent_id);
    if (argc > 3) parse(argv[3], domain);

    Recorder r;
    r.out.open(argv[1], std::ios::trunc);
    r.out.imbue(std::locale::classic());
    if (not r.out) { std::println(stderr, "graph_recorder: cannot open {}", argv[1]); return 1; }
    r.out << "t_ms,event,agent,a_id,a_name,a_type,b_id,b_name,edge_type,attr,value\n";

    r.G = std::make_shared<DSR::DSRGraph>("graph_recorder", agent_id, std::string{}, true, static_cast<int8_t>(domain));
    std::println("graph_recorder: joined DSR as agent {} on domain {}, writing {}", agent_id, domain, argv[1]);

    // QUEUED to the main thread (the context object `&app` lives there); the payloads are copied at emit time.
    QObject::connect(r.G.get(), &DSR::DSRGraph::update_node_signal, &app,
                     [&](std::uint64_t id, const std::string& type, DSR::SignalInfo i) { r.on_node(id, type, i); }, Qt::QueuedConnection);
    QObject::connect(r.G.get(), &DSR::DSRGraph::update_node_attr_signal, &app,
                     [&](std::uint64_t id, const std::vector<std::string>& names, DSR::SignalInfo i)
                     {
                         for (const auto& a : names)
                             if (a == "state" or a == "room_id" or a == "parent" or a == "recall_late") r.on_attr(id, a, i);
                     }, Qt::QueuedConnection);
    QObject::connect(r.G.get(), &DSR::DSRGraph::update_edge_signal, &app,
                     [&](std::uint64_t f, std::uint64_t t, const std::string& type, DSR::SignalInfo i) { r.on_edge(f, t, type, i); },
                     Qt::QueuedConnection);
    QObject::connect(r.G.get(), &DSR::DSRGraph::deleted_edge_signal, &app,
                     [&](const DSR::Edge& e, DSR::SignalInfo i) { r.on_edge_deleted(e, i); }, Qt::QueuedConnection);
    QObject::connect(r.G.get(), &DSR::DSRGraph::deleted_node_signal, &app,
                     [&](const DSR::Node& n, DSR::SignalInfo i) { r.on_node_deleted(n, i); }, Qt::QueuedConnection);

    QTimer::singleShot(3000, &app, [&] { r.snapshot(); std::println("graph_recorder: snapshot of {} nodes", r.nodes.size()); });
    QTimer flush;
    QObject::connect(&flush, &QTimer::timeout, &app, [&] { r.out.flush(); });
    flush.start(1000);

    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);
    const int rc = app.exec();
    r.out.flush();
    r.G.reset();   // leave the DDS domain cleanly
    std::println("graph_recorder: stopped");
    return rc;
}
