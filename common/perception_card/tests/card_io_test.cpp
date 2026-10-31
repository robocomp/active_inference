// Card file I/O: atomic write, read, corrupt -> nullopt, tree scan.   LANG=es_ES.UTF-8 ./card_io_test
#include "perception_card/card_io.h"
#include <clocale>
#include <cstdio>
#include <fstream>
#include <map>
#include <type_traits>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
using namespace rc::card;
namespace fs = std::filesystem;

// Stub graph: one "agent" node with a string attribute map, counting writes.
struct StubNode { std::uint64_t id_ = 7; std::optional<std::string> card; std::string presence; std::uint64_t stamp = 0; };
struct StubAtt {};
struct StubPres {};
struct StubStamp {};
struct StubA { using card = StubAtt; using presence = StubPres; using stamp = StubStamp; };
struct StubGraph
{
    std::vector<StubNode> nodes; int writes = 0;
    std::optional<StubNode> get_node(std::uint64_t id)
    { for (auto& n : nodes) if (n.id_ == id) return n; return std::nullopt; }
    std::vector<StubNode> get_nodes_by_type(const std::string&) { return nodes; }
    template <class A> auto get_attrib_by_name(const StubNode& n)
    {
        if constexpr (std::is_same_v<A, StubAtt>) return n.card;
        else if constexpr (std::is_same_v<A, StubPres>) return std::optional<std::string>(n.presence);
        else return std::optional<std::uint64_t>(n.stamp);
    }
    template <class A> void add_or_modify_attrib_local(StubNode& n, const std::string& s) { n.card = s; }
    void update_node(const StubNode& n) { for (auto& m : nodes) if (m.id_ == n.id_) m = n; ++writes; }
};

static Card make(const std::string& name)
{
    Card c;
    c.concept_ = name; c.graph_type = name; c.level = Level::Countable;
    c.channels = {{"zed", Role::Both, true}};
    c.geometry = {0.7f, 0.05f, 2.0f, 0.0f, 2.05f};
    SubtypeParams d; d.env = {0.12f, 0.55f, 0.07f}; d.q_return = 0.85f;
    d.terms = {{"range_conf", {{"r0", 4.5f}, {"k", 0.123456f}}}};
    c.subtypes[""] = d;
    c.calibration = "fitted";
    return c;
}

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const fs::path root = fs::temp_directory_path() / "card_io_test_tree";
    fs::remove_all(root);
    fs::create_directories(root);

    const Card a = make("door");
    CHECK(write_card_file(root / "door_concept", a));      // creates etc/
    CHECK(fs::exists(root / "door_concept/etc/perception_card.json"));
    CHECK(not fs::exists(root / "door_concept/etc/perception_card.json.tmp"));
    const auto r = read_card_file(root / "door_concept");
    CHECK(r.has_value());
    CHECK(r and to_json(*r) == to_json(a));
    CHECK(r and r->subtypes.at("").terms[0].params.at("k") == 0.123456f);

    { // a relative blind_spot_map_path is resolved against agent_dir on write; an absolute one is left alone
        Card m = make("table");  m.calibration = "blind_spot_map";  m.blind_spot_map_path = "etc/bsm.csv";
        CHECK(write_card_file(root / "table_concept", m));
        const auto rm = read_card_file(root / "table_concept");
        CHECK(rm and fs::path(rm->blind_spot_map_path).is_absolute());
        CHECK(rm and rm->blind_spot_map_path == (fs::absolute(root / "table_concept") / "etc/bsm.csv").lexically_normal().string());
        m.blind_spot_map_path = "/abs/bsm.csv";
        CHECK(write_card_file(root / "table_concept", m));
        const auto ra = read_card_file(root / "table_concept");
        CHECK(ra and ra->blind_spot_map_path == "/abs/bsm.csv");
        fs::remove_all(root / "table_concept");
    }
    CHECK(write_card_file(root / "bottle_concept", make("bottle")));
    CHECK(write_card_file(root / "not_an_agent", make("zzz")));      // ignored: no _concept suffix
    fs::create_directories(root / "empty_concept");                  // no card: skipped silently
    fs::create_directories(root / "bad_concept/etc");
    { std::ofstream(root / "bad_concept/etc/perception_card.json") << "{ not json"; }
    CHECK(not read_card_file(root / "bad_concept").has_value());
    CHECK(not read_card_file(root / "empty_concept").has_value());

    const auto all = read_all_cards(root);
    CHECK(all.size() == 2);
    CHECK(all.size() == 2 and all[0].concept_ == "bottle" and all[1].concept_ == "door");

    { // graph publication: one write for two identical publishes; absent node -> false; read prefers the graph
        StubGraph g;
        CHECK(not publish_card<StubAtt>(g, 7, a) and g.writes == 0);
        g.nodes.push_back(StubNode{});
        CHECK(publish_card<StubAtt>(g, 7, a) and g.writes == 1);
        CHECK(publish_card<StubAtt>(g, 7, a) and g.writes == 1);
        Card a2 = a;  a2.calibration = "other";
        CHECK(publish_card<StubAtt>(g, 7, a2) and g.writes == 2);
        bool fg = false;
        const auto rg = read_card<StubA>(g, "door", root, &fg);
        CHECK(rg and fg and rg->calibration == "other");
        const auto rf = read_card<StubA>(g, "bottle", root, &fg);          // not in the graph -> file
        CHECK(rf and not fg and rf->concept_ == "bottle");
        StubGraph empty;                                            // no attribute anywhere -> file fallback
        const auto re = read_card<StubA>(empty, "door", root, &fg);
        CHECK(re and not fg and re->calibration == "fitted");
        CHECK(not read_card<StubA>(empty, "nothing", root, &fg).has_value());
        std::size_t ng = 9;
        const auto all2 = read_all_cards<StubA>(g, root, &ng);      // door from the graph, bottle from its file
        CHECK(ng == 1 and all2.size() == 2 and all2[0].concept_ == "bottle" and all2[1].calibration == "other");
        // two nodes claim "door": the live one wins even with an older heartbeat; else the newer heartbeat
        Card stale = a;  stale.calibration = "stale";
        StubNode n1; n1.id_ = 1; n1.card = to_json(stale); n1.presence = "Degraded"; n1.stamp = 900;
        StubNode n2; n2.id_ = 2; n2.card = to_json(a);     n2.presence = "Operating"; n2.stamp = 100;
        StubGraph two; two.nodes = {n1, n2};
        CHECK(read_card<StubA>(two, "door", root).value().calibration == "fitted");
        two.nodes[1].presence = "Degraded";
        CHECK(read_card<StubA>(two, "door", root).value().calibration == "stale");
    }
    { // determinism: the same card built twice from separate objects serialises byte-identically (publish compares strings)
        const Card x = make("door"), y = make("door");
        CHECK(to_json(x) == to_json(y));
        CHECK(graph_json(x) == graph_json(y));
    }

    fs::remove_all(root);
    std::printf(bad ? "FAILED %d\n" : "card_io_test OK\n", bad);
    return bad ? 1 : 0;
}
