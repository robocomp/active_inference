// Offline test of src/human_graph.h: the human at the chat on a standalone DSR graph (private DDS domain 63).
#include "../src/human_graph.h"
#include "../src/memory_graph.h"
#include "../../language_agent/src/ltom_query_dsr.h"   // the reader on the other side of this graph

#include <memory>
#include <print>

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

namespace
{
template <typename O> std::string str(const O& o) { if (not o) return "<none>"; return std::string(o.value()); }
constexpr std::uint64_t uid_of(int n) { return (std::uint64_t{20} << 40) | static_cast<std::uint64_t>(n); }

ltom::Entry chair(int n)
{
    ltom::Entry e;
    e.uid = uid_of(n); e.cls = "chair"; e.name = "chair_" + std::to_string(n);
    if (n == 34) { e.color_name = "magenta"; e.color_prob = 0.74f; }   // the question names it by colour
    e.lives.push_back({.born_ms = 1000, .first_seen_ms = 1100, .last_seen_ms = 5000, .end_ms = 5200, .end = ltom::End::death});
    e.keyframes.push_back({.t_ms = 1100, .room_key = 1, .pose = {1.0, 2.0, 0.5, 0.01, 0.01, 0.1}});
    return e;
}

struct Fixture
{
    std::shared_ptr<DSR::DSRGraph> G;
    ltom::Repertory rep{3000};
    ltom::MemoryGraph mem;
    ltom::HumanGraph human;
    ltom::HumanBook book;
    Fixture() : G(std::make_shared<DSR::DSRGraph>("ltom_human_test", 63, std::string(LTOM_ROOT_JSON), true, static_cast<int8_t>(63), DSR::Queue)),
                mem(G), human(G)
    {
        rep.insert_loaded(chair(34));
        rep.insert_loaded(chair(35));
        mem.sync(rep.entries(), {});
    }
    std::uint64_t id(const std::string& n) { const auto x = G->get_node(n); return x ? x->id() : 0; }
    bool owns(const std::string& obj) { return G->get_edge(id("human_1"), id(obj), "owns").has_value(); }
    void claim(ltom::ClaimKind k, std::uint64_t uid, std::uint64_t stamp, std::string value = {})
    { book.apply(ltom::Claim{k, std::move(value), uid, "words", stamp}, rep, 1000 + static_cast<std::int64_t>(stamp)); }
};

void test_person_node_with_name()
{
    Fixture f;
    f.human.sync(f.book, std::nullopt);
    const auto n = f.G->get_node("human_1");
    CHECK(n.has_value() and n->type() == "person");
    CHECK(n and str(f.G->get_attrib_by_name<person_role_att>(*n)) == "developer");
    CHECK(n and not f.G->get_attrib_by_name<person_name_att>(*n).has_value());
    f.claim(ltom::ClaimKind::name, 0, 1, "Pablo");
    f.human.sync(f.book, std::nullopt);
    CHECK(str(f.G->get_attrib_by_name<person_name_att>(*f.G->get_node("human_1"))) == "Pablo");
    CHECK(f.G->get_nodes_by_type("person").size() == 1);
    // Hangs from root by a non-geometric `has` edge: the chat partner is not a place, so no RT edge.
    CHECK(f.G->get_edge(f.id("root"), f.id("human_1"), "has").has_value());
    CHECK(not f.G->get_edge(f.id("root"), f.id("human_1"), "RT").has_value());
    f.human.sync(f.book, std::nullopt);   // repeatable
    CHECK(f.G->get_edges_by_type("has").size() == 1);
}
void test_owns_edge_to_the_memory_node()
{
    Fixture f;
    f.claim(ltom::ClaimKind::owns, uid_of(34), 1);
    f.human.sync(f.book, std::nullopt);
    CHECK(f.owns("chair_34"));
    CHECK(not f.owns("chair_35"));
    f.human.sync(f.book, std::nullopt);   // repeatable
    CHECK(f.G->get_edges_by_type("owns").size() == 1);
}
void test_retracted_link_not_drawn()
{
    Fixture f;
    f.claim(ltom::ClaimKind::owns, uid_of(34), 1);
    f.human.sync(f.book, std::nullopt);
    CHECK(f.owns("chair_34"));
    f.claim(ltom::ClaimKind::disowns, uid_of(34), 2);
    f.human.sync(f.book, std::nullopt);
    CHECK(not f.owns("chair_34"));
    CHECK(f.G->get_edges_by_type("owns").empty());
}
void test_owned_node_that_appears_later_gets_its_edge()
{
    Fixture f;
    f.rep.insert_loaded(chair(36));       // in the book's repertory, not yet on the graph
    f.claim(ltom::ClaimKind::owns, uid_of(36), 1);
    f.human.sync(f.book, std::nullopt);
    CHECK(f.id("chair_36") == 0);
    f.mem.sync(f.rep.entries(), {});
    f.human.sync(f.book, std::nullopt);
    CHECK(f.owns("chair_36"));
}
void test_answer_published()
{
    Fixture f;
    f.human.sync(f.book, ltom::Answer{77, "recorded"});
    const auto n = f.G->get_node("human_1");
    CHECK(n and str(f.G->get_attrib_by_name<human_claim_answer_att>(*n)) == "recorded");
    CHECK(n and f.G->get_attrib_by_name<human_claim_answer_stamp_att>(*n) == std::optional<std::uint64_t>(77));
}
void test_read_claim()
{
    Fixture f;
    CHECK(not ltom::read_claim(*f.G).has_value());          // no chat_intention node
    auto n = DSR::Node::create<intention_node_type>("chat_intention");
    f.G->add_or_modify_attrib_local<human_claim_kind_att>(n, std::string("owns"));
    f.G->add_or_modify_attrib_local<human_claim_uid_att>(n, uid_of(34));
    f.G->add_or_modify_attrib_local<human_claim_words_att>(n, std::string("that's my chair"));
    const auto id = f.G->insert_node(n);
    CHECK(id.has_value());
    CHECK(not ltom::read_claim(*f.G).has_value());          // no stamp: incomplete
    auto m = *f.G->get_node("chat_intention");
    f.G->add_or_modify_attrib_local<human_claim_stamp_att>(m, std::uint64_t{77});
    f.G->update_node(m);
    const auto c = ltom::read_claim(*f.G);
    CHECK(c.has_value());
    CHECK(c and c->kind == ltom::ClaimKind::owns and c->value.empty() and c->object_uid == uid_of(34));
    CHECK(c and c->words == "that's my chair" and c->stamp == 77);
    auto k = *f.G->get_node("chat_intention");
    f.G->add_or_modify_attrib_local<human_claim_kind_att>(k, std::string("bogus"));
    f.G->update_node(k);
    CHECK(not ltom::read_claim(*f.G).has_value());          // unknown kind
}
void test_language_agent_reads_the_human()
{
    Fixture f;
    CHECK(not rc::chat::load_human(*f.G).name.has_value());       // no human_1 yet: empty
    CHECK(rc::chat::load_human(*f.G).owned_uids.empty());
    f.claim(ltom::ClaimKind::name, 0, 1, "Pablo");
    f.claim(ltom::ClaimKind::owns, uid_of(34), 2);
    f.claim(ltom::ClaimKind::owns, uid_of(35), 3);
    f.claim(ltom::ClaimKind::disowns, uid_of(35), 4);
    f.human.sync(f.book, ltom::Answer{4, "recorded"});
    const auto h = rc::chat::load_human(*f.G);
    CHECK(h.name == std::optional<std::string>("Pablo"));
    CHECK(h.role == "developer");
    CHECK(h.owned_uids == std::set<std::uint64_t>{uid_of(34)});
    CHECK(h.answer_stamp == std::optional<std::uint64_t>(4) and h.answer == "recorded");
    // the human is not a remembered object
    const auto mem = rc::chat::load_remembered(*f.G, 10000);
    CHECK(std::ranges::none_of(mem.objects, [](const auto& o) { return o.name == "human_1"; }));
}
// ── Changed object (2026-11-15): questions and beliefs on the memory graph, as language_agent reads them ─────
ltom::Sighting live57(float same, float left)
{
    ltom::Sighting s; s.uid = uid_of(57); s.cls = "chair"; s.name = "chair_57"; s.room_key = 1; s.born_ms = 6000;
    s.pose = {3.0, -1.0, 0.0, 0.01, 0.01, 0.1};
    s.link = ltom::LinkObs{uid_of(34), same, left};
    return s;
}
void test_question_node_published_and_removed()
{
    Fixture f;
    const std::vector<ltom::Sighting> v{live57(0.9f, 0.8f)};
    f.rep.observe(v, {20}, {uid_of(57)}, 7000);
    f.mem.sync(f.rep.entries(), {uid_of(57)});
    f.human.sync_questions(f.rep.questions());
    CHECK(f.G->get_node("ltom_question_7000").has_value());
    const auto qs = rc::chat::load_questions(*f.G);
    CHECK(qs.size() == 1);
    if (qs.size() == 1)
    {
        CHECK(qs[0].stamp == 7000 and qs[0].memory_uid == uid_of(34) and qs[0].live_uid == uid_of(57));
        CHECK(qs[0].room == "room_1" and qs[0].cls == "chair" and qs[0].colour == "magenta" and qs[0].memory_name == "chair_34");
        CHECK(qs[0].old_xy.x == 1.0f and qs[0].old_xy.y == 2.0f and qs[0].new_xy.x == 3.0f and not qs[0].again);
    }
    f.human.sync_questions(f.rep.questions());   // repeatable
    CHECK(f.G->get_nodes_by_type("metaconcept").size() >= 1);
    f.book.apply(ltom::Claim{ltom::ClaimKind::moved, "yes:7000", uid_of(34), "yes, I moved it", 9}, f.rep, 8000);
    f.human.sync_questions(f.rep.questions());
    CHECK(not f.G->get_node("ltom_question_7000").has_value());
    CHECK(rc::chat::load_questions(*f.G).empty());
}
void test_belief_node_lines()
{
    Fixture f;
    const std::vector<ltom::Sighting> v{live57(0.9f, 0.8f)};
    f.rep.observe(v, {20}, {uid_of(57)}, 7000);
    f.book.apply(ltom::Claim{ltom::ClaimKind::moved, "yes:7000", uid_of(34), "yes, I moved it", 9}, f.rep, 8000);
    f.human.sync(f.book, std::nullopt, &f.rep);
    const auto bs = rc::chat::load_beliefs(*f.G);
    const auto it = std::ranges::find(bs, std::string("moves_things"), &rc::chat::BeliefFacts::key);
    CHECK(it != bs.end());
    if (it != bs.end())
    {
        CHECK(it->yes == 1 and it->no == 0 and it->evidence.size() == 1);
        CHECK(it->evidence.size() == 1 and it->evidence[0].contains(" | yes | yes, I moved it | chair_34"));
    }
    f.human.sync(f.book, std::nullopt, &f.rep);   // repeatable
    CHECK(rc::chat::load_beliefs(*f.G).size() == bs.size());
}
}   // namespace

int main()
{
    test_person_node_with_name();
    test_owns_edge_to_the_memory_node();
    test_retracted_link_not_drawn();
    test_owned_node_that_appears_later_gets_its_edge();
    test_answer_published();
    test_read_claim();
    test_language_agent_reads_the_human();
    test_question_node_published_and_removed();
    test_belief_node_lines();
    std::println("{}", failures == 0 ? "ALL PASS" : "FAILED");
    return failures == 0 ? 0 : 1;
}
