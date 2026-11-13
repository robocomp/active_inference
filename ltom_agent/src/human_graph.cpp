#include "human_graph.h"

#include <print>
#include <set>

namespace ltom
{
namespace
{
constexpr const char* kHumanNode = "human_1";
constexpr const char* kChatNode = "chat_intention";   // language_agent's own node on the working graph
constexpr float kHumanX = -300.f, kHumanY = 120.f;    // the canvas: left of room_1, on the rooms' row
}

std::optional<Claim> read_claim(DSR::DSRGraph& working)
{
    const auto n = working.get_node(kChatNode);
    if (not n) return std::nullopt;
    const auto kind = working.get_attrib_by_name<human_claim_kind_att>(*n);
    const auto stamp = working.get_attrib_by_name<human_claim_stamp_att>(*n);
    if (not kind or not stamp) return std::nullopt;
    const auto k = parse_claim_kind(kind.value().get());
    if (not k) return std::nullopt;
    Claim c;
    c.kind = *k;
    c.stamp = stamp.value();
    c.value = working.get_attrib_by_name<human_claim_value_att>(*n).value_or("");
    c.object_uid = working.get_attrib_by_name<human_claim_uid_att>(*n).value_or(0);
    c.words = working.get_attrib_by_name<human_claim_words_att>(*n).value_or("");
    return c;
}

HumanGraph::HumanGraph(std::shared_ptr<DSR::DSRGraph> memory) : G_(std::move(memory)) {}

void HumanGraph::sync(const HumanBook& book, const std::optional<Answer>& last)
{
    auto node = G_->get_node(kHumanNode);
    if (not node)
    {
        const auto root = G_->get_node("root");
        if (not root) return;
        auto n = DSR::Node::create<person_node_type>(kHumanNode);
        G_->add_or_modify_attrib_local<parent_att>(n, root->id());
        G_->add_or_modify_attrib_local<level_att>(n, 1);
        G_->add_or_modify_attrib_local<pos_x_att>(n, kHumanX);
        G_->add_or_modify_attrib_local<pos_y_att>(n, kHumanY);
        G_->add_or_modify_attrib_local<person_role_att>(n, book.human().role);
        if (const auto nm = book.name()) G_->add_or_modify_attrib_local<person_name_att>(n, *nm);
        if (not G_->insert_node(n)) { std::println(stderr, "⚠ ltom: cannot insert {}", kHumanNode); return; }
        node = G_->get_node(kHumanNode);
        if (not node) return;
    }
    // The chat partner hangs from root by a non-geometric `has` edge: not a place, so no RT edge (no pose to claim).
    // With a body (real Shadow) it will hang by RT from the room the robot believes the person is in.
    if (const auto root = G_->get_node("root"); root and not G_->get_edge(root->id(), node->id(), "has"))
        G_->insert_or_assign_edge(DSR::Edge::create<has_edge_type>(root->id(), node->id()));

    // Attributes: written only when they differ from what the node already says.
    bool changed = false;
    auto set_str = [&]<typename Att>(Att, const std::string& v)
    {
        const auto cur = G_->get_attrib_by_name<Att>(*node);
        if (cur and std::string(cur.value()) == v) return;
        G_->add_or_modify_attrib_local<Att>(*node, v);
        changed = true;
    };
    set_str(person_role_att{}, book.human().role);
    if (const auto nm = book.name()) set_str(person_name_att{}, *nm);
    if (last)
    {
        if (G_->get_attrib_by_name<human_claim_answer_stamp_att>(*node) != std::optional<std::uint64_t>(last->stamp))
        {
            G_->add_or_modify_attrib_local<human_claim_answer_stamp_att>(*node, last->stamp);
            changed = true;
        }
        set_str(human_claim_answer_att{}, last->text);
    }
    if (changed) G_->update_node(*node);

    // `owns` edges: exactly the memory nodes of the owned (not retracted) uids that are on the graph now.
    std::set<std::uint64_t> owned_uids;
    for (const Ownership* o : book.owned()) owned_uids.insert(o->object_uid);
    std::set<std::uint64_t> want;
    if (not owned_uids.empty())
        for (const auto& n : G_->get_nodes_by_type("object"))
            if (const auto uid = G_->get_attrib_by_name<object_uid_att>(n); uid and owned_uids.contains(uid.value()))
                want.insert(n.id());
    const auto human_id = node->id();
    std::set<std::uint64_t> have;
    for (const auto& e : DSR::DSRGraph::get_node_edges_by_type(*node, "owns")) have.insert(e.to());
    for (const auto to : have)
        if (not want.contains(to)) G_->delete_edge(human_id, to, "owns");
    for (const auto to : want)
        if (not have.contains(to))
            G_->insert_or_assign_edge(DSR::Edge::create<owns_edge_type>(human_id, to));
}
}
