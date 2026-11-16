#include "human_graph.h"
#include <algorithm>
#include <format>
#include <ctime>

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

namespace
{
// "2026-11-15 10:02": local time, as the person reads it.
std::string local_minute(std::int64_t t_ms)
{
    const std::time_t t = static_cast<std::time_t>(t_ms / 1000);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M", &tm);
    return buf;
}
std::string object_name(const Repertory* r, std::uint64_t uid)
{
    if (r != nullptr)
        if (const auto e = r->entries().find(uid); e != r->entries().end() and not e->second.name.empty()) return e->second.name;
    return std::to_string(uid);
}
constexpr const char* kBeliefPrefix = "ltom_belief_";
constexpr const char* kQuestionPrefix = "ltom_question_";
}

std::string belief_evidence_text(const Belief& b, const Repertory* r)
{
    std::string out;
    for (const BeliefRecord& x : b.records)
        out += std::format("{} | {} | {} | {}\n", local_minute(x.said.t_ms), x.yes ? "yes" : "no", x.said.words, object_name(r, x.object_uid));
    for (const Said& f : b.forgotten)
        out += std::format("{} | forgotten | {} | -\n", local_minute(f.t_ms), f.words);
    return out;
}

void HumanGraph::sync_questions(const std::map<std::uint64_t, Question>& open)
{
    const auto root = G_->get_node("root");
    if (not root) return;
    std::set<std::string> want;
    for (const auto& [stamp, q] : open) want.insert(kQuestionPrefix + std::to_string(stamp));
    for (const auto& n : G_->get_nodes_by_type("metaconcept"))
        if (n.name().starts_with(kQuestionPrefix) and not want.contains(n.name())) G_->delete_node(n.id());
    for (const auto& [stamp, q] : open)
    {
        const std::string name = kQuestionPrefix + std::to_string(stamp);
        if (G_->get_node(name)) continue;   // a question never changes once open
        auto n = DSR::Node::create<metaconcept_node_type>(name);
        G_->add_or_modify_attrib_local<parent_att>(n, root->id());
        G_->add_or_modify_attrib_local<level_att>(n, 1);
        G_->add_or_modify_attrib_local<question_kind_att>(n, std::string("moved"));
        G_->add_or_modify_attrib_local<question_live_uid_att>(n, q.live_uid);
        G_->add_or_modify_attrib_local<question_memory_uid_att>(n, q.memory_uid);
        G_->add_or_modify_attrib_local<question_old_xy_att>(n, std::vector<float>{static_cast<float>(q.old_kf.pose.x), static_cast<float>(q.old_kf.pose.y)});
        G_->add_or_modify_attrib_local<question_new_xy_att>(n, std::vector<float>{static_cast<float>(q.new_kf.pose.x), static_cast<float>(q.new_kf.pose.y)});
        G_->add_or_modify_attrib_local<question_room_att>(n, std::string("room_") + std::to_string(q.room_key));
        G_->add_or_modify_attrib_local<question_again_att>(n, q.again);
        G_->add_or_modify_attrib_local<question_stamp_att>(n, q.stamp);
        if (G_->insert_node(n)) std::println("[ltom] question opened {} (memory {} as live {})", name, q.memory_uid, q.live_uid);
        else std::println(stderr, "⚠ ltom: cannot insert {}", name);
    }
}

void HumanGraph::sync(const HumanBook& book, const std::optional<Answer>& last, const Repertory* r)
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

    // Beliefs learned from the person's answers: one `ltom_belief_<key>` metaconcept each, hanging from human_1 by
    // a `has` edge, with the counts and the dated evidence lines (never a bare number).
    for (const Belief& b : book.human().beliefs)
    {
        const std::string name = kBeliefPrefix + b.key;
        const int yes = static_cast<int>(std::ranges::count_if(b.records, &BeliefRecord::yes));
        const int no = static_cast<int>(b.records.size()) - yes;
        const std::string ev = belief_evidence_text(b, r);
        auto bn = G_->get_node(name);
        if (not bn)
        {
            auto n = DSR::Node::create<metaconcept_node_type>(name);
            G_->add_or_modify_attrib_local<parent_att>(n, human_id);
            G_->add_or_modify_attrib_local<level_att>(n, 2);
            G_->add_or_modify_attrib_local<belief_yes_att>(n, yes);
            G_->add_or_modify_attrib_local<belief_no_att>(n, no);
            G_->add_or_modify_attrib_local<belief_evidence_att>(n, ev);
            if (not G_->insert_node(n)) { std::println(stderr, "⚠ ltom: cannot insert {}", name); continue; }
            bn = G_->get_node(name);
            if (not bn) continue;
        }
        else if (G_->get_attrib_by_name<belief_yes_att>(*bn) != std::optional<int>(yes) or
                 G_->get_attrib_by_name<belief_no_att>(*bn) != std::optional<int>(no) or
                 G_->get_attrib_by_name<belief_evidence_att>(*bn).value_or("") != ev)
        {
            G_->add_or_modify_attrib_local<belief_yes_att>(*bn, yes);
            G_->add_or_modify_attrib_local<belief_no_att>(*bn, no);
            G_->add_or_modify_attrib_local<belief_evidence_att>(*bn, ev);
            G_->update_node(*bn);
        }
        if (not G_->get_edge(human_id, bn->id(), "has"))
            G_->insert_or_assign_edge(DSR::Edge::create<has_edge_type>(human_id, bn->id()));
    }
}
}
