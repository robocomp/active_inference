#include "room_proposal.h"
#include "room_eviction.h"     // RoomEviction::rt_attributes
#include "room_recall.h"       // RoomRecall::live_name_of

#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QDebug>
#include <QString>

#include <format>

namespace ltsm
{
namespace
{
bool is_proposal(DSR::DSRGraph &g, const DSR::Node &n)
{
    return n.type() == "metaconcept" and g.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) == RoomProposal::kSubtype;
}
}   // namespace

RoomProposal::RoomProposal(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> mem)
    : live_(std::move(live)), mem_(std::move(mem)) {}

std::uint64_t RoomProposal::post(std::uint64_t P, const std::string &K_mem, const Mat::RTMat &P_from_K, const std::vector<float> &prior)
{
    const auto Pn = live_->get_node(P);
    const auto K = mem_->get_node(K_mem);
    if (not Pn.has_value() or not K.has_value()) return 0;
    const auto px = mem_->get_attrib_by_name<delimiting_polygon_x_att>(K.value());
    const auto py = mem_->get_attrib_by_name<delimiting_polygon_y_att>(K.value());
    if (not px.has_value() or not py.has_value()) return 0;

    // Superseded: ours, any other proposal under P, and a stale namesake (the name must be free for the new node).
    const std::string name = "proposal_" + RoomRecall::live_name_of(K_mem);
    withdraw();
    for (const auto &n : live_->get_nodes_by_type("metaconcept"))
        if (is_proposal(*live_, n) and (live_->get_attrib_by_name<parent_att>(n).value_or(0) == P or n.name() == name))
        {
            qInfo().noquote() << QString::fromStdString(std::format("[proposal] {} superseded: deleted", n.name()));
            live_->delete_node(n.id());
        }

    DSR::Node n;
    n.type("metaconcept");
    n.name(name);
    live_->add_or_modify_attrib_local<object_subtype_att>(n, std::string(kSubtype));
    live_->add_or_modify_attrib_local<state_att>(n, std::string(kPending));
    live_->add_or_modify_attrib_local<parent_att>(n, P);
    live_->add_or_modify_attrib_local<level_att>(n, live_->get_node_level(Pn.value()).value_or(0) + 1);
    live_->add_or_modify_attrib_local<room_id_att>(n, mem_->get_attrib_by_name<room_id_att>(K.value()).value_or(0));
    live_->add_or_modify_attrib_local<delimiting_polygon_x_att>(n, std::vector<float>(px->get()));
    live_->add_or_modify_attrib_local<delimiting_polygon_y_att>(n, std::vector<float>(py->get()));
    if (not prior.empty()) live_->add_or_modify_attrib_local<recall_pose_prior_att>(n, std::vector<float>(prior));
    live_->add_or_modify_attrib_local<pos_x_att>(n, live_->get_attrib_by_name<pos_x_att>(Pn.value()).value_or(0.f) + 60.f);
    live_->add_or_modify_attrib_local<pos_y_att>(n, live_->get_attrib_by_name<pos_y_att>(Pn.value()).value_or(0.f) + 60.f);
    const auto id = live_->insert_node(n);
    if (not id.has_value()) return 0;
    if (not live_->insert_or_assign_edge(DSR::Edge::create<RT_edge_type>(P, id.value(), RoomEviction::rt_attributes(P_from_K))))
    {
        live_->delete_node(id.value());
        return 0;
    }
    id_ = id.value();
    P_ = P;
    qInfo().noquote() << QString::fromStdString(std::format("[proposal] {} posted under {}: {} is remembered {} (pending room_concept's verdict)",
                                                            name, Pn->name(), Pn->name(), K_mem));
    return id_;
}

RoomProposal::Verdict RoomProposal::verdict() const
{
    if (id_ == 0) return Verdict::Gone;
    const auto n = live_->get_node(id_);
    if (not n.has_value() or not live_->get_node(P_).has_value()) return Verdict::Gone;
    const auto s = live_->get_attrib_by_name<state_att>(n.value()).value_or(std::string(kPending));
    if (s == kAccepted) return Verdict::Accepted;
    if (s == kRejected) return Verdict::Rejected;
    return Verdict::Pending;
}

void RoomProposal::withdraw()
{
    if (id_ != 0)
        if (const auto n = live_->get_node(id_); n.has_value() and live_->delete_node(id_))
            qInfo().noquote() << QString::fromStdString(std::format("[proposal] {} withdrawn", n->name()));
    id_ = 0;
    P_ = 0;
}

int RoomProposal::sweep_stale()
{
    int n = 0;
    for (const auto &node : live_->get_nodes_by_type("metaconcept"))
        if (is_proposal(*live_, node) and live_->delete_node(node.id())) ++n;
    id_ = 0;
    P_ = 0;
    return n;
}
}   // namespace ltsm
