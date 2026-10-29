#pragma once
// Graph half of the proposal stage (plan Task 2.3): writes mount_proposal / _version / _ready on the agent's OWN
// node, type-attributed, never an RT edge (robot_concept is their only writer).
#include "../../common/sensor_calib/proposal_publisher.h"

#include <dsr/api/dsr_api.h>

namespace sc
{
inline bool write_proposal(DSR::DSRGraph& G, std::uint64_t own_node_id, const calib::ProposalPayload& p)
{
    auto n = G.get_node(own_node_id);
    if (not n)
        return false;
    G.add_or_modify_attrib_local<mount_proposal_att>(*n, std::vector<float>(p.mount_proposal));
    G.add_or_modify_attrib_local<mount_proposal_version_att>(*n, int(p.version));
    G.add_or_modify_attrib_local<mount_proposal_ready_att>(*n, bool(p.ready));
    return G.update_node(*n);
}
}   // namespace sc
