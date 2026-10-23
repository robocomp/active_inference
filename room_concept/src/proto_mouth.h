#pragma once
// The DSR write of a proto room's MOUTH pose (Task M), factored out of the birth site so a graph-level test can run
// the very code the birth executes. Pure geometry/covariance live in door_crossing.h (mouth_of).
#include <dsr/api/dsr_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>
#include <cstdint>
#include <vector>
#include "door_crossing.h"

namespace rc::crossing
{
    /// ⚠ NODE-LEVEL OVERLOAD of the RT attribute names: rt_translation / rt_rotation_euler_xyz / rt_covariance are edge
    /// attributes everywhere else. On the proto room NODE they mean "this room's frame in the frame of the room named by
    /// interaction_node_id" (the room that owns the crossed aperture). Nothing reads RT attributes off nodes; ltsm erases
    /// them when it archives the room (twin()). Covariance sources: see mouth_of. `sigma_tangent_known` <= 0 (door_concept
    /// stated none) falls to the layout bar; `bar` = the layout's publish_corner_sigma.
    inline Mouth write_mouth(DSR::DSRGraph& G, DSR::Node& proto, const Crossing& c, float sigma_tangent_known, float bar,
                             std::uint64_t old_room_id)
    {
        const Mouth mouth = mouth_of(c, sigma_tangent_known > 0.f ? sigma_tangent_known : bar, bar);
        G.add_or_modify_attrib_local<rt_translation_att>(proto, std::vector<float>{mouth.x, mouth.y, 0.f});
        G.add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(proto, std::vector<float>{0.f, 0.f, mouth.yaw});
        if (mouth.has_cov)
            G.add_or_modify_attrib_local<rt_covariance_att>(proto, std::vector<float>(mouth.cov.begin(), mouth.cov.end()));
        G.add_or_modify_attrib_local<interaction_node_id_att>(proto, old_room_id);
        return mouth;
    }
}
