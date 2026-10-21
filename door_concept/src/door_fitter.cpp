/*
 * door_fitter.cpp — door_concept's DSR-bound side of the fit: the instance map, ensure_instance's graph read and the
 * camera pose for the remembered-door look (the existence probes moved into the DoorCore, Task 5). The FIT itself — observe /
 * run_inference / estimate_phi / the pre-belief geometry and read-back / the ROI / the chain covariance — moved into the
 * DSR-free DoorCore (src/core/door_fit.{h,cpp}, Task 4b of the door-core replay plan).
 */

#include "door_fitter.h"
#include "core/door_core.h"   // point_in_polygon — the containment rule, shared with the birth test
#include "door_standpoint.h"
#include "door_phi_posterior.h"   // the phi channel as pure functions (tested offline)
#include "door_opening_edges.h"   // try96: wall registration, the opening's through-span, the registration offset

#include "../../common/exclusion/exclusion.h"   // rc::exclusion — the SHARED no-two-objects rule

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <limits>
#include <cstdint>
#include <print>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <array>
#include <opencv2/imgproc.hpp>   // fillConvexPoly / bitwise_and — the rasterised phi score
#include "../../common/rt_query_probe/rt_query_probe.h"
#include "../../common/room_resolve/room_resolve.h"   // rc::room::current_room_frame — the room is room_1/room_2/…, never the literal "room"

// (MaskCells — the hashed mask-pixel cells the existence probes share — moved with them: src/core/door_existence.cpp.)

namespace rc {

DoorFitter::DoorFitter(std::shared_ptr<DSR::DSRGraph> graph,
                         DSR::InnerEigenAPI* inner_eigen,
                         DoorConfig& cfg,
                         MaskIngestor* mask_ingestor,
                         DoorSceneGraph* scene_graph,
                         rc::door::core::DoorFit& fit)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(cfg),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph), fit_(fit)
{}

// ─── Instance lifecycle ──────────────────────────────────────────────────────

bool DoorFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_id)
{
    room_node_id_ = room_id;
    if (instances_.count(node.id()))
        return false;

    DoorState init_state;
    init_state.cx        = 0.0f;
    init_state.cy        = 0.0f;
    init_state.yaw       = 0.0f;
    init_state.w         = cfg_.door_prior_w_m;
    init_state.h         = cfg_.door_prior_h_m;
    init_state.thickness = cfg_.door_thickness_m;

    // Standard DSR geometry attrs → panel dims: width_m→w, depth_m→thickness, height_m→h.
    if (auto v = G_->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.w         = v.value();
    if (auto v = G_->get_attrib_by_name<depth_m_att> (node); v.has_value()) init_state.thickness = v.value();
    if (auto v = G_->get_attrib_by_name<height_m_att>(node); v.has_value()) init_state.h         = v.value();

    // Read RT pose from room→door edge
    if (room_node_id_ != 0)
    {
        if (const auto edge = G_->get_edge(room_node_id_, node.id(), "RT"); edge.has_value())
        {
            if (const auto tr = G_->get_attrib_by_name<rt_translation_att>(edge.value()); tr.has_value())
            {
                const auto& tvec = tr.value().get();
                if (tvec.size() >= 2) { init_state.cx = tvec[0]; init_state.cy = tvec[1]; }
            }
            if (const auto rot = G_->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge.value()); rot.has_value())
            {
                const auto& rvec = rot.value().get();
                if (rvec.size() >= 3) init_state.yaw = rvec[2];
            }
        }
    }

    // The DSR-free rest — the birth seed (noted by the DoorCore at birth) overrides the centre, non-finite fields are
    // reset, the model is built and the cached geometry seeded from the box — is the core's (DoorFit::init_instance).
    DoorInstance inst;
    fit_.init_instance(inst, node.id(), node.name(), init_state);
    inst.affordance.init(G_, node.id(), node.name(), "door");

    instances_.emplace(node.id(), std::move(inst));
    std::print("door_concept: created instance for node '{}' id={}\n", node.name(), node.id());
    return true;
}


// ─── RGB-mask ROI projection (active-perception aid) ──────────────────────────

std::optional<Eigen::Matrix4d> DoorFitter::room_T_zed_matrix(std::uint64_t pose_ts_ms) const
{
    if (not inner_eigen_)
        return std::nullopt;
    // Pin the moving room→body hop to the frame's capture stamp (Nearest); keep the rigid body→zed mount
    // at latest (it carries only a bootstrap stamp — a pinned query would fail). ts=0 → current pose.
    // ★LISTENING ONLY — no behaviour change. Asks cortex what this query DID (see
    // common/rt_query_probe/rt_query_probe.h). A timestamped query that falls outside the ring
    // returns the end block and is indistinguishable from a success, so a fitter cannot currently
    // tell whether it is placing detections at the pose the camera actually had. One throttled line
    // per 15 s appends a row to etc/rt_query_probe.csv saying whether this call site ever clamps,
    // by how much, and — the column that makes it readable — how fast the robot was moving in the
    // same window. A clamp only costs geometry while the robot moves; the controller's rate goes
    // 1-3% parked to 35% moving, so a clamp share without a motion column cannot be interpreted.
    static rc::rtprobe::Probe rt_probe{"door_fitter room<-body"};
    DSR::RT_API::TimeQueryInfo rt_info;
    const auto rtb = inner_eigen_->get_transformation_matrix(rc::room::current_room_frame(*G_), "body", pose_ts_ms, "RT",
                                                             DSR::RT_API::TimeQuery::Interpolated,
                                                             &rt_info);
    rt_probe.note(rt_info, G_);
    const auto btz = inner_eigen_->get_transformation_matrix("body", "zed", 0);
    if (not (rtb.has_value() and btz.has_value()))
        return std::nullopt;
    const auto to_mat4 = [](const Mat::RTMat& T)
    {
        Eigen::Matrix4d m;
        const auto& s = T.matrix();
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m(i, j) = s(i, j);   // no aligned load
        return m;
    };
    return to_mat4(rtb.value()) * to_mat4(btz.value());
}

// (compute_silhouette_existence, compute_frame_probe, door_view_obliquity and aperture_band_offset moved into the
//  DSR-free DoorCore — src/core/door_existence.cpp, Task 5 — where they project with the cycle's snapshot camera.)

bool DoorFitter::point_in_room(const Eigen::Vector2f& q, float margin_m) const
{
    // ONE rule for the containment prior: the DoorCore's birth test and this one (core/door_core.cpp).
    return rc::door::core::point_in_polygon(room_polygon(), q, margin_m);
}

}  // namespace rc
