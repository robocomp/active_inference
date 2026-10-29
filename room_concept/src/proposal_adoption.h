/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
// ADOPTING OR REJECTING A RECOGNITION PROPOSAL (room-transition protocol, spec C2.7).
// ltsm's recogniser, convinced that the provisional room P is a remembered room K, posts ONE proposal under P:
//   node `proposal_<K live name>`, type metaconcept, object_subtype "room_proposal", parent = P,
//   RT edge P→proposal = T(P←K), room_id = K's, delimiting_polygon_x/y = K's outline (K frame),
//   recall_pose_prior = the robot's pose in K (10 floats per mode: w x y th Sxx Sxy Sxth Syy Syth Sthth),
//   state = "pending".
// room_concept answers by writing `state` = "accepted" | "rejected" on that node, and on accept turns P into K IN PLACE:
// same node id, renamed to K's live name, room_id and outline K's, its own RT children re-expressed by T(K←P), the
// `proto` self-edge removed. The node is ltsm's: this agent reads it and writes `state`, never deletes it.
// This header is the graph-side half that needs no RoomConcept (so it is testable on a bare DSR graph,
// tools/proposal_adoption_test.cpp); the fit that DECIDES is RoomConcept's (seed_reloc_prior + late_recall_verdict),
// driven by RoomSceneGraph::step_proposal (room_scene_graph_late_recall.cpp).
#pragma once

#include "reloc_search.h"   // rc::reloc::PriorMode

#include <dsr/api/dsr_api.h>

#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rc::proposal
{
inline constexpr std::string_view kSubtype  = "room_proposal";
inline constexpr std::string_view kPending  = "pending";
inline constexpr std::string_view kAccepted = "accepted";
inline constexpr std::string_view kRejected = "rejected";
inline constexpr std::string_view kPrefix   = "proposal_";

struct Proposal
{
    std::uint64_t node_id    = 0;
    std::uint64_t parent     = 0;        ///< P: the provisional room the proposal is about
    std::uint64_t room_index = 0;        ///< K's room_id
    std::string   live_name;             ///< K's live name (the node name without "proposal_")
    std::vector<Eigen::Vector2f> outline_K;                    ///< K's outline in K's own frame
    Eigen::Affine2f T_P_K = Eigen::Affine2f::Identity();       ///< K's frame in P's published frame (RT P→proposal)
    std::vector<float> pose_prior;       ///< packed robot-in-K modes, 10 floats each
    bool well_formed = false;            ///< outline ≥ 3, prior non-empty and %10, RT edge readable
};

/// Every room_proposal whose state is "pending" (malformed ones too, flagged, so they can be answered).
[[nodiscard]] std::vector<Proposal> pending(DSR::DSRGraph& G);

enum class Admission { Judge, NotProvisional, NameTaken, Malformed };
[[nodiscard]] const char* to_string(Admission a);

/// May this agent judge `p`? Only a well-formed proposal about THIS agent's provisional room (`provisional_room`, 0 =
/// none), which still carries its `proto` self-edge, and whose K name no OTHER node holds (the rename must be possible).
[[nodiscard]] Admission admit(DSR::DSRGraph& G, const Proposal& p, std::uint64_t provisional_room);

/// Write `state` on the proposal node (update_node). False if the node is gone.
bool answer(DSR::DSRGraph& G, std::uint64_t proposal_id, std::string_view state);

/// THE ACCEPTED TRANSFORM (Task 20). ltsm's RT edge P→proposal holds its PRIOR T(P←K); the fit that decides may correct it
/// (recog_proto6: a half turn). The accepted fact is the fit's: the robot's pose in P just before adoption (`robot_P`, what
/// was published under P) and its fitted pose in K (`robot_K`) are the same physical pose, so T(K←P) = robot_K ∘ robot_P⁻¹,
/// i.e. T(P←K) = robot_P ∘ robot_K⁻¹.
[[nodiscard]] Eigen::Affine2f accepted_T_P_K(const Eigen::Affine2f& robot_P, const Eigen::Affine2f& robot_K);

/// The ACCEPTED verdict, as ONE fact for every reader, in ONE update_node of the proposal NODE: rt_translation [x y z] and
/// rt_rotation_euler_xyz [0 0 yaw] = the accepted T(P←K); rt_covariance (36, row-major [x y z rx ry rz], SE(2) on rows/cols
/// 0,1,5) = the covariance of the robot's FITTED pose in K, in K's frame (`robot_K_cov`, [x y θ]) — the uncertainty T(P←K)
/// inherits (the survey pose robot_P is its reference); and state = "accepted". One DDS sample, so a reader that sees
/// "accepted" sees this T. Readers take T ONLY from these node attributes (absent ⇒ malformed). The RT edge P→proposal is
/// overwritten too, for viewers/history only: edge and node updates travel on different, unordered DDS topics.
/// False if the node or P is gone.
bool answer_accepted(DSR::DSRGraph& G, const Proposal& p, const Eigen::Affine2f& T_P_K_accepted,
                     const Eigen::Matrix3f& robot_K_cov);

struct Adoption
{
    bool ok = false;
    std::string why;        ///< on failure: what refused it (nothing was written)
    int reexpressed = 0;    ///< RT edges re-expressed by T(K←P)
};

/// Accept: P becomes K in place, by `p.T_P_K` — the caller sets it to the ACCEPTED transform (accepted_T_P_K), not ltsm's prior. Rename + room_id + outline in ONE update_node (same id) FIRST — a refused rename (the
/// name held by another id: cortex returns false; a cortex without rename support throws) leaves the graph untouched
/// and returns ok=false. Then P's OWN RT children are re-expressed by T(K←P): the robot and room_concept's walls,
/// whether hung from P or from P's floor (floor←room stays the identity: the floor IS the room's frame). The robot edge
/// is found in EITHER direction: P→robot holds T(P←robot) and is LEFT-composed (T(K←P)·T(P←robot)); robot→P (this fleet:
/// the robot is the RT root) holds T(robot←P) and is RIGHT-composed (T(robot←P)·T(P←K)), its child-axes twist carried by
/// the adjoint Ad(T(K←P)) and its parent-frame pose covariance by the composition's Jacobian. EVERY block of
/// each edge's history ring is re-expressed in place (same slots and timestamps; pose covariance rotated), so no reader
/// can interpolate across the P and K frames. Other agents' children are theirs to move. Last, the `proto` self-edge is removed.
/// The verdict (`state`) is NOT written here: the caller answers once its own state has followed.
Adoption adopt_in_place(DSR::DSRGraph& G, const Proposal& p, std::uint64_t robot_id);

// ── pure frame algebra ──────────────────────────────────────────────────────────────────────────────────────────
/// Unpack robot-in-K modes and express them in the frame `T_dst_K` maps K into: mean composed, covariance rotated.
/// Modes with a non-finite mean/covariance or non-positive weight are dropped.
[[nodiscard]] std::vector<rc::reloc::PriorMode> modes_in(const std::vector<float>& packed, const Eigen::Affine2f& T_dst_K);
/// Planar pose (x, y, yaw) re-expressed: T · pose.
[[nodiscard]] Eigen::Vector3f transform_pose(const Eigen::Affine2f& T, const Eigen::Vector3f& pose);
[[nodiscard]] float angle_of(const Eigen::Affine2f& T);
}   // namespace rc::proposal
