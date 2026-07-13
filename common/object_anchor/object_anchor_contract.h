/*
 *  object_anchor_contract.h — SHARED producer/consumer contract for object pose anchors.
 *
 *  A "concept agent" (table_concept, chair_concept, …) fits an object in the ROOM frame and
 *  publishes it as a DSR node. A localizer (room_concept) can then use that object as an SE(2)
 *  pose landmark to tighten the robot↔room estimate. For that to carry information, the producer
 *  must also publish, EACH FRAME, its raw observation of the object expressed in the ROBOT base
 *  frame (z_o) — the map pose alone is circular (it was built from the robot pose).
 *
 *  This header is the single source of truth for that contract, included by BOTH sides:
 *    - producers call to_robot_frame() + write_observation() from their scene-graph collaborator;
 *    - the consumer (room_concept/object_anchor_source) reads the same attribute names.
 *  Keeping the names here means the two ends can never silently drift.
 *
 *  Header-only (inline); DSR + Eigen only, no torch. See room_concept/src/object_anchor_*.{h,cpp}
 *  for the consumer/factor side and CLAUDE.md for the SLAM-consistency rationale (one-frame delay,
 *  keep the map pose detached, object weight < walls).
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_eigen_defs.h>
#include <dsr/api/dsr_inner_eigen_api.h>

namespace rc::object_anchor
{
    // The producer/consumer contract is the type-attributed `obj_obs_robot_att` / `obj_obs_robot_cov_att`
    // (registered in cortex dsr_attr_name.h): compile-time checked, so the two ends can never drift.
    //   obj_obs_robot     = [x, y] or [x, y, yaw] in the robot frame (z_o)
    //   obj_obs_robot_cov = [σx², σy²] or [σx², σy², σyaw²] (optional R_o)

    /// Express a room-frame SE(2) fit (THIS frame) in the robot base frame → the observation z_o.
    /// `robot_frame` is the localizer's base node (e.g. "body" — the frame whose room←robot pose the
    /// localizer optimizes). `ts` pins the RT chain to the capture stamp (ts!=0 ⇒ no InnerEigen cache).
    /// Returns nullopt if the chain is unavailable. Mean only — supply R_o separately if wanted.
    inline std::optional<Eigen::Vector3f> to_robot_frame(DSR::InnerEigenAPI& inner,
                                                         const std::string&     robot_frame,
                                                         const Eigen::Vector3f& pose_room,
                                                         std::uint64_t          ts,
                                                         const std::string&     room_frame = "room")
    {
        // M = robot_frame ← room_frame : maps a room point/heading into the robot base frame.
        const auto m = inner.get_transformation_matrix(robot_frame, room_frame, ts);
        if (not m.has_value())
            return std::nullopt;
        const Mat::RTMat& M = m.value();
        const Mat::Vector3d p = M * Mat::Vector3d(pose_room.x(), pose_room.y(), 0.0);
        const double yaw_m = std::atan2(M.linear()(1, 0), M.linear()(0, 0));
        const double yaw   = std::remainder(static_cast<double>(pose_room.z()) + yaw_m, 2.0 * M_PI);
        return Eigen::Vector3f{static_cast<float>(p.x()), static_cast<float>(p.y()),
                               static_cast<float>(yaw)};
    }

    /// Write z_o (+ optional diagonal R_o) onto the object's node. Does NOT call update_node — the
    /// caller (the agent's single DSR writer) batches it with its other per-frame attribute writes.
    /// has_orientation=false ⇒ write only [x, y] (a single-view centroid gives a clean position but a
    /// biased yaw); the consumer then treats it as a 2-DOF position landmark.
    inline void write_observation(DSR::DSRGraph& G, DSR::Node& node,
                                  const Eigen::Vector3f&                obs_robot,
                                  bool                                  has_orientation = true,
                                  const std::optional<Eigen::Vector3f>& var_robot = std::nullopt)
    {
        std::vector<float> obs = has_orientation
            ? std::vector<float>{obs_robot.x(), obs_robot.y(), obs_robot.z()}
            : std::vector<float>{obs_robot.x(), obs_robot.y()};
        G.add_or_modify_attrib_local<obj_obs_robot_att>(node, std::move(obs));
        if (var_robot.has_value())
        {
            std::vector<float> cov = has_orientation
                ? std::vector<float>{var_robot->x(), var_robot->y(), var_robot->z()}
                : std::vector<float>{var_robot->x(), var_robot->y()};
            G.add_or_modify_attrib_local<obj_obs_robot_cov_att>(node, std::move(cov));
        }
    }
} // namespace rc::object_anchor
