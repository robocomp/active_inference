#pragma once

#include <genericworker.h>

#include <optional>

#include "controller_runtime_types.h"
#include "../../common/affordance_manager/affordance_manager.h"

class ControllerWorldModel
{
public:
    void set_params(const ControllerParams *params);
    void set_affordance_manager(rc::AffordanceManager *affordance_manager);
    void set_dependencies(std::shared_ptr<DSR::DSRGraph> graph, DSR::InnerEigenAPI *inner_eigen_api);

    bool refresh_graph_state();
    const ControllerGraphState &graph_state() const { return graph_state_; }

    std::optional<std::vector<Eigen::Vector2f>> read_room_polygon() const;
    std::optional<ControllerRobotPose> read_robot_pose_in_room(
        std::uint64_t timestamp_ms,
        const std::optional<std::uint64_t> &last_lidar_timestamp_ms) const;
    std::optional<ControllerPoseUncertainty> read_pose_uncertainty() const;

    /// Age, in ms, of the pose currently on the room<-robot RT edge, measured against ITS OWN validity
    /// stamp rather than against when we noticed it change.
    ///
    /// room_concept writes the edge through the RT_API timestamped overload with `res.timestamp_ms` —
    /// the LOCALISATION stamp, derived from the lidar scan that produced the pose. So the edge already
    /// carries the capture time of the scan behind it, and comparing that to the wall clock gives
    /// end-to-end LIDAR -> pose -> controller latency as a MEASUREMENT instead of a residual. Both
    /// agents share a machine, so system_clock is common and the subtraction is meaningful.
    /// nullopt when the edge carries no timestamp history (nothing to measure, which is not the same
    /// as zero latency).
    std::optional<std::uint64_t> pose_stamp_age_ms(std::uint64_t now_ms) const;
    std::optional<ControllerTargetInfo> read_target_in_room(std::uint64_t timestamp_ms) const;
    // Room-frame XY of an arbitrary node (e.g. an affordance's parent object), via the RT tree. Used
    // to re-aim a repaired affordance target's heading at the object it observes.
    std::optional<Eigen::Vector2f> read_node_room_xy(std::uint64_t node_id, std::uint64_t timestamp_ms) const;

    static bool same_target_instance(const ControllerTargetInfo &lhs, const ControllerTargetInfo &rhs);

private:
    const ControllerParams *params_ = nullptr;
    rc::AffordanceManager *affordance_manager_ = nullptr;
    std::shared_ptr<DSR::DSRGraph> graph_;
    DSR::InnerEigenAPI *inner_eigen_api_ = nullptr;
    ControllerGraphState graph_state_;
};