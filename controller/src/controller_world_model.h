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
    std::optional<ControllerTargetInfo> read_target_in_room(std::uint64_t timestamp_ms) const;

    static bool same_target_instance(const ControllerTargetInfo &lhs, const ControllerTargetInfo &rhs);

private:
    const ControllerParams *params_ = nullptr;
    rc::AffordanceManager *affordance_manager_ = nullptr;
    std::shared_ptr<DSR::DSRGraph> graph_;
    DSR::InnerEigenAPI *inner_eigen_api_ = nullptr;
    ControllerGraphState graph_state_;
};