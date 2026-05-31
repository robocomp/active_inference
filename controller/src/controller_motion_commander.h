#pragma once

#include <genericworker.h>

#include <functional>

#include "controller_runtime_types.h"

class ControllerWorldModel;

class ControllerMotionCommander
{
public:
    using CommandTextSink = std::function<void(const QString &)>;

    void set_params(const ControllerParams *params);
    void set_dependencies(std::shared_ptr<DSR::DSRGraph> graph,
                          const ControllerWorldModel *world_model,
                          RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy,
                          int agent_id,
                          CommandTextSink command_text_sink);

    void apply_uncertainty_speed_limit(float &adv_mps, float &side_mps, float &rot_rps) const;
    void send_speed_command(float adv_mps, float side_mps, float rot_rps);
    void stop_robot();

private:
    static float ramp_uncertainty_scale(float value, float slow_threshold, float stop_threshold, float min_scale);
    static float preserve_sign_clamp(float value, float max_abs);
    static std::uint64_t current_time_ms();

    void publish_robot_reference_speed(float adv_mps, float side_mps, float rot_rps, std::uint64_t timestamp_ms);

    static constexpr const char *kRobotRefAdvSpeedAttr = "robot_ref_adv_speed";
    static constexpr const char *kRobotRefRotSpeedAttr = "robot_ref_rot_speed";
    static constexpr const char *kRobotRefSideSpeedAttr = "robot_ref_side_speed";
    static constexpr const char *kRobotRefSpeedTimestampAttr = "robot_ref_speed_timestamp";

    const ControllerParams *params_ = nullptr;
    std::shared_ptr<DSR::DSRGraph> graph_;
    const ControllerWorldModel *world_model_ = nullptr;
    RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy_;
    int agent_id_ = 0;
    CommandTextSink command_text_sink_;
    bool stop_command_latched_ = false;
    bool has_last_speed_command_ = false;
    Eigen::Vector3f last_speed_command_ = Eigen::Vector3f::Zero();
};