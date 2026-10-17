#pragma once
// The frame-local state of ONE RoomConcept::update() cycle, handed from phase to phase.
// Refactor step 2b (docs/refactor_proposal_2026-10-13.md): update() was a single 1014-line body whose
// `// =====` sections shared these locals; each section is now a private update_* method and this struct
// holds exactly the locals that crossed a section boundary — nothing that a phase keeps to itself.
#include "room_concept.h"

namespace rc
{
    struct RoomConcept::UpdateCycle
    {
        const std::pair<std::vector<Eigen::Vector3f>, std::int64_t>& lidar;
        const std::vector<rc::VelocityCommand>& velocity_history;
        const std::vector<rc::OdometryReading>& odometry_history;

        UpdateResult res;                                   // what update() returns
        MotionPriorSelection motion_prior_selection;         // update_predict
        Eigen::Vector2f pred_pos = Eigen::Vector2f::Zero();  // update_predict
        float pred_theta = 0.f;                              // update_predict
        std::vector<Eigen::Vector3f> sampled_points;        // update_build_slot: the scan every later phase uses
        Eigen::Vector3f slot_odom_delta = Eigen::Vector3f::Zero();       // update_build_slot: the new slot's motion
        Eigen::Matrix3f slot_motion_cov = Eigen::Matrix3f::Identity();   //   factor (stride-spanned)
        bool window_slid = false;                            // update_build_slot: an old slot was dropped

        [[nodiscard]] const OdometryPrior& selected_prior() const { return motion_prior_selection.selected_prior; }
    };
} // namespace rc
