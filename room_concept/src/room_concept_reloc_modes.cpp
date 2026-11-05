// RoomConcept, relocalisation part 2 (split out of room_concept_reloc.cpp at Task R1b to keep both under the
// file-length lint): the hand-over's entry and the door anchor's odometry (the entry prior of the rival belief).

#include "room_concept.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <QDebug>

namespace rc
{
    void RoomConcept::note_passage_handover(const rc::passage::Passage& p)
    {
        { std::scoped_lock lk(passage_mutex_); rt_.pending_passage = p; }
        belief_reset_requested_.store(true, std::memory_order_relaxed);
    }

    void RoomConcept::step_door_anchor(const UpdateResult& res)
    {
        // Carry the door anchor by THIS frame's odometry increment: the motion prior's prediction minus the pose it was
        // built on (pred[k] − est[k−1], the publisher's measured-motion identity), in the robot frame of est[k−1], with
        // the selected motion prior's covariance rotated into that frame. Never the tracker's correction: the anchor
        // exists for the frames in which the tracker cannot be trusted.
        if (rt_.door_anchor.empty() or not rt_.anchor_prev_valid)
            return;
        const Eigen::Vector3f& b = rt_.anchor_prev_est;
        const float c = std::cos(b.z()), sn = std::sin(b.z());
        const Eigen::Vector2f dw(res.pred_x - b.x(), res.pred_y - b.y());
        const Eigen::Vector3f u(c * dw.x() + sn * dw.y(), -sn * dw.x() + c * dw.y(),
                                std::remainder(res.pred_theta - b.z(), 2.f * static_cast<float>(M_PI)));
        Eigen::Matrix3f Su = Eigen::Matrix3f::Zero();
        if (last_selected_prior_.valid and last_selected_prior_.covariance_eigen.allFinite())
        {
            Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
            R(0, 0) = c; R(0, 1) = sn; R(1, 0) = -sn; R(1, 1) = c;   // world → robot frame of est[k−1]
            Su = R * last_selected_prior_.covariance_eigen * R.transpose();
        }
        if (u.allFinite())
        {
            for (auto& an : rt_.door_anchor) rc::passage::propagate(an, u, Su);
            ++rt_.door_anchor_frames;
        }
    }

} // namespace rc
