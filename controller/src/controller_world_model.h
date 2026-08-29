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
    // The graph itself, for the one-shot reads that belong to the session rather than here — currently the
    // robot's mesh `path` (controller_robot_body.h). Null until set_dependencies has run.
    DSR::DSRGraph *graph() const { return graph_.get(); }

    std::optional<std::vector<Eigen::Vector2f>> read_room_polygon() const;
    // ── TWO POSES, BECAUSE TWO CONSUMERS WANT DIFFERENT INSTANTS ─────────────────────────────────
    // This one is SCAN-ALIGNED: it queries the RT tree at the last LiDAR stamp, so the pose it returns
    // is contemporaneous with the observations that produced the current obstacle set. Everything that
    // reasons about that set together with where the robot was when it was taken should keep using it.
    std::optional<ControllerRobotPose> read_robot_pose_in_room(
        std::uint64_t timestamp_ms,
        const std::optional<std::uint64_t> &last_lidar_timestamp_ms) const;

    // ...and this one is the FRESHEST pose the RT tree holds, for the control law, whose error terms
    // are about where the robot IS. Measured 2026-08-16: the newest room<-robot block already leads the
    // scan the loop was pinning to by 33 ms at the median and 68 ms at p90, and that lead was simply
    // being declined — it is a real block, not an extrapolation, so recovering it costs nothing and
    // introduces no model. Against a feedback term worth 2.78 rad/s per metre of cross-track, ~35 ms at
    // 0.35 m/s is 12 mm is 0.034 rad/s of demand that was being asked for on stale evidence.
    // ★It is the SAME query chain with the scan pin removed, so the fallbacks (exact ts, then 0) and the
    // Interpolated/Nearest choice are unchanged; only the instant asked for differs.
    std::optional<ControllerRobotPose> read_robot_pose_latest(std::uint64_t timestamp_ms) const;
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
    // The two public pose readers differ ONLY in which instants they try, so that is the only thing they
    // state: the guard, the TimeQuery choice, the RT walk and the matrix->pose decode live here once.
    // They were two copies, and the one that gets forgotten on the next change is the one feeding the
    // control law. Tried in order; the first instant that resolves wins.
    std::optional<ControllerRobotPose> pose_from_rt(std::initializer_list<std::uint64_t> instants) const;

    const ControllerParams *params_ = nullptr;
    rc::AffordanceManager *affordance_manager_ = nullptr;
    std::shared_ptr<DSR::DSRGraph> graph_;
    DSR::InnerEigenAPI *inner_eigen_api_ = nullptr;
    // Owned RT accessor: read_pose_uncertainty() needs get_edge_RT_covariance(), which is the only thing
    // that knows how rt_covariance is packed (ring vs single 36-block, head index, timestamp selection).
    // Hand-indexing the flat attribute is what produced the yaw-read-as-z bug this replaced.
    std::unique_ptr<DSR::RT_API> rt_api_;
    ControllerGraphState graph_state_;
};