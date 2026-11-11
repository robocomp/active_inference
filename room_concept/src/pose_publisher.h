#pragma once

// ── PUBLISHING THE ROBOT'S POSE, LIFTED OUT OF SpecificWorker ────────────────────────────────────
// Everything the agent writes about WHERE THE ROBOT IS: the corrected pose at LiDAR rate, the
// predicted pose at IMU rate between corrections, the kinematic clamp that refuses a physically
// impossible correction, the append-only record of the discontinuities it catches, and the pose
// trace both writers share.
//
// ⚠ TWO THREADS LIVE IN HERE and that is the reason this class exists as a unit rather than as loose
// methods. maybe_publish_corrected_pose() runs on the LOCALISER thread and publish_predicted_tick()
// on the IMU-INGEST thread; they share last_published_pose_/_ts_ms_/_est_, the propagator, the written ticks, the
// trace and the jump log, and publish_mutex_ is what makes that safe. Keeping the state and both
// writers in one class is what keeps that invariant reviewable — it was spread across 500 lines of a
// 3000-line file, where the only thing tying the two threads together was a comment.

#include <Eigen/Dense>
#include <atomic>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <functional>
#include <optional>

#include "room_config.h"
#include "update_result.h"   // UpdateResult crosses the hook below, so the full type is needed
#include "pose_propagator.h"

namespace DSR { class DSRGraph; class RT_API; }
namespace rc
{
    class RoomSceneGraph;
    class RoomViewer;

    class PosePublisher
    {
    public:
        /// ⚠ NO RT_API PARAMETER, DELIBERATELY. One was passed until 2026-09-13 and was always null:
        /// the worker injected rt_api_.get() before assigning it. Nothing here needs it — this class
        /// publishes through RoomSceneGraph, which holds the RT_API and receives it after the
        /// assignment. A null member behind a constructor signature that claims the dependency is
        /// satisfied is worse than no member: it reads as wired. If this class ever needs the RT API,
        /// add the parameter back AND construct below the worker's assignment.
        PosePublisher(std::shared_ptr<DSR::DSRGraph> graph, RoomConfig& params,
                      RoomConcept& room, RoomViewer** viewer, const std::atomic<bool>& shutting_down)
            : G(std::move(graph)), params(params), room_concept_(room),
              viewer_slot_(viewer), shutting_down_(shutting_down) {}

        /// The scene graph is built after this object, so it is handed over rather than injected.
        void set_scene_graph(RoomSceneGraph* sg) { scene_graph_ = sg; }

        /// The base's measured twist, as the graph reports it. Written by the node-attribute slot on the
        /// main thread; the correction's twist when there is no previous publish to difference against.
        void set_robot_speed(float adv, float side, float rot)
        { last_robot_adv_speed_ = adv; last_robot_side_speed_ = side; last_robot_rot_speed_ = rot; }

        /// Called on every accepted CORRECTED publish, on the localiser thread. The worker hangs its
        /// ground-truth comparison here: that is a diagnostic ABOUT a published pose, not part of
        /// publishing one, and routing it through a hook keeps this class from depending on it.
        void set_on_corrected_published(std::function<void(const loc::UpdateResult&)> f)
        { on_corrected_ = std::move(f); }

        bool maybe_publish_corrected_pose();
        /// IMU-ingest thread, once per sample: wall stamp + raw gyro. Always feeds the propagator; writes the edge only
        /// with RoomConcept.PredictedTicks.
        void publish_predicted_tick(std::int64_t imu_ts_ms, float gyro_z);
        /// Main thread (odometry slot): one wheel sample, WALL stamp, for the propagator.
        void add_odometry(std::int64_t ts_ms, float adv, float side, float rot);
        void update_rt_rate_readout(std::int64_t now_ms, bool on_gui_thread);
        /// One-shot: read the base's DECLARED capability from the graph and make it the clamp's
        /// bound. Lives here because the clamp it configures lives here — the bound is a physical fact
        /// about the robot, which is the whole reason the jump log cannot be quietly mis-set.
        void apply_base_capability_to_pose_clamp();
        void log_pose_trace(int type, std::int64_t valid_ts_ms,
                            const Eigen::Affine2f& pose, float innov_norm);

    private:
        RoomViewer* viewer() const { return viewer_slot_ != nullptr ? *viewer_slot_ : nullptr; }

        std::shared_ptr<DSR::DSRGraph> G;
        RoomConfig&     params;
        RoomConcept&    room_concept_;
        RoomSceneGraph* scene_graph_ = nullptr;
        RoomViewer**    viewer_slot_ = nullptr;
        const std::atomic<bool>& shutting_down_;

        float last_robot_adv_speed_  = 0.f;
        float last_robot_side_speed_ = 0.f;
        float last_robot_rot_speed_  = 0.f;

        std::int64_t last_dsr_publish_try_ms_  = 0;
        std::int64_t last_dsr_published_ts_ms_ = 0;

        // Kinematic clamp state: a correction is refused when it implies motion the base cannot make.
        std::optional<Eigen::Affine2f> last_published_pose_;
        std::int64_t                   last_published_ts_ms_ = 0;
        long                           pose_clamp_hits_      = 0;
        long                           pose_clamp_mrej_      = 0;   // measured motions rejected as outside the dynamics
        std::optional<Eigen::Vector3f> last_published_est_;
        std::uint64_t                  last_published_reloc_epoch_ = 0;   // see UpdateResult::reloc_epoch

        /// Serialises the two writer threads over every shared member below and above it.
        std::mutex publish_mutex_;

        Eigen::Matrix3f last_published_cov_ = Eigen::Matrix3f::Identity();

        // ── THE POSE BETWEEN CORRECTIONS (pose_propagator.h) ─────────────────────────────────────────
        // propagator_ integrates IMU + wheels from the last correction; propagator_mutex_ guards it (the odometry
        // slot feeds it from the main thread). Lock order: publish_mutex_ before propagator_mutex_.
        // written_ticks_: stamps of the NON-correction blocks on the edge — mode b's ticks and every correction's
        // spread/tail blocks — ascending, last second; under publish_mutex_. In mode a (no tick writes) it holds the
        // previous corrections' tail blocks only, so the next tail is the newest-IMU block and the spread moves the
        // previous correction's tail block(s) in (t_k, t_c).
        PosePropagator           propagator_;
        std::mutex               propagator_mutex_;
        std::deque<std::int64_t> written_ticks_;
        // The estimator's pending re-anchor as it stood after the last correction's update(): a different one now means
        // its internal frame moved between t_k and t_c, and the old base cannot be differenced against the correction.
        std::optional<Eigen::Matrix3f> reanchor_after_last_;
        bool                           reanchor_seen_ = false;

        // etc/publish_timing.csv — one row per published correction (enter_ms,exit_ms,valid_ts_ms,n_spread,n_tail).
        std::ofstream publish_timing_;
        bool          publish_timing_open_attempted_ = false;

        std::ofstream pose_trace_;
        bool          pose_trace_open_attempted_ = false;

        // ── APPEND-ONLY RECORD OF LOCALISER DISCONTINUITIES (etc/pose_jumps.csv) ─────────────────
        // Append-only on purpose: pose_trace.csv is truncated every start, and two corrections of
        // 1.4 m and 2.0 m seen on 2026-09-10 were unrecoverable by the time anyone went back for
        // them. The trigger is the base's own DECLARED CAPABILITY, not a tuned number, so it cannot
        // be quietly mis-set and cannot discard a real error the way a magnitude cutoff would; with
        // no capability channel up it stays silent and says so once. Compared only against the
        // previous point of the SAME type — a corrected pose following a predicted one is entitled
        // to step, and that is a correction, not a discontinuity.
        struct TracePoint
        {
            std::int64_t wall_ms = 0, valid_ts_ms = 0;
            float x = 0.f, y = 0.f, th = 0.f, innov = 0.f;
            bool  set = false;
        };
        std::ofstream pose_jump_log_;
        bool          pose_jump_log_attempted_ = false;
        bool          pose_jump_no_capability_warned_ = false;
        std::int64_t  pose_jump_run_id_ = 0;
        TracePoint    prev_trace_[2];   // indexed by type: 0 corrected, 1 predicted

        std::atomic<int> rt_corr_count_ {0};   ///< written from the localiser thread, read from compute()
        std::int64_t rt_rate_window_start_ms_ = 0;
        bool         pose_clamp_from_capability_ = false;
        std::function<void(const loc::UpdateResult&)> on_corrected_;
    };
}   // namespace rc
