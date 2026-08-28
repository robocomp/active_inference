/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	\brief
	@author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
//#define HIBERNATION_ENABLED

#include <genericworker.h>
#include <fstream>
#include "buffer_types.h"
#include "room_concept.h"
#include "svg_room_loader.h"
#include "epistemic_controller.h"
#include "room_scene_graph.h"
#include "lidar_ingestor.h"
#include "imu_ingestor.h"
#include "camera_ingestor.h"
#include "image_edge_source.h"
#include "room_viewer.h"
#include "room_config.h"
#include "../../common/affordance_manager/affordance_manager.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <limits>
#include <QPointer>
#include <fps/fps.h>
#include "custom_widget.h"
#include "ui_localUI.h"

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 *
 * Runs the SDF-based room localizer (RoomConcept) and publishes the estimated
 * robot pose into the DSR graph as an RT edge on the world→robot (pre-stable)
 * or room→robot (post-stable) relationship.
 */
class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
        ~SpecificWorker();

        void JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data);
        bool is_shutting_down() const noexcept { return shutting_down_.load(); }

    public slots:
        void initialize();
        void compute();
        void emergency();
        void restore();
        int  startup_check();

        void modify_node_slot(std::uint64_t id, const std::string &type);
        void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names);
        void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
        void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
        void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
        void del_node_slot(std::uint64_t from){};

        // twopi relayout of the DSR graph viewer; MOVED out of the regenerated genericworker
        // (robocompdsl clobbers hand edits there) — mirror robot_concept. Uses the inherited
        // find_graph_viewer(). Injected into RoomSceneGraph as the relayout callback.
        void trigger_graph_layout_twopi();

    private:
        rc::RoomConfig params;   // worker config; loaded by rc::load_room_config()

        bool startup_check_flag;
        AgentPresenceCoordinator presence_coordinator_;
        bool owned_nodes_cleaned_ = false;

        // ── Velocity / odometry buffers (thread-safe) ──────────────────────────
        rc::VelocityBuffer velocity_buffer_{20};
        // Sized in SECONDS OF HISTORY, not in samples, because that is what the integrator needs: it
        // must find samples bracketing [last accepted pose, this sweep], and that span stretches
        // whenever a sweep is late or the localiser stalls. 20 entries was ~2.0 s at the old ~10 Hz;
        // at 50 Hz the same 20 entries are 0.4 s, and the cap is HARD (doublebuffer_sync.h pop_front
        // evicts the oldest silently), so the bracket would fail as a coverage refusal rather than as
        // an error anyone could see. 128 restores ~2.6 s at 50 Hz and still covers 6.4 s if a bridge
        // is republishing at the old rate.
        rc::OdometryBuffer odometry_buffer_{128};
        // Same rule: the IMU runs ~125 Hz and must span the gap between two lidar sweeps (50-100 ms)
        // with slack for a late one. 256 entries is ~2.0 s.
        rc::ImuBuffer imu_buffer_{256};
        // Fitted from the (wall, sim) stamp pairs arriving on every odometry sample; used to put the
        // lidar sweep bounds on the same clock as the rates integrated between them.
        rc::SimClockMap sim_clock_;
        std::int64_t last_imu_sim_ts_ = 0;    // dedup: the graph re-signals on unrelated attribute writes
        std::uint64_t last_robot_ref_speed_timestamp_ = 0;
        std::uint64_t last_robot_current_speed_timestamp_ = 0;
        float last_robot_adv_speed_  = 0.f;   // robot-frame forward velocity (m/s), updated from DSR
        float last_robot_side_speed_ = 0.f;   // robot-frame lateral velocity (m/s)
        float last_robot_rot_speed_  = 0.f;   // robot-frame angular velocity (rad/s)
        // Last COMMANDED velocity seen on the robot node, mirrored so every odometry sample can be
        // labelled with the command in force when it was produced. Selecting "zero command" rows is
        // the whole basis of a rest-noise measurement, and a command sampled later would label the
        // wrong rows at a start or stop.
        float last_cmd_adv_  = 0.f;
        float last_cmd_side_ = 0.f;
        float last_cmd_rot_  = 0.f;
        std::int64_t last_cmd_ts_ms_ = 0;
        // Per-sample odometry log (RoomConcept.OdomSampleLog). Opened lazily on the first sample.
        std::ofstream odom_sample_log_;
        std::uint64_t odom_sample_seq_ = 0;

        std::atomic<bool> pose_saved_{false};
        void save_robot_pose_once();

        // ── LiDAR acquisition (decoupled ingest thread + buffer + health) ──────
        std::unique_ptr<rc::LidarIngestor> lidar_ingestor_;
        // IMU on the media plane. Always created: the graph no longer carries the IMU at all.
        std::unique_ptr<rc::ImuIngestor>   imu_ingestor_;
        // RGB edge alignment (ImageEdge.enable). Null unless the feature is switched on, so the
        // whole subsystem — subscriber, thread, extraction — costs exactly nothing when off.
        std::unique_ptr<rc::CameraIngestor> camera_ingestor_;
        std::unique_ptr<rc::ImageEdgeSource> image_edge_source_;
        bool         image_edge_bound_ = false;      // bind_camera() succeeded (retried until it does)
        std::int64_t last_image_edge_ms_ = 0;
        std::int64_t last_image_edge_log_ms_ = 0;
        // Previous localizer result, for the ego-motion twist that feeds the dt nuisance column.
        // Differencing two published poses is a real measurement and needs no graph read.
        Eigen::Vector3f image_edge_prev_pose_ = Eigen::Vector3f::Zero();
        std::int64_t    image_edge_prev_ts_   = 0;
        void pump_image_edges();                     // extraction, once per compute() tick

        // ── LiDAR stream gate on Waiting→Operating ─────────────────────────────
        // Without LiDAR the localizer can never stabilize, so Operating would be a lie. True when the
        // media plane is advertised in the graph; `why` receives the reason when it is not.
        [[nodiscard]] bool lidar_stream_ready(std::string* why = nullptr) const;
        // Operating: has the stream gone silent past LidarStallTimeoutMs? Grace-counted from Operating
        // entry so a normal warm-up (subscriber discovery + first sweep) is not mistaken for a stall.
        [[nodiscard]] bool lidar_stream_stalled(std::int64_t* age_ms_out = nullptr) const;
        std::int64_t operating_since_ms_    = 0;   // wall clock of the last Operating entry
        std::int64_t last_wait_log_ms_      = 0;   // throttles the "still waiting" line
        bool         lidar_stall_reported_  = false; // one presenceLost per stall episode
        bool         degraded_from_lidar_   = false; // makes the Degraded log name the real reason

        // ── Compute-loop pacing / timing telemetry (worker-owned) ──────────────
        std::atomic<bool> operating_compute_queued_{false};
        std::int64_t      last_affordance_monitor_ms_ = 0;
        std::int64_t      last_dsr_publish_try_ms_    = 0;
        std::int64_t      last_compute_timing_log_ms_ = 0;
        std::string pose_file_path() const;

        // Publish the latest corrected pose (robot↔room RT) to the DSR graph if it is fresh. Called
        // both from compute() and — the instant the localizer produces a result — from a
        // Qt::QueuedConnection posted by room_concept_'s on_result_ready callback. Both run on the MAIN
        // thread (the queued hop marshals the localizer-thread trigger), so the publish bookkeeping is
        // race-free and the timestamp dedup makes whichever path fires second a no-op. Returns true iff
        // it actually published this call.
        bool maybe_publish_corrected_pose();

        // ── Localizer ──────────────────────────────────────────────────────────
        rc::RoomConcept room_concept_;
        bool room_initialized_from_svg_polygon_ = false;
        // Room contour AS HANDED TO THE LOCALIZER (already recentred when RECENTER_ROOM_POLYGON).
        // The viewer/camera overlay must reuse THIS, not re-load the SVG, or it would draw the
        // outline in the un-shifted frame.
        std::vector<Eigen::Vector2f> room_polygon_;
        // Old-frame coordinates of the new origin; zero when no recentring was applied.
        Eigen::Vector2f room_polygon_offset_ = Eigen::Vector2f::Zero();
        void initialize_room_model_from_svg();
        void save_robot_pose_on_exit() const;

        // ── Epistemic controller ────────────────────────────────────────────────
        rc::EpistemicController epistemic_controller_;
        bool self_target_active_ = false;

        // ── GUI / visualization (2-D viewer, plots, camera projection window) ───
        std::unique_ptr<rc::RoomViewer> viewer_;
        FPSCounter fps_counter_;

        void request_shutdown();
        void cleanup_owned_nodes();
        void waiting_enter();
        void waiting_loop();
        void operating_enter();
        void operating_loop();
        void degraded_enter();
        void degraded_loop();
        void on_optional_peer_lost(const std::string &name, std::uint32_t id);
        void on_optional_peer_ready(const std::string &name, std::uint32_t id);
        void cleanup_self_agent_node();

        // ── DSR scene-graph writer (robot-pose RT, room/wall/affordance nodes) ──
        std::unique_ptr<DSR::RT_API>            rt_api_;          // worker-owned, injected
        std::unique_ptr<rc::RoomSceneGraph> scene_graph_;
        std::int64_t last_dsr_published_ts_ms_ = 0;

        // Kinematic clamp state (see the clamp block in maybe_publish_corrected_pose). The clamp is
        // relative to what was actually PUBLISHED last, not to the optimizer's previous output, because
        // the invariant it enforces is about the stream consumers see.
        std::optional<Eigen::Affine2f> last_published_pose_;
        std::int64_t                   last_published_ts_ms_ = 0;
        long                           pose_clamp_hits_      = 0;
        // The localiser's PRE-CLAMP estimate for the last published frame (room frame, x/y/theta).
        // The predictor builds each prediction on the previous CORRECTED estimate, so
        // pred[k] - est[k-1] is exactly the sensor increment for this interval -- the motion the
        // clamp must let through untouched. Differencing two consecutive PREDICTIONS instead looks
        // equivalent but is not: it evaluates to increment + innovation[k-1], which would smuggle
        // the previous frame's correction through unbounded. See the clamp block.
        std::optional<Eigen::Vector3f> last_published_est_;


        // Pose trace CSV (etc/pose_trace.csv): logs CORRECTED (20 Hz, compute) and PREDICTED (60 Hz,
        // tick) poses with timestamps so the intermediate dead-reckoned poses can be compared against
        // the optimizer corrections (diagnose the noise predict-publish injects). Both writers run on
        // this->thread() → single ofstream, no lock. type: 0=corrected, 1=predicted.
        std::ofstream pose_trace_;
        bool          pose_trace_open_attempted_ = false;
        void log_pose_trace(int type, std::int64_t valid_ts_ms,
                            const Eigen::Affine2f& pose, float innov_norm);

        // Per-tick compute-timing CSV (etc/compute_timing.csv): exposes WHERE compute() stalls (viewer
        // vs dsr vs loc_fetch) so we can see why the corrected publish drops below the optimizer rate.
        std::ofstream compute_csv_;
        bool          compute_csv_open_attempted_ = false;

        // ── Ground-truth comparison (SIMULATION ONLY) ────────────────────────────────────────────
        // Logs the localiser's published pose beside the Webots supervisor pose that robot_concept
        // writes onto the robot node as robot_gt_*. The point is a witness from OUTSIDE the
        // estimator: a wrong pose fitted well scores exactly like a right one on the SDF residual,
        // which is how a 0.35 rad yaw error hid behind an SDF of 0.009 for a whole session.
        // ⚠ The two poses are in DIFFERENT FRAMES — GT is world, the estimate is room — so a
        // CONSTANT offset between them is expected and benign (the room frame's own orientation).
        // What matters is whether that offset stays constant: fit offset+gain over many rows and
        // look at the RESIDUAL. Never compare two single readings; that is how three wrong

    // ── robot_gt_angle arrives with an INVERTED SIGN (measured 2026-08-28, 9258 rows) ────────────
    // Position is a clean pure translation: gt_x = est_x + 0.53, gt_y = est_y, slope +1 on both.
    // Heading is not: gt_theta = -est_theta - 89.2 deg. No rigid transform maps position that way
    // and angle this way, and the position half is verifiably right, so the ANGLE is wrong — the
    // signature (a clean reflection, not a rotation) is what you get extracting a Webots axis-angle
    // assuming +Z when the node turns about -Z. The defect is in robot_concept, which publishes it.
    // Differencing est - gt raw yields garbage that LOOKS like a wild localiser: it reported heading
    // "errors" of +201/+224/-119 deg before this was found.
    // ★ SELF-CHECKING, not a silent negation. Both conventions are scored every cycle by how CONSTANT
    //   the resulting offset is; gt_convention_report() names the winner. If robot_concept is fixed,
    //   the winner flips and the log says so, instead of this correction inverting a correct signal.
    double gt_sum_diff_c_ = 0, gt_sum_diff_s_ = 0;   ///< circular accumulators for est - gt
    double gt_sum_sum_c_  = 0, gt_sum_sum_s_  = 0;   ///< and for est + gt
    long   gt_n_ = 0;
    long   gt_report_at_ = 200;
    void   gt_convention_report(float est_th, float gt_th);
        // conclusions got drawn by hand on 2026-08-22.
        // Gated on the attributes EXISTING, so on the real robot nothing is written at all.
        std::ofstream gt_csv_;
        bool          gt_csv_open_attempted_ = false;
        void          log_ground_truth(const rc::RoomConcept::UpdateResult &res);

        // RT publish-rate monitor (shown in the window title at ~1 Hz so it can be watched visually).
        int          rt_corr_count_           = 0;   // corrected RT publishes this window
        std::int64_t rt_rate_window_start_ms_ = 0;
        void update_rt_rate_readout(std::int64_t now_ms, bool on_gui_thread);

        std::atomic<bool> shutting_down_{false};

    signals:
        void presenceReady();
        void presenceLost();
};

#endif // SPECIFICWORKER_H
