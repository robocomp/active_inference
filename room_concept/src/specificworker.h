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
#include "mount_lidar_pair.h"
#include "mount_calibrator.h"
#include "pose_publisher.h"
#include "calib_channels.h"
#include "ground_truth_log.h"
#include "camera_calibration.h"
#include <map>
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
        std::int64_t last_image_edge_ms_ = 0;
        // Previous localizer result, for the ego-motion twist that feeds the dt nuisance column.
        // Differencing two published poses is a real measurement and needs no graph read.

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
        std::int64_t      last_compute_timing_log_ms_ = 0;
        std::string pose_file_path() const;

        // Publish the latest corrected pose (robot↔room RT) to the DSR graph if it is fresh. Called
        // ONLY from room_concept_'s on_result_ready callback, on the LOCALIZER thread (2026-09-03: the
        // old Qt::QueuedConnection marshal to the main thread, and compute()'s own redundant call, are
        // both removed -- see the FIX notes at set_on_result_ready and in compute()). Returns true iff
        // it actually published this call. Takes publish_mutex_, shared with publish_predicted_tick().
        bool maybe_publish_corrected_pose();

        // FIX 2026-09-04: publish a dead-reckoned pose on EVERY IMU sample (~100 Hz on this robot,
        // vs the lidar-paced ~20 Hz of maybe_publish_corrected_pose), so the graph is never more than
        // one IMU period stale for consumers that read it directly rather than extrapolating the
        // twist themselves. Extrapolates from the last REAL (SDF-corrected) publish using the
        // corrected twist cached at that publish (last_pub_adv_/_side_/_rot_) -- NOT from the previous
        // predicted tick, so small per-tick errors cannot compound between real corrections. Takes
        // publish_mutex_: a real correction landing mid-tick must never be overwritten by a stale
        // prediction, and the two run on different threads (localizer vs imu-ingest) so this is a
        // genuine race, not a formality. Wired from ImuIngestor::set_on_new_sample() in initialize().
        void publish_predicted_tick(std::int64_t imu_ts_ms);

        // ── Localizer ──────────────────────────────────────────────────────────
        rc::RoomConcept room_concept_;
        bool room_initialized_from_svg_polygon_ = false;
        // Room contour AS HANDED TO THE LOCALIZER (already recentred when RECENTER_ROOM_POLYGON).
        // The viewer/camera overlay must reuse THIS, not re-load the SVG, or it would draw the
        // outline in the un-shifted frame.
        // Old-frame coordinates of the new origin; zero when no recentring was applied.
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

        // Kinematic clamp state (see the clamp block in maybe_publish_corrected_pose). The clamp is
        // relative to what was actually PUBLISHED last, not to the optimizer's previous output, because
        // the invariant it enforces is about the stream consumers see.
        // The localiser's PRE-CLAMP estimate for the last published frame (room frame, x/y/theta).
        // The predictor builds each prediction on the previous CORRECTED estimate, so
        // pred[k] - est[k-1] is exactly the sensor increment for this interval -- the motion the
        // clamp must let through untouched. Differencing two consecutive PREDICTIONS instead looks
        // equivalent but is not: it evaluates to increment + innovation[k-1], which would smuggle
        // the previous frame's correction through unbounded. See the clamp block.

        // Shared between maybe_publish_corrected_pose() (localizer thread) and publish_predicted_tick()
        // (imu-ingest thread): both read/write last_published_pose_/_ts_ms_/_est_ above and both call
        // into scene_graph_, so both must hold this for their whole critical section.
        // The corrected twist from the last REAL publish (see the FIX note on maybe_publish_corrected_
        // pose in specificworker.cpp), cached here so publish_predicted_tick() has something to
        // extrapolate with between lidar cycles without recomputing it.
        // Covariance from the last REAL publish, reused as-is for predicted ticks (not grown with
        // dt yet -- a known simplification; see the FIX note in publish_predicted_tick()).


        // Pose trace CSV (etc/pose_trace.csv): logs CORRECTED (~20 Hz, localizer thread, via
        // maybe_publish_corrected_pose) and PREDICTED (~100 Hz, imu-ingest thread, via
        // publish_predicted_tick -- wired 2026-09-04) poses with timestamps so the intermediate
        // dead-reckoned poses can be compared against the optimizer corrections (diagnose the noise
        // predict-publish injects). Both writers now run on DIFFERENT threads, serialized by
        // publish_mutex_ (NOT thread-affinity -- that assumption held only until the predicted writer
        // existed). type: 0=corrected, 1=predicted.
        void log_pose_trace(int type, std::int64_t valid_ts_ms,
                            const Eigen::Affine2f& pose, float innov_norm);

        // ── APPEND-ONLY RECORD OF LOCALISER DISCONTINUITIES (etc/pose_jumps.csv) ─────────────────
        // ★IT IS APPEND-ONLY BECAUSE THAT IS THE ENTIRE POINT. pose_trace.csv is truncated on every
        // start, and so is the controller's overlay log. On 2026-09-10 two corrections of 1.4 m and
        // 2.0 m were observed, and by the time anyone went back for them both files had been
        // rewritten by later runs -- the events were real, unrecoverable, and could not be quoted.
        // A failure this rare cannot be gone back for; it has to be caught the first time.
        // ★THE TRIGGER IS THE BASE'S OWN DECLARED CAPABILITY, NOT A TUNED NUMBER. A step is recorded
        // when the pose moves faster than the machine can physically move (common/robot_capability:
        // max_linear_speed_mps / max_rot_speed_rps, published by robot_concept off the base config).
        // That is a physical fact about the robot, so it cannot be quietly mis-set, and it cannot
        // discard a real error the way a magnitude cutoff would. If the capability channel is not up
        // the log stays SILENT and says so once -- guessing a bound would produce either a flood or
        // an empty file, and both would read as evidence.
        // ★Compared only against the previous point of the SAME type: a corrected pose following a
        // predicted one is entitled to step, and that is a correction, not a discontinuity.
        // Runs on both writer threads, serialized by publish_mutex_ exactly as pose_trace_ is.

        // Per-tick compute-timing CSV (etc/compute_timing.csv): exposes WHERE compute() stalls (viewer
        // vs dsr vs loc_fetch) so we can see why the corrected publish drops below the optimizer rate.
        // ★ Every duration column is MICROSECONDS (2026-09-03). They were integer milliseconds off
        //   QElapsedTimer::elapsed(), and since every stage is sub-ms the section columns had read
        //   exactly 0 for the life of the file — the CSV bounded compute() at ~3 ms but could never
        //   say which stage owned it. Any older compute_timing.csv on disk is in the ms schema and
        //   its section columns are all zeros; do not compare the two files column-for-column.
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
    // ── Stage 2: camera-vs-LiDAR mount calibration, pose-free (mount_lidar_pair.h) ───────────────
    // TWO accumulators on purpose. `mp_win_` resets every window and is directly comparable with
    // stage 1's per-window solve; `mp_pool_` never resets. Pooling is only legitimate if the pose was
    // the dominant between-window nuisance, which is a CLAIM — running both is what tests it.
    // ── One calibration channel per camera ───────────────────────────────────────────────────────
    // The driving camera keeps the members below; every OTHER camera in ImageEdge.calibCameras gets
    // one of these — its own ingestor, its own extraction, its own evidence file. Calibration and
    // driving are different jobs and need not use the same sensor.
    // CalibChannel now lives in rc::CalibChannels, with the rest of the RGB plumbing.
    /// Cost of the two per-tick camera pumps, in nanoseconds accumulated between reports. They run
    /// whether or not the optimiser will, so at 100% early exit they ARE the CPU.
    qint64 pump_ns_edge_ = 0, pump_ns_calib_ = 0;
    int    pump_ticks_ = 0;
    qint64 pump_report_ms_ = 0;
    /// Put each triple point in the ROOM by intersecting its measured ray with the plane it lies in.
    ///
    /// ★ NO DEPTH NEEDED, and that is the point: a triple point sits at a KNOWN height — the floor,
    ///   or the ceiling — so the ray through its measured pixel meets that plane at exactly one
    ///   place. This works for the panorama, which has no depth stream at all, and is better than
    ///   depth even where depth exists: the height is exact while ZED depth carries a measured
    ///   ~0.4% bias. It is also what the 2-D canvas needs to draw a camera corner beside its LiDAR
    ///   one, which until now it silently skipped for want of a range.
    static void place_triple_points_in_room(rc::ImageEdgeObs& obs, const rc::CameraIngestor& ing,
                                            const Eigen::Vector3f& pose);

    // Camera<->LiDAR mount calibration: rc::MountCalibrator (src/mount_calibrator.{h,cpp}).
    // Owns its accumulators, its pair logs, the per-camera publish bookkeeping and the sensor-triangle
    // loop closure; borrows only the graph, the config and the viewer slot.
    std::unique_ptr<rc::MountCalibrator> mount_;
    /// Pose publishing: rc::PosePublisher (src/pose_publisher.{h,cpp}). Owns the corrected and
    /// predicted writers, the clamp, the jump log and the mutex that makes the two threads safe.
    std::unique_ptr<rc::PosePublisher> pose_pub_;
    /// The RGB channels: rc::CalibChannels (src/calib_channels.{h,cpp}). Owns the driving camera and
    /// every calibration-only camera, their extractors, evidence and pair logs, and the room polygon
    /// they project.
    std::unique_ptr<rc::CalibChannels> calib_;
    /// Ground-truth grading: rc::GroundTruthLog (src/ground_truth_log.{h,cpp}).
    std::unique_ptr<rc::GroundTruthLog> gt_log_;
    // ── REVIEW 2026-09-13, FOUR REVIEWERS AGAINST THE 951e464 BASELINE ───────────────────────────
    // The extraction came out behaviour-preserving: 22 of 34 moved bodies byte-identical, 5
    // mechanical-only, every remaining difference deliberate and listed in the commits. No function
    // was silently dropped; every member removed from this header reappears in exactly ONE new header
    // with the same initialiser; all 254 long string literals, every params.X read and every output
    // file survive; the imbue(locale::classic()) count is 8 before and 8 after, so the locale rule
    // holds. No duplicated live state remains here — the names shared with the collaborator headers
    // are borrowed references, not copies.
    // WHAT THE REVIEW LEFT STANDING, deliberately unfixed because that pass changed nothing:
    //   · SIX METHOD DECLARATIONS BELOW HAVE NO DEFINITION ANYWHERE — maybe_publish_corrected_pose,
    //     publish_predicted_tick, log_pose_trace, update_rt_rate_readout, place_triple_points_in_room
    //     and write_pair_row all moved to collaborators. Nothing calls them on `this`, so there is no
    //     link error; the hazard is that this header still ADVERTISES that the worker publishes poses,
    //     writes the trace and places triple points. A future edit calling one compiles and fails to
    //     link, or passes review looking correct.
    //   · MUCH OF THE COMMENTARY BELOW describes state that is now in pose_publisher.h,
    //     mount_calibrator.h or ground_truth_log.h — the clamp, the pose trace, the append-only jump
    //     log, mp_win_/mp_pool_, the gt heading convention. The members are gone; the prose is not.
    //     Read it as history. Following it would invite re-adding exactly the copies the warning
    //     above this line forbids.
    //   · three members are dead and were ALREADY dead at the baseline, so they are not refactor
    //     damage: last_imu_sim_ts_, last_image_edge_ms_, self_target_active_ (0 readers, 0 writers).
    //   · shutting_down_ is declared AFTER pose_pub_ and gt_log_, which hold a reference to it, so on
    //     a real destructor run they would outlive their referent. Moot only because request_shutdown()
    //     ends in std::_Exit and no destructor body ever runs.
    // ⚠ NOTHING of a collaborator's state may be kept here after it moves. A copy left behind is
    // written by the worker and read by nobody, which is exactly how the commanded twist silently
    // became zero mid-refactor: it compiles, and the regression only shows at run time.
    /// Raw view of viewer_, kept in step with it, so collaborators constructed BEFORE the viewer can
    /// still reach it later without owning it or being rebuilt when it appears.
    rc::RoomViewer* viewer_raw_slot_ = nullptr;
    static void write_pair_row(std::ofstream& csv, const std::string& cam, std::int64_t ts,
                               const rc::mount::PairObs& pr, bool ceiling, float angle_deg,
                               float assoc_chi2, int n_rivals, float runnerup_chi2,
                               const Eigen::Vector3f& corr);

        // conclusions got drawn by hand on 2026-08-22.
        // Gated on the attributes EXISTING, so on the real robot nothing is written at all.
        // ── THE POSE CLAMP TAKES ITS BOUND FROM THE ROBOT, ONCE ──────────────────────────────────
        // Reads robot_max_linear_speed / robot_max_rot_speed off the robot node and installs them as
        // POSE_CLAMP_V_MAX / POSE_CLAMP_W_MAX. One-shot, and NOT in initialize(): the robot node may
        // not have synced from the persistent server yet at that point, and a clamp taken from a node
        // that is not there is silently the config fallback for the life of the process.

        // RT publish-rate monitor (shown in the window title at ~1 Hz so it can be watched visually).
        void update_rt_rate_readout(std::int64_t now_ms, bool on_gui_thread);

        std::atomic<bool> shutting_down_{false};

    signals:
        void presenceReady();
        void presenceLost();
};

#endif // SPECIFICWORKER_H
