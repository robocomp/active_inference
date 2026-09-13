// ── THE DSR GRAPH SLOTS ──────────────────────────────────────────────────────────────────────────
// SpecificWorker's node/attribute callbacks, in their own translation unit. Same class, and
// deliberately so: they are coordination — they take what the graph says and hand it to whichever
// collaborator owns that quantity — so they belong to the worker rather than to a new class of their
// own. They are 220 lines of dispatch, which is why they are not in specificworker.cpp.
//
// ⚠ QUEUED, NEVER DirectConnection. These run on the main thread because the connects are queued;
// under DirectConnection they would run on the FastDDS reader thread and corrupt the heap (CLAUDE.md).

#include "specificworker.h"

#include "calib_channels.h"
#include "ground_truth_log.h"
#include "mount_calibrator.h"
#include "pose_publisher.h"

#include <dsr/api/dsr_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QDateTime>
#include <QDebug>
#include <QString>

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names)
{
    if (!G || id == 0)
        return;

    const auto touches_any = [&att_names](std::initializer_list<const char*> names)
    {
        return std::ranges::any_of(names, [&att_names](const char* name)
        {
            return std::find(att_names.begin(), att_names.end(), name) != att_names.end();
        });
    };

    // LiDAR is read from the media plane (pumped in compute), not the DSR graph laser_* attrs.

    // ★ OUR OWN AFFORDANCE NODE — handled BEFORE the robot-id filter below, which would otherwise
    // discard it. The consumer clears epistemic_pending when it finishes; being told the moment that
    // happens is what removes the missed-edge race that wedged both agents (549 s with zero
    // completions detected while the consumer was completing continuously).
    if (std::ranges::any_of(att_names, [](const std::string& n)
                            { return n == "epistemic_pending" or n == "aff_outcome"; }))
        scene_graph_->on_affordance_attr_changed(id);

    if (const auto robot_id = scene_graph_->robot_id(); robot_id != 0 && id != robot_id)
        return;

    const bool touches_current_speed = touches_any({
        "robot_current_advance_speed",
        "robot_current_side_speed",
        "robot_current_angular_speed",
        "robot_current_speed_timestamp"
    });
    const bool touches_ref_speed = touches_any({
        "robot_ref_adv_speed",
        "robot_ref_side_speed",
        "robot_ref_rot_speed",
        "robot_ref_speed_timestamp"
    });
    if (not touches_current_speed and not touches_ref_speed)
        return;

    const auto node_opt = G->get_node(id);
    if (!node_opt.has_value())
        return;

    const auto &robot_node = node_opt.value();

    // No inertial branch here: the IMU rides the media plane (rc/imu/data) and never touches the
    // graph, so this slot has nothing to do for it. ImuIngestor owns that channel; the commentary
    // about why the gyro matters for yaw now lives with it, next to the code that uses it.

    if (touches_current_speed)
    {
        if (auto adv_value = G->get_attrib_by_name<robot_current_advance_speed_att>(robot_node); adv_value.has_value())
        {
            if (auto side_value = G->get_attrib_by_name<robot_current_side_speed_att>(robot_node); side_value.has_value())
            {
                if (auto rot_value = G->get_attrib_by_name<robot_current_angular_speed_att>(robot_node); rot_value.has_value())
                {
                    if (auto ts_value = G->get_attrib_by_name<robot_current_speed_timestamp_att>(robot_node); ts_value.has_value())
                    {
                        const auto source_ts = static_cast<std::uint64_t>(ts_value.value());
                        if (source_ts > 0 and source_ts > last_robot_current_speed_timestamp_)
                        {
                            static std::mt19937 gen{std::random_device{}()};
                            const float nf = params.ODOMETRY_NOISE_FACTOR;

                            auto add_noise = [&](float value) -> float {
                                if (nf <= 0.f || value == 0.f) return value;
                                std::normal_distribution<float> dist(0.f, std::abs(value) * nf);
                                return value + dist(gen);
                            };

                            rc::OdometryReading odom;
                            odom.adv = add_noise(adv_value.value());
                            odom.side = add_noise(side_value.value());
                            odom.rot = add_noise(rot_value.value());
                            odom.source_ts_ms = static_cast<std::int64_t>(source_ts);
                            // The producer's own clock, where it has one. A simulator's velocities are
                            // per SIMULATION second, so integrating them over wall-clock intervals
                            // under-counts by the sim/wall ratio; this is the stamp that makes the
                            // interval agree with the rate. Absent on real hardware, where the two
                            // clocks are the same thing and integration_ts_ms() falls back cleanly.
                            if (const auto sim_ts = G->get_attrib_by_name<robot_current_speed_sim_timestamp_att>(robot_node);
                                sim_ts.has_value())
                                odom.sim_ts_ms = static_cast<std::int64_t>(sim_ts.value());
                            if (const auto sim_flag = G->get_attrib_by_name<robot_current_speed_simulated_att>(robot_node);
                                sim_flag.has_value())
                                odom.simulated = sim_flag.value();
                            // ── The producer's OWN per-sample velocity variance ────────────────────
                            // robot_concept forwards FullPoseEuler::velCov here as {adv, side, rot},
                            // already de-crossed from the body indices (m11 = forward on this +Y robot).
                            // Absent attribute = this producer never published it; a NEGATIVE entry =
                            // it published and said "unknown" for that channel. Both leave the fields
                            // at -1 and the preintegrator falls back to its asserted constant, so a
                            // producer that says nothing is not silently treated as a perfect one.
                            //
                            // NOT NOISE-FACTOR-ADJUSTED on purpose: ODOMETRY_NOISE_FACTOR above injects
                            // SYNTHETIC noise for robustness tests, and pretending the sensor reported
                            // that would let a test knob masquerade as a measurement.
                            if (const auto vvar = G->get_attrib_by_name<robot_current_speed_variance_att>(robot_node);
                                vvar.has_value())
                            {
                                const auto& v = vvar.value().get();
                                if (v.size() >= 3)
                                {
                                    const auto stated = [](float x) { return x >= 0.f ? x : -1.f; };
                                    odom.var_adv  = stated(v[0]);
                                    odom.var_side = stated(v[1]);
                                    odom.var_rot  = stated(v[2]);
                                }
                            }
                            if (odom.simulated and odom.sim_ts_ms > 0)
                                sim_clock_.observe(odom.source_ts_ms, odom.sim_ts_ms);
                            odom.recv_ts_ms = static_cast<std::int64_t>(
                                std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count());
                            odom.timestamp = std::chrono::high_resolution_clock::time_point(
                                std::chrono::milliseconds(source_ts));

                            room_concept_.record_odometry_ingress("dsr_current_speed",
                                                                  adv_value.value(),
                                                                  rot_value.value(),
                                                                  odom.adv,
                                                                  odom.rot,
                                                                  odom.source_ts_ms);
                            // ── PER-SAMPLE log, for measuring the stream's own statistics ──────
                            // Written HERE, where samples arrive, and not from the per-cycle debug
                            // row: that row keeps only the latest sample and is emitted once per
                            // lidar sweep, so it aliases this stream onto a slower one. An
                            // autocorrelation computed on aliased rows answers a different question
                            // than the one asked, and looks perfectly reasonable doing it.
                            //
                            // `seq` is the arrival counter and `source_ts_ms` the producer's stamp:
                            // together they make a DROPPED sample visible as a stamp gap without a
                            // seq gap, which matters because a dropped sample changes the effective
                            // sampling interval and therefore the very quantity being measured.
                            if (params.ODOM_SAMPLE_LOG)
                            {
                                if (not odom_sample_log_.is_open())
                                {
                                    odom_sample_log_.open("etc/odom_samples.csv",
                                                          std::ios::out | std::ios::trunc);
                                    // Locale-proof the WRITE side: these machines run es_ES, where a
                                    // stray comma separator would silently turn every value into a
                                    // column break. See CLAUDE.md.
                                    odom_sample_log_.imbue(std::locale::classic());
                                    odom_sample_log_ << "seq,recv_ts_ms,source_ts_ms,sim_ts_ms,simulated,"
                                                        "adv,side,rot,var_adv,var_side,var_rot,"
                                                        "cmd_adv,cmd_side,cmd_rot,cmd_ts_ms\n";
                                }
                                odom_sample_log_ << odom_sample_seq_++ << ','
                                                 << odom.recv_ts_ms   << ','
                                                 << odom.source_ts_ms << ','
                                                 << odom.sim_ts_ms    << ','
                                                 << (odom.simulated ? 1 : 0) << ','
                                                 << odom.adv << ',' << odom.side << ',' << odom.rot << ','
                                                 << odom.var_adv << ',' << odom.var_side << ',' << odom.var_rot << ','
                                                 << last_cmd_adv_ << ',' << last_cmd_side_ << ','
                                                 << last_cmd_rot_ << ',' << last_cmd_ts_ms_ << '\n';
                                odom_sample_log_.flush();   // 50 lines/s; a lost tail costs more
                            }
                            odometry_buffer_.put<0>(std::move(odom), static_cast<std::uint64_t>(odom.recv_ts_ms));
                            last_robot_current_speed_timestamp_ = source_ts;
                            // Straight to the publisher, which is the only reader: a prediction
                            // extrapolates the twist cached at the last correction, so a copy kept
                            // here and never forwarded would silently make every predicted pose
                            // extrapolate a ZERO twist.
                            pose_pub_->set_robot_speed(adv_value.value(), side_value.value(),
                                                       rot_value.value());
                        }
                    }
                }
            }
        }
    }

    if (touches_ref_speed)
    {
        if (auto adv_value = G->get_attrib_by_name<robot_ref_adv_speed_att>(robot_node); adv_value.has_value())
        {
            if (auto side_value = G->get_attrib_by_name<robot_ref_side_speed_att>(robot_node); side_value.has_value())
            {
                if (auto rot_value = G->get_attrib_by_name<robot_ref_rot_speed_att>(robot_node); rot_value.has_value())
                {
                    if (auto ts_value = G->get_attrib_by_name<robot_ref_speed_timestamp_att>(robot_node); ts_value.has_value())
                    {
                        const auto source_ts = static_cast<std::uint64_t>(ts_value.value());
                        if (source_ts > 0 and source_ts > last_robot_ref_speed_timestamp_)
                        {
                            rc::VelocityCommand cmd;
                            cmd.adv_y = adv_value.value();
                            cmd.adv_x = side_value.value();
                            cmd.rot = rot_value.value();
                            cmd.source_ts_ms = static_cast<std::int64_t>(source_ts);
                            cmd.recv_ts_ms = static_cast<std::int64_t>(
                                std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count());
                            cmd.timestamp = std::chrono::high_resolution_clock::time_point(
                                std::chrono::milliseconds(source_ts));

                            room_concept_.record_command_ingress("dsr_ref",
                                                                 adv_value.value(),
                                                                 rot_value.value(),
                                                                 cmd.adv_y,
                                                                 cmd.rot,
                                                                 cmd.source_ts_ms);
                            last_cmd_adv_    = cmd.adv_y;
                            last_cmd_side_   = cmd.adv_x;
                            last_cmd_rot_    = cmd.rot;
                            last_cmd_ts_ms_  = cmd.source_ts_ms;
                            velocity_buffer_.put<0>(std::move(cmd), source_ts);
                            last_robot_ref_speed_timestamp_ = source_ts;
                        }
                    }
                }
            }
        }
    }
}


void SpecificWorker::modify_node_slot(std::uint64_t /*id*/, const std::string& /*type*/)
{
    // LiDAR now arrives via the media plane (pumped in compute); the graph 'laser' node is not read.
}

// Re-run the graph viewer's twopi layout now and again once queued, so it also settles after the
// pending node/edge update signals are processed. No-op when Agent.graph is off (no viewer widget).
// Lives here (not in the regenerated genericworker) so robocompdsl regeneration cannot clobber it;
// mirrors robot_concept. find_graph_viewer() is the inherited GenericWorker helper.
