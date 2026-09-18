/*
 *    Copyright (C) 2026 by YOUR NAME HERE
 *
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
#include "specificworker.h"

#include "mount_calibrator.h"
#include "pose_publisher.h"
#include "calib_channels.h"
#include "ground_truth_log.h"
#include <limits>
#include "../../common/robot_capability/robot_capability.h"
#include "../../common/room_resolve/room_resolve.h"   // rc::room::current_room_frame (the room is room_1, room_2, …)

#include "image_edge_ops.h"   // xyz_from_pixel_depth(): model-aware, unlike cortex's pinhole-only version
#include <filesystem>
#include <locale>

#include <algorithm>
#include <print>
#include <sys/resource.h>   // getrusage — process CPU% readout
#include <cstdio>   // std::remove — the Reset must delete the saved evidence
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
#include <random>
#include <stdexcept>
#include <QElapsedTimer>
#include <QCoreApplication>
#include <fstream>
#include <unordered_set>
#include <QDir>
#include <QFile>
#include <QDateTime>
#include <QMetaObject>
#include <QFileInfo>
#include <QThread>
#include <QVBoxLayout>

#include <variant>

///////////////////////////////////////////////////////////////////////////////
SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check)
    : GenericWorker(configLoader, tprx)
{
    this->startup_check_flag = startup_check;
    if (this->startup_check_flag)
    {
        this->startup_check();
    }
    else
    {
#ifdef HIBERNATION_ENABLED
        hibernationChecker.start(500);
#endif
        const int period = configLoader.get<int>("Period.Compute");

        states["Waiting"] = std::make_unique<GRAFCETStep>("Waiting", period,
            std::bind(&SpecificWorker::waiting_loop, this),
            std::bind(&SpecificWorker::waiting_enter, this));
        states["Operating"] = std::make_unique<GRAFCETStep>("Operating", period,
            std::bind(&SpecificWorker::operating_loop, this),
            std::bind(&SpecificWorker::operating_enter, this));
        states["Degraded"] = std::make_unique<GRAFCETStep>("Degraded", period,
            std::bind(&SpecificWorker::degraded_loop, this),
            std::bind(&SpecificWorker::degraded_enter, this));

        states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
        states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
        states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
        states["Degraded"]->addTransition(states["Degraded"].get(), SIGNAL(entered()), states["Waiting"].get());

        statemachine.addState(states["Waiting"].get());
        statemachine.addState(states["Operating"].get());
        statemachine.addState(states["Degraded"].get());

        statemachine.setChildMode(QState::ExclusiveStates);
        statemachine.start();
        auto error = statemachine.errorString();
        if (error.length() > 0) { qWarning() << error; throw error; }
    }
}

///////////////////////////////////////////////////////////////////////////////
SpecificWorker::~SpecificWorker()
{
    request_shutdown();
}

// initialize() and request_shutdown() live in specificworker_startup.cpp — same class, separate
// translation unit. Startup is ~440 lines of coordination and does not belong beside the per-tick
// loop; the order dependencies it carries are documented at the top of that file.

void SpecificWorker::compute()
{
    const auto now_ms = QDateTime::currentMSecsSinceEpoch();
    QElapsedTimer compute_timer;
    compute_timer.start();
    // ── The ceiling the LiDAR measured, into the model, EVERY CYCLE ──────────────────────────────
    // The producer is a LiDAR-thread atomic that only becomes valid once the ceiling check has seen
    // a scan, so a one-shot copy at start-up can only ever move a zero. From here the existing
    // machinery does the rest: room_scene_graph publishes it on the room node (debounced, so an
    // attribute is not rewritten every frame) and pump_image_edges projects the wall-ceiling contour
    // at it instead of at the stated constant.
    if (lidar_ingestor_)
    {
        room_concept_.set_measured_ceiling(
            lidar_ingestor_->measured_ceiling_z_.load(std::memory_order_relaxed));
        // The width travels with the value. A consumer that may REFINE this number needs to know how
        // much it is allowed to move, and one that only draws it can ignore the second number.
        room_concept_.set_measured_ceiling_sigma(
            lidar_ingestor_->measured_ceiling_sigma_.load(std::memory_order_relaxed));
    }
    auto init_time = std::chrono::steady_clock::now();
    // MICROSECONDS, not milliseconds. These were qint64 *_ms read off QElapsedTimer::elapsed(),
    // which is integer ms — and every stage here is sub-millisecond, so every section column in
    // etc/compute_timing.csv had been exactly 0 for the life of the file. The CSV could report that
    // compute() cost ~3 ms but never which stage, which is precisely the split any decoupling work
    // needs to judge itself on. nsecsElapsed() costs the same and resolves it.
    qint64 t_affordance_us = 0;
    qint64 t_loc_fetch_us = 0;
    qint64 t_viewer_us = 0;
    qint64 t_dsr_us = 0;
    qint64 t_ui_us = 0;
    qint64 t_health_us = 0;
    bool   did_publish = false;   // a corrected RT block was published this tick (for compute_timing.csv)

    // ── OPEN DOORWAYS, AT THE RATE A DOOR ACTUALLY MOVES ────────────────────────────────────────
    // 1 Hz, not the 20 Hz of the scan. Two different things change at two different rates: the
    // APERTURE is static — door_concept's invariant is that it does not move when the leaf swings, so
    // its segment is resolved once per door and cached — while door_open_prob changes over SECONDS, at
    // the speed of a door. Reading the graph at scan rate would be 20 Hz of get_nodes_by_type plus a
    // transform walk per door to watch a quantity that moves once a second.
    // The staleness this buys is bounded and harmless: a door caught mid-swing is weighted as it was
    // up to a second ago. Because the weight is continuous (1 - p_open) that is a slightly wrong
    // weight for one second, not a wrong decision — which is the advantage of having no threshold in
    // there at all.
    if (inner_eigen_ and (last_door_scan_ms_ == 0 or now_ms - last_door_scan_ms_ >= 1000))
    {
        last_door_scan_ms_ = now_ms;
        auto open_doors = doors_.refresh(*G, *inner_eigen_, rc::room::current_room_frame(*G));
        static std::size_t last_n = std::numeric_limits<std::size_t>::max();
        if (open_doors.size() != last_n)
        {
            last_n = open_doors.size();
            QString which;
            for (const auto& d : open_doors)
                which += QString(" %1(p=%2)").arg(QString::fromStdString(d.name)).arg(d.p_open, 0, 'f', 2);
            qInfo().noquote() << QString("[room][doors] %1 open aperture(s) now discounting the scan:%2"
                                         " | chain walks so far %3")
                                     .arg(open_doors.size()).arg(which.isEmpty() ? " -" : which)
                                     .arg(doors_.resolves());
        }
        // ── WHY A DOOR DID NOT BECOME AN APERTURE ───────────────────────────────────────────────
        // Four stages can fail silently here and from the outside all four look the same: the scan is
        // not filtered. One row per refresh naming the count at each stage, so the answer is read
        // rather than guessed.
        {
            static std::ofstream dcsv;
            if (not dcsv.is_open())
            {
                std::filesystem::create_directories("tmp");
                dcsv.open("tmp/door_filter.csv", std::ios::out | std::ios::trunc);
                if (dcsv.is_open())
                {
                    dcsv.imbue(std::locale::classic());
                    dcsv << "ts_ms,objects,named_door,subtype_ok,believed_open,no_width,no_transform,"
                            "emitted,resolves,door,p_open,ax,ay,bx,by\n";
                }
            }
            if (dcsv.is_open())
            {
                const auto& t = doors_.tally();
                const auto row = [&](const char* name, float p, float ax, float ay, float bx, float by)
                {
                    dcsv << now_ms << ',' << t.objects << ',' << t.named_door << ',' << t.subtype_ok
                         << ',' << t.believed_open << ',' << t.no_width << ',' << t.no_transform
                         << ',' << t.emitted << ',' << doors_.resolves() << ',' << name << ','
                         << p << ',' << ax << ',' << ay << ',' << bx << ',' << by << '\n';
                };
                if (open_doors.empty()) row("-", -1.f, 0.f, 0.f, 0.f, 0.f);
                for (const auto& d : open_doors)
                    row(d.name.c_str(), d.p_open, d.a.x(), d.a.y(), d.b.x(), d.b.y());
                dcsv.flush();
            }
        }
        room_concept_.set_door_apertures(std::move(open_doors));
    }

    if (last_affordance_monitor_ms_ == 0 || now_ms - last_affordance_monitor_ms_ >= 200)
    {
        QElapsedTimer section_timer;
        section_timer.start();
        scene_graph_->monitor_affordance();
        t_affordance_us = section_timer.nsecsElapsed() / 1000;
        last_affordance_monitor_ms_ = now_ms;
    }

    // LiDAR is drained by lidar_ingestor_'s dedicated ingest thread (started at Operating-enter), which
    // pumps a fresh scan to the localizer with ~0-2 ms latency instead of this ~16 ms tick. compute()
    // only reads the resulting buffer/result below.

    // ── WHERE THE CPU GOES WHEN NOTHING IS BEING OPTIMISED ──────────────────────────────────────
    // At 100% early exit the localiser update is ~1.6 ms at 20 Hz — 3% of a core — so the rest of
    // the CPU is not the thing being watched. Both pumps below run EVERY compute tick regardless of
    // whether the optimiser will: the driving camera's extraction, and then the SAME extraction
    // again for each entry in ImageEdge.calibCameras (zed + ricoh today, so three per tick).
    // Their only live consumers while the gate holds are the calibration accumulator and the canvas
    // — the loss sees them just on optimised frames. Reported as ms per second, i.e. percent of one
    // core, so the answer is a number rather than an argument about which pump is heavy.
    {
        QElapsedTimer t; t.start();
        calib_->pump_image_edges();
        const auto t_edge = t.nsecsElapsed();
        t.restart();
        calib_->pump_calib_channels();   // extra cameras: calibration only, never the pose
        const auto t_calib = t.nsecsElapsed();
        pump_ns_edge_ += t_edge; pump_ns_calib_ += t_calib; ++pump_ticks_;
        const auto now_ms = QDateTime::currentMSecsSinceEpoch();
        if (pump_report_ms_ == 0) pump_report_ms_ = now_ms;
        else if (now_ms - pump_report_ms_ > 5000)
        {
            const double secs = 1e-3 * static_cast<double>(now_ms - pump_report_ms_);
            // ★ The CONVERSION ratio belongs on the same line as the extraction cost, because the
            //   two answer one question between them. The pumps measured 5.5% of a core, which is
            //   NOT where this agent's CPU goes; the frames those pumps read are pulled and greyed
            //   on the per-camera ingest threads, and that is what minConvertIntervalMs throttles.
            //   Printing converted/delivered per camera is what makes the throttle falsifiable —
            //   a ratio of 1.00 means it never fired, and no CPU number would have said so.
            QString conv;
            conv += QString::fromStdString(calib_->convert_stats_line());
            qInfo().nospace().noquote()
                << "[pumps] over " << secs << " s: image_edge " << (1e-6 * pump_ns_edge_ / secs)
                << " ms/s  calib_channels " << (1e-6 * pump_ns_calib_ / secs)
                << " ms/s  (= % of one core) | " << pump_ticks_ << " ticks | conv" << conv
                << " (converted/delivered)";
            pump_report_ms_ = now_ms; pump_ns_edge_ = pump_ns_calib_ = 0; pump_ticks_ = 0;
        }
    }

    QElapsedTimer section_timer;
    section_timer.start();
    const auto loc_res  = room_concept_.get_last_result();
    const bool have_loc = loc_res.has_value() && loc_res->ok;
    t_loc_fetch_us = section_timer.nsecsElapsed() / 1000;

    const Eigen::Affine2f pose_for_draw = viewer_->best_available_pose(loc_res, have_loc);
    
    // ── Update 2-D viewer ─────────────────────────────────────────────────
    const Eigen::Affine2f loc_pose = have_loc ? loc_res->robot_pose : pose_for_draw;
    const bool use_loc = have_loc && !loc_res->lidar_scan.empty();

    std::vector<Eigen::Vector3f> lidar_for_canvas;
    if (use_loc)
        lidar_for_canvas = loc_res->lidar_scan;
    else
    {
        const auto& [lidar_from_buffer] = lidar_ingestor_->buffer().read_last();
        if (lidar_from_buffer.has_value())
            lidar_for_canvas = lidar_from_buffer->first;
    }

    // ── THE FILTER, MADE VISIBLE ────────────────────────────────────────────────────────────────
    // The canvas draws what the SDF keeps. If a door is open and the returns beyond it are still on
    // screen, the filter is not working — which is a stronger and faster test than any log line, and
    // it is why this is done here rather than left to a CSV.
    {
        int removed = 0;
        lidar_for_canvas = room_concept_.filter_through_door(lidar_for_canvas, &removed);
        static int last_removed = -1;
        if (removed != last_removed and (removed > 0 or last_removed > 0))
        {
            last_removed = removed;
            qInfo().noquote() << QString("[room][doors] %1 scan point(s) dropped from the canvas as "
                                         "through-the-door returns").arg(removed);
        }
    }

    const bool on_gui_thread = (QThread::currentThread() == this->thread());
    if (on_gui_thread)
    {
        section_timer.restart();
        viewer_->update_viewer(loc_res, have_loc, pose_for_draw, lidar_for_canvas, loc_pose, use_loc);
        viewer_->draw_landmarks(scene_graph_->pinned_landmarks(), scene_graph_->pinned_measured(), pose_for_draw);
        t_viewer_us = section_timer.nsecsElapsed() / 1000;
    }

    // ── DSR graph update (only on fresh localization frames) ──────────────
    // In PreserveBootstrapRoom mode the room is a static prior: do NOT create/reparent the
    // room or write robot->room (that would clobber the fixed low-cov bootstrap edge). The
    // localizer still runs for the viewer; it just doesn't touch the graph.
    // Publish near the lidar rate (~60 ms) so the RT timestamped history is dense enough for
    // consumers to bracket a recent lidar-stamped query (e.g. the controller's overlay) instead of
    // clamping to a stale block. Steady RT updates on one edge — not join/leave churn — so low risk.
    {
        section_timer.restart();
        // FIX 2026-09-03: this redundant call REMOVED. The localizer's own on_result_ready callback
        // now calls maybe_publish_corrected_pose() directly (no more QueuedConnection marshal to this
        // thread — see the FIX note at set_on_result_ready), so it is no longer guaranteed to run on
        // the main thread. Calling it again from here would race the localizer thread's call on the
        // dedup state (last_dsr_publish_try_ms_/last_dsr_published_ts_ms_, not atomic) and on the
        // ofstream members touched inside dsr_update_calibration/dsr_update_affordance — exactly the
        // class of bug the old comment's "whichever fires first wins" was quietly relying on both
        // firing on the same thread to make safe. did_publish is no longer meaningful per compute()
        // tick now that publishing is fully decoupled from the compute cadence; see pose_trace.csv /
        // optimizer_timing.csv for the real publish timing.
        did_publish = false;
        t_dsr_us = section_timer.nsecsElapsed() / 1000;
    }

    // Visual RT-rate monitor: refresh the custom-widget readout once per second (low freq, cheap).
    pose_pub_->update_rt_rate_readout(now_ms, on_gui_thread);

    if (on_gui_thread)
    {
        section_timer.restart();
        viewer_->update_ui(loc_res);
        // Room-stabilization indicator: AMBER while a global grid search is relocating the robot,
        // GREEN once the room node exists in the graph (the very condition downstream consumers gate
        // on), RED while stable frames are still accumulating.
        viewer_->set_room_stable(scene_graph_->room_node_created(),
                                 scene_graph_->stable_frames(),
                                 params.STABLE_FRAMES_REQUIRED,
                                 room_concept_.is_grid_searching());
        t_ui_us = section_timer.nsecsElapsed() / 1000;
    }

    t_health_us = 0;

    const auto total_ms = compute_timer.elapsed();
    // Sub-millisecond resolution total. MICROSECONDS — named total_us everywhere it is emitted so it
    // cannot be mistaken for milliseconds. total_ms below is kept ONLY for the >50 ms stall trigger.
    const auto elapsed_since_init_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - init_time).count();

    // Per-tick compute-timing CSV — shows whether compute() holds the period (≈50 ms) or stalls, and
    // which section is responsible, explaining publish-rate loss vs the optimizer's production rate.
    if (not compute_csv_open_attempted_)
    {
        compute_csv_open_attempted_ = true;
        compute_csv_.open("etc/compute_timing.csv", std::ios::out | std::ios::trunc);
        if (compute_csv_.is_open())
            compute_csv_ << "wall_ms,total_us,affordance_us,loc_fetch_us,viewer_us,dsr_us,ui_us,did_publish,gui_thread\n";
    }
    if (compute_csv_.is_open())
    {
        compute_csv_ << now_ms << ',' << elapsed_since_init_us << ',' << t_affordance_us << ',' << t_loc_fetch_us
                     << ',' << t_viewer_us << ',' << t_dsr_us << ',' << t_ui_us << ','
                     << (did_publish ? 1 : 0) << ',' << (on_gui_thread ? 1 : 0) << '\n';
        compute_csv_.flush();
    }

    if (last_compute_timing_log_ms_ == 0 || now_ms - last_compute_timing_log_ms_ >= 3000 || total_ms > 50)
    {
        last_compute_timing_log_ms_ = now_ms;
        qInfo() << "[Timing][compute]"
                << "total_us=" << elapsed_since_init_us   // MICROSECONDS, as are every section below
                << "affordance_us=" << t_affordance_us
                << "loc_fetch_us=" << t_loc_fetch_us
                << "viewer_us=" << t_viewer_us
                << "dsr_us=" << t_dsr_us
                << "ui_us=" << t_ui_us
                << "health_us=" << t_health_us
                << "gui_thread=" << on_gui_thread;
    }
    fps_counter_.print("[Compute]", 3000);
}

void SpecificWorker::initialize_room_model_from_svg()
{
    // The scenario names the FILE; LayoutDir says where the layouts live. An explicit path in the
    // config (anything containing a separator) wins, so a one-off layout outside the folder still
    // works without a config-shape change.
    const std::string svg_path =
        (params.ROOM_LAYOUT_SVG.find('/') != std::string::npos or params.LAYOUT_DIR.empty())
            ? params.ROOM_LAYOUT_SVG
            : params.LAYOUT_DIR + "/" + params.ROOM_LAYOUT_SVG;
    auto room_polygon = rc::SvgRoomLoader::load_polygon_points(
        svg_path, "room_contour", false, true);
    if (room_polygon.size() < 3)
        qWarning() << "[room] layout" << QString::fromStdString(svg_path)
                   << "gave fewer than 3 points — check RoomConcept.LayoutDir and the scenario's"
                      " RoomLayoutSvg. The layouts moved to active_inference/layouts on 2026-08-29.";
    if (room_polygon.size() >= 3)
    {
        // Move the room-frame origin onto the layout's geometric centre. Everything downstream is
        // origin-agnostic (the grid search derives its box from the polygon bbox, the planner grid
        // carries its own origin, and the DSR `room` node republishes these very vertices as
        // delimiting_polygon_x/y), so the shift propagates on its own.
        const Eigen::Vector2f offset = params.RECENTER_ROOM_POLYGON
                                   ? rc::SvgRoomLoader::recenter_to_bbox_center(room_polygon)
                                   : Eigen::Vector2f::Zero();
        if (offset.norm() > 1e-3f)
            qInfo() << "[room] Room polygon recentred on its bbox centre: origin moved by"
                    << offset.x() << offset.y() << "m."
                    << "Poses in the room frame are shifted by that amount — a seed pose or object"
                    << "RT edges saved before this change are stale.";

        calib_->set_room_polygon(room_polygon, offset);
        room_concept_.configure_room_from_polygon(calib_->room_polygon());
        room_initialized_from_svg_polygon_ = true;
        return;
    }
    calib_->set_room_polygon({}, Eigen::Vector2f::Zero());
    room_concept_.configure_room_from_rect(params.GRID_MAX_DIM.width(), params.GRID_MAX_DIM.height());
    room_initialized_from_svg_polygon_ = false;
    qWarning() << "SVG polygon not loaded; using rectangular fallback.";
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::save_robot_pose_on_exit() const
{
    // An estimated map defines its own frame each run; a seed pose from it would mislead a later
    // Given-mode start against the layout (and an Estimate-mode start ignores seeds anyway).
    if (room_concept_.estimating()) return;
    Eigen::Vector3f pose = Eigen::Vector3f::Zero();
    if (const auto loc = room_concept_.get_last_result(); loc.has_value() && loc->ok)
    {
        pose[0] = loc->state[2]; pose[1] = loc->state[3]; pose[2] = loc->state[4];
    }
    else if (room_concept_.is_initialized())
    {
        const auto state = room_concept_.get_current_state();
        pose[0] = state[2]; pose[1] = state[3]; pose[2] = state[4];
    }
    else return;

    const QString qpath = QString::fromStdString(pose_file_path());
    QDir().mkpath(QFileInfo(qpath).absolutePath());
    std::ofstream out(qpath.toStdString(), std::ios::trunc);
    if (!out.is_open()) { qWarning() << "Cannot open pose file:" << qpath; return; }
    out << pose[0] << ' ' << pose[1] << ' ' << pose[2] << '\n';
}

void SpecificWorker::save_robot_pose_once()
{
    if (pose_saved_.exchange(true)) return;
    save_robot_pose_on_exit();
}

std::string SpecificWorker::pose_file_path() const
{
    auto find_etc_upwards = [](const QString& start) -> QString {
        QDir dir(start);
        for (int depth = 0; depth < 8; ++depth)
        {
            const QString etc_dir = dir.absoluteFilePath("etc");
            if (QDir(etc_dir).exists()) return etc_dir;
            if (!dir.cdUp()) break;
        }
        return {};
    };
    const QString from_app = find_etc_upwards(QCoreApplication::applicationDirPath());
    if (!from_app.isEmpty()) return (from_app + "/last_robot_pose.txt").toStdString();
    const QString from_cwd = find_etc_upwards(QDir::currentPath());
    if (!from_cwd.isEmpty()) return (from_cwd + "/last_robot_pose.txt").toStdString();
    return (QDir(QCoreApplication::applicationDirPath() + "/../etc").absolutePath()
            + "/last_robot_pose.txt").toStdString();
}


///////////////////////////////////////////////////////////////////////////////
/// SLOTS from GUI and DSR signals
///////////////////////////////////////////////////////////////////////////////
/// @brief ///////////DSR callback triggered when a node is modified. We check if it's the robot node and if the current speed attributes have been updated, then we read them and push them to the odometry buffer with some optional noise added.

// The DSR graph slots (modify_node_attrs_slot, modify_node_slot) live in specificworker_slots.cpp —
// same class, separate translation unit: they are 220 lines of dispatch, and dispatch is what this
// worker is for, so they stay its methods rather than becoming a class of their own.

void SpecificWorker::trigger_graph_layout_twopi()
{
    auto graph_viewer_owner = find_graph_viewer("");
    if (not graph_viewer_owner)
        return;

    QWidget* graph_widget = graph_viewer_owner->get_widget(DSR::DSRViewer::view::graph);
    auto* graph_viewer = qobject_cast<DSR::GraphViewer*>(graph_widget);
    if (not graph_viewer)
        return;

    // Run now and once queued, so layout also happens after pending node/edge
    // update signals are processed by the viewer.
    graph_viewer->compute_layout("twopi");
    QMetaObject::invokeMethod(graph_viewer,
                              [graph_viewer]() { graph_viewer->compute_layout("twopi"); },
                              Qt::QueuedConnection);
}
///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
}

void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
}

int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}


///////////////////////////////////////////////////////////////////////////////
/// ICE INTERFACE CALLBACKS
///////////////////////////////////////////////////////////////////////////////

void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data)
{
    rc::VelocityCommand cmd;
    float raw_adv_y = 0.f;
    float raw_rot = 0.f;
    for (const auto& axis : data.axes)
    {
        if      (axis.name == "rotate")
        {
            raw_rot = axis.value;
            cmd.rot = axis.value;
        }
        else if (axis.name == "advance")
        {
            raw_adv_y = axis.value / 1000.0f;
            cmd.adv_y = raw_adv_y;
        }
        else if (axis.name == "side")    cmd.adv_x = 0.0f;
    }
    cmd.timestamp  = std::chrono::high_resolution_clock::now();
    cmd.recv_ts_ms = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    room_concept_.record_command_ingress("joystick",
                                         raw_adv_y,
                                         raw_rot,
                                         cmd.adv_y,
                                         cmd.rot,
                                         cmd.recv_ts_ms);
    velocity_buffer_.put<0>(std::move(cmd), static_cast<std::uint64_t>(cmd.recv_ts_ms));
}
