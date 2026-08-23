/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_viewer.h.
 */

#include "room_viewer.h"

#include <algorithm>
#include <cmath>

#include <QByteArray>
#include <QColor>
#include <QSettings>
#include <QVBoxLayout>
#include <QtCore/qdebug.h>

#include "epistemic_controller.h"
#include "camera_visualizer.h"
#include "../../common/media_transport/media_transport.h"   // descriptor_from_graph

namespace rc
{

namespace
{
constexpr auto kWinSettingsOrg = "RoboComp";
constexpr auto kWinSettingsApp = "room_concept";
// _h2 suffix: the layout changed from a vertical stack (canvas over timeseries) to a horizontal
// split (canvas | timeseries column). Old persisted geometry/splitter state is for the tall layout
// and would fight the new compact default — new keys let the new layout start fresh, then persist.
constexpr auto kWinSettingsKey = "RoomLayoutWindow_geometry_h2";
constexpr auto kSplitterStateKey = "RoomLayoutWindow_splitterState_h2";
}  // namespace

RoomViewer::~RoomViewer()
{
    save_window_geometry();
}

void RoomViewer::restore_window_geometry()
{
    if (!custom_widget_)
        return;
    QSettings settings(kWinSettingsOrg, kWinSettingsApp);
    const QByteArray geom = settings.value(kWinSettingsKey).toByteArray();
    if (!geom.isEmpty())
        custom_widget_->restoreGeometry(geom);
    else
        custom_widget_->resize(700, 430);   // compact: square canvas on the left + timeseries column on the right
}

void RoomViewer::save_window_geometry() const
{
    if (!custom_widget_)
        return;
    QSettings settings(kWinSettingsOrg, kWinSettingsApp);
    settings.setValue(kWinSettingsKey, custom_widget_->saveGeometry());
    // Persist the splitter division separately — saveGeometry() only stores the window rect, not the
    // splitter handle position, so remembering where the user dragged it needs saveState().
    if (custom_widget_->splitter != nullptr)
        settings.setValue(kSplitterStateKey, custom_widget_->splitter->saveState());
    settings.sync();
}

RoomViewer::RoomViewer(std::shared_ptr<DSR::DSRGraph> graph,
                                       rc::RoomConfig& params,
                                       const std::vector<Eigen::Vector2f>& room_polygon,
                                       bool has_room_polygon,
                                       rc::RoomConcept& room_concept,
                                       rc::EpistemicController& epistemic)
    : params_(&params), has_room_polygon_(has_room_polygon),
      room_concept_(&room_concept), epistemic_(&epistemic), graph_(graph)
{
    // Own top-level window (parent == nullptr), NOT docked into the DSR graph viewer. This
    // decouples the layout GUI from Agent.graph, so the agent runs with graph=false (no
    // DSRViewer created at all). Mirrors retina's independent custom drawing windows.
    custom_widget_ = new Custom_widget();
    custom_widget_->setWindowTitle(QStringLiteral("room_concept — layout"));
    restore_window_geometry();
    custom_widget_->show();
    viewer_2d_ = new rc::Viewer2D(custom_widget_->frame, params_->GRID_MAX_DIM, true);
    viewer_2d_->show();
    viewer_2d_->add_robot(params_->ROBOT_WIDTH, params_->ROBOT_LENGTH, 0.f, 0.f, QColor("blue"));

    // RT publish-rate readout in the controls row (updated ~1 Hz from the worker's compute loop).
    rt_rate_label_ = new QLabel(QStringLiteral("RT: --"), custom_widget_);
    rt_rate_label_->setStyleSheet("QLabel { font-weight: bold; }");
    if (custom_widget_->controlsLayout != nullptr)
        custom_widget_->controlsLayout->addWidget(rt_rate_label_);

    // Self-driven 1 Hz object/obstacle BB overlay (GUI-thread timer; thread-safe, decoupled
    // from the compute loop). Objects render blue, obstacles red.
    viewer_2d_->start_semantic_bbox_overlay(graph, 1000);

    // Free-Energy time series stacked in the right-hand frame of the custom widget (was the lower
    // frame before the layout went horizontal). Keep a minimum width so the narrow right column can't
    // collapse the plots to unreadable slivers when the canvas grows / the splitter is dragged left.
    custom_widget_->frame_series->setMinimumWidth(200);
    if (custom_widget_->frame_series->layout() == nullptr)
    {
        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(2, 2, 2, 2);
        series_layout->setSpacing(2);
        custom_widget_->frame_series->setLayout(series_layout);
    }
    ts_plot_fe_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
    ts_plot_fe_->set_visible_window(60.f);
    // FE series hidden for now — it overlapped/obscured the early-exit decision variable below.
    // Re-add ts_plot_fe_->add_series("FE", QColor(255, 170, 0), 1.8f, 0); (and its add_point) to restore.
    // Early-exit decision variable: mean |SDF| at the odometry-predicted pose (meters). When it sits
    // BELOW the dashed threshold line the optimizer is skipped (prediction trusted); when it rises
    // above, Adam runs. The straight line is the base trust threshold sigma_sdf*prediction_trust_factor
    // (a small rotation-dependent boost is added at runtime, so brief crossings during turns are
    // expected). Same axis as FE — both are SDF-energy quantities in meters.
    ts_plot_fe_->add_series("pred |SDF|", QColor(0, 150, 70), 1.6f, 0);
    if (room_concept_ != nullptr)
    {
        const float thr = room_concept_->params.sigma_sdf * room_concept_->params.prediction_trust_factor;
        ts_plot_fe_->set_reference_line(thr, QColor(200, 60, 60), "opt threshold");
    }
    custom_widget_->frame_series->layout()->addWidget(ts_plot_fe_);

    // Localization confidence plot (raw, fixed 0..1 scale): -log10(det Σ_pose)/12, clamped. 1 = tightly
    // localized, 0 = uncertain. On its OWN axis so it's read directly (the old FE-scaled cov line dropped
    // with the FE level even when confidence was high — misleading; removed). Stacked directly under FE.
    ts_plot_conf_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
    ts_plot_conf_->set_visible_window(60.f);
    ts_plot_conf_->set_y_range(0.f, 1.f);
    ts_plot_conf_->add_series("confidence", QColor(0, 190, 255), 1.8f, 0);
    custom_widget_->frame_series->layout()->addWidget(ts_plot_conf_);

    // Pipeline-rate plot (Hz over time): RT-publish rate (corrected pose) and optimizer rate. Separate
    // plot because the Y units differ (Hz vs free energy). No "predicted" series — predicted poses are no
    // longer published (the optimizer output is the RT).
    ts_plot_rates_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
    ts_plot_rates_->set_visible_window(60.f);
    ts_plot_rates_->set_y_range(0.f, 22.f);   // fixed Hz scale (optimizer tops out ~20 Hz)
    ts_plot_rates_->add_series("RT publish Hz", QColor(46, 204, 113), 1.8f, 0);   // corrected-pose publishes
    ts_plot_rates_->add_series("optimizer Hz",  QColor(230, 126, 34), 1.6f, 0);   // loc-thread solve rate
    custom_widget_->frame_series->layout()->addWidget(ts_plot_rates_);

    // ── Ground truth vs estimate (SIMULATION ONLY) ────────────────────────────────────────────
    // The localiser cannot be graded on its own residual: a confidently wrong pose scores like a
    // right one (measured 2026-08-22, SDF 0.009 with the yaw 0.35 rad out). robot_concept publishes
    // the Webots supervisor pose as robot_gt_* while the producer reports simulated; this plots it
    // against what the localiser publishes, so the two are visible side by side rather than inferred.
    // ★ x/y (metres) and theta (radians) share ONE axis on purpose: in an 8x6 room they span
    // comparable ranges, and separating them would hide the thing worth seeing — whether GT and
    // estimate move TOGETHER. A constant offset between the theta pair is EXPECTED and benign: the
    // room frame's orientation is arbitrary, room_concept picks it from its own fit. Only a
    // CHANGING gap is a defect.
    ts_plot_gt_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
    ts_plot_gt_->set_visible_window(60.f);
    // GT solid/dark, estimate a CONTRASTING hue rather than a pale tint of the same one: the pale
    // version of a colour is invisible against its own partner when the two coincide, which is the
    // normal case for x and y. Distinct hues stay readable whether they overlap or diverge.
    ts_plot_gt_->add_series("gt x",      QColor(200,  30,  30), 2.0f, 0);   // deep red
    ts_plot_gt_->add_series("est x",     QColor(255, 140,   0), 1.4f, 0);   // orange
    ts_plot_gt_->add_series("gt y",      QColor(  0, 140,  60), 2.0f, 0);   // deep green
    ts_plot_gt_->add_series("est y",     QColor(190, 210,   0), 1.4f, 0);   // olive/yellow
    ts_plot_gt_->add_series("gt theta",  QColor( 30,  90, 200), 2.0f, 0);   // deep blue
    ts_plot_gt_->add_series("est theta", QColor(190,  90, 220), 1.4f, 0);   // violet
    custom_widget_->frame_series->layout()->addWidget(ts_plot_gt_);

    // Integrated odometry AT THE POINT THE PREDICTOR USES IT: the selected motion prior's delta_pose,
    // i.e. the increment the predicted pose is actually built from. Deliberately not a velocity sample
    // and not the raw stream — those are one and two steps upstream, and a prediction can be wrong
    // because the integral over the window is wrong even when every sample in it was fine (a stale
    // window bound, a clipped segment, a dropped sample). This is the last quantity before the
    // prediction, so a bad prediction can be attributed here or to the optimiser, not left ambiguous.
    // Plotted per prediction interval, in metres and radians, on ONE axis so the two stay comparable.
    // ★RANGE IS MEASURED, NOT ASSUMED. It was ±0.20 on the estimate that both channels "sit around
    // 0.08" at 0.19 s and 0.4 m/s, but the interval is shorter than that in practice, so the traces
    // used ~12% of the half-height and were unreadable. Over 95k logged samples spanning the fastest
    // tour to date (0.75 m/s peak):
    //     d|xy|     med 0.0003  p90 0.0230  p99 0.0495  max 0.0704
    //     dtheta    med 0.0001  p90 0.0180  p99 0.0356  max 0.0480
    // ±0.10 doubles the trace and still leaves 42% headroom above the all-time max, so it magnifies
    // without ever clipping — re-measure before tightening it further, because the increments scale
    // with speed and a faster route would spend that headroom.
    ts_plot_odo_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
    ts_plot_odo_->set_visible_window(60.f);
    // The odometry increments that used to live here were diagnostic scaffolding for the frame
    // work and are now uninteresting. What matters is whether the online motion calibration is
    // actually learning, and the honest signal for that is the PRECISION of each parameter, not its
    // value: a value that stops moving has either converged or simply not been asked, and only the
    // precision tells those apart. Plotted as log10(1/sigma^2) because precision spans decades as a
    // parameter goes from "unknown" to "pinned". The three curves are NOT comparable to each other
    // (yaw is rad, the scales are dimensionless) -- read each against its own trend.
    ts_plot_odo_->set_y_range(0.f, 12.f);
    ts_plot_odo_->add_series("prec fwd scale", QColor(41, 128, 185), 1.8f, 0);
    ts_plot_odo_->add_series("prec gyro scale", QColor(192, 57, 43), 1.6f, 0);
    ts_plot_odo_->add_series("prec yaw offset", QColor(241, 196, 15), 1.6f, 0);
    ts_plot_odo_->set_reference_line(0.f, QColor(170, 170, 170), "");
    custom_widget_->frame_series->layout()->addWidget(ts_plot_odo_);

    // Horizontal split: the (roughly square) room canvas on the left takes the larger share, the
    // timeseries column on the right stays narrower. On resize the canvas absorbs most of the extra
    // width (stretch 3 vs 2) so the plots keep a readable-but-compact width. Thicken the handle and
    // border both frames. Then restore the user's last splitter drag if we have one; otherwise start
    // from the ~60/40 default. saveState()/restoreState() persists the handle position across
    // sessions (saveGeometry() does not cover it).
    if (custom_widget_->splitter != nullptr)
    {
        custom_widget_->splitter->setHandleWidth(6);
        custom_widget_->splitter->setStretchFactor(0, 3);   // room canvas grows more on resize
        custom_widget_->splitter->setStretchFactor(1, 2);   // timeseries column grows less
        custom_widget_->splitter->setStyleSheet("QSplitter::handle { background-color: #5a5f64; }");
        QSettings settings(kWinSettingsOrg, kWinSettingsApp);
        const QByteArray split_state = settings.value(kSplitterStateKey).toByteArray();
        if (!split_state.isEmpty())
            custom_widget_->splitter->restoreState(split_state);
        else
            custom_widget_->splitter->setSizes({10000, 6500});   // first run → ~60/40 (canvas | series)
    }
    for (auto* f : {custom_widget_->frame, custom_widget_->frame_series})
        if (f != nullptr)
        {
            f->setFrameShape(QFrame::Box);
            f->setLineWidth(2);
        }

    if (has_room_polygon_ && room_polygon.size() >= 3)
        viewer_2d_->draw_room_polygon(room_polygon, false);

    // Camera-projection window: overlays the room layout on the live RGB image.
    camera_viz_ = std::make_unique<rc::CameraVisualizer>(graph, room_polygon, params_->OVERLAY_OBJECT_TYPES, nullptr);

    // Bring up the RGB media plane. The subscriber is created lazily by the camera
    // visualizer once the "zed" node + media descriptor exist, reading the DDS
    // domain/topic straight from that JSON descriptor (no config). This just starts
    // the always-on drain/discovery timer.
    camera_viz_->start_media_plane();
    camera_media_plane_initialized_ = true;
    qInfo() << "[room][camera] RGB media plane discovery started (waits for 'zed' descriptor)";
}

Eigen::Affine2f RoomViewer::best_available_pose(
    const std::optional<rc::RoomConcept::UpdateResult>& loc_res, bool have_loc) const
{
    if (have_loc)
        return loc_res->robot_pose;
    if (room_concept_ && room_concept_->is_initialized())
    {
        const auto s = room_concept_->get_current_state();
        Eigen::Affine2f p = Eigen::Affine2f::Identity();
        p.translation() = Eigen::Vector2f(s[2], s[3]);
        p.linear() = Eigen::Rotation2Df(s[4]).toRotationMatrix();
        return p;
    }
    return Eigen::Affine2f::Identity();
}

void RoomViewer::update_viewer(const std::optional<rc::RoomConcept::UpdateResult>& loc_res, bool have_loc,
                                       const Eigen::Affine2f& pose_for_draw,
                                       const std::vector<Eigen::Vector3f>& lidar_for_canvas,
                                       const Eigen::Affine2f& loc_pose, bool use_loc)
{
    if (!viewer_2d_)
        return;

    viewer_2d_->update_frame({
        .lidar_points     = lidar_for_canvas,
        .display_pose     = pose_for_draw,
        .covariance       = have_loc ? loc_res->covariance : Eigen::Matrix3f::Identity(),
        .max_lidar_points = params_->MAX_LIDAR_DRAW_POINTS,
        .have_loc         = have_loc,
        .is_initialized   = room_concept_ && room_concept_->is_initialized(),
        .has_room_polygon = has_room_polygon_,
        .room_width       = have_loc ? loc_res->state[0] : 0.f,
        .room_length      = have_loc ? loc_res->state[1] : 0.f,
        .loc_pose         = loc_pose,
        .use_loc_pose     = use_loc,
    });

    update_epistemic_overlay();

    if (have_loc && !loc_res->corner_matches.empty())
        viewer_2d_->draw_corners(loc_res->corner_matches, pose_for_draw);
    else
        viewer_2d_->draw_corners({}, pose_for_draw);

    // Object-anchor overlay (fridge, …): pinned p_o, this frame's z_o, the sight line and the residual
    // between them. Display-only — a copy taken under the localizer's lock, drawn without holding it.
    if (room_concept_)
        viewer_2d_->draw_object_anchors(room_concept_->object_anchors(), pose_for_draw,
                                        room_concept_->object_landmarks());
    else
        viewer_2d_->draw_object_anchors({}, pose_for_draw);

    // Feed the same matched corners to the RGB camera overlay (translucent uncertainty circles).
    if (camera_viz_)
        camera_viz_->set_corner_matches(have_loc ? loc_res->corner_matches
                                                 : std::vector<rc::CornerDetector::CornerMatch>{});
}

void RoomViewer::update_epistemic_overlay()
{
    if (!viewer_2d_ || !epistemic_)
        return;

    // Score-grid / IoR-grid cell colouring of the room space is intentionally NOT drawn:
    // the room space is now conveyed by robot→corner and robot→landmark sight lines instead.
    const auto& planner = epistemic_->epistemic_planner();

    const auto& current_target = planner.current_target();
    if (current_target.has_value() && !current_target->rotate_in_place)
    {
        viewer_2d_->draw_selected_grid_cell(current_target->position, planner.cell_size());
        viewer_2d_->update_target_marker(current_target->position.x(),
                                         current_target->position.y(),
                                         true);
    }
    else
    {
        viewer_2d_->draw_selected_grid_cell(std::nullopt, planner.cell_size());
        viewer_2d_->update_target_marker(0.f, 0.f, false);
    }
}

void RoomViewer::draw_landmarks(const std::vector<Eigen::Vector2f>& landmarks_world,
                                const std::vector<char>& measured,
                                const Eigen::Affine2f& robot_pose)
{
    if (!viewer_2d_)
        return;
    viewer_2d_->draw_landmark_lines(landmarks_world, measured, robot_pose.translation());
}

void RoomViewer::add_rate_samples(float corr_hz, float opt_hz)
{
    if (ts_plot_rates_)
    {
        ts_plot_rates_->add_point("RT publish Hz", corr_hz);
        ts_plot_rates_->add_point("optimizer Hz",  opt_hz);
    }
}

void RoomViewer::set_rt_rate_text(const QString& text)
{
    if (rt_rate_label_)
        rt_rate_label_->setText(text);
}

void RoomViewer::set_room_stable(bool stable, int stable_frames, int frames_required, bool searching)
{
    if (custom_widget_ == nullptr or custom_widget_->lbl_room_stable == nullptr)
        return;

    auto* lbl = custom_widget_->lbl_room_stable;

    // SEARCHING outranks both other states: while a global grid search runs the pose is being
    // relocated wholesale, so "ROOM STABLE" would be actively misleading — the room node still
    // exists, but the robot's place in it is exactly what is currently in question.
    const int state = searching ? 2 : (stable ? 1 : 0);

    const QString text = searching ? QStringLiteral("SEARCHING…")
                       : stable    ? QStringLiteral("ROOM STABLE")
                                   : QStringLiteral("STABILIZING %1/%2")
                                         .arg(stable_frames).arg(std::max(1, frames_required));

    // Repainting the stylesheet every frame forces a full re-parse + relayout in Qt, so only touch the
    // widget when the displayed state actually changes.
    if (lbl->text() == text and room_stable_shown_ == state)
        return;
    lbl->setText(text);

    if (room_stable_shown_ != state)
    {
        // amber (searching) / green (stable) / red (stabilizing)
        static constexpr const char* kStyle =
            "QLabel { background-color: %1; color: white; border: 1px solid %2;"
            " border-radius: 4px; font-weight: bold; }";
        const QString bg     = state == 2 ? "#c9791a" : state == 1 ? "#1e8b3a" : "#b02020";
        const QString border = state == 2 ? "#8c520f" : state == 1 ? "#145c26" : "#7a1616";
        lbl->setStyleSheet(QString(kStyle).arg(bg, border));
        room_stable_shown_ = state;
    }
}

void RoomViewer::update_ui(const std::optional<rc::RoomConcept::UpdateResult>& loc_res)
{
    if (!loc_res.has_value() || !ts_plot_fe_)
        return;
    // FE series hidden for now (see constructor) — no add_point("FE", ...) while it's not registered.

    // Early-exit decision variable (predicted-pose mean |SDF|). NaN on frames where the gate never
    // ran (warmup / no odometry) — skip those so the line doesn't spike to a garbage sample.
    if (std::isfinite(loc_res->early_exit_metric))
        ts_plot_fe_->add_point("pred |SDF|", loc_res->early_exit_metric);

    // Localization confidence from the pose covariance determinant: small det (well-localized) → high.
    // det ~ 1e-8..1e-10 well-localized, ~1e-4 uncertain → -log10(det) ~ 4..10, mapped to [0,1] by /12.
    // Plotted raw on its own fixed 0..1 axis (ts_plot_conf_) — 1 = tight, 0 = uncertain.
    const float det_cov = std::max(1e-12f, std::abs(loc_res->covariance.determinant()));
    const float conf = std::clamp(-std::log10(det_cov) / 12.f, 0.f, 1.f);
    if (ts_plot_conf_)
        ts_plot_conf_->add_point("confidence", conf);

    // Ground truth beside the estimate. Absent attributes (real robot) => nothing plotted.
    if (ts_plot_gt_ and graph_)
    {
        if (const auto robots = graph_->get_nodes_by_type("robot"); not robots.empty())
        {
            const auto &rn = robots.front();
            const auto gx = graph_->get_attrib_by_name<robot_gt_x_att>(rn);
            const auto gy = graph_->get_attrib_by_name<robot_gt_y_att>(rn);
            const auto ga = graph_->get_attrib_by_name<robot_gt_angle_att>(rn);
            if (gx.has_value() and gy.has_value() and ga.has_value())
            {
                const auto &pose = loc_res->robot_pose;
                ts_plot_gt_->add_point("gt x",      gx.value());
                ts_plot_gt_->add_point("est x",     pose.translation().x());
                ts_plot_gt_->add_point("gt y",      gy.value());
                ts_plot_gt_->add_point("est y",     pose.translation().y());
                ts_plot_gt_->add_point("gt theta",  ga.value());
                ts_plot_gt_->add_point("est theta",
                                       std::atan2(pose.linear()(1, 0), pose.linear()(0, 0)));
            }
        }
    }

    if (not calib_viewer_.isNull())
    {
        Eigen::Matrix<float, rc::calib::P_COUNT, 1> v, sg;
        v  << loc_res->calib_k_v - 1.f, loc_res->calib_yaw,
              loc_res->calib_k_w - 1.f, loc_res->calib_b_omega;
        sg << loc_res->calib_sigma_k_v, loc_res->calib_sigma_yaw,
              loc_res->calib_sigma_k_w, loc_res->calib_sigma_b_omega;
        calib_viewer_->update_values(v, sg, loc_res->calib_informed,
                                     loc_res->calib_condition, loc_res->calib_episodes);
    }

    if (ts_plot_odo_ != nullptr)   // loc_res is guaranteed by the early return above
    {
        // log10 precision. sigma == 0 means the calibrator has not been configured (feature off),
        // which must read as "no information" rather than as infinite confidence.
        const auto log_prec = [](float sigma) -> float
        { return sigma > 1e-9f ? std::log10(1.f / (sigma * sigma)) : 0.f; };
        ts_plot_odo_->add_point("prec fwd scale",  log_prec(loc_res->calib_sigma_k_v));
        ts_plot_odo_->add_point("prec gyro scale", log_prec(loc_res->calib_sigma_k_w));
        ts_plot_odo_->add_point("prec yaw offset", log_prec(loc_res->calib_sigma_yaw));
    }
}

void RoomViewer::show_calibration()
{
    if (calib_viewer_.isNull())
        calib_viewer_ = new rc::CalibrationViewer(custom_widget_);
    calib_viewer_->show();
    calib_viewer_->raise();
    calib_viewer_->activateWindow();
}

void RoomViewer::show_camera()
{
    if (camera_viz_)
    {
        camera_viz_->update_frame();
        camera_viz_->show();
        camera_viz_->raise();
        camera_viz_->activateWindow();
    }
}

void RoomViewer::toggle_lidar_points(bool checked)
{
    if (viewer_2d_)
        viewer_2d_->set_lidar_points_visible(checked);
}

void RoomViewer::on_robot_moved(QPointF scene_pos)
{
    // Shift+Left: move robot to clicked position, keep current heading. push_command
    // (thread-safe queue) — never set_robot_pose() directly from the GUI thread while
    // the localization thread may be mid-backward().
    if (!room_concept_)
        return;
    const auto state = room_concept_->get_current_state();
    room_concept_->push_command(rc::RoomConcept::CmdSetPose{
        static_cast<float>(scene_pos.x()), static_cast<float>(scene_pos.y()), state[4]});
}

void RoomViewer::on_robot_rotated(QPointF scene_pos)
{
    // Ctrl+Left: rotate robot to face the clicked point, keep current position.
    if (!room_concept_)
        return;
    const auto state = room_concept_->get_current_state();
    const float rx = state[2];
    const float ry = state[3];
    const float theta = std::atan2(static_cast<float>(scene_pos.y()) - ry,
                                   static_cast<float>(scene_pos.x()) - rx);
    room_concept_->push_command(rc::RoomConcept::CmdSetPose{rx, ry, theta});
}

}  // namespace rc
