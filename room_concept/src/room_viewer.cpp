/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_viewer.h.
 */

#include "room_viewer.h"

#include <algorithm>
#include <cmath>
#include <numbers>

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

    // ── THE LOCALIZATION METRIC ──────────────────────────────────────────────────────────────────
    // What the estimator is GRADED on, rather than one of its inputs. All three series are mm per
    // metre travelled: normalised by motion because a parked robot predicts nothing and would score
    // perfectly for standing still, and relative (not |est - gt|) because ground truth is the world
    // frame while the estimate is the room frame, whose orientation room_concept picks for itself —
    // an absolute difference measures that arbitrary choice, not the localiser. See
    // localization_drift.h for the two heading conventions this has to undo, one of which cost a
    // 27x wrong number when it was missed.
    ts_plot_loc_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
    ts_plot_loc_->set_visible_window(60.f);
    ts_plot_loc_->add_series("pred drift mm/m", QColor(0, 190, 255), 1.8f, 5);
    ts_plot_loc_->add_series("pose drift mm/m", QColor(46, 204, 113), 1.6f, 5);
    ts_plot_loc_->add_series("correction mm/m", QColor(230, 126, 34), 1.4f, 5);
    custom_widget_->frame_series->layout()->addWidget(ts_plot_loc_);
    // ── What each legend entry MEANS, on hover ───────────────────────────────────────────────────
    // A series name is a label, not an explanation. None of these say what units they are in, which
    // direction is good, or what a reader should do about a value — and the plots are read by people
    // who did not write the estimator.
    ts_plot_fe_->set_series_tooltip("pred |SDF|", QStringLiteral(
        "Mean |signed distance| of this sweep's laser points, placed at the PREDICTED pose, in metres.\n"
        "If the prediction is right the points lie on walls and this is near zero.\n\n"
        "The dashed line is the trust threshold: below it the optimizer is SKIPPED and the prediction\n"
        "is published as-is — which is the large majority of cycles. Above it the map argues with the\n"
        "prediction inside Gauss-Newton.\n\n"
        "The threshold widens while turning (0.2 m per radian) because a heading error pivots the whole\n"
        "scan: 0.02 rad at 5 m already displaces the points 10 cm on a prediction that is perfectly good."));
    ts_plot_conf_->set_series_tooltip("confidence", QStringLiteral(
        "Localisation confidence, 0..1, raw and unsmoothed.\n\n"
        "Read it as a trend, not a value: what matters is whether it is recovering or decaying, and\n"
        "how it moves when the robot turns or enters a corridor. A high number is not a guarantee —\n"
        "the localiser has been measured jumping several metres while reporting a tight sigma."));
    ts_plot_rates_->set_series_tooltip("RT publish Hz", QStringLiteral(
        "How often a CORRECTED pose is published to the graph, in Hz.\n\n"
        "This is the rate consumers actually see. It should track the laser sweep rate; a drop means\n"
        "the localiser is not keeping up, not that the robot stopped."));
    ts_plot_rates_->set_series_tooltip("optimizer Hz", QStringLiteral(
        "How often Gauss-Newton actually RUNS, in Hz — not how often a pose is published.\n\n"
        "Normally a small fraction of the publish rate, because the prediction usually passes the SDF\n"
        "check and the optimizer is skipped. That gap is the point: between firings the pose runs\n"
        "open-loop on dead reckoning, and those are the only cycles where an accumulating channel\n"
        "error is visible before a correction wipes it.\n\n"
        "A SUSTAINED rise means the prediction has stopped being good enough — either the motion model\n"
        "drifted or the prior loosened."));
    ts_plot_loc_->set_series_tooltip("pred drift mm/m", QStringLiteral(
        "LOCALIZATION ERROR OF THE MOTION MODEL, in millimetres per metre travelled.\n\n"
        "Over each metre of true travel, how far the PREDICTED motion departs from the true motion.\n"
        "This is the channel a wrong odometry scale or gyro scale shows up in first, and the one the\n"
        "self-calibration is trying to reduce.\n\n"
        "Relative, not absolute: ground truth is the world frame and the estimate is the room frame,\n"
        "whose orientation this agent picks from its own fit. The offset between them is arbitrary and\n"
        "carries no information, so only the motion is compared.\n\n"
        "Simulation only — it needs robot_gt_*, which exists only while the producer reports simulated."));
    ts_plot_loc_->set_series_tooltip("pose drift mm/m", QStringLiteral(
        "The same measure applied to the PUBLISHED pose — what every consumer of the RT tree sees.\n\n"
        "Measured 2026-08-29: on the 96.2% of cycles that early-exit, the optimizer does not run and\n"
        "the correction is EXACTLY zero, so the published pose IS the raw prediction and this line sits\n"
        "on top of the blue one. Them SEPARATING is the news: it means the optimizer started working.\n\n"
        "If this stays flat while pred drift rises, the model got worse and the optimizer is paying for\n"
        "it — look at the orange line, which is where that cost appears."));
    ts_plot_loc_->set_series_tooltip("correction mm/m", QStringLiteral(
        "How far the optimizer had to MOVE the pose, per metre travelled: |published - predicted|.\n\n"
        "The effort channel. An uncalibrated robot need not localise visibly worse — it can localise\n"
        "just as well by working harder — and this is where that work becomes visible. Error either\n"
        "reaches the output or shows up here; the two together are conserved.\n\n"
        "Near zero while the early-exit gate holds, because on those cycles nothing is corrected."));


    // ── The calibration window exists from startup, hidden ───────────────────────────────────────
    // It used to be constructed on the first press of the Calib button, so its traces began at that
    // moment and the six parameters looked as though they started learning when you opened it. They
    // have been learning since the first cycle: motion_calib_.observe() runs in the update path and
    // knows nothing about this window. Building it here costs one hidden QDialog and makes the trace
    // an honest record of the whole run rather than of how long you have been watching.
    calib_viewer_ = new rc::CalibrationViewer(custom_widget_);
    calib_viewer_->hide();
    // ★ EVERY CAMERA GETS ITS COLUMN NOW, fed or not. Declaring the list here rather than letting a
    // column appear when the first solve arrives is what lets the window show "no evidence" for a
    // camera that has never produced a pair: a column that does not exist cannot say anything, and
    // an absent column and a silent one would be indistinguishable. The driving camera goes first.
    {
        std::vector<std::string> cams{params_->IMAGE_EDGE_CAMERA};
        for (const auto& c : params_->CALIB_CAMERAS)
            if (c != params_->IMAGE_EDGE_CAMERA) cams.push_back(c);
        calib_viewer_->set_cameras(cams, params_->IMAGE_EDGE_CAMERA);
    }
    // The window asks; the localiser thread acts. RoomConcept queues it rather than touching the
    // estimator from the GUI thread.
    if (room_concept_ != nullptr)
        calib_viewer_->set_reset_handler([this]
        {
            room_concept_->request_calibration_reset();
            if (on_camera_reset_) on_camera_reset_();   // both blocks, or neither
        });

    // ── Ground truth vs estimate (SIMULATION ONLY) ────────────────────────────────────────────
    // The localiser cannot be graded on its own residual: a confidently wrong pose scores like a
    // right one (measured 2026-08-22, SDF 0.009 with the yaw 0.35 rad out). robot_concept publishes
    // the Webots supervisor pose as robot_gt_* while the producer reports simulated; this plots it
    // The odometry-increment panel that used to sit here is gone: it was scaffolding for the frame
    // work, and the calibration parameters that replaced it now live in their own window behind the
    // Calib button, where each one gets a full-width trace, its uncertainty and an "is this being
    // taught" lamp. Four cramped traces sharing one strip could show none of that.

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
    // ★ The camera node is passed EXPLICITLY. It has a default of "zed", but the old call passed
    //   `nullptr` in the position the node name now occupies — and nullptr converts to std::string
    //   through the const char* constructor, so it compiles cleanly and is undefined behaviour at
    //   runtime. A defaulted std::string parameter added ahead of a pointer parameter is a trap;
    //   naming both arguments removes it.
    camera_viz_ = std::make_unique<rc::CameraVisualizer>(
        graph, room_polygon, params_->OVERLAY_OBJECT_TYPES, std::string("zed"), nullptr);
    ricoh_viz_ = std::make_unique<rc::CameraVisualizer>(
        graph, room_polygon, params_->OVERLAY_OBJECT_TYPES, std::string("ricoh"), nullptr);

    // Bring up the RGB media plane. The subscriber is created lazily by the camera
    // visualizer once the "zed" node + media descriptor exist, reading the DDS
    // domain/topic straight from that JSON descriptor (no config). This just starts
    // the always-on drain/discovery timer.
    camera_viz_->start_media_plane();
    // ★ THE SECOND WINDOW NEEDS THIS TOO. Without it the ricoh visualiser is constructed, shows,
    //   and reports "Media plane RGB subscriber not initialized" for ever — because the ingest
    //   thread that performs discovery is never started, so discovery is never even attempted.
    //   The failure is indistinguishable from a producer that is not publishing, which is what made
    //   it look like a DDS problem: the diagnostics added to try_discover_media_plane() could not
    //   fire, since nothing called it.
    ricoh_viz_->start_media_plane();
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

    // RGB triple points beside them. Read through triple_points(), NOT image_edges(): that holder is
    // emptied by take_image_edges() the moment a slot consumes it, so peeking there would draw
    // corners only on the tick the viewer happened to win the race against the solver.
    if (room_concept_)
        viewer_2d_->draw_rgb_corners(room_concept_->triple_points(), pose_for_draw);

    // Object-anchor overlay (fridge, …): pinned p_o, this frame's z_o, the sight line and the residual
    // between them. Display-only — a copy taken under the localizer's lock, drawn without holding it.
    if (room_concept_)
        viewer_2d_->draw_object_anchors(room_concept_->object_anchors(), pose_for_draw,
                                        room_concept_->object_landmarks());
    else
        viewer_2d_->draw_object_anchors({}, pose_for_draw);

    // Feed the same matched corners to the RGB camera overlay (translucent uncertainty circles).
    const auto ms = have_loc ? loc_res->corner_matches
                             : std::vector<rc::CornerDetector::CornerMatch>{};
    if (camera_viz_) camera_viz_->set_corner_matches(ms, pose_for_draw);
    if (ricoh_viz_)  ricoh_viz_->set_corner_matches(ms, pose_for_draw);
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
    // RGB projection agreement, appended only when a NEW cycle produced one. Without the stamp test
    // a stalled camera would draw a flat line at its last value, which reads as "steady" rather than
    // "stopped" — the two must not look alike.

    // Localization confidence from the pose covariance determinant: small det (well-localized) → high.
    // det ~ 1e-8..1e-10 well-localized, ~1e-4 uncertain → -log10(det) ~ 4..10, mapped to [0,1] by /12.
    // Plotted raw on its own fixed 0..1 axis (ts_plot_conf_) — 1 = tight, 0 = uncertain.
    const float det_cov = std::max(1e-12f, std::abs(loc_res->covariance.determinant()));
    const float conf = std::clamp(-std::log10(det_cov) / 12.f, 0.f, 1.f);
    if (ts_plot_conf_)
        ts_plot_conf_->add_point("confidence", conf);

    // ── FEED THE LOCALIZATION METRIC ─────────────────────────────────────────────────────────────
    // ⚠ This replaced a block that fed ts_plot_gt_, a plot that was DECLARED and USED but never
    // CONSTRUCTED — so the pointer was always null and none of it ever ran. Ground truth has not
    // been on screen at all. (Found 2026-08-30.)
    //
    // Absent attributes (real robot) => nothing plotted, which is the honest appearance of a
    // measurement that needs a simulator.
    if (ts_plot_loc_ and graph_)
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
                const double est_th = std::atan2(pose.linear()(1, 0), pose.linear()(0, 0));
                // The raw published values go in; localization_drift.h owns the sign flip on
                // robot_gt_angle and the +90 deg on the room-frame heading, so no caller can apply
                // either of them twice.
                const auto dp = drift_pred_.push(gx.value(), gy.value(), ga.value(),
                                                 loc_res->pred_x, loc_res->pred_y, loc_res->pred_theta);
                const auto de = drift_pose_.push(gx.value(), gy.value(), ga.value(),
                                                 pose.translation().x(), pose.translation().y(), est_th);
                if (dp.has_trans) ts_plot_loc_->add_point("pred drift mm/m", static_cast<float>(dp.mm_per_m));
                if (de.has_trans) ts_plot_loc_->add_point("pose drift mm/m", static_cast<float>(de.mm_per_m));

                // Correction effort, on the same 1 m yardstick so it is directly comparable with
                // the two drift series rather than being a differently-scaled third thing.
                corr_sum_m_ += std::hypot(pose.translation().x() - loc_res->pred_x,
                                          pose.translation().y() - loc_res->pred_y);
                if (have_last_gt_)
                    corr_dist_m_ += std::hypot(gx.value() - last_gt_x_, gy.value() - last_gt_y_);
                last_gt_x_ = gx.value(); last_gt_y_ = gy.value(); have_last_gt_ = true;
                if (corr_dist_m_ >= 1.0)
                {
                    ts_plot_loc_->add_point("correction mm/m",
                                            static_cast<float>(corr_sum_m_ / corr_dist_m_ * 1000.0));
                    corr_sum_m_ = corr_dist_m_ = 0.0;
                }

                // Say it ONCE, loudly, and stop plotting — a metric quietly measuring the wrong
                // thing is worse than a gap in the plot, because only one of the two looks wrong.
                if (drift_pred_.suspect() and not drift_suspect_logged_)
                {
                    drift_suspect_logged_ = true;
                    qCritical() << "[loc-metric] DISABLED:" << drift_pred_.suspect_reason()
                                << "| measured mean(course-theta): gt"
                                << drift_pred_.measured_gt_offset() * 180.0 / std::numbers::pi << "deg, est"
                                << drift_pred_.measured_est_offset() * 180.0 / std::numbers::pi
                                << "deg (both should be ~0 AFTER the declared corrections). Fix the "
                                   "convention in localization_drift.h; do not widen the tolerance.";
                }
            }
        }
    }

    if (not calib_viewer_.isNull())
    {
        // The whole vector, straight from the solve. Copying parameter by parameter into
        // UpdateResult fields was already awkward at four and would not survive six -- and every
        // hand-copied field is a chance to pass a literal 0 for a sigma, which once rendered the one
        // parameter we had never measured as the most certain thing on the screen.
        calib_viewer_->update_values(loc_res->calib_value, loc_res->calib_sigma,
                                     loc_res->calib_informed,
                                     loc_res->calib_condition, loc_res->calib_episodes);
    }

}

void RoomViewer::show_calibration()
{
    // Constructed at startup, not here — see the constructor. This only raises it.
    if (calib_viewer_.isNull())
        calib_viewer_ = new rc::CalibrationViewer(custom_widget_);
    calib_viewer_->show();
    calib_viewer_->raise();
    calib_viewer_->activateWindow();
}

void RoomViewer::show_ricoh()
{
    if (ricoh_viz_)
    {
        ricoh_viz_->update_frame();
        ricoh_viz_->show();
        ricoh_viz_->raise();
        ricoh_viz_->activateWindow();
    }
}

void RoomViewer::set_triple_points(std::vector<rc::TriplePoint> pts, const std::string& from_camera)
{
    if (camera_viz_) camera_viz_->set_triple_points(pts, from_camera);
    if (ricoh_viz_)  ricoh_viz_->set_triple_points(std::move(pts), from_camera);
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


void RoomViewer::set_camera_calibration(const Eigen::Matrix<float, rc::camcal::P_COUNT, 1>& value,
                                        const Eigen::Matrix<float, rc::camcal::P_COUNT, 1>& sigma,
                                        int informed_mask, float condition, long pairs,
                                        const std::string& camera)
{
    if (calib_viewer_)
        calib_viewer_->update_camera(value, sigma, informed_mask, condition, pairs, camera);
}

void RoomViewer::set_loop_closure(double du_deg, double dv_deg, double sd_du, double sd_dv, long n)
{
    if (calib_viewer_) calib_viewer_->update_loop_closure(du_deg, dv_deg, sd_du, sd_dv, n);
}

}  // namespace rc
