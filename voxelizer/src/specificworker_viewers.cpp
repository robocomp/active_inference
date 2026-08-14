/*
 * specificworker_viewers.cpp — custom drawing-window setup (Voxel3D GL + YOLO raster).
 *
 * Split out of initialize() to keep specificworker.cpp lean. The widgets attach to
 * the DSR GUI (when present) in their OWN top-level windows, never docked into the
 * graph-viewer window — a GL surface compositing under the graph view's churn-driven
 * repaints corrupts the backing store and crashes the process.
 */

#include "specificworker.h"
#include "perception_worker.h"
#include "semantic_stage.h"
#include "sam2_stage.h"

#include <QCoreApplication>
#include <cmath>
#include <print>

#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QMainWindow>
#include <QSettings>
#include <QByteArray>

#include "yolo_viewer.h"
#include "image_popup_viewer.h"
#include "scene_processor.h"
#include "yolo_semantic.h"
#include "graph_publisher.h"

namespace
{
// Persist the external (own-window) viewers' geometry under the same QSettings group the
// VoxelOpenGLViewer uses for its camera state, so position/size survive restarts.
constexpr auto kWinSettingsOrg   = "RoboComp";
constexpr auto kWinSettingsApp   = "robot_concept";
constexpr auto kWinSettingsGroup = "VoxelOpenGLViewer";

bool restore_external_window_geometry(QWidget* win, const QString& key)
{
    if (win == nullptr)
        return false;
    QSettings settings(kWinSettingsOrg, kWinSettingsApp);
    settings.beginGroup(kWinSettingsGroup);
    const QByteArray geom = settings.value(key + "_geometry").toByteArray();
    settings.endGroup();
    return not geom.isEmpty() and win->restoreGeometry(geom);
}
} // namespace

void SpecificWorker::save_external_window_geometry() const
{
    QSettings settings(kWinSettingsOrg, kWinSettingsApp);
    settings.beginGroup(kWinSettingsGroup);
    if (voxel3d_window_ != nullptr)
        settings.setValue("Voxel3DWindow_geometry", voxel3d_window_->saveGeometry());
    if (yolo_window_ != nullptr)
        settings.setValue("YOLOWindow_geometry", yolo_window_->saveGeometry());
    if (ricoh_window_ != nullptr)
        settings.setValue("Ricoh360Window_geometry", ricoh_window_->saveGeometry());
    settings.endGroup();
}

void SpecificWorker::setup_custom_viewers()
{
    // NOTE: these are shown as standalone top-level windows, independent of the DSR graph viewer.
    // The DSR node-link graph view is disabled (Agent.graph=false) because it crashes in
    // paintAndFlush under participant churn (e.g. bottle_concept joining). All consumers are null-guarded.

    // Ricoh 360 panorama popup: a raster window, created HIDDEN and toggled by a
    // button in the Voxel3D top bar (below). Kept a plain top-level label window
    // like the no-controls YOLO case; frames are pushed from on_render_tick.
    if (params.SHOW_RICOH_VIEWER)
    {
        ricoh_viewer_ = std::make_unique<rc::ImagePopupViewer>(nullptr);

        // Wrap the raster label in a panel with a control row so the panorama gets its own "Models"
        // toggle (equirectangular projection of the DSR scene), mirroring the ZED popup.
        auto* ricoh_panel = new QWidget(nullptr);
        auto* ricoh_layout = new QVBoxLayout(ricoh_panel);
        ricoh_layout->setContentsMargins(6, 6, 6, 6);
        ricoh_layout->setSpacing(6);

        auto* controls = new QHBoxLayout();
        controls->setContentsMargins(0, 0, 0, 0);
        controls->setSpacing(8);

        auto* models_btn = new QPushButton(ricoh_model_overlay_enabled_ ? "Models: ON" : "Models: OFF", ricoh_panel);
        models_btn->setCheckable(true);
        models_btn->setChecked(ricoh_model_overlay_enabled_);
        models_btn->setCursor(Qt::PointingHandCursor);
        models_btn->setStyleSheet(QString(
            "QPushButton { border: 2px solid %1; border-radius: 4px; padding: 3px 8px; }"
            "QPushButton:checked { background-color: %1; color: #101010; }").arg("#FFC864"));
        connect(models_btn, &QPushButton::toggled, this, [this, models_btn](bool checked)
        {
            ricoh_model_overlay_enabled_ = checked;
            // Invalidate only when NO consumer is left — RoomΔ reads this same cache (compute()).
            if (not checked and not ricoh_lidar_overlay_enabled_ and ricoh_room_mode_ == 0)
                ricoh_scene_.valid = false;
            models_btn->setText(checked ? "Models: ON" : "Models: OFF");
        });

        // Lidar reprojection overlay: lidar points projected into the panorama, coloured by range
        // (sparse depth / calibration check). Starts OFF.
        auto* lidar_btn = new QPushButton(ricoh_lidar_overlay_enabled_ ? "Lidar: ON" : "Lidar: OFF", ricoh_panel);
        lidar_btn->setCheckable(true);
        lidar_btn->setChecked(ricoh_lidar_overlay_enabled_);
        lidar_btn->setCursor(Qt::PointingHandCursor);
        lidar_btn->setStyleSheet(QString(
            "QPushButton { border: 2px solid %1; border-radius: 4px; padding: 3px 8px; }"
            "QPushButton:checked { background-color: %1; color: #101010; }").arg("#8C9EC7"));
        connect(lidar_btn, &QPushButton::toggled, this, [this, lidar_btn](bool checked)
        {
            ricoh_lidar_overlay_enabled_ = checked;
            if (not checked and not ricoh_model_overlay_enabled_ and ricoh_room_mode_ == 0)
                ricoh_scene_.valid = false;
            lidar_btn->setText(checked ? "Lidar: ON" : "Lidar: OFF");
        });

        // Monocular-depth ramp (yolo26l-depth, [RicohDepth]). The toggle also ENABLES THE STAGE, so the
        // model does no work while the overlay is off — same contract as the ZED semantic toggle.
        // ★Gated on the CONFIG FLAG, not on ricoh_worker_->stage("depth"): setup_custom_viewers() runs
        // at specificworker.cpp:267, a hundred lines BEFORE the ricoh worker is constructed, so probing
        // the worker here would always see nullptr and the button would silently never exist. The
        // lambda resolves the stage lazily instead — by the time anyone can click, the worker is up.
        QPushButton* depth_btn = nullptr;
        if (params.RICOH_DEPTH_ENABLED)
        {
            depth_btn = new QPushButton(ricoh_depth_overlay_enabled_ ? "Depth: ON" : "Depth: OFF", ricoh_panel);
            depth_btn->setCheckable(true);
            depth_btn->setChecked(ricoh_depth_overlay_enabled_);
            depth_btn->setCursor(Qt::PointingHandCursor);
            depth_btn->setToolTip("Monocular depth per 120° strip — RELATIVE within a strip, not comparable "
                                  "across the white seam lines (each strip has its own scale).");
            depth_btn->setStyleSheet(QString(
                "QPushButton { border: 2px solid %1; border-radius: 4px; padding: 3px 8px; }"
                "QPushButton:checked { background-color: %1; color: #101010; }").arg("#7ED0C0"));
            connect(depth_btn, &QPushButton::toggled, this, [this, depth_btn](bool checked)
            {
                ricoh_depth_overlay_enabled_ = checked;
                sync_ricoh_depth_stage();   // stage runs if EITHER overlay or collection wants it
                depth_btn->setText(checked ? "Depth: ON"
                                   : (ricoh_depth_collect_enabled_ ? "Depth: (auto)" : "Depth: OFF"));
            });
        }

        // Dataset collection + map rebuild. Both live next to the Depth toggle because they only mean
        // anything while depth is running: collection needs the model's output, and the map corrects it.
        QPushButton* collect_btn = nullptr;
        QPushButton* rebuild_btn = nullptr;
        if (params.RICOH_DEPTH_ENABLED)
        {
            collect_btn = new QPushButton("Collect: OFF", ricoh_panel);
            collect_btn->setCheckable(true);
            collect_btn->setCursor(Qt::PointingHandCursor);
            collect_btn->setToolTip("Append LiDAR-anchored depth samples to etc/ricoh_depth_dataset.csv.\n"
                                    "Frames are kept only when the robot has MOVED (>10 cm or >5°), so drive "
                                    "it around — standing still adds nothing.");
            collect_btn->setStyleSheet(QString(
                "QPushButton { border: 2px solid %1; border-radius: 4px; padding: 3px 8px; }"
                "QPushButton:checked { background-color: %1; color: #101010; }").arg("#E8B84B"));
            connect(collect_btn, &QPushButton::toggled, this, [this, collect_btn, depth_btn](bool checked)
            {
                // Arming Collect ENABLES THE DEPTH MODEL by itself — a dataset sample is (what the
                // model predicted, what the LiDAR measured), so with no depth map there is nothing to
                // pair and collection would silently do nothing. Show it on the Depth button too, so
                // the UI cannot report a state the system is not in.
                arm_depth_collection(checked);
                collect_btn->setText(checked ? "Collect: REC" : "Collect: OFF");
                if (depth_btn != nullptr)
                    depth_btn->setText(ricoh_depth_overlay_enabled_ ? "Depth: ON"
                                       : (checked ? "Depth: (auto)" : "Depth: OFF"));
            });

            rebuild_btn = new QPushButton("Rebuild map", ricoh_panel);
            rebuild_btn->setCursor(Qt::PointingHandCursor);
            rebuild_btn->setToolTip("Reload the whole dataset, drop duplicate poses, refit the correction "
                                    "map, save it to etc/ricoh_depth_map.csv and apply it live.");
            rebuild_btn->setStyleSheet(
                "QPushButton { border: 2px solid #B08CD9; border-radius: 4px; padding: 3px 8px; }"
                "QPushButton:pressed { background-color: #B08CD9; color: #101010; }");
            connect(rebuild_btn, &QPushButton::clicked, this, [this, rebuild_btn]
            {
                // Refitting reads the entire CSV and can take a moment on a big set; say so rather
                // than letting the UI look wedged.
                rebuild_btn->setEnabled(false);
                rebuild_btn->setText("Fitting…");
                QCoreApplication::processEvents();
                const std::string err = rebuild_depth_fit_map();
                // ★On failure show the ERROR, never depth_fit_map_ — that still holds the map loaded
                // at startup, so printing it would report a stale fit as if it were the new one.
                if (not err.empty())
                {
                    rebuild_btn->setText("FAILED: " + QString::fromStdString(err));
                    rebuild_btn->setStyleSheet(
                        "QPushButton { border: 2px solid #E05252; border-radius: 4px; padding: 3px 8px; }");
                }
                else
                {
                    // ★Show the TYPICAL ERROR, not `a`. `a` is a shape parameter of the fit and says
                    // nothing about how wrong a metre reading is; median |d−d*|/d* does, and δ1 says
                    // how often it is in the right ballpark. The log line carries the tail.
                    rebuild_btn->setText(QString("Map: err %1% (%2 m) · δ1 %3%")
                                             .arg(100.0 * depth_fit_map_.med_rel,   0, 'f', 0)
                                             .arg(depth_fit_map_.med_abs_m,          0, 'f', 2)
                                             .arg(100.0 * depth_fit_map_.delta125,  0, 'f', 0));
                    rebuild_btn->setStyleSheet(
                        "QPushButton { border: 2px solid #B08CD9; border-radius: 4px; padding: 3px 8px; }"
                        "QPushButton:pressed { background-color: #B08CD9; color: #101010; }");
                }
                rebuild_btn->setEnabled(true);
            });
        }

        // ── Offline ENRICHMENT (rc::depth::DatasetEnricher) ─────────────────────────────────────
        // Same workflow slot as "Rebuild map", but the map is refitted against the ROOM BELIEF as well
        // as the LiDAR: every saved panorama is re-run through the depth model AND a semantic
        // segmenter, the room envelope is ray-cast through it, and the ceiling/floor/wall pixels
        // become synthetic samples where helios never reaches. MINUTES — the pass runs on its own
        // thread with its own ONNX sessions and this lambda only pumps the event loop, so the window
        // stays responsive and the button reports frames done.
        QPushButton* enrich_btn = nullptr;
        if (params.RICOH_DEPTH_ENABLED and params.SEMANTIC_SEG_ENABLED)
        {
            enrich_btn = new QPushButton("Enrich map", ricoh_panel);
            enrich_btn->setCursor(Qt::PointingHandCursor);
            enrich_btn->setToolTip(
                "Add room-belief supervision to the correction map (MINUTES — two networks over every "
                "saved panorama).\n"
                "The LiDAR only ever anchors a horizon stripe out to ~10 m, so the ceiling half of the\n"
                "band has NO samples and the ct*t² term had to be deleted. This re-runs yolo26l-depth +\n"
                "yolo26l-sem on etc/depth_frames/, ray-casts the room envelope, and emits synthetic\n"
                "samples on the shell — weighted by the belief's own σ, the ray's incidence angle and\n"
                "the segmenter's confidence, so a grazing wall or a cluttered floor fades out instead\n"
                "of being gated.\n"
                "★It FIRST measures whether re-running the model reproduces the stored field, and\n"
                "refuses to mix two different fields. Read the console for that number and the A/B.");
            enrich_btn->setStyleSheet(
                "QPushButton { border: 2px solid #6FA8DC; border-radius: 4px; padding: 3px 8px; }"
                "QPushButton:pressed { background-color: #6FA8DC; color: #101010; }");
            connect(enrich_btn, &QPushButton::clicked, this, [this, enrich_btn, rebuild_btn]
            {
                enrich_btn->setEnabled(false);
                if (rebuild_btn != nullptr)
                    rebuild_btn->setEnabled(false);
                const std::string err = run_depth_enrichment([enrich_btn](const std::string& s)
                { enrich_btn->setText(QString::fromStdString(s)); });
                if (not err.empty())
                {
                    // Same contract as the rebuild: on failure show the ERROR, never depth_fit_map_,
                    // which still holds the map that was already loaded.
                    enrich_btn->setText("FAILED: " + QString::fromStdString(err));
                    enrich_btn->setStyleSheet(
                        "QPushButton { border: 2px solid #E05252; border-radius: 4px; padding: 3px 8px; }");
                }
                else
                {
                    enrich_btn->setText(QString("Enriched: err %1% (%2 m) · δ1 %3%%4")
                                            .arg(100.0 * depth_fit_map_.med_rel,  0, 'f', 0)
                                            .arg(depth_fit_map_.med_abs_m,        0, 'f', 2)
                                            .arg(100.0 * depth_fit_map_.delta125, 0, 'f', 0)
                                            .arg(depth_fit_map_.ct_active ? " · ct ON" : ""));
                    enrich_btn->setStyleSheet(
                        "QPushButton { border: 2px solid #6FA8DC; border-radius: 4px; padding: 3px 8px; }"
                        "QPushButton:pressed { background-color: #6FA8DC; color: #101010; }");
                }
                enrich_btn->setEnabled(true);
                if (rebuild_btn != nullptr)
                    rebuild_btn->setEnabled(true);
            });
        }

        // Monocular model vs the ROOM BELIEF, in equirect. Same idea as the ZED window's RoomΔ, but
        // here BOTH sides are predictions — see the compose site for how to read it.
        QPushButton* rroom_btn = nullptr;
        if (params.RICOH_DEPTH_ENABLED and params.ZED_ROOM_DEPTH_ENABLED)
        {
            rroom_btn = new QPushButton("RoomΔ: OFF", ricoh_panel);
            rroom_btn->setCursor(Qt::PointingHandCursor);
            rroom_btn->setToolTip("Cycle: off → room-belief predicted range → difference vs the monocular model.\n"
                                  "NEITHER side is a measurement: this is one prediction against another.\n"
                                  "Furniture is bright by construction (the model sees it, the envelope does not);\n"
                                  "what matters is disagreement on the SHELL — a wall or floor lighting up.\n"
                                  "PREDICTED is pure geometry and needs nothing but the room belief.\n"
                                  "DIFF additionally needs Depth ON and a fitted+anchored map, since the\n"
                                  "monocular model has to be metric before it can be subtracted.");
            rroom_btn->setStyleSheet(
                "QPushButton { border: 2px solid #C0A0E0; border-radius: 4px; padding: 3px 8px; }"
                "QPushButton:pressed { background-color: #C0A0E0; color: #101010; }");
            connect(rroom_btn, &QPushButton::clicked, this, [this, rroom_btn]
            {
                ricoh_room_mode_ = (ricoh_room_mode_ + 1) % 3;
                // Symmetric with the Models/Lidar toggles: stop the per-frame scene copy once the
                // last consumer of the cache is gone. compute() re-fills it on the next tick.
                if (ricoh_room_mode_ == 0 and not ricoh_model_overlay_enabled_
                    and not ricoh_lidar_overlay_enabled_)
                    ricoh_scene_.valid = false;
                rroom_btn->setText(ricoh_room_mode_ == 0 ? "RoomΔ: OFF"
                                   : ricoh_room_mode_ == 1 ? "RoomΔ: PREDICTED" : "RoomΔ: DIFF");
            });
        }

        controls->addWidget(models_btn);
        controls->addWidget(lidar_btn);
        if (depth_btn != nullptr)
            controls->addWidget(depth_btn);
        if (rroom_btn != nullptr)
            controls->addWidget(rroom_btn);
        if (collect_btn != nullptr)
            controls->addWidget(collect_btn);
        if (rebuild_btn != nullptr)
            controls->addWidget(rebuild_btn);
        if (enrich_btn != nullptr)
            controls->addWidget(enrich_btn);
        controls->addStretch(1);

        ricoh_layout->addLayout(controls);
        ricoh_layout->addWidget(ricoh_viewer_.get(), 1);   // reparents the label into the panel

        ricoh_panel->setWindowTitle("Ricoh 360");
        ricoh_window_ = ricoh_panel;
        if (not restore_external_window_geometry(ricoh_window_, "Ricoh360Window"))
            ricoh_window_->resize(960, 500);   // half the 1920x960 panorama + control row
        // Intentionally NOT shown here — hidden until the top-bar button turns it on.
        qInfo() << __FUNCTION__ << "Ricoh 360 viewer created (hidden; toggle from the Voxel3D top bar)";
    }

    // The voxel viewer is a QOpenGLWidget; disabling it (Voxel.show_voxel_viewer=false) is the robust
    // production setting. All viewer consumers are null-guarded.
    if (params.SHOW_VOXEL_VIEWER)
    {
        auto* voxel_panel = new QWidget(nullptr);
        auto* panel_layout = new QVBoxLayout(voxel_panel);
        panel_layout->setContentsMargins(6, 6, 6, 6);
        panel_layout->setSpacing(6);

        // Two stacked rows instead of one long row: a single row of 9 buttons was setting the
        // window's minimum width. Row 1 = perception clouds + fitted geometry, row 2 = occupancy /
        // belief layers, labels and the popup-window toggles.
        auto* controls_layout = new QVBoxLayout();
        controls_layout->setContentsMargins(0, 0, 0, 0);
        controls_layout->setSpacing(4);

        auto* controls_row1 = new QHBoxLayout();
        controls_row1->setContentsMargins(0, 0, 0, 0);
        controls_row1->setSpacing(8);

        auto* controls_row2 = new QHBoxLayout();
        controls_row2->setContentsMargins(0, 0, 0, 0);
        controls_row2->setSpacing(8);

        // ZED popup toggle — opens the ZED RGB window (YOLO seg overlay lives inside it). Created here;
        // the window itself is built (hidden) in the SHOW_YOLO_VIEWER block below, so the lambda
        // resolves yolo_window_ at click time.
        QPushButton* yolo_btn = nullptr;
        if (params.SHOW_YOLO_VIEWER)
        {
            yolo_btn = new QPushButton("ZED: OFF", voxel_panel);
            yolo_btn->setCheckable(true);
            yolo_btn->setCursor(Qt::PointingHandCursor);
        }

        // Ricoh 360 popup toggle — only when the popup window was created.
        QPushButton* ricoh_btn = nullptr;
        if (params.SHOW_RICOH_VIEWER and ricoh_window_ != nullptr)
        {
            ricoh_btn = new QPushButton("Ricoh360: OFF", voxel_panel);
            ricoh_btn->setCheckable(true);
            ricoh_btn->setCursor(Qt::PointingHandCursor);
        }

        // ★THE 3-D LAYER TOGGLES MOVED OUT WITH THE VIEW. Lidar/Models/Masks/Residual/Grid/Field/Labels
        // drove the GL widget that now lives in the `viewer3d` agent; they are its buttons, not the
        // voxelizer's. What stays here is what acts on THIS agent: the two camera windows below, whose
        // toggles also gate real work (a hidden ricoh popup stops decoding).
        // Row 2 — the camera-window toggles.
        if (yolo_btn)  controls_row2->addWidget(yolo_btn);
        if (ricoh_btn) controls_row2->addWidget(ricoh_btn);
        controls_row2->addStretch(1);

        controls_layout->addLayout(controls_row1);
        controls_layout->addLayout(controls_row2);

        panel_layout->addLayout(controls_layout);
        panel_layout->addStretch(1);

        if (yolo_btn)
            connect(yolo_btn, &QPushButton::toggled, this, [this, yolo_btn](bool checked)
            {
                if (yolo_window_)
                    yolo_window_->setVisible(checked);
                yolo_btn->setText(checked ? "ZED: ON" : "ZED: OFF");
            });

        if (ricoh_btn)
            connect(ricoh_btn, &QPushButton::toggled, this, [this, ricoh_btn](bool checked)
            {
                if (ricoh_window_)
                    ricoh_window_->setVisible(checked);
                // Gate decoding: a hidden popup only poll-discards (no clone cost).
                if (scene_processor)
                    scene_processor->set_ricoh_wanted(checked);
                ricoh_btn->setText(checked ? "Ricoh360: ON" : "Ricoh360: OFF");
            });

        // Standalone top-level window (voxel_panel was created parentless → it IS a top-level window).
        voxel_panel->setWindowTitle("Voxel3D");
        voxel3d_window_ = voxel_panel;
        // Restore the last geometry; otherwise open at ~half the default window size.
        if (not restore_external_window_geometry(voxel3d_window_, "Voxel3DWindow"))
            voxel3d_window_->resize(450, 360);
        voxel3d_window_->show();
        qInfo() << __FUNCTION__ << "Voxel3D GL viewer shown in its own window";
    }

    // YOLO viewer (raster QLabel) — independently gated.
    if (params.SHOW_YOLO_VIEWER)
    {
        yolo_viewer_ = std::make_unique<rc::YoloViewer>(nullptr);

        // Wrap the raster label in a panel so we can host a control row above it. The row always
        // carries the YOLO-silhouette and model-projection toggles; the dense semantic-seg toggle is
        // added only when the *-sem model is loaded (it drives that extra model).
        auto* yolo_panel = new QWidget(nullptr);
        auto* yolo_layout = new QVBoxLayout(yolo_panel);
        yolo_layout->setContentsMargins(6, 6, 6, 6);
        yolo_layout->setSpacing(6);

        auto* controls = new QHBoxLayout();
        controls->setContentsMargins(0, 0, 0, 0);
        controls->setSpacing(8);

        const auto accent = [](QPushButton* b, const char* hex)
        {
            b->setStyleSheet(QString(
                "QPushButton { border: 2px solid %1; border-radius: 4px; padding: 3px 8px; }"
                "QPushButton:checked { background-color: %1; color: #101010; }").arg(hex));
        };

        // YOLO seg-detection overlay toggle — starts pushed (masks/bboxes drawn by default).
        auto* yolo_overlay_btn = new QPushButton(yolo_overlay_enabled_ ? "YOLO: ON" : "YOLO: OFF", yolo_panel);
        yolo_overlay_btn->setCheckable(true);
        yolo_overlay_btn->setChecked(yolo_overlay_enabled_);
        yolo_overlay_btn->setCursor(Qt::PointingHandCursor);
        accent(yolo_overlay_btn, "#64C8FF");
        connect(yolo_overlay_btn, &QPushButton::toggled, this, [this, yolo_overlay_btn](bool checked)
        {
            yolo_overlay_enabled_ = checked;
            yolo_overlay_btn->setText(checked ? "YOLO: ON" : "YOLO: OFF");
        });

        // Model-instance projection overlay toggle — starts OFF (projects the graph model BBs on demand).
        auto* models_overlay_btn = new QPushButton(model_overlay_enabled_ ? "Models: ON" : "Models: OFF", yolo_panel);
        models_overlay_btn->setCheckable(true);
        models_overlay_btn->setChecked(model_overlay_enabled_);
        models_overlay_btn->setCursor(Qt::PointingHandCursor);
        accent(models_overlay_btn, "#FFC864");
        connect(models_overlay_btn, &QPushButton::toggled, this, [this, models_overlay_btn](bool checked)
        {
            model_overlay_enabled_ = checked;
            models_overlay_btn->setText(checked ? "Models: ON" : "Models: OFF");
        });

        controls->addWidget(yolo_overlay_btn);
        controls->addWidget(models_overlay_btn);

        // Semantic overlay: shown when the model is configured. The SemanticStage lives in the ZED worker
        // (created after this in initialize()), so the toggle reaches it lazily; the class-name table is
        // fed once the worker exists (see initialize()).
        if (params.SEMANTIC_SEG_ENABLED)
        {
            auto* sem_btn = new QPushButton(semantic_overlay_enabled_ ? "Semantic: ON" : "Semantic: OFF", yolo_panel);
            sem_btn->setCheckable(true);
            sem_btn->setChecked(semantic_overlay_enabled_);
            sem_btn->setCursor(Qt::PointingHandCursor);
            accent(sem_btn, "#64C8FF");
            connect(sem_btn, &QPushButton::toggled, this, [this, sem_btn](bool checked)
            {
                // The toggle controls ONLY the overlay display. Keep the model running whenever its output
                // is consumed downstream (publish_masks → furniture masks; publish_node → residual's semantic
                // node); it may only be gated off in pure display-only mode. (Mirrors the SAM2 toggle.)
                semantic_overlay_enabled_ = checked;
                if (zed_worker_)
                    if (auto* s = dynamic_cast<rc::SemanticStage*>(zed_worker_->stage("semantic")))
                        s->set_enabled(checked or params.SEMANTIC_PUBLISH_MASKS or params.SEMANTIC_PUBLISH_NODE);
                sem_btn->setText(checked ? "Semantic: ON" : "Semantic: OFF");
            });
            controls->addWidget(sem_btn);
        }

        // SAM2 mask-refinement overlay (magenta). Toggle gates the Sam2Stage's enabled flag → the heavy
        // 1024² encoder only runs while shown. Stage lives in the ZED worker (created after this).
        if (params.SAM2_ENABLED)
        {
            auto* sam2_btn = new QPushButton(sam2_overlay_enabled_ ? "SAM2: ON" : "SAM2: OFF", yolo_panel);
            sam2_btn->setCheckable(true);
            sam2_btn->setChecked(sam2_overlay_enabled_);
            sam2_btn->setCursor(Qt::PointingHandCursor);
            accent(sam2_btn, "#FF28DC");   // magenta — matches the overlay tint
            connect(sam2_btn, &QPushButton::toggled, this, [this, sam2_btn](bool checked)
            {
                sam2_overlay_enabled_ = checked;
                if (zed_worker_)
                    if (auto* s = dynamic_cast<rc::Sam2Stage*>(zed_worker_->stage("sam2")))
                        // Keep running if publish_refined needs it; otherwise gate the heavy encoder.
                        s->set_enabled(checked or params.SAM2_PUBLISH_REFINED);
                sam2_btn->setText(checked ? "SAM2: ON" : "SAM2: OFF");
            });
            controls->addWidget(sam2_btn);
        }

        // Model-depth vs ZED-depth overlay. NOT checkable — it CYCLES off → model → diff, because the
        // three states are a sequence a person steps through while looking, not a binary. The button
        // also drives the ZED DepthStage's enable flag, so the model only runs while being looked at.
        if (params.ZED_ROOM_DEPTH_ENABLED)
        {
            auto* zdepth_btn = new QPushButton("Room\u0394: OFF", yolo_panel);
            zdepth_btn->setCursor(Qt::PointingHandCursor);
            zdepth_btn->setToolTip("Cycle: off → room-belief predicted depth → difference vs the ZED's measured depth.\n"
                                   "The room ENVELOPE only (floor/walls/ceiling), so a well-fitted room goes\n"
                                   "BLACK and furniture stays bright — that is the residual, not an error.\n"
                                   "Red = belief predicts farther than measured, blue = nearer.\n"
                                   "Hover to read predicted, measured and their difference at a pixel.");
            accent(zdepth_btn, "#7ED0C0");
            connect(zdepth_btn, &QPushButton::clicked, this, [this, zdepth_btn]
            {
                zed_depth_mode_ = (zed_depth_mode_ + 1) % 3;   // no model to enable — pure geometry
                zdepth_btn->setText(zed_depth_mode_ == 0 ? "Room\u0394: OFF"
                                    : zed_depth_mode_ == 1 ? "Room\u0394: PREDICTED" : "Room\u0394: DIFF");
            });
            controls->addWidget(zdepth_btn);
        }
        controls->addStretch(1);

        yolo_layout->addLayout(controls);
        yolo_layout->addWidget(yolo_viewer_.get(), 1);   // reparents the label into the panel

        yolo_panel->setWindowTitle("ZED");
        yolo_window_ = yolo_panel;   // the panel is the top-level window

        // Restore the last geometry; otherwise size the RGB window to the camera image on first frame.
        if (not restore_external_window_geometry(yolo_window_, "YOLOWindow"))
            yolo_window_needs_image_size_ = true;
        // Start HIDDEN — shown only when the "YOLO" button in the Voxel3D top bar is toggled on.
        qInfo() << __FUNCTION__ << "YOLO viewer created (hidden; toggle from the Voxel3D top bar)";
    }
}
