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

#include "voxel_opengl_viewer.h"
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
            if (not checked and not ricoh_lidar_overlay_enabled_) ricoh_scene_.valid = false;
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
            if (not checked and not ricoh_model_overlay_enabled_) ricoh_scene_.valid = false;
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
                if (ricoh_worker_)
                    if (auto* s = ricoh_worker_->stage("depth"))
                        s->set_enabled(checked);   // atomic; read on the worker thread
                depth_btn->setText(checked ? "Depth: ON" : "Depth: OFF");
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
            connect(collect_btn, &QPushButton::toggled, this, [this, collect_btn](bool checked)
            {
                ricoh_depth_collect_enabled_ = checked;
                if (checked)
                    depth_collect_session_ = 0;
                else
                    std::println("[depth-collect] stopped — {} frames added this session", depth_collect_session_);
                collect_btn->setText(checked ? "Collect: REC" : "Collect: OFF");
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

        controls->addWidget(models_btn);
        controls->addWidget(lidar_btn);
        if (depth_btn != nullptr)
            controls->addWidget(depth_btn);
        if (collect_btn != nullptr)
            controls->addWidget(collect_btn);
        if (rebuild_btn != nullptr)
            controls->addWidget(rebuild_btn);
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

        auto* controls_layout = new QHBoxLayout();
        controls_layout->setContentsMargins(0, 0, 0, 0);
        controls_layout->setSpacing(8);

        auto* lidar_btn = new QPushButton("Lidar: OFF", voxel_panel);
        lidar_btn->setCheckable(true);
        lidar_btn->setCursor(Qt::PointingHandCursor);

        // Fitted models (table mesh + bottle cylinder + graph boxes) — default ON.
        auto* models_btn = new QPushButton("Models: ON", voxel_panel);
        models_btn->setCheckable(true);
        models_btn->setChecked(true);
        models_btn->setCursor(Qt::PointingHandCursor);

        auto* masks_btn = new QPushButton("Masks: OFF", voxel_panel);
        masks_btn->setCheckable(true);
        masks_btn->setCursor(Qt::PointingHandCursor);

        // table_concept residual debug cloud — toggle, default OFF.
        auto* residual_btn = new QPushButton("Residual: OFF", voxel_panel);
        residual_btn->setCheckable(true);
        residual_btn->setChecked(false);
        residual_btn->setCursor(Qt::PointingHandCursor);

        // Occupancy-grid display (residual_concept's rebuilt safety layer) — ON by default.
        auto* grid_btn = new QPushButton("Grid: ON", voxel_panel);
        grid_btn->setCheckable(true);
        grid_btn->setChecked(true);
        grid_btn->setCursor(Qt::PointingHandCursor);

        // Beta belief-field heatmap (hue=P risk, brightness=confidence) — ON by default.
        auto* field_btn = new QPushButton("Field: ON", voxel_panel);
        field_btn->setCheckable(true);
        field_btn->setChecked(true);
        field_btn->setCursor(Qt::PointingHandCursor);

        // Node-name text labels drawn in the 3D view (debug reference) — ON by default.
        auto* labels_btn = new QPushButton("Labels: ON", voxel_panel);
        labels_btn->setCheckable(true);
        labels_btn->setChecked(true);
        labels_btn->setCursor(Qt::PointingHandCursor);

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

        // Tint each point toggle with the colour of the points it shows in the viewer:
        // a coloured outline always (so the association is visible) filled when ON.
        // Voxels are multi-category (no single colour) and Fuse/Clear are actions, so
        // they keep the default style.
        auto accent = [](QPushButton* b, const char* hex)
        {
            b->setStyleSheet(QString(
                "QPushButton { border: 2px solid %1; border-radius: 4px; padding: 3px 8px; }"
                "QPushButton:checked { background-color: %1; color: #101010; }").arg(hex));
        };
        accent(lidar_btn,     "#8C9EC7");  // lidar: slate blue-gray
        accent(models_btn,    "#FFC864");  // models: table-mesh amber
        accent(masks_btn,     "#EBEBF2");  // mask: white
        accent(residual_btn,  "#2633CC");  // residual: dark blue
        accent(grid_btn,      "#CC8C0D");  // occupancy grid: amber
        accent(field_btn,     "#D64550");  // belief field: risk red
        accent(labels_btn,    "#DDDDDD");  // node-name labels: light grey
        if (yolo_btn)  accent(yolo_btn,  "#64C8FF");  // yolo: light blue
        if (ricoh_btn) accent(ricoh_btn, "#00D4BB");  // ricoh: teal

        controls_layout->addWidget(lidar_btn);
        controls_layout->addWidget(models_btn);
        controls_layout->addWidget(masks_btn);
        controls_layout->addWidget(residual_btn);
        controls_layout->addWidget(grid_btn);
        controls_layout->addWidget(field_btn);
        controls_layout->addWidget(labels_btn);
        if (yolo_btn)  controls_layout->addWidget(yolo_btn);
        if (ricoh_btn) controls_layout->addWidget(ricoh_btn);
        controls_layout->addStretch(1);

        voxel_viewer_gl = std::make_unique<rc::VoxelOpenGLViewer>(nullptr);
        voxel_viewer_gl->set_perf_log(params.PERF_LOG);   // per-paint CSV probe off unless perf logging on
        voxel_viewer_gl->load_robot_mesh("meshes/shadow.obj");
        // Furniture display meshes are no longer hardcoded here: each concept agent publishes mesh_path /
        // mesh_texture_path on its node, and the viewer loads them on demand (cached) — see update_graph_boxes.

        panel_layout->addLayout(controls_layout);
        panel_layout->addWidget(voxel_viewer_gl.get(), 1);

        connect(lidar_btn, &QPushButton::toggled, this, [this, lidar_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_lidar(checked);
            lidar_btn->setText(checked ? "Lidar: ON" : "Lidar: OFF");
        });

        connect(models_btn, &QPushButton::toggled, this, [this, models_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_models(checked);
            models_btn->setText(checked ? "Models: ON" : "Models: OFF");
        });

        connect(masks_btn, &QPushButton::toggled, this, [this, masks_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_masks(checked);
            masks_btn->setText(checked ? "Masks: ON" : "Masks: OFF");
        });

        connect(residual_btn, &QPushButton::toggled, this, [this, residual_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_residual(checked);
            residual_btn->setText(checked ? "Residual: ON" : "Residual: OFF");
        });

        connect(grid_btn, &QPushButton::toggled, this, [this, grid_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_grid(checked);
            grid_btn->setText(checked ? "Grid: ON" : "Grid: OFF");
        });

        connect(field_btn, &QPushButton::toggled, this, [this, field_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_field(checked);
            field_btn->setText(checked ? "Field: ON" : "Field: OFF");
        });

        connect(labels_btn, &QPushButton::toggled, this, [this, labels_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_labels(checked);
            labels_btn->setText(checked ? "Labels: ON" : "Labels: OFF");
        });

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
