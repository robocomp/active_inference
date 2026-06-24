/*
 * specificworker_viewers.cpp — custom drawing-window setup (Voxel3D GL + YOLO raster).
 *
 * Split out of initialize() to keep specificworker.cpp lean. The widgets attach to
 * the DSR GUI (when present) in their OWN top-level windows, never docked into the
 * graph-viewer window — a GL surface compositing under the graph view's churn-driven
 * repaints corrupts the backing store and crashes the process.
 */

#include "specificworker.h"

#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QMainWindow>
#include <QSettings>
#include <QByteArray>

#include "voxel_processor.h"
#include "voxel_opengl_viewer.h"
#include "yolo_viewer.h"
#include "graph_publisher.h"

namespace
{
// Persist the external (own-window) viewers' geometry under the same QSettings group the
// VoxelOpenGLViewer uses for its camera state, so position/size survive restarts.
constexpr auto kWinSettingsOrg   = "RoboComp";
constexpr auto kWinSettingsApp   = "robot_concept";
constexpr auto kWinSettingsGroup = "VoxelOpenGLViewer";

bool restore_external_window_geometry(QMainWindow* win, const QString& key)
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
    settings.endGroup();
}

void SpecificWorker::setup_custom_viewers()
{
    if (graph_viewers.empty())
        return;

    // The voxel viewer is a QOpenGLWidget; disabling it (Voxel.show_voxel_viewer=false) is the robust
    // production setting (graph stays available for debugging). All viewer consumers are null-guarded.
    if (params.SHOW_VOXEL_VIEWER)
    {
        const std::string viewer_key = graph_viewers.contains("")
            ? std::string("") : graph_viewers.begin()->first;

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

        auto* lidar_voxels_btn = new QPushButton("Fuse Lidar3D: ON", voxel_panel);
        lidar_voxels_btn->setCheckable(true);
        lidar_voxels_btn->setChecked(include_lidar3d_in_voxels_);
        lidar_voxels_btn->setCursor(Qt::PointingHandCursor);

        auto* voxels_btn = new QPushButton("Voxels: ON", voxel_panel);
        voxels_btn->setCheckable(true);
        voxels_btn->setChecked(true);
        voxels_btn->setCursor(Qt::PointingHandCursor);

        auto* masks_btn = new QPushButton("Masks: OFF", voxel_panel);
        masks_btn->setCheckable(true);
        masks_btn->setCursor(Qt::PointingHandCursor);

        // table_concept point clouds — independently toggled (default ON).
        auto* candidate_btn = new QPushButton("Candidate: ON", voxel_panel);
        candidate_btn->setCheckable(true);
        candidate_btn->setChecked(true);
        candidate_btn->setCursor(Qt::PointingHandCursor);

        auto* rfe_btn = new QPushButton("RFE: ON", voxel_panel);
        rfe_btn->setCheckable(true);
        rfe_btn->setChecked(true);
        rfe_btn->setCursor(Qt::PointingHandCursor);

        auto* residual_btn = new QPushButton("Residual: ON", voxel_panel);
        residual_btn->setCheckable(true);
        residual_btn->setChecked(true);
        residual_btn->setCursor(Qt::PointingHandCursor);

        auto* clear_voxels_btn = new QPushButton("Clear Voxels", voxel_panel);
        clear_voxels_btn->setCursor(Qt::PointingHandCursor);

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
        accent(masks_btn,     "#EBEBF2");  // mask: white
        accent(candidate_btn, "#33D9FF");  // candidate: cyan
        accent(rfe_btn,       "#FF8C1A");  // rfe: orange
        accent(residual_btn,  "#F233D9");  // residual: magenta

        controls_layout->addWidget(lidar_btn);
        controls_layout->addWidget(lidar_voxels_btn);
        controls_layout->addWidget(voxels_btn);
        controls_layout->addWidget(masks_btn);
        controls_layout->addWidget(candidate_btn);
        controls_layout->addWidget(rfe_btn);
        controls_layout->addWidget(residual_btn);
        controls_layout->addWidget(clear_voxels_btn);
        controls_layout->addStretch(1);

        voxel_viewer_gl = std::make_unique<rc::VoxelOpenGLViewer>(nullptr);
        voxel_viewer_gl->load_robot_mesh("meshes/shadow.obj");

        panel_layout->addLayout(controls_layout);
        panel_layout->addWidget(voxel_viewer_gl.get(), 1);

        connect(lidar_btn, &QPushButton::toggled, this, [this, lidar_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_lidar(checked);
            lidar_btn->setText(checked ? "Lidar: ON" : "Lidar: OFF");
        });

        connect(lidar_voxels_btn, &QPushButton::toggled, this, [this, lidar_voxels_btn](bool checked)
        {
            include_lidar3d_in_voxels_ = checked;
            lidar_voxels_btn->setText(checked ? "Fuse Lidar3D: ON" : "Fuse Lidar3D: OFF");
            // Toggling only gates FUTURE lidar fusion; voxels already fused into the grid would
            // linger and keep being drawn. Purge the grid so the displayed voxel set immediately
            // reflects the new setting (it repopulates from the next RGBD/lidar frames).
            if (voxel_processor && voxel_viewer_gl)
            {
                voxel_processor->clear_state(voxel_viewer_gl.get());
                if (graph_publisher_)
                    graph_publisher_->refresh_voxels_node();
            }
        });

        connect(voxels_btn, &QPushButton::toggled, this, [this, voxels_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_voxels(checked);
            voxels_btn->setText(checked ? "Voxels: ON" : "Voxels: OFF");
        });

        connect(masks_btn, &QPushButton::toggled, this, [this, masks_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_masks(checked);
            masks_btn->setText(checked ? "Masks: ON" : "Masks: OFF");
        });

        connect(candidate_btn, &QPushButton::toggled, this, [this, candidate_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_candidate(checked);
            candidate_btn->setText(checked ? "Candidate: ON" : "Candidate: OFF");
        });

        connect(rfe_btn, &QPushButton::toggled, this, [this, rfe_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_rfe(checked);
            rfe_btn->setText(checked ? "RFE: ON" : "RFE: OFF");
        });

        connect(residual_btn, &QPushButton::toggled, this, [this, residual_btn](bool checked)
        {
            if (voxel_viewer_gl)
                voxel_viewer_gl->set_show_residual(checked);
            residual_btn->setText(checked ? "Residual: ON" : "Residual: OFF");
        });

        connect(clear_voxels_btn, &QPushButton::clicked, this, [this]
        {
            if (!voxel_processor || !voxel_viewer_gl)
                return;
            voxel_processor->clear_state(voxel_viewer_gl.get());
            if (graph_publisher_)
                graph_publisher_->refresh_voxels_node();
        });

        // Own top-level window, NOT docked into the DSR graph-viewer window. add_custom_widget_in_own_window
        // reparents voxel_panel into the holder, so voxel_panel->window() is that holder.
        graph_viewers.at(viewer_key)->add_custom_widget_in_own_window("Voxel3D", voxel_panel);
        voxel3d_window_ = qobject_cast<QMainWindow*>(voxel_panel->window());
        // Restore the last geometry; otherwise open at ~half the default window size.
        if (voxel3d_window_ != nullptr and not restore_external_window_geometry(voxel3d_window_, "Voxel3DWindow"))
            voxel3d_window_->resize(450, 360);
        qInfo() << __FUNCTION__ << "Voxel3D GL viewer attached in its own window";
    }

    // YOLO viewer (raster QLabel) — independently gated.
    if (params.SHOW_YOLO_VIEWER)
    {
        const std::string viewer_key = graph_viewers.contains("")
            ? std::string("") : graph_viewers.begin()->first;
        yolo_viewer_ = std::make_unique<rc::YoloViewer>(nullptr);
        graph_viewers.at(viewer_key)->add_custom_widget_in_own_window("YOLO", yolo_viewer_.get());
        yolo_window_ = qobject_cast<QMainWindow*>(yolo_viewer_->window());
        // Restore the last geometry; otherwise size the RGB window to the camera image on first frame.
        if (yolo_window_ != nullptr and not restore_external_window_geometry(yolo_window_, "YOLOWindow"))
            yolo_window_needs_image_size_ = true;
        qInfo() << __FUNCTION__ << "YOLO viewer attached in its own window";
    }
}
