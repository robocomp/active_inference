#pragma once

#include <QLabel>
#include <QPixmap>
#include <opencv2/core.hpp>

#include <chrono>

class QResizeEvent;

namespace rc
{

// Minimal raster image popup (a QLabel that scales a BGR cv::Mat to fit, keeping
// aspect ratio) — used for the RGBD_360 panorama window. Mirrors YoloViewer's
// QImage→QPixmap path but carries no detections/overlays. Push frames from the Qt
// main thread only (e.g. on_render_tick).
class ImagePopupViewer final : public QLabel
{
    Q_OBJECT
public:
    explicit ImagePopupViewer(QWidget* parent = nullptr);

    // Update with a BGR (CV_8UC3) frame. Empty frames are ignored. Draws a small
    // display-rate FPS chip so a live/stalled stream is obvious at a glance.
    void update_image(const cv::Mat& bgr);

    // Hover readout: hand it the CORRECTED metric field (CV_32FC1, natural log of range in metres,
    // NaN where no view covers) and hovering reports the depth under the cursor. Pass active=false to
    // switch it off. The Mat is CLONED — it crosses no thread boundary here (both this and the
    // producer run on the GUI thread) but the caller's copy is a scratch buffer it rewrites per frame.
    void set_depth_readout(const cv::Mat& metric_log_range, bool active);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;

private:
    QPixmap last_pixmap_;
    std::chrono::steady_clock::time_point last_frame_time_{};
    float fps_ema_ = 0.f;
    cv::Mat depth_log_range_;      // CV_32FC1 panorama-sized, ln(metres)
    bool    depth_active_ = false;
};

} // namespace rc
