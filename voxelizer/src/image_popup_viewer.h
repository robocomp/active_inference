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

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    QPixmap last_pixmap_;
    std::chrono::steady_clock::time_point last_frame_time_{};
    float fps_ema_ = 0.f;
};

} // namespace rc
