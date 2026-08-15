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

    // Hover readout. Both fields are CV_32FC1 natural-log RANGE IN METRES, NaN where undefined:
    //   room_log_range  — what the ROOM BELIEF predicts (the envelope ray-cast). Metric by construction.
    //   model_log_range — what the MONOCULAR model believes, and ONLY once it is anchored; uncorrected
    //                     it is a per-view relative scale and a metre reading off it would be invented.
    // Either may be empty; whichever are present are reported, and when BOTH are, so is their
    // difference (model − room, matching compose_difference's sign: + ⇒ the model reads farther).
    // They need not share a resolution — each is sampled through its own normalised coordinates.
    // Pass active=false to switch the readout off; that is also what RELEASES the cloned fields, so it
    // must be reached on every frame that draws no depth. The Mats are CLONED: they cross no thread
    // boundary (both this and the producer run on the GUI thread) but the caller's copies are scratch
    // buffers it rewrites per frame.
    void set_depth_readout(const cv::Mat& room_log_range, const cv::Mat& model_log_range, bool active);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;

private:
    QPixmap last_pixmap_;
    std::chrono::steady_clock::time_point last_frame_time_{};
    float fps_ema_ = 0.f;
    cv::Mat room_log_range_;       // CV_32FC1, ln(metres) — room-belief envelope
    cv::Mat model_log_range_;      // CV_32FC1, ln(metres) — anchored monocular model
    bool    depth_active_ = false;
};

} // namespace rc
