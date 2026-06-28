#pragma once

#include <QLabel>
#include <QPixmap>
#include <opencv2/core.hpp>

#include <chrono>
#include <vector>

struct SegDetection;

namespace rc
{

class YoloViewer final : public QLabel
{
    Q_OBJECT
public:
    explicit YoloViewer(QWidget* parent = nullptr);

    // Call from compute() after each YOLO pass; thread must be Qt main thread.
    void update_frame(const cv::Mat& rgb, const std::vector<SegDetection>& detections);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    QPixmap render_frame(const cv::Mat& rgb, const std::vector<SegDetection>& detections) const;
    static cv::Vec3b class_color(int class_id);

    QPixmap last_pixmap_;

    // Display-rate (update_frame call rate) shown as an on-image FPS overlay. EMA over inter-call
    // intervals; 0 until the second frame.
    std::chrono::steady_clock::time_point last_frame_time_{};
    float fps_ema_ = 0.f;
};

} // namespace rc
