#pragma once

#include <QLabel>
#include <QPixmap>
#include <opencv2/core.hpp>

#include <chrono>
#include <string>
#include <vector>

class QMouseEvent;
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

    // Semantic hover-readout: the ADE20K-150 class-id table (set once) and the per-frame dense label
    // map (CV_8UC1, image resolution). When `active`, hovering shows the class name under the cursor.
    void set_class_names(std::vector<std::string> names) { class_names_ = std::move(names); }
    void update_semantic(const cv::Mat& labels, bool active);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;

private:
    QPixmap render_frame(const cv::Mat& rgb, const std::vector<SegDetection>& detections) const;
    static cv::Vec3b class_color(int class_id);

    QPixmap last_pixmap_;

    // Semantic hover-readout state.
    cv::Mat                  semantic_labels_;   // CV_8UC1 dense class-id map (clone of the last frame)
    bool                     semantic_active_ = false;
    std::vector<std::string> class_names_;

    // Display-rate (update_frame call rate) shown as an on-image FPS overlay. EMA over inter-call
    // intervals; 0 until the second frame.
    std::chrono::steady_clock::time_point last_frame_time_{};
    float fps_ema_ = 0.f;
};

} // namespace rc
