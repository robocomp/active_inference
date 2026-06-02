#pragma once

#include <QLabel>
#include <QPixmap>
#include <opencv2/core.hpp>

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
};

} // namespace rc
