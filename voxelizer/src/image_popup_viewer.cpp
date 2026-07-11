#include "image_popup_viewer.h"

#include <opencv2/imgproc.hpp>
#include <QImage>
#include <QResizeEvent>

namespace rc
{

ImagePopupViewer::ImagePopupViewer(QWidget* parent)
    : QLabel(parent)
{
    setAlignment(Qt::AlignCenter);
    setMinimumSize(320, 160);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setText("Waiting for Ricoh 360 frame…");
}

void ImagePopupViewer::update_image(const cv::Mat& bgr)
{
    if (bgr.empty() || bgr.type() != CV_8UC3)
        return;

    // Display-rate EMA from the inter-call interval.
    const auto now = std::chrono::steady_clock::now();
    if (last_frame_time_.time_since_epoch().count() != 0)
    {
        const double dt_ms = std::chrono::duration<double, std::milli>(now - last_frame_time_).count();
        if (dt_ms > 0.0)
        {
            const float inst = static_cast<float>(1000.0 / dt_ms);
            fps_ema_ = (fps_ema_ > 0.f) ? (0.9f * fps_ema_ + 0.1f * inst) : inst;
        }
    }
    last_frame_time_ = now;

    // BGR → RGB for Qt, with an FPS chip drawn on a copy.
    cv::Mat canvas;
    cv::cvtColor(bgr, canvas, cv::COLOR_BGR2RGB);
    const std::string fps_text = cv::format("%.1f FPS  %dx%d", fps_ema_, canvas.cols, canvas.rows);
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    cv::rectangle(canvas, cv::Point(6, 6), cv::Point(6 + ts.width + 10, 6 + ts.height + 12),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(canvas, fps_text, cv::Point(11, 6 + ts.height + 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    const QImage qimg(canvas.data, canvas.cols, canvas.rows,
                      static_cast<int>(canvas.step), QImage::Format_RGB888);
    last_pixmap_ = QPixmap::fromImage(qimg.copy());   // .copy() detaches from canvas memory
    setPixmap(last_pixmap_.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ImagePopupViewer::resizeEvent(QResizeEvent* event)
{
    QLabel::resizeEvent(event);
    if (!last_pixmap_.isNull())
        setPixmap(last_pixmap_.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

} // namespace rc
